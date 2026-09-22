"""FastAPI backend — all routes, SSE streaming, background tasks."""
import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sqlalchemy import cast, select
from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.database import AsyncSessionLocal, get_db
from backend.memory.constraint_tracker import (
    deactivate_constraint,
    extract_and_save_constraints,
    get_session_constraints,
)
from backend.memory.factual_compressor import maybe_compress
from backend.memory.obsidian_writer import write_note
from backend.memory.session_store import (
    delete_session,
    get_or_create_session,
    get_session_messages,
    list_sessions,
    load_session_context,
    save_message,
)
from backend.models import Constraint, Document, EvaluationResult, EvaluationRun, Message
from backend.retrieval.cbr_enhancer import augment_query, seed_case_library
from backend.retrieval.hybrid_retriever import retrieve
from backend.routing.moe_generator import generate_response
from backend.routing.self_router import route
from backend.schemas import (
    ChatRequest,
    ConstraintDelete,
    ConstraintResponse,
    DocumentResponse,
    EvalResultResponse,
    EvalRunResponse,
    HealthStatus,
    MessageResponse,
    SessionCreate,
    SessionResponse,
)

app = FastAPI(title="Legal RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Seed the CBR case library on app startup, logging but not failing on error."""
    try:
        seed_case_library()
        print("[startup] Case library seeded")
    except Exception as exc:
        print(f"[startup] Warning: could not seed case library: {exc}")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Ping Qdrant, Postgres, Ollama and the compute cluster and report their reachability."""
    # Qdrant
    qdrant_ok = "ok"
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"http://{settings.qdrant_host}:{settings.qdrant_port}/healthz")
            qdrant_ok = "ok" if r.status_code == 200 else "error"
    except Exception:
        qdrant_ok = "error"

    # Postgres
    pg_ok = "ok"
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(select(1))
    except Exception:
        pg_ok = "error"

    # Ollama
    ollama_ok = "ok"
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_ok = "ok" if r.status_code == 200 else "error"
    except Exception:
        ollama_ok = "error"

    # Cluster (squeue ping)
    cluster_ok = "unknown"
    try:
        import subprocess
        # `squeue --version` is a cheap way to confirm SSH + Slurm are reachable
        # without submitting an actual job just to test connectivity.
        r = subprocess.run(
            ["ssh", settings.cluster_ssh_alias, "squeue --version"],
            capture_output=True, text=True, timeout=5,
        )
        cluster_ok = "ok" if r.returncode == 0 else "unreachable"
    except Exception:
        cluster_ok = "unreachable"

    return HealthStatus(
        qdrant=qdrant_ok, postgres=pg_ok, ollama=ollama_ok, cluster=cluster_ok
    )


# ---------------------------------------------------------------------------
# Chat — SSE stream
# ---------------------------------------------------------------------------

async def _sse_event(event: str, data: Any) -> str:
    """Format a single named payload as a Server-Sent Events wire message."""
    payload = json.dumps(data) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


@app.post("/chat")
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    """Stream a chat turn end-to-end over SSE: load context, retrieve, route, generate, persist."""
    async def event_stream():
        """Drive the chat pipeline and yield SSE-formatted events as each stage completes."""
        try:
            # 1. Load or create session
            session_id = await get_or_create_session(
                req.session_id,
                user_id=req.user_id,
                title=req.title or req.message[:60],
            )

            # 2. Load session context (DB only — no LLM calls)
            ctx = await load_session_context(session_id, req.user_id)

            # 3. Load existing constraints from DB (no LLM extraction here)
            constraints = await get_session_constraints(session_id)
            for c in constraints:
                yield await _sse_event("constraint", {
                    "id": str(c.id),
                    "type": c.constraint_type,
                    "value": c.value,
                    "is_permanent": c.is_permanent,
                })

            # 4. Retrieve from both collections (hybrid RRF) — computed now since
            # has_app_docs/sources feed routing and generation, but the SSE
            # "source" events themselves aren't sent to the client until after
            # the "route" event, matching the spec's event ordering.
            sources = retrieve(req.message)
            has_app_docs = any(s.get("source_type") == "app_docs" for s in sources)

            # 5. Route (keyword scan + complexity heuristic — no LLM call)
            decision = route(req.message, has_app_docs=has_app_docs)

            # 6. Generate
            full_response_tokens: list[str] = []
            temporal_result: dict | None = None

            async for event_type, data, extra in generate_response(
                req.message, ctx, decision, sources, use_mixtral=req.use_mixtral
            ):
                if event_type == "token":
                    full_response_tokens.append(data)
                    yield await _sse_event("token", {"text": data})
                elif event_type == "route":
                    yield await _sse_event("route", {
                        "path": data,
                        "expert_role": extra.get("expert_role"),
                        "confidence": extra.get("confidence"),
                    })
                    for s in sources:
                        yield await _sse_event("source", s)
                elif event_type == "temporal_facts":
                    temporal_result = extra
                    yield await _sse_event("temporal_facts", extra)
                elif event_type == "status":
                    yield await _sse_event("status", {"message": data})
                elif event_type == "done":
                    temporal_result = (extra or {}).get("temporal_result")
                elif event_type == "error":
                    yield await _sse_event("error", {"message": data})

            full_response = "".join(full_response_tokens).strip()

            # 7. Save messages
            await save_message(session_id, "user", req.message)
            saved_msg = await save_message(
                session_id, "assistant", full_response,
                model_used="mixtral:8x7b" if req.use_mixtral else settings.model_medium,
                route_path=decision.path,
                sources=sources,
                temporal_facts=temporal_result,
            )

            # 8. Background tasks (non-blocking — don't delay the response)
            active_constraint_dicts = [
                {"type": c.constraint_type, "value": c.value, "is_permanent": c.is_permanent}
                for c in constraints
            ]
            # Constraint extraction runs after response is sent
            background_tasks.add_task(
                extract_and_save_constraints, session_id, req.message, req.user_id
            )
            # History compression — ctx.turn_count was read before this turn's
            # messages were saved, and every turn saves exactly 2 rows (user +
            # assistant), so the post-save count is ctx.turn_count + 2, not +1.
            # (+1 is always odd, since ctx.turn_count is always even, so
            # `turn_count % 10 == 0` could never fire — compression silently
            # never ran.)
            background_tasks.add_task(maybe_compress, session_id, ctx.turn_count + 2)
            # Obsidian note
            background_tasks.add_task(
                write_note,
                session_id=session_id,
                session_title=ctx.title or "Chat",
                query=req.message,
                response=full_response,
                model_used=saved_msg.model_used or "",
                route_path=decision.path,
                sources=sources,
                temporal_facts=temporal_result,
                active_constraints=active_constraint_dicts,
            )

            yield await _sse_event("saved", {"session_id": str(session_id)})
            yield await _sse_event("done", {
                "session_id": str(session_id),
                "message_id": str(saved_msg.id),
                "temporal_result": temporal_result,
            })

        except Exception as exc:
            import traceback
            traceback.print_exc()
            yield await _sse_event("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@app.get("/sessions", response_model=list[SessionResponse])
async def get_sessions(user_id: str = settings.default_user_id):
    """List all chat sessions belonging to a user."""
    sessions = await list_sessions(user_id)
    return sessions


@app.post("/sessions", response_model=SessionResponse)
async def create_session(body: SessionCreate, db: AsyncSession = Depends(get_db)):
    """Create and persist a new chat session."""
    from backend.models import Session
    sess = Session(
        user_id=body.user_id,
        title=body.title,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(sess)
    await db.commit()
    await db.refresh(sess)
    return sess


@app.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
async def get_messages(session_id: uuid.UUID):
    """Return all messages belonging to a session, in chronological order."""
    return await get_session_messages(session_id)


@app.delete("/sessions/{session_id}")
async def remove_session(session_id: uuid.UUID):
    """Delete a session and its associated messages/constraints."""
    await delete_session(session_id)
    return {"ok": True}


@app.get("/sessions/{session_id}/constraints", response_model=list[ConstraintResponse])
async def get_constraints(session_id: uuid.UUID):
    """List the active constraints currently tracked for a session."""
    return await get_session_constraints(session_id)


@app.delete("/sessions/{session_id}/constraints")
async def remove_constraint(session_id: uuid.UUID, body: ConstraintDelete):
    """Deactivate (soft-delete) a single tracked constraint."""
    await deactivate_constraint(body.constraint_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    source_type: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Save uploaded file then trigger ingestion in background."""
    project_root = Path(__file__).parent.parent
    dest_dir = project_root / "data" / source_type
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    content = await file.read()
    dest_path.write_bytes(content)

    # Run ingest in background using the venv Python
    async def _ingest():
        """Invoke ingest.py as a subprocess against the just-uploaded file."""
        import subprocess
        # Use the venv's own interpreter explicitly rather than sys.executable/PATH —
        # BackgroundTasks run in-process and can't assume the venv is on PATH here,
        # but ingest.py needs the same installed deps as the API process.
        venv_python = str(project_root / ".venv" / "bin" / "python")
        subprocess.run(
            [venv_python, str(project_root / "backend" / "ingest.py"),
             "--source", source_type, "--file", str(dest_path)],
            cwd=str(project_root),
            capture_output=False,
        )

    background_tasks.add_task(_ingest)
    return {"filename": file.filename, "source_type": source_type, "status": "ingesting"}


@app.get("/documents", response_model=list[DocumentResponse])
async def get_documents(db: AsyncSession = Depends(get_db)):
    """List all ingested documents, most recently ingested first.

    Each document is annotated with `sessions_using`: the count of distinct
    chat sessions whose messages cite it, so the frontend can warn about the
    real impact of deleting it *before* the (destructive) delete happens.
    This reuses the same read-only JSONB containment check that
    delete_document() uses afterwards to compute `sessions_deleted` — it
    just runs it against SELECTs only, with nothing removed.
    """
    result = await db.execute(select(Document).order_by(Document.ingested_at.desc()))
    docs = list(result.scalars().all())

    responses = []
    for doc in docs:
        cited = await db.execute(
            select(Message.session_id)
            .where(
                Message.sources.op("@>")(
                    cast(json.dumps([{"filename": doc.filename}]), PG_JSONB)
                )
            )
            .distinct()
        )
        responses.append(
            DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                source_type=doc.source_type,
                file_path=doc.file_path,
                chunk_count=doc.chunk_count,
                ingested_at=doc.ingested_at,
                qdrant_ids=doc.qdrant_ids,
                sessions_using=len(cited.all()),
            )
        )
    return responses


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Remove a document's vectors, file, citing sessions, and DB row.

    Each cleanup step below is wrapped in its own try/except so that a failure
    in one (e.g. Qdrant unreachable) doesn't block the others — the document
    row is still removed and the caller sees which steps actually succeeded.
    """
    from backend.models import Session as ChatSession

    doc = await db.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    filename = doc.filename
    collection = "legal_docs" if doc.source_type == "legal" else "app_docs"

    # 1. Remove vectors from Qdrant
    vectors_deleted = False
    try:
        qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        qdrant.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
            ),
        )
        vectors_deleted = True
    except Exception as exc:
        print(f"[delete] Qdrant cleanup failed for {filename}: {exc}")

    # 2. Delete physical file from disk
    file_deleted = False
    try:
        fp = Path(doc.file_path)
        if fp.exists():
            fp.unlink()
            file_deleted = True
    except Exception as exc:
        print(f"[delete] File removal failed for {doc.file_path}: {exc}")

    # 3. Find sessions whose messages cited this document and delete them
    # Deleting the document invalidates any answer that cited it, so those
    # sessions are removed too rather than left pointing at a dangling source.
    # `sources` is a JSONB array of dicts; the Postgres `@>` containment operator
    # matches rows where any element contains this filename, which plain
    # SQLAlchemy filtering on JSON columns can't express — hence the raw cast.
    cited = await db.execute(
        select(Message.session_id)
        .where(
            Message.sources.op("@>")(
                cast(json.dumps([{"filename": filename}]), PG_JSONB)
            )
        )
        .distinct()
    )
    session_ids = [row[0] for row in cited.all()]
    sessions_deleted = 0
    for sid in session_ids:
        sess = await db.get(ChatSession, sid)
        if sess:
            await db.delete(sess)   # cascade deletes messages + constraints
            sessions_deleted += 1

    # 4. Remove document row
    await db.delete(doc)
    await db.commit()

    return {
        "ok": True,
        "filename": filename,
        "vectors_deleted": vectors_deleted,
        "file_deleted": file_deleted,
        "sessions_deleted": sessions_deleted,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

_eval_running: dict[str, bool] = {}  # "{dataset}:{config}" → is running

# The finalised dataset set — the earlier 7-dataset suite
# (legalbench-rag/lexrag/chronoqa/lexglue) has been removed from the
# codebase entirely, so those names are no longer accepted here. "locomo"
# and "locomoplus" are two separate benchmarks (factual vs. cognitive
# memory), not one merged dataset — see benchmarks.py's module docstring.
VALID_EVAL_DATASETS = {"contractnli", "timeqa", "locomo", "locomoplus", "casehold", "cuad", "all"}
VALID_EVAL_CONFIGS = {
    "vanilla_rag", "long_ctx_only", "self_route_base",
    "full_system", "mixtral_cluster", "all",
}


@app.get("/evaluation/results", response_model=list[EvalResultResponse])
async def get_eval_results(db: AsyncSession = Depends(get_db), limit: int | None = None):
    """List per-sample evaluation results, most recent first, optionally capped by limit."""
    q = select(EvaluationResult).order_by(EvaluationResult.run_at.desc())
    if limit and limit > 0:
        q = q.limit(limit)
    result = await db.execute(q)
    return list(result.scalars().all())


@app.get("/evaluation/runs", response_model=list[EvalRunResponse])
async def get_eval_runs(db: AsyncSession = Depends(get_db)):
    """Final aggregated metric per (run, dataset, config, metric) — one row per
    run, not per data point."""
    result = await db.execute(
        select(EvaluationRun).order_by(EvaluationRun.run_at.desc())
    )
    return list(result.scalars().all())


@app.post("/evaluation/run")
async def run_evaluation(
    background_tasks: BackgroundTasks,
    dataset: str = "contractnli",
    config: str = "all",
    n: int | None = None,
):
    """Trigger a benchmark run in the background.

    Query params
    ------------
    dataset : contractnli | timeqa | locomo | locomoplus | casehold | cuad | all
    config  : vanilla_rag | long_ctx_only | self_route_base | full_system | mixtral_cluster | all
    n       : max samples per dataset (omit for all)
    """
    if dataset not in VALID_EVAL_DATASETS:
        raise HTTPException(400, f"dataset must be one of {VALID_EVAL_DATASETS}")
    if config not in VALID_EVAL_CONFIGS:
        raise HTTPException(400, f"config must be one of {VALID_EVAL_CONFIGS}")

    run_key = f"{dataset}:{config}"
    if _eval_running.get(run_key):
        return {"status": "already_running", "dataset": dataset, "config": config}

    _eval_running[run_key] = True  # mark before background task so status poll sees it immediately
    run_id = str(uuid.uuid4())

    async def _run():
        """Run the requested benchmark(s)/config(s) and import results into Postgres."""
        try:
            from backend.evaluation.benchmarks import (
                run_benchmark, import_to_postgres,
                CONFIGS as ALL_CONFIGS, ALL_DATASETS,
            )
            datasets = ALL_DATASETS if dataset == "all" else [dataset]
            configs  = ALL_CONFIGS  if config  == "all" else [config]
            all_rows: list[dict] = []
            for ds in datasets:
                rows = run_benchmark(ds, configs=configs, run_key=run_key, limit=n)
                all_rows.extend(rows)
            if all_rows:
                await import_to_postgres(all_rows, run_id=run_id)
            print(f"[eval] Finished {run_key}: {len(all_rows)} rows (run {run_id})")
        except Exception as exc:
            print(f"[eval] run failed ({run_key}): {exc}")
        finally:
            _eval_running[run_key] = False

    background_tasks.add_task(_run)
    return {"status": "started", "dataset": dataset, "config": config, "n": n, "run_id": run_id}


@app.get("/evaluation/status")
async def eval_status():
    """Report which benchmark runs are currently in progress.

    There is no dataset-download endpoint — data/eval/*.json is populated
    manually (see backend/evaluation/download_datasets.py and the
    _load_contractnli/_load_timeqa docstrings in benchmarks.py), not via
    an in-app feature.
    """
    return {"running": [k for k, v in _eval_running.items() if v]}


@app.post("/evaluation/stop")
async def stop_evaluation(dataset: str, config: str = "all"):
    """Request cancellation of a running benchmark.  Checked between samples."""
    from backend.evaluation.benchmarks import request_stop
    run_key = f"{dataset}:{config}"
    if not _eval_running.get(run_key):
        return {"status": "not_running", "dataset": dataset, "config": config}
    request_stop(run_key)
    return {"status": "stop_requested", "dataset": dataset, "config": config}


@app.delete("/evaluation/results")
async def delete_eval_results(
    db: AsyncSession = Depends(get_db),
    dataset: str | None = None,
):
    """Bulk-delete evaluation results.

    Query params
    ------------
    dataset : if given, only delete results for that dataset; omit to delete all.
    """
    from sqlalchemy import delete as sa_delete
    q = sa_delete(EvaluationResult)
    q_runs = sa_delete(EvaluationRun)
    if dataset and dataset != "all":
        q = q.where(EvaluationResult.dataset == dataset)
        q_runs = q_runs.where(EvaluationRun.dataset == dataset)
    result = await db.execute(q)
    await db.execute(q_runs)
    await db.commit()
    return {"deleted": result.rowcount, "dataset": dataset or "all"}
