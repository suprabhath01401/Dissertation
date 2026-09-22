"""MoE generator — assembles system prompt, streams locally or dispatches to cluster."""
import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langchain_ollama import OllamaLLM

from backend.config import settings
from backend.memory.session_store import SessionContext
from backend.routing.cluster_dispatcher import dispatch_to_cluster_sync
from backend.routing.self_router import RouteDecision
from backend.temporal.logic_circuits import run_temporal_pipeline
from backend.temporal.parser import extract_temporal_data

_llm_medium: OllamaLLM | None = None


def _get_llm_medium() -> OllamaLLM:
    """Lazily instantiate and cache the shared local Ollama LLM client."""
    global _llm_medium
    if _llm_medium is None:
        _llm_medium = OllamaLLM(
            model=settings.model_medium,
            base_url=settings.ollama_base_url,
        )
    return _llm_medium


def _format_constraints(constraints) -> str:
    """Render a list of constraint objects as indented bullet lines for the prompt."""
    if not constraints:
        return ""
    lines = []
    for c in constraints:
        lines.append(f"  - {c.constraint_type}: {c.value}")
    return "\n".join(lines)


def _format_sources(sources: list[dict]) -> str:
    """Render retrieved source chunks as labeled, citation-ready text blocks for the prompt."""
    if not sources:
        return "No relevant documents found."
    parts = []
    for s in sources:
        src_type = s.get("source_type", "")
        fname = s.get("filename", "")
        page = s.get("page", "")
        text = s.get("text", "")
        label = "LEGAL DOCUMENT" if src_type == "legal" else "APP DOCUMENTATION"
        parts.append(f"[{label} — {fname}, page {page}]:\n{text}")
    return "\n\n".join(parts)


def build_system_prompt(
    ctx: SessionContext,
    decision: RouteDecision,
    sources: list[dict],
    temporal_result: dict | None,
    query: str,
) -> str:
    """Assemble system prompt in priority order per Section 10 spec."""
    sections: list[str] = []

    # 1. Permanent constraints
    if ctx.permanent_constraints:
        sections.append(
            "[PERMANENT CONSTRAINTS]:\n"
            + _format_constraints(ctx.permanent_constraints)
        )

    # 2. Session constraints (non-permanent)
    session_only = [c for c in ctx.session_constraints if not c.is_permanent]
    if session_only:
        sections.append(
            "[SESSION CONSTRAINTS]:\n" + _format_constraints(session_only)
        )

    # 3. Expert role
    sections.append(
        f"[EXPERT ROLE]: You are a {decision.expert_role}.\n"
        "STRICT RULES:\n"
        "1. Answer ONLY using information from the [RETRIEVED CONTEXT] below.\n"
        "2. Do NOT use any prior knowledge or training data.\n"
        "3. If the answer is not in the retrieved context, say exactly: "
        "'I could not find this information in the provided documents.'\n"
        "4. Always cite the source document and page number."
    )

    # 4. Conversation summary
    if ctx.summary:
        sections.append(f"[CONVERSATION SUMMARY]:\n{ctx.summary}")

    # 5. Retrieved context
    sections.append(f"[RETRIEVED CONTEXT]:\n{_format_sources(sources)}")

    # 6. Temporal facts
    if temporal_result:
        tf_lines = []
        parallel = temporal_result.get("parallel", {})
        series = temporal_result.get("series", {})
        if parallel.get("earliest"):
            tf_lines.append(f"  Earliest date: {parallel['earliest']}")
        if parallel.get("latest"):
            tf_lines.append(f"  Latest date: {parallel['latest']}")
        if series.get("deadline"):
            tf_lines.append(
                f"  Deadline ({series['rule_applied']}): {series['deadline']}"
            )
        if parallel.get("conflicts"):
            tf_lines.append(f"  Conflicts: {', '.join(parallel['conflicts'])}")
        if tf_lines:
            sections.append("[TEMPORAL FACTS]:\n" + "\n".join(tf_lines))

    # 7. Recent messages
    if ctx.recent_messages:
        history_lines = []
        for m in ctx.recent_messages:
            history_lines.append(f"{m.role.upper()}: {m.content}")
        sections.append("[RECENT MESSAGES]:\n" + "\n".join(history_lines))

    return "\n\n".join(sections)


async def generate_response(
    query: str,
    ctx: SessionContext,
    decision: RouteDecision,
    sources: list[dict],
    use_mixtral: bool = False,
) -> AsyncIterator[tuple[str, str, dict | None]]:
    """
    Async generator yielding (event_type, data, extra) tuples.
    event_types: "route" | "temporal_facts" | "status" | "token" | "done" | "error"

    Event order: "route" is always yielded first, as soon as the routing
    decision is known — before temporal extraction, prompt assembly, or
    generation. "temporal_facts" follows if decision.path == "temporal".
    "status" follows if this turn is being dispatched to the cluster.
    "token" events stream the generated answer. The turn ends with either
    "done" (success) or "error" (cluster dispatch raised unexpectedly).
    """
    # Mixtral is used whenever the user explicitly toggled it on, OR the
    # router itself decided this query needs the cluster — both cases take
    # the identical dispatch_to_cluster_sync() code path.
    route_to_cluster = use_mixtral or decision.path == "cluster"

    # 1. Routing decision — yielded first, before any extraction/prompt/
    #    generation work begins.
    if use_mixtral:
        # Reported as "cluster" — the routing badge is a 3-way
        # local/cluster/temporal enum regardless of whether the cluster was
        # chosen by explicit toggle or by the router.
        yield ("route", "cluster", {"expert_role": decision.expert_role, "confidence": 1.0})
    else:
        yield ("route", decision.path, {"expert_role": decision.expert_role, "confidence": decision.confidence})

    temporal_result: dict | None = None

    # 2. Temporal path: extract + run circuits
    if decision.path == "temporal":
        combined_text = query + "\n\n" + "\n".join(s.get("text", "") for s in sources)
        td = extract_temporal_data(combined_text)
        temporal_result = run_temporal_pipeline(td)
        yield ("temporal_facts", "", temporal_result)

    system_prompt = build_system_prompt(ctx, decision, sources, temporal_result, query)
    payload = {"query": query, "system_prompt": system_prompt}

    llm = _get_llm_medium()

    if route_to_cluster:
        # Block on the cluster — runs in a thread so SSE loop stays alive
        yield ("status", "Submitting to Mixtral 8×7B on the Warwick cluster…", None)
        try:
            response = await asyncio.to_thread(
                dispatch_to_cluster_sync,
                payload,
                llm.invoke,
                600,   # 10-minute timeout for interactive Mixtral
            )
        except Exception as exc:
            yield ("error", f"Cluster error: {exc}", None)
            return
    else:
        full_prompt = f"{system_prompt}\n\nUser: {query}\n\nAssistant:"
        response = str(llm.invoke(full_prompt))

    for token in response.split():
        yield ("token", token + " ", None)
        await asyncio.sleep(0)

    yield ("done", "", {"temporal_result": temporal_result})
