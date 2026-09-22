# Legal RAG

A modular legal RAG (Retrieval-Augmented Generation) conversational agent — hybrid retrieval, SAC chunking, adaptive routing, expert prompting, temporal reasoning, and a benchmark evaluation framework comparing multiple RAG configurations (including Warwick GPU cluster inference for Mixtral 8x7B and Llama-3.1-8B).

## Prerequisites

- **Python 3.13** (a `.venv` is expected at the project root)
- **Docker** + Docker Compose (for Postgres and Qdrant)
- **[Ollama](https://ollama.com)** running locally, with these models pulled:
  ```bash
  ollama pull llama3.2:3b
  ollama pull llama3.1:8b
  ollama pull nomic-embed-text
  ```
- **[bun](https://bun.sh)** (frontend package manager/dev server)

## 1. Clone and install

```bash
git clone <repo-url> legal_rag
cd legal_rag

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd frontend
bun install
cd ..
```

## 2. Configure environment

Copy `.env` (or create one) at the project root. Key variables — all have sane local defaults in `backend/config.py`, so a bare `.env` works for local dev:

```dotenv
DATABASE_URL=postgresql+asyncpg://postgres:legalrag@localhost:5432/legalrag
QDRANT_HOST=localhost
QDRANT_PORT=6333
OLLAMA_BASE_URL=http://localhost:11434
MODEL_SMALL=llama3.2:3b
MODEL_MEDIUM=llama3.1:8b
MODEL_EMBED=nomic-embed-text
```

Only needed if you use the Warwick GPU cluster or Obsidian note-writing features — see [Cluster inference](#cluster-inference-warwick-gpu-cluster) and `backend/config.py` for the full list (`CLUSTER_*`, `OBSIDIAN_*`, `SLURM_ACCOUNT`, etc).

## 3. Start infrastructure

```bash
docker compose up -d
```

This starts:
- **Postgres** (`localhost:5432`, db `legalrag`, user `postgres` / password `legalrag`)
- **Qdrant** (`localhost:6333`)

Apply migrations:

```bash
alembic upgrade head
```

Initialize Qdrant collections (`legal_docs`, `app_docs`, `case_library`):

```bash
python backend/ingest.py --init-collections
```

## 4. Ingest documents

Drop legal source documents (PDF, `.md`, `.txt`) into `data/legal/`, then:

```bash
python backend/ingest.py --source legal
```

Or ingest a single file:

```bash
python backend/ingest.py --source legal --file path/to/doc.pdf
```

Uploading via the frontend's Upload modal does this automatically in the background.

## 5. Run the app

**Backend** (from the project root, with `.venv` activated):
```bash
uvicorn backend.main:app --reload --port 8000
```

**Frontend** (separate terminal):
```bash
cd frontend
bun run dev
```

Open **http://localhost:5173** — Vite proxies `/api/*` to the backend on port 8000.

Check backend health directly at `http://localhost:8000/health` (reports Qdrant, Postgres, Ollama, and cluster status).

## Project layout

```
backend/
  main.py                  FastAPI app — chat, sessions, documents, evaluation routes
  config.py                Settings (env-driven, see backend/config.py for all fields)
  models.py / schemas.py   SQLAlchemy models / Pydantic schemas
  ingest.py                Document ingestion CLI (PDF/md/txt -> Qdrant)
  retrieval/                Hybrid retriever, SAC chunker, CBR case-library enhancer
  routing/                  Self-router, MoE prompt generator, cluster dispatcher
  memory/                    Session store, constraint tracker, factual compressor, Obsidian writer
  temporal/                  Temporal fact extraction + logic circuits
  evaluation/                Benchmark runner, dataset loaders, metrics
frontend/
  src/components/           React components (chat, dashboard, evaluation, uploads)
  src/api/client.ts          Backend API client
alembic/versions/           DB migrations
jobs/                        Slurm sbatch scripts for the Warwick GPU cluster
data/legal/, data/app_docs/  Ingested source documents
data/eval/                   Cached benchmark datasets (JSON)
```

## Evaluation framework

The system compares 5 RAG configurations (`vanilla_rag`, `long_ctx_only`, `self_route_base`, `full_system`, `mixtral_cluster`) across 6 benchmark datasets (`contractnli`, `timeqa`, `locomo`, `locomoplus`, `casehold`, `cuad`).

There is no in-app or CLI "download all" feature. `contractnli.json` and `timeqa.json` have no automated fetcher — see `backend/evaluation/benchmarks.py`'s `_load_contractnli`/`_load_timeqa` docstrings for their sources and expected schema, and place the files at `data/eval/contractnli.json` / `data/eval/timeqa.json`. `locomo`/`locomoplus` (from `xjtuleeyf/Locomo-Plus` on GitHub) and `casehold`/`cuad` (from HuggingFace) are already cached under `data/eval/`.

**Run an evaluation:**
```bash
# CLI
python backend/evaluation/benchmarks.py --dataset contractnli --config full_system

# Or via the UI: Evaluation tab -> select dataset/config -> Run Evaluation
```

Results are saved to Postgres (`evaluation_results` — per-sample rows, `evaluation_runs` — aggregated final score per run) and shown on the Evaluation dashboard's bar chart, Run Summary table, and per-sample table.

### Cluster inference (Warwick GPU cluster)

`mixtral_cluster` dispatches generation to the Warwick GPU cluster (kudu-taught) over SSH via Slurm, instead of running locally. Requires:

1. SSH access configured (`warwick-cluster` alias, via a `warwick-remote` jump host — see `backend/config.py`'s `cluster_*` settings and `~/.ssh/config`)
2. The project synced to the cluster:
   ```bash
   rsync -az --exclude=.env --exclude=data/ --exclude=__pycache__ --exclude=.git --exclude=.venv \
     ./ warwick-cluster:/dcs/pg25/u5754610/Desktop/test/legal_rag/
   ```
3. Mixtral weights pre-downloaded on the cluster (`sbatch jobs/download_mixtral.sbatch`)

If cluster dispatch fails or times out, `mixtral_cluster` falls back to local generation automatically.

Github link: https://github.com/suprabhath01401/Dissertation

## Troubleshooting

- **`/health` shows Ollama down**: confirm `ollama serve` is running and the models above are pulled (`ollama list`).
- **Postgres/Qdrant connection errors**: confirm `docker compose ps` shows both containers healthy, and `.env`/`config.py` ports match.
- **Alembic errors on startup**: run `alembic current` to check the applied revision, `alembic upgrade head` to bring it current.
- **Frontend can't reach backend**: confirm the backend is running on port 8000 — Vite's dev proxy (`frontend/vite.config.ts`) expects it there.
