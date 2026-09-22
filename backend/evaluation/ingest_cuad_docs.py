"""Ingest CUAD's own contracts into a dedicated eval_docs Qdrant collection,
so the cuad eval config can do open-book retrieval instead of being handed
the answer-bearing contract text directly.

NOT part of the standard CUAD flow: the finalised benchmark spec scores CUAD
closed-book (the contract excerpt is given directly with the question, same
protocol as ContractNLI/TimeQA — see backend/evaluation/benchmarks.py's
_score_cuad), so this script and the eval_docs collection it builds are
unused by default. Kept for anyone who wants to re-enable the open-book
variant; running it has no effect on a standard benchmark run.

CUAD's local cache (data/eval/cuad.json) has one row per (contract, question)
pair — many questions share the same contract. This script dedupes down to
the unique contracts and indexes each one once, so retrieval has an actual
multi-document corpus to search rather than a single trivially-matching
document.

Usage:
    python backend/evaluation/ingest_cuad_docs.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import hashlib
import json
import uuid

from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from backend.config import settings
from backend.ingest import _bm25_sparse

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "eval"
COLLECTION = "eval_docs"


def ensure_eval_docs_collection(qdrant: QdrantClient) -> None:
    """Create the eval_docs Qdrant collection (dense + sparse vectors) if it doesn't exist yet.

    A separate collection from production `legal_docs` is used deliberately, so CUAD's
    unrelated contract corpus never mixes into (and pollutes retrieval results for) the
    real legal-document collection used outside of evaluation.
    """
    existing = {c.name for c in qdrant.get_collections().collections}
    if COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)},
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
        )
        print(f"[ingest_cuad_docs] Created collection: {COLLECTION}")


def load_unique_cuad_contracts() -> list[str]:
    """Load cuad.json and dedupe its (contract, question) rows down to unique contract texts.

    cuad.json has one row per QA pair, so many rows repeat the same contract context; a
    plain set() over `context` collapses that back to the 42 distinct contracts so each is
    indexed (and embedded) exactly once instead of redundantly per question.
    """
    path = DATA_DIR / "cuad.json"
    if not path.exists():
        raise SystemExit(f"{path} not found — run download_datasets.py --datasets cuad first")
    samples = json.loads(path.read_text())
    contexts = {s["context"] for s in samples if s.get("context")}
    return sorted(contexts)


def main() -> None:
    """Ensure the eval_docs collection exists, then embed and upsert all unique CUAD contracts into it."""
    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    ensure_eval_docs_collection(qdrant)

    contracts = load_unique_cuad_contracts()
    print(f"[ingest_cuad_docs] {len(contracts)} unique CUAD contracts to index")

    # Idempotent: clear any previously-ingested cuad points before re-upserting.
    qdrant.delete(
        collection_name=COLLECTION,
        points_selector=Filter(must=[FieldCondition(key="dataset", match=MatchValue(value="cuad"))]),
    )

    embedder = OllamaEmbeddings(model=settings.model_embed, base_url=settings.ollama_base_url)
    print(f"[ingest_cuad_docs] Embedding {len(contracts)} contracts…")
    dense_vecs = embedder.embed_documents(contracts)

    points = []
    for text, dense in zip(contracts, dense_vecs):
        doc_id = hashlib.sha1(text.encode("utf-8")).hexdigest()
        sparse_idx, sparse_val = _bm25_sparse(text)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": dense, "sparse": {"indices": sparse_idx, "values": sparse_val}},
                payload={"dataset": "cuad", "doc_id": doc_id, "text": text},
            )
        )

    qdrant.upsert(collection_name=COLLECTION, points=points)
    print(f"[ingest_cuad_docs] Upserted {len(points)} contracts to {COLLECTION}")


if __name__ == "__main__":
    main()
