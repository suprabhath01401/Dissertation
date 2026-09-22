"""Case-Based Reasoning query augmentation.

Startup: embed all query_examples from case_library.json → upsert to Qdrant.
Per query: find top-2 similar cases, append their relevant_clauses to the query.
"""
import json
import uuid
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from backend.config import settings

_CASE_LIBRARY_PATH = Path(__file__).parent.parent.parent / "data" / "case_library.json"

_qdrant: QdrantClient | None = None
_embedder: OllamaEmbeddings | None = None


def _get_qdrant() -> QdrantClient:
    """Lazily create and cache a singleton Qdrant client."""
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return _qdrant


def _get_embedder() -> OllamaEmbeddings:
    """Lazily create and cache a singleton Ollama embeddings client."""
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbeddings(
            model=settings.model_embed,
            base_url=settings.ollama_base_url,
        )
    return _embedder


def load_case_library() -> list[dict]:
    """Read and parse the case library JSON file from disk."""
    return json.loads(_CASE_LIBRARY_PATH.read_text(encoding="utf-8"))


def seed_case_library() -> None:
    """Embed query_examples and upsert to Qdrant case_library collection."""
    qdrant = _get_qdrant()
    embedder = _get_embedder()
    cases = load_case_library()

    # Ensure collection exists
    existing = {c.name for c in qdrant.get_collections().collections}
    if "case_library" not in existing:
        qdrant.create_collection(
            collection_name="case_library",
            vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)},
        )

    texts = [c["query_example"] for c in cases]
    vectors = embedder.embed_documents(texts)

    points = [
        PointStruct(
            # Deterministic id derived from case_type (not random uuid4) so
            # re-seeding overwrites the same points instead of duplicating them.
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, c["case_type"])),
            vector={"dense": vec},
            payload={
                "case_type": c["case_type"],
                "query_example": c["query_example"],
                "relevant_clauses": c["relevant_clauses"],
                "resolution_template": c["resolution_template"],
            },
        )
        for c, vec in zip(cases, vectors)
    ]
    qdrant.upsert(collection_name="case_library", points=points)
    print(f"[cbr] Seeded {len(points)} cases into case_library")


def augment_query(query: str, top_k: int = 2) -> str:
    """
    Search case_library for similar cases, append relevant_clauses to query.
    Returns the augmented query string.
    """
    qdrant = _get_qdrant()
    embedder = _get_embedder()

    vec = embedder.embed_query(query)
    hits = qdrant.query_points(
        collection_name="case_library",
        query=vec,
        using="dense",
        limit=top_k,
        with_payload=True,
    ).points

    if not hits:
        return query

    all_clauses: list[str] = []
    for hit in hits:
        clauses = hit.payload.get("relevant_clauses", [])
        all_clauses.extend(clauses)

    # Deduplicate while preserving order (seen.add() always returns None, so
    # "seen.add(c)" is falsy and never short-circuits the `in` check away).
    seen: set[str] = set()
    unique_clauses = [c for c in all_clauses if not (c in seen or seen.add(c))]

    # CBR augmentation: graft precedent clause text onto the raw user query so
    # downstream retrieval has extra lexical/semantic overlap with the legal
    # language of similar past cases, steering it toward the same provisions.
    augmented = f"{query} {' '.join(unique_clauses)}"
    return augmented
