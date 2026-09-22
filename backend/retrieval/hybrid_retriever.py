"""Hybrid retrieval: dense + sparse → Qdrant native RRF fusion.

Queries both legal_docs and app_docs every turn, merges, reorders
(lost-in-the-middle), and tags source_type on each result.
"""
import math
from collections import Counter

from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

from backend.config import settings

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


def _bm25_sparse(text: str) -> tuple[list[int], list[float]]:
    """Build a term-frequency-weighted sparse vector (indices, values) for a query string."""
    from collections import defaultdict
    tokens = text.lower().split()
    tf = Counter(tokens)
    bucket: dict[int, float] = defaultdict(float)
    for token, count in tf.items():
        # Hash each token into a fixed-size bucket instead of maintaining a real
        # vocabulary/IDF index: this avoids a persistent term-to-id mapping that
        # would need to be built and kept in sync with the indexed corpus, at
        # the cost of occasional hash collisions between unrelated tokens.
        h = abs(hash(token)) % 30000
        bucket[h] += count * math.log(1 + count)
    return list(bucket.keys()), list(bucket.values())


def _lost_in_middle_reorder(results: list[dict]) -> list[dict]:
    """Rank-1 → index 0, rank-2 → last, remainder fill middle."""
    if len(results) <= 2:
        return results
    reordered = [None] * len(results)
    # LLMs attend less to content buried in the middle of a long context
    # ("lost in the middle"). Placing the two highest-scored chunks at the
    # very start and very end of the prompt puts them where attention is
    # strongest, instead of losing them among lower-ranked results.
    reordered[0] = results[0]
    reordered[-1] = results[1]
    middle_slots = [i for i in range(1, len(results) - 1)]
    for slot, item in zip(middle_slots, results[2:]):
        reordered[slot] = item
    return [r for r in reordered if r is not None]


def _query_collection(
    qdrant: QdrantClient,
    collection: str,
    dense_vec: list[float],
    sparse_idx: list[int],
    sparse_val: list[float],
    limit: int,
) -> list:
    """Hybrid RRF query with dense+sparse; falls back to dense-only on error."""
    try:
        # RRF fusion: independently over-fetch (limit * 2) candidates from the
        # dense and sparse prefetch legs, then let Qdrant's native Reciprocal
        # Rank Fusion combine them by rank (not raw score) so two searches on
        # different scales (cosine similarity vs. sparse dot product) can be
        # merged fairly before truncating down to the requested `limit`.
        return qdrant.query_points(
            collection_name=collection,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=limit * 2),
                Prefetch(
                    query=SparseVector(indices=sparse_idx, values=sparse_val),
                    using="sparse",
                    limit=limit * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        ).points
    except Exception as e:
        print(f"[retriever] {collection} hybrid failed ({e}), falling back to dense")
        try:
            return qdrant.query_points(
                collection_name=collection,
                query=dense_vec,
                using="dense",
                limit=limit,
                with_payload=True,
            ).points
        except Exception as e2:
            print(f"[retriever] {collection} dense fallback failed: {e2}")
            return []


def retrieve(
    query: str,
    top_k: int | None = None,
    collections: tuple[str, ...] = ("legal_docs", "app_docs"),
) -> list[dict]:
    """
    Query the given collections (legal_docs + app_docs by default) via hybrid
    RRF, merge, reorder, tag source.

    Returns list of dicts:
      {text, filename, page, chunk_index, source_type, sac_summary?, score}
    """
    top_k = top_k or settings.retrieval_top_k
    qdrant = _get_qdrant()
    embedder = _get_embedder()

    dense_vec = embedder.embed_query(query)
    sparse_idx, sparse_val = _bm25_sparse(query)

    all_results: list[dict] = []

    for collection in collections:
        hits = _query_collection(qdrant, collection, dense_vec, sparse_idx, sparse_val, top_k)
        for hit in hits:
            p = hit.payload or {}
            all_results.append(
                {
                    "text": p.get("text", ""),
                    "filename": p.get("filename", ""),
                    "page": p.get("page", 0),
                    "chunk_index": p.get("chunk_index", 0),
                    "source_type": p.get("source_type", collection.replace("_docs", "")),
                    "sac_summary": p.get("sac_summary"),
                    "score": hit.score if hit.score is not None else 0.0,
                }
            )

    # Sort merged list by score descending, keep top_k
    all_results.sort(key=lambda x: x["score"], reverse=True)
    all_results = all_results[:top_k]

    return _lost_in_middle_reorder(all_results)
