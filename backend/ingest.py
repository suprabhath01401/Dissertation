"""
Ingest CLI — runs on the personal laptop.
Usage:
    python backend/ingest.py --source all
    python backend/ingest.py --source legal --file path/to/doc.pdf
    python backend/ingest.py --source app_docs --file path/to/doc.md
    python backend/ingest.py --init-collections   # create Qdrant collections only
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

import fitz  # pymupdf — better font/encoding handling than pypdf

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
from sqlalchemy import select
from tqdm import tqdm

from backend.config import settings
from backend.database import AsyncSessionLocal
from backend.models import Document
from backend.retrieval.sac_chunker import chunk_app_document_paged, chunk_legal_document_paged

# ---------------------------------------------------------------------------
# Qdrant client (local Docker)
# ---------------------------------------------------------------------------

QDRANT = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

EMBED_MODEL = OllamaEmbeddings(
    model=settings.model_embed,
    base_url=settings.ollama_base_url,
)

COLLECTIONS = {
    "legal_docs": {"source_type": "legal"},
    "app_docs": {"source_type": "app_docs"},
}


# ---------------------------------------------------------------------------
# Collection bootstrap
# ---------------------------------------------------------------------------

def ensure_collections() -> None:
    """Create the legal_docs, app_docs and case_library Qdrant collections if they don't already exist."""
    existing = {c.name for c in QDRANT.get_collections().collections}

    for name in ("legal_docs", "app_docs"):
        if name not in existing:
            QDRANT.create_collection(
                collection_name=name,
                vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)},
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
                },
            )
            print(f"[ingest] Created collection: {name}")

    if "case_library" not in existing:
        QDRANT.create_collection(
            collection_name="case_library",
            vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)},
        )
        print("[ingest] Created collection: case_library")


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_pages(file_path: Path) -> list[tuple[int, str]]:
    """Return list of (1-indexed page_num, text). Non-PDFs: [(1, full_text)]."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        doc = fitz.open(str(file_path))
        pages = [(i + 1, page.get_text()) for i, page in enumerate(doc)]
        doc.close()
        return pages
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return [(1, text)]


# ---------------------------------------------------------------------------
# BM25 sparse vector (simple term-frequency approach)
# ---------------------------------------------------------------------------

def _bm25_sparse(text: str) -> tuple[list[int], list[float]]:
    """TF sparse vector for Qdrant BM25-style index. Deduplicates colliding hash buckets."""
    import math
    from collections import Counter, defaultdict
    tokens = text.lower().split()
    tf = Counter(tokens)
    bucket: dict[int, float] = defaultdict(float)
    for token, count in tf.items():
        # Hash each token into one of a fixed 30k buckets rather than maintaining a global
        # vocabulary/IDF table (no corpus-wide fitting step needed at ingest time). Two
        # different tokens can collide into the same bucket; summing (+=) into `bucket`
        # instead of overwriting keeps the vector still valid, just slightly noisier.
        h = abs(hash(token)) % 30000
        bucket[h] += count * math.log(1 + count)
    indices = list(bucket.keys())
    values = list(bucket.values())
    return indices, values



# ---------------------------------------------------------------------------
# Core ingest logic
# ---------------------------------------------------------------------------

async def ingest_file(file_path: Path, source_type: str) -> None:
    """Extract, chunk, embed and upsert a single document into Qdrant, then upsert its row in Postgres."""
    filename = file_path.name
    collection = "legal_docs" if source_type == "legal" else "app_docs"

    async with AsyncSessionLocal() as session:
        existing = await session.scalar(
            select(Document).where(Document.filename == filename)
        )

    # Re-ingesting a file produces a brand-new set of chunk texts/point IDs, so the old
    # points for this filename must be purged first — otherwise Qdrant would keep the
    # stale chunks alongside the new ones (there's no upsert-by-filename in Qdrant, only
    # by point ID) and searches would return duplicate/outdated content.
    if existing:
        print(f"[ingest] Removing old vectors for {filename}…")
        QDRANT.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
            ),
        )

    print(f"[ingest] Processing {filename} ({source_type})")
    pages = extract_pages(file_path)

    if source_type == "legal":
        chunks = chunk_legal_document_paged(pages, filename)
    else:
        chunks = chunk_app_document_paged(pages, filename)

    if not chunks:
        print(f"[ingest] No chunks produced for {filename}, skipping.")
        return

    texts = [c["text"] for c in chunks]
    print(f"[ingest] Embedding {len(texts)} chunks…")
    dense_vecs = EMBED_MODEL.embed_documents(texts)

    points = []
    qdrant_ids = []
    for chunk, dense in tqdm(zip(chunks, dense_vecs), total=len(chunks)):
        point_id = str(uuid.uuid4())
        qdrant_ids.append(point_id)
        sparse_idx, sparse_val = _bm25_sparse(chunk["text"])

        payload = {
            "source_type": source_type,
            "filename": filename,
            "page": chunk["page"],          # actual PDF page number (1-indexed)
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        if chunk.get("sac_summary"):
            payload["sac_summary"] = chunk["sac_summary"]

        points.append(
            PointStruct(
                id=point_id,
                vector={"dense": dense, "sparse": {"indices": sparse_idx, "values": sparse_val}},
                payload=payload,
            )
        )

    QDRANT.upsert(collection_name=collection, points=points)
    print(f"[ingest] Upserted {len(points)} points to {collection}")

    async with AsyncSessionLocal() as db:
        if existing:
            existing.chunk_count = len(chunks)
            existing.qdrant_ids = qdrant_ids
            existing.ingested_at = datetime.now(timezone.utc)
        else:
            doc = Document(
                filename=filename,
                source_type=source_type,
                file_path=str(file_path),
                chunk_count=len(chunks),
                ingested_at=datetime.now(timezone.utc),
                qdrant_ids=qdrant_ids,
            )
            db.add(doc)
        await db.commit()
    print(f"[ingest] Postgres row saved for {filename}")


async def ingest_directory(directory: Path, source_type: str) -> None:
    """Ingest every PDF/Markdown/text file found directly in a directory, in sorted order."""
    patterns = ["*.pdf", "*.md", "*.txt"]
    files = []
    for pat in patterns:
        files.extend(directory.glob(pat))
    if not files:
        print(f"[ingest] No files found in {directory}")
        return
    for f in sorted(files):
        await ingest_file(f, source_type)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """CLI entry point: parse args and dispatch to single-file or directory ingestion."""
    parser = argparse.ArgumentParser(description="Legal RAG ingest pipeline")
    parser.add_argument("--source", choices=["legal", "app_docs", "all"], default="all")
    parser.add_argument("--file", type=Path, help="Ingest a single file")
    parser.add_argument("--init-collections", action="store_true", help="Only create Qdrant collections")
    args = parser.parse_args()

    ensure_collections()

    if args.init_collections:
        print("[ingest] Collections initialised. Done.")
        return

    project_root = Path(__file__).parent.parent

    if args.file:
        src = args.source if args.source != "all" else "legal"
        await ingest_file(args.file, src)
    elif args.source in ("legal", "all"):
        await ingest_directory(project_root / "data" / "legal", "legal")

    if not args.file and args.source in ("app_docs", "all"):
        await ingest_directory(project_root / "data" / "app_docs", "app_docs")


if __name__ == "__main__":
    asyncio.run(main())
