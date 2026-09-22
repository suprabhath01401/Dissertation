"""SAC (Summary-Augmented Chunking) for legal documents only.

Each chunk is prefixed with a 150-char document summary generated once per
document by llama3.2:3b. App docs use plain chunking (no SAC).
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM

from backend.config import settings

_MIN_CHUNK_CHARS = 80   # skip header-only fragments shorter than this

_llm_small: OllamaLLM | None = None


def _get_llm() -> OllamaLLM:
    """Lazily create and cache a singleton small Ollama LLM used for summaries."""
    global _llm_small
    if _llm_small is None:
        _llm_small = OllamaLLM(
            model=settings.model_small,
            base_url=settings.ollama_base_url,
        )
    return _llm_small


def generate_sac_summary(doc_text: str) -> str:
    """Call llama3.2:3b once to produce a ≤150-char document summary."""
    llm = _get_llm()
    prompt = (
        f"Summarise this legal document in {settings.sac_summary_max_chars} "
        f"characters or fewer. Return only the summary, no preamble:\n\n"
        f"{doc_text[:3000]}"
    )
    summary = llm.invoke(prompt)
    return str(summary).strip()[: settings.sac_summary_max_chars]


def _make_splitter() -> RecursiveCharacterTextSplitter:
    """Build a recursive character splitter configured from settings."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )


def chunk_legal_document_paged(pages: list[tuple[int, str]], filename: str) -> list[dict]:
    """
    SAC chunking with accurate per-page numbering.

    Uses create_documents so each chunk inherits the page number of the PDF
    page it came from. Skips fragments shorter than _MIN_CHUNK_CHARS (headers,
    footers, page reference lines).
    """
    # Chunk-then-summarize: the whole document is summarized once (cheap, single
    # LLM call) up front, before splitting, so every resulting chunk can carry
    # the same document-level summary rather than paying for a summary per chunk.
    full_text = "\n\n".join(text for _, text in pages)
    summary = generate_sac_summary(full_text)

    splitter = _make_splitter()
    texts = [text for _, text in pages]
    metadatas = [{"page": page_num} for page_num, _ in pages]
    lang_docs = splitter.create_documents(texts, metadatas=metadatas)

    chunks = []
    chunk_index = 0
    for doc in lang_docs:
        raw = doc.page_content
        if len(raw.strip()) < _MIN_CHUNK_CHARS:
            continue
        # SAC: prefix the document summary onto each chunk's embedded text so a
        # chunk retrieved in isolation still carries whole-document context (what
        # the source document is about), improving embedding relevance for
        # chunks whose local text alone would be ambiguous out of context.
        chunks.append({
            "text": f"[DOC: {summary}] {raw}",
            "sac_summary": summary,
            "chunk_index": chunk_index,
            "page": doc.metadata.get("page", 1),
            "filename": filename,
        })
        chunk_index += 1
    return chunks


def chunk_app_document_paged(pages: list[tuple[int, str]], filename: str) -> list[dict]:
    """Plain chunking for app docs with accurate per-page numbering."""
    splitter = _make_splitter()
    texts = [text for _, text in pages]
    metadatas = [{"page": page_num} for page_num, _ in pages]
    lang_docs = splitter.create_documents(texts, metadatas=metadatas)

    chunks = []
    chunk_index = 0
    for doc in lang_docs:
        raw = doc.page_content
        if len(raw.strip()) < _MIN_CHUNK_CHARS:
            continue
        chunks.append({
            "text": raw,
            "sac_summary": None,
            "chunk_index": chunk_index,
            "page": doc.metadata.get("page", 1),
            "filename": filename,
        })
        chunk_index += 1
    return chunks


def chunk_legal_document(doc_text: str, filename: str) -> list[dict]:
    """Backward-compat wrapper."""
    return chunk_legal_document_paged([(1, doc_text)], filename)


def chunk_app_document(doc_text: str, filename: str) -> list[dict]:
    """Backward-compat wrapper."""
    return chunk_app_document_paged([(1, doc_text)], filename)
