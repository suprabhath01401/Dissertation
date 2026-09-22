"""
Generate a dissertation-level PDF documentation for the Legal RAG system.
Run: python generate_dissertation_doc.py
Output: Legal_RAG_System_Documentation.pdf
"""
import sys
from pathlib import Path
from fpdf import FPDF

OUT = Path(__file__).parent / "Legal_RAG_System_Documentation.pdf"

# ── helpers ──────────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    """Replace common non-latin1 characters so core fonts don't crash."""
    replacements = {
        "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--", "\u00e9": "e", "\u00ef": "i",
        "\u00e0": "a", "\u00e2": "a", "\u00ea": "e", "\u00ee": "i",
        "\u00f4": "o", "\u00fb": "u", "\u00e8": "e", "\u00e7": "c",
        "\u00c9": "E", "\u00c0": "A", "\u00d4": "O", "\u00c7": "C",
        "\u00d7": "x", "\u2026": "...", "\u00b7": "*", "\u00b0": "deg",
        "\u0142": "l", "\u00f3": "o",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


class Doc(FPDF):
    TITLE_COLOR  = (26, 58, 108)    # dark navy
    HEAD1_COLOR  = (26, 58, 108)
    HEAD2_COLOR  = (52, 90, 160)
    HEAD3_COLOR  = (80, 120, 190)
    BODY_COLOR   = (30, 30, 30)
    CODE_BG      = (245, 245, 245)
    TABLE_HEAD   = (220, 230, 245)
    TABLE_ALT    = (245, 248, 255)
    ACCENT       = (70, 130, 200)

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(20, 20, 20)
        self._toc: list[tuple[int, str, int]] = []   # (level, title, page)
        self._chapter_num = 0

    # ── header / footer ──────────────────────────────────────────────────────

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 6, "Legal RAG System -- Dissertation Documentation", align="L")
        self.cell(0, 6, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(2)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 5, "University of Warwick -- Department of Computer Science", align="C")

    # ── typography helpers ────────────────────────────────────────────────────

    def h1(self, text: str):
        self._chapter_num += 1
        full = f"Chapter {self._chapter_num}: {text}"
        self._toc.append((1, full, self.page_no()))
        self.ln(4)
        self.set_fill_color(*self.TITLE_COLOR)
        self.rect(20, self.get_y(), 170, 10, style="F")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(255, 255, 255)
        self.set_xy(23, self.get_y() + 1.5)
        self.cell(164, 7, clean(full), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*self.BODY_COLOR)
        self.ln(4)

    def h2(self, text: str):
        self._toc.append((2, text, self.page_no()))
        self.ln(3)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*self.HEAD2_COLOR)
        self.cell(0, 6, clean(text), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*self.HEAD2_COLOR)
        self.line(20, self.get_y(), 120, self.get_y())
        self.ln(3)
        self.set_text_color(*self.BODY_COLOR)

    def h3(self, text: str):
        self.ln(2)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*self.HEAD3_COLOR)
        self.cell(0, 5, clean(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        self.set_text_color(*self.BODY_COLOR)

    def body(self, text: str, indent: float = 0):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*self.BODY_COLOR)
        if indent:
            self.set_x(20 + indent)
        self.multi_cell(170 - indent, 5.5, clean(text))
        self.ln(1)

    def bullet(self, text: str, level: int = 0):
        indent = 5 + level * 5
        marker = "-" if level == 0 else "+"
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*self.BODY_COLOR)
        self.set_x(20 + indent)
        self.cell(5, 5.5, marker)
        self.multi_cell(165 - indent, 5.5, clean(text))

    def code(self, text: str):
        self.set_fill_color(*self.CODE_BG)
        self.set_draw_color(200, 200, 200)
        self.set_font("Courier", "", 8)
        self.set_text_color(60, 60, 60)
        self.rect(22, self.get_y(), 166, len(text.splitlines()) * 4.5 + 4, style="FD")
        self.set_xy(24, self.get_y() + 2)
        for line in text.splitlines():
            self.cell(162, 4.5, clean(line), new_x="LMARGIN", new_y="NEXT")
            self.set_x(24)
        self.ln(3)
        self.set_text_color(*self.BODY_COLOR)

    def table(self, headers: list[str], rows: list[list[str]],
              col_widths: list[float] | None = None):
        if col_widths is None:
            w = 170 / len(headers)
            col_widths = [w] * len(headers)
        # header row
        self.set_fill_color(*self.TABLE_HEAD)
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(*self.TITLE_COLOR)
        for h, cw in zip(headers, col_widths):
            self.cell(cw, 7, clean(h), border=1, fill=True)
        self.ln()
        # data rows
        self.set_font("Helvetica", "", 8.5)
        for ri, row in enumerate(rows):
            fill = ri % 2 == 1
            self.set_fill_color(*(self.TABLE_ALT if fill else (255, 255, 255)))
            self.set_text_color(*self.BODY_COLOR)
            for cell, cw in zip(row, col_widths):
                self.cell(cw, 6, clean(str(cell)), border=1, fill=fill)
            self.ln()
        self.ln(3)

    def note(self, text: str):
        self.set_fill_color(255, 251, 235)
        self.set_draw_color(230, 180, 60)
        y = self.get_y()
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 80, 0)
        self.multi_cell(166, 5.5, clean("Note: " + text), border="LRBT", fill=True, new_x="LMARGIN")
        self.ln(2)
        self.set_text_color(*self.BODY_COLOR)

    def sep(self):
        self.ln(3)
        self.set_draw_color(220, 220, 220)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(3)


# ── title page ────────────────────────────────────────────────────────────────

def title_page(doc: Doc):
    doc.add_page()
    doc.set_fill_color(*doc.TITLE_COLOR)
    doc.rect(0, 0, 210, 60, style="F")
    doc.set_xy(20, 15)
    doc.set_font("Helvetica", "B", 22)
    doc.set_text_color(255, 255, 255)
    doc.cell(170, 12, "Legal RAG System", align="C", new_x="LMARGIN", new_y="NEXT")
    doc.set_xy(20, 30)
    doc.set_font("Helvetica", "", 13)
    doc.cell(170, 8, "Adaptive Multi-Expert Retrieval-Augmented Generation", align="C",
             new_x="LMARGIN", new_y="NEXT")
    doc.set_xy(20, 42)
    doc.set_font("Helvetica", "I", 10)
    doc.cell(170, 6, "for Legal Document Question Answering", align="C")

    doc.set_text_color(*doc.BODY_COLOR)
    doc.set_xy(20, 75)
    doc.set_font("Helvetica", "", 10)
    lines = [
        ("University of Warwick", "B"),
        ("Department of Computer Science", ""),
        ("", ""),
        ("Dissertation Technical Documentation", "B"),
        ("MSc / Final Year Project", ""),
        ("", ""),
        ("April 2026", "I"),
    ]
    for text, style in lines:
        doc.set_font("Helvetica", style, 10)
        doc.set_x(20)
        doc.cell(170, 7, text, align="C", new_x="LMARGIN", new_y="NEXT")

    # architecture summary box
    doc.set_xy(30, 135)
    doc.set_fill_color(240, 245, 255)
    doc.set_draw_color(*doc.HEAD2_COLOR)
    doc.rect(30, 135, 150, 80, style="FD")
    doc.set_xy(33, 139)
    doc.set_font("Helvetica", "B", 9)
    doc.set_text_color(*doc.HEAD2_COLOR)
    doc.cell(144, 6, "System at a Glance", new_x="LMARGIN", new_y="NEXT")
    doc.set_xy(33, doc.get_y() + 1)
    items = [
        "Backend  : FastAPI + SQLAlchemy + Qdrant",
        "Frontend : React 18 + TypeScript + Vite + Tailwind CSS",
        "Models   : llama3.2:3b / llama3.1:8b (local) + Mixtral 8x7B (cluster)",
        "Retrieval: Hybrid RRF (dense nomic-embed-text + sparse BM25)",
        "Chunking : SAC (Summary-Augmented Chunking) + per-page tracking",
        "Memory   : Postgres constraints + session summaries + Obsidian vault",
        "Temporal : LLM extraction + pure-Python parallel/series circuits",
        "Cluster  : Warwick DCS (kudu-taught) -- Slurm + HuggingFace transformers",
        "Eval     : 5 configs x 5 metrics x 4 benchmark datasets",
    ]
    doc.set_font("Courier", "", 8)
    doc.set_text_color(40, 40, 40)
    for item in items:
        doc.set_x(34)
        doc.cell(140, 5, item, new_x="LMARGIN", new_y="NEXT")


# ── chapter content ────────────────────────────────────────────────────────────

def ch_introduction(doc: Doc):
    doc.add_page()
    doc.h1("Introduction and Motivation")

    doc.h2("1.1 Motivation")
    doc.body(
        "Legal professionals and researchers regularly interact with large corpora of complex legal "
        "documents -- court orders, statutes, case precedents, and procedural filings. Traditional "
        "keyword search is insufficient because legal reasoning requires understanding context, "
        "temporal relationships between events, jurisdictional constraints, and the interplay of "
        "multiple clauses. Large Language Models (LLMs) offer a promising path but hallucinate "
        "unless grounded in retrieved evidence."
    )
    doc.body(
        "This system addresses these limitations by building a complete Retrieval-Augmented Generation "
        "(RAG) pipeline specifically designed for legal document question answering. It combines "
        "hybrid vector retrieval, adaptive model routing, structured temporal logic, and a persistent "
        "constraint memory -- all accessible through a streaming chat interface."
    )

    doc.h2("1.2 Problem Statement")
    doc.body(
        "Given a natural language query over a collection of legal documents, the system must:"
    )
    for item in [
        "Retrieve the most relevant passages using both semantic similarity (dense) and keyword overlap (sparse BM25)",
        "Determine the type of legal question (temporal, contractual, case-law, procedural) and assign an expert reasoning role",
        "Generate a grounded, faithful answer that cites source documents and page numbers",
        "Remember user constraints (e.g. jurisdiction, party names) across a multi-turn conversation",
        "Reason about dates, deadlines, and time-bars using structured temporal logic",
        "Dispatch computationally expensive queries to a GPU cluster (Mixtral 8x7B) while always providing a fast local response",
    ]:
        doc.bullet(item)

    doc.h2("1.3 Key Contributions")
    doc.table(
        ["Contribution", "Description"],
        [
            ["Hybrid RRF Retrieval", "Dense (768-dim nomic-embed-text) + sparse BM25 fused via Reciprocal Rank Fusion"],
            ["SAC Chunking", "Summary-Augmented Chunking prefixes each chunk with a document-level summary for better embedding context"],
            ["Per-page PDF tracking", "pymupdf extraction preserves real page numbers; short header fragments filtered at 80 chars"],
            ["Adaptive MoE Router", "Zero-cost temporal keyword scan + complexity heuristic; routes to temporal, local, or cluster"],
            ["Temporal Logic Circuits", "Pure-Python parallel (earliest/latest/conflicts) and series (deadline) circuits -- no LLM tokens"],
            ["Constraint Memory", "Permanent and session-scoped constraints extracted by llama3.2:3b, stored in Postgres, injected into every prompt"],
            ["Mixtral Integration", "Blocking cluster dispatch via Slurm + HuggingFace transformers; also selectable in chat UI"],
            ["Evaluation Framework", "5 configs x 5 metrics x 4 datasets with in-app run trigger and live bar chart dashboard"],
        ],
        [55, 115],
    )


def ch_architecture(doc: Doc):
    doc.add_page()
    doc.h1("System Architecture")

    doc.h2("2.1 High-Level Overview")
    doc.body(
        "The system follows a request-response pipeline where each chat message triggers a series of "
        "asynchronous stages. The FastAPI backend streams results back to the React frontend using "
        "Server-Sent Events (SSE), so the user sees sources, routing decisions, and tokens as they "
        "are produced -- before the full response is complete."
    )
    doc.code(
        "User query\n"
        "    |\n"
        "    v\n"
        "[1] Load session context (Postgres) -- constraints, summary, recent messages\n"
        "    |\n"
        "    v\n"
        "[2] Hybrid Retrieval (Qdrant) -- dense + sparse RRF -> top-8 chunks\n"
        "    |\n"
        "    v\n"
        "[3] Adaptive Router -- temporal / local / cluster path\n"
        "    |\n"
        "    v\n"
        "[4] Temporal Pipeline (if temporal path) -- parser + logic circuits\n"
        "    |\n"
        "    v\n"
        "[5] Build System Prompt -- 7-section priority assembly\n"
        "    |\n"
        "    v\n"
        "[6] Generate Response\n"
        "        llama3.1:8b (local, default)\n"
        "     OR Mixtral 8x7B (cluster, blocking, user-selected)\n"
        "    |\n"
        "    v\n"
        "[7] Stream tokens via SSE -> React frontend\n"
        "    |\n"
        "    v\n"
        "[8] Background tasks: constraint extraction, compression, Obsidian note"
    )

    doc.h2("2.2 Technology Stack")
    doc.table(
        ["Layer", "Technology", "Purpose"],
        [
            ["API", "FastAPI 0.111 + Uvicorn", "Async HTTP server, SSE streaming"],
            ["Database", "PostgreSQL 15 + asyncpg + SQLAlchemy 2.0", "Sessions, messages, constraints, documents, eval results"],
            ["Vector DB", "Qdrant (Docker)", "Dense + sparse vectors for legal_docs, app_docs, case_library"],
            ["Embeddings", "nomic-embed-text (768-dim, Ollama)", "Query and document embedding"],
            ["Small LLM", "llama3.2:3b (Ollama)", "Temporal extraction, constraint detection, SAC summaries"],
            ["Medium LLM", "llama3.1:8b (Ollama)", "Main response generation, history compression, eval judge"],
            ["Large LLM", "Mixtral 8x7B (Warwick cluster)", "Optional high-quality generation, cluster evaluation config"],
            ["Migrations", "Alembic", "Version-controlled Postgres schema"],
            ["PDF parsing", "pymupdf (fitz)", "Per-page text extraction with correct font decoding"],
            ["Chunking", "LangChain RecursiveCharacterTextSplitter", "512-char chunks with 50-char overlap"],
            ["Frontend", "React 18 + TypeScript + Vite + Tailwind CSS", "Chat UI, eval dashboard, document manager"],
            ["Charts", "Recharts", "Bar chart comparison in eval dashboard"],
            ["Notes", "Obsidian vault (markdown files)", "Persistent research notes from every chat turn"],
        ],
        [28, 60, 82],
    )

    doc.h2("2.3 Docker Services")
    doc.body("Two services are run via docker-compose:")
    doc.bullet("Qdrant  -- port 6333, stores all vector collections")
    doc.bullet("PostgreSQL 15  -- port 5432, stores all relational data")
    doc.body(
        "Ollama runs natively on the host machine (not in Docker) to allow direct GPU access for "
        "llama3.2:3b, llama3.1:8b, and nomic-embed-text."
    )


def ch_ingestion(doc: Doc):
    doc.add_page()
    doc.h1("Data Ingestion Pipeline")

    doc.h2("3.1 Overview")
    doc.body(
        "Documents are ingested via a CLI script (backend/ingest.py) or through the Upload button "
        "in the frontend. The pipeline extracts text per PDF page, chunks it, embeds it, and stores "
        "vectors in Qdrant alongside metadata in Postgres."
    )

    doc.h2("3.2 PDF Text Extraction")
    doc.body(
        "Early versions used pypdf which produced garbled text (character-level spacing corruption) "
        "for ICC-style PDFs that use non-standard embedded font encoding. The system was migrated to "
        "pymupdf (fitz) which uses a more robust text ordering algorithm."
    )
    doc.body("Extract pages returns a list of (page_number, text) tuples -- 1-indexed:")
    doc.code(
        "doc = fitz.open(file_path)\n"
        "pages = [(i+1, page.get_text()) for i, page in enumerate(doc)]"
    )
    doc.note(
        "This preserves actual PDF page numbers in chunk metadata, so citations like "
        "'see page 3' refer to the real PDF page, not an internal chunk index."
    )

    doc.h2("3.3 Summary-Augmented Chunking (SAC)")
    doc.body(
        "For legal documents, each chunk is prefixed with a 150-character document-level summary "
        "generated once by llama3.2:3b. This augmentation improves dense embedding quality because "
        "the embedding model can relate individual chunk text back to the document's overall topic."
    )
    doc.body("The chunking uses LangChain's create_documents API to preserve per-page metadata:")
    doc.code(
        "# Each page becomes one 'document'; chunks inherit its page number\n"
        "lang_docs = splitter.create_documents(texts, metadatas=[{'page': p} for p,_ in pages])\n"
        "# Filter out header-only fragments\n"
        "chunks = [d for d in lang_docs if len(d.page_content.strip()) >= 80]"
    )

    doc.h2("3.4 BM25 Sparse Vectors")
    doc.body(
        "Alongside dense embeddings, a BM25-style sparse vector is computed for each chunk using "
        "term-frequency weighting. Hash collisions are resolved by summing into a defaultdict, "
        "producing a 30,000-bucket sparse representation:"
    )
    doc.code(
        "bucket = defaultdict(float)\n"
        "for token, count in tf.items():\n"
        "    h = abs(hash(token)) % 30000\n"
        "    bucket[h] += count / total * math.log(1 + count)"
    )
    doc.body(
        "Both dense and sparse vectors are stored in Qdrant using named vector spaces "
        "('dense' and 'sparse') within the same collection point."
    )

    doc.h2("3.5 Qdrant Collections")
    doc.table(
        ["Collection", "Vector Type", "Contents"],
        [
            ["legal_docs", "dense (768) + sparse (BM25)", "Ingested legal PDFs -- ICC decisions, statutes, case law"],
            ["app_docs", "dense (768) + sparse (BM25)", "Application documentation, user manuals, procedural guides"],
            ["case_library", "dense (768)", "22 pre-seeded CBR cases for query augmentation"],
        ],
        [35, 55, 80],
    )

    doc.h2("3.6 Postgres Document Record")
    doc.body(
        "Each ingested file gets a row in the documents table with: filename, source_type, "
        "file_path, chunk_count, ingested_at, and qdrant_ids (JSON array of all point UUIDs). "
        "On re-ingestion, old Qdrant points are deleted by filename filter before new ones are "
        "upserted -- preventing duplicate vectors from accumulating."
    )

    doc.h2("3.7 Re-ingestion Safety")
    doc.body("When a document is re-ingested:")
    for step in [
        "Old Qdrant vectors deleted via Filter(filename == doc.filename)",
        "Fresh PDF extraction with pymupdf",
        "New SAC summary generated by llama3.2:3b",
        "New chunks embedded and upserted",
        "Postgres row updated with new chunk_count and qdrant_ids",
    ]:
        doc.bullet(f"Step {step[0]}: {step[3:]}" if step[0].isdigit() else step)


def ch_retrieval(doc: Doc):
    doc.add_page()
    doc.h1("Hybrid Retrieval System")

    doc.h2("4.1 Why Hybrid Retrieval")
    doc.body(
        "Dense-only retrieval using semantic embeddings excels at conceptual similarity but fails "
        "on specific legal terms. For example, 'amicus curiae' or 'paragraph 6(i) of the Order' "
        "are exact phrases that dense embedding may score poorly if the training distribution "
        "under-represents ICC procedural language. BM25 sparse retrieval provides exact term "
        "matching that complements semantic search."
    )
    doc.body(
        "Qdrant's native Reciprocal Rank Fusion (RRF) fuses both ranked lists into a single "
        "score without requiring calibration of score scales between the two methods."
    )

    doc.h2("4.2 Retrieval Flow")
    doc.code(
        "Query text\n"
        "   |\n"
        "   +-- embed_query() --> dense_vec (768-dim float list)\n"
        "   +-- _bm25_sparse() --> sparse_idx, sparse_val (bucket hash + TF-IDF)\n"
        "   |\n"
        "   v\n"
        "For each collection in [legal_docs, app_docs]:\n"
        "   Prefetch(dense_vec, limit=top_k*2)   -- candidate pool\n"
        "   Prefetch(SparseVector(...), limit=top_k*2)\n"
        "   FusionQuery(RRF) --> merged, limit=top_k\n"
        "   |\n"
        "   v\n"
        "Merge results from both collections\n"
        "Sort by score descending, keep top_k=8\n"
        "   |\n"
        "   v\n"
        "_lost_in_middle_reorder()  --> final list"
    )

    doc.h2("4.3 Lost-in-the-Middle Reordering")
    doc.body(
        "Research (Liu et al., 2023) shows that LLMs attend most strongly to context at the "
        "beginning and end of a prompt, with middle positions receiving less attention. To mitigate "
        "this, retrieved chunks are reordered so the highest-scoring chunk is placed first, the "
        "second-highest last, and remaining chunks fill the middle positions."
    )
    doc.code(
        "reordered[0]  = results[0]   # highest score  -> first position\n"
        "reordered[-1] = results[1]   # second score   -> last position\n"
        "reordered[1:-1] = results[2:]  # rest fill middle"
    )

    doc.h2("4.4 Case-Based Reasoning (CBR) Enhancement")
    doc.body(
        "A library of 22 legal case templates is embedded into Qdrant at startup. For each query, "
        "the top-2 most similar cases are retrieved and their 'relevant_clauses' are appended to "
        "the query before retrieval. This expands the query with domain-specific legal terminology "
        "without requiring the user to know the correct phrasing."
    )
    doc.body("Example CBR case structure:")
    doc.code(
        '{\n'
        '  "case_type": "contract_termination",\n'
        '  "query_example": "Can a party terminate the contract early?",\n'
        '  "relevant_clauses": ["termination clause", "notice period", "material breach"],\n'
        '  "resolution_template": "Review termination conditions in clause X"\n'
        '}'
    )


def ch_routing(doc: Doc):
    doc.add_page()
    doc.h1("Adaptive Query Routing (MoE Router)")

    doc.h2("5.1 Design Philosophy")
    doc.body(
        "The router must be zero-cost for the common case: no LLM call, no network round-trip. "
        "It uses a two-step heuristic: a keyword scan for temporal queries, and a complexity "
        "heuristic based on query length and hedge words for local vs cluster routing."
    )

    doc.h2("5.2 Routing Decision Tree")
    doc.code(
        "query\n"
        "  |\n"
        "  +-- contains any TEMPORAL_KEYWORDS?\n"
        "  |       Yes --> path='temporal', expert='temporal reasoning expert', confidence=1.0\n"
        "  |\n"
        "  +-- _score_confidence(query)\n"
        "        word_count > 40 OR n_complex_signals >= 2\n"
        "            --> confidence=0.5 --> path='cluster'\n"
        "        else\n"
        "            --> confidence=0.85 --> path='local'"
    )

    doc.h2("5.3 Temporal Keywords")
    doc.body(
        "The following keywords trigger the temporal reasoning path (zero LLM cost):"
    )
    keywords = [
        "deadline", "prior to", "effective date", "within", "days of",
        "statute of limitations", "before", "after", "calculated from",
        "expiry", "expires", "signed on", "commencing", "termination date",
        "notice period", "limitation period", "time-bar", "accrual",
    ]
    for i in range(0, len(keywords), 4):
        doc.bullet("  |  ".join(keywords[i:i+4]))

    doc.h2("5.4 Complexity Signals")
    doc.body("These words in a query (2 or more) trigger routing to the cluster:")
    signals = ["analyse", "analyze", "compare", "explain in detail", "what are all",
               "summarise", "summarize", "critically", "implications", "comprehensive",
               "what is the relationship", "how does", "argue"]
    doc.code("  |  ".join(signals))

    doc.h2("5.5 Expert Role Assignment")
    doc.table(
        ["Condition", "Expert Role"],
        [
            ["has_app_docs=True (app docs retrieved)", "software documentation expert"],
            ["'contract' or 'clause' in query", "contract law expert"],
            ["'case' or 'precedent' in query", "case law expert"],
            ["temporal keywords present", "temporal reasoning expert"],
            ["default", "legal assistant"],
        ],
        [90, 80],
    )


def ch_generation(doc: Doc):
    doc.add_page()
    doc.h1("Response Generation (MoE Generator)")

    doc.h2("6.1 System Prompt Assembly")
    doc.body(
        "The system prompt is assembled from up to 7 sections in strict priority order. "
        "Higher-priority sections (permanent constraints) override lower ones when the model "
        "must choose what to respect."
    )
    doc.table(
        ["Priority", "Section", "Content"],
        [
            ["1 (highest)", "[PERMANENT CONSTRAINTS]", "Cross-session rules that always apply (jurisdiction, party role, etc.)"],
            ["2", "[SESSION CONSTRAINTS]", "Temporary rules for this session only"],
            ["3", "[EXPERT ROLE]", "Role + 4 strict rules: use only retrieved context, cite sources, no prior knowledge"],
            ["4", "[CONVERSATION SUMMARY]", "Compressed factual summary of earlier turns (written by llama3.1:8b)"],
            ["5", "[RETRIEVED CONTEXT]", "Top-8 chunks with filename, page, source type labels"],
            ["6", "[TEMPORAL FACTS]", "Earliest/latest dates, deadline, conflicts from temporal circuits (if temporal path)"],
            ["7 (lowest)", "[RECENT MESSAGES]", "Last 4 messages verbatim for immediate conversational context"],
        ],
        [22, 42, 106],
    )

    doc.h2("6.2 Strict Grounding Rules")
    doc.body("Every prompt in full_system mode contains these non-negotiable instructions:")
    for rule in [
        "Answer ONLY using information from the [RETRIEVED CONTEXT] below.",
        "Do NOT use any prior knowledge or training data.",
        "If the answer is not in the retrieved context, say exactly: 'I could not find this information in the provided documents.'",
        "Always cite the source document and page number.",
    ]:
        doc.bullet(rule)

    doc.h2("6.3 Generation Paths")
    doc.table(
        ["Path", "Model", "When Used", "Latency"],
        [
            ["local (default)", "llama3.1:8b (Ollama)", "All queries unless user selects Mixtral", "5-30 seconds"],
            ["mixtral_cluster (user-selected)", "Mixtral 8x7B (HuggingFace transformers)", "User clicks Mixtral toggle in UI", "5-10 minutes"],
            ["cluster (background)", "Mixtral 8x7B", "Auto-routed complex queries -- fire-and-forget", "Not used for response"],
            ["temporal (pre-generation)", "Pure Python circuits", "Any query with temporal keywords", "< 1 second"],
        ],
        [32, 52, 55, 31],
    )

    doc.h2("6.4 Mixtral Cluster Dispatch")
    doc.body(
        "When the user selects Mixtral mode, the system runs dispatch_to_cluster_sync() in a "
        "thread pool (asyncio.to_thread) so the SSE event loop remains unblocked. The sequence:"
    )
    for step in [
        "Write payload JSON to /tmp/legalrag_cluster/input_{uuid}.json",
        "SCP input file to cluster: /dcs/pg25/u5754610/.../shared/infer_requests/",
        "Submit sbatch job: mixtral_inference.sbatch (Slurm, wmlg-ada partition)",
        "Poll squeue every 2 seconds until job disappears (up to 600 seconds)",
        "SCP output file back from cluster: shared/infer_results/output_{uuid}.json",
        "Stream response tokens back to client via SSE",
    ]:
        doc.bullet(step)


def ch_memory(doc: Doc):
    doc.add_page()
    doc.h1("Memory and Constraint System")

    doc.h2("7.1 Constraint Types")
    doc.body(
        "The system maintains two tiers of constraints that are injected into every system prompt "
        "ahead of the retrieved context. Constraints are extracted automatically from the user's "
        "messages by llama3.2:3b as a background task after each response."
    )
    doc.table(
        ["Type", "Scope", "Example", "Storage"],
        [
            ["Permanent", "All sessions, all time", "'Always apply English law'", "constraints table, is_permanent=True"],
            ["Session", "Current session only", "'The party I represent is the defendant'", "constraints table, is_permanent=False"],
        ],
        [22, 35, 60, 53],
    )

    doc.h2("7.2 Constraint Extraction")
    doc.body(
        "After each assistant response, a background task calls extract_and_save_constraints(). "
        "This function sends the user's message to llama3.2:3b with a prompt asking it to identify "
        "any constraints (jurisdiction, party role, confidentiality requirements, etc.) in JSON format. "
        "Detected constraints are saved to Postgres and streamed to the frontend as SSE events at "
        "the start of the next turn."
    )

    doc.h2("7.3 Session Context Loading")
    doc.body("At the start of each request, load_session_context() assembles:")
    doc.bullet("All active permanent constraints (across all sessions for this user)")
    doc.bullet("All active session constraints for the current session")
    doc.bullet("The last 4 messages (max_context_messages=4)")
    doc.bullet("The session summary (if compression has been triggered)")

    doc.h2("7.4 Conversation Compression")
    doc.body(
        "After every 10 turns (turn_count % 10 == 0), maybe_compress() is triggered as a "
        "background task. It sends all messages except the last 4 to llama3.1:8b with a "
        "prompt asking for a concise factual summary that preserves: legal conclusions, "
        "constraints set, key facts, and document references. The summary is stored in "
        "sessions.summary and prepended to the next prompt."
    )

    doc.h2("7.5 Obsidian Vault Writer")
    doc.body(
        "After each chat turn, a background task writes a Markdown note to the configured Obsidian "
        "vault path. Each note contains: session title, query, response, model used, routing path, "
        "source citations, temporal facts, and active constraints. This creates a persistent, "
        "searchable research trail that survives session deletion."
    )
    doc.code(
        "# Note structure\n"
        "---\n"
        "session: <title>\n"
        "date: 2026-04-21\n"
        "model: llama3.1:8b\n"
        "route: local\n"
        "---\n"
        "## Query\n"
        "<user question>\n\n"
        "## Response\n"
        "<assistant answer>\n\n"
        "## Sources\n"
        "- filename.pdf, page 3 (score 0.50)\n\n"
        "## Active Constraints\n"
        "- jurisdiction: England and Wales"
    )


def ch_temporal(doc: Doc):
    doc.add_page()
    doc.h1("Temporal Reasoning Engine")

    doc.h2("8.1 Overview")
    doc.body(
        "Legal documents frequently require temporal reasoning: computing deadlines, checking whether "
        "a limitation period has expired, or ordering events in a legal sequence. The system handles "
        "this with a two-stage pipeline: LLM-based extraction followed by pure-Python deterministic "
        "circuits. The circuits perform no LLM calls, ensuring accuracy and speed."
    )

    doc.h2("8.2 Temporal Data Extraction")
    doc.body(
        "When the router detects temporal keywords, extract_temporal_data() sends the query plus "
        "retrieved context to llama3.2:3b with a structured JSON extraction prompt. The model "
        "extracts three categories:"
    )
    doc.table(
        ["Category", "Description", "Example"],
        [
            ["explicit_dates", "ISO date strings with context", '{"date": "2025-10-08", "context": "Order issued"}'],
            ["trigger_events", "Named events with optional date", '{"event": "contract signed", "date": "2024-01-15"}'],
            ["relative_refs", "Duration or timing phrases", '"within 30 days", "prior to expiry"'],
        ],
        [30, 60, 80],
    )
    doc.body(
        "All extracted dates are validated via python-dateutil and converted to ISO-8601 format. "
        "Invalid dates are silently dropped rather than propagated as errors."
    )

    doc.h2("8.3 Parallel Circuit")
    doc.body(
        "The parallel circuit finds the temporal bounds of all explicit dates and detects conflicts "
        "with relative references:"
    )
    doc.code(
        "parallel_circuit(temporal_data) --> {\n"
        "  'earliest': '2025-10-08',\n"
        "  'latest':   '2026-04-13',\n"
        "  'span_days': 187,\n"
        "  'conflicts': ['Date span (187 days) exceeds reference within 30 days']\n"
        "}"
    )

    doc.h2("8.4 Series Circuit")
    doc.body(
        "The series circuit calculates a legal deadline by finding the earliest trigger event "
        "and adding the rule_days (extracted from relative_refs, defaulting to 30):"
    )
    doc.code(
        "series_circuit(temporal_data, rule_days=30) --> {\n"
        "  'trigger_event': 'Order issued',\n"
        "  'trigger_date':  '2025-10-08',\n"
        "  'deadline':      '2025-11-07',   # +30 days\n"
        "  'rule_applied':  '+30 days'\n"
        "}"
    )

    doc.h2("8.5 Temporal Facts in System Prompt")
    doc.body(
        "The output of both circuits is injected into the [TEMPORAL FACTS] section of the system "
        "prompt, giving the LLM pre-computed, verified date information rather than asking it to "
        "reason about dates itself -- which is a known source of hallucination."
    )


def ch_evaluation(doc: Doc):
    doc.add_page()
    doc.h1("Evaluation Framework")

    doc.h2("9.1 Purpose")
    doc.body(
        "The evaluation framework exists to provide empirical evidence that the engineering choices "
        "in this system (hybrid retrieval, SAC chunking, adaptive routing, expert prompting, session "
        "memory) produce measurably better answers than simpler baselines. It compares five system "
        "configurations across thirteen metrics on five established, English-language benchmark "
        "datasets from the legal NLP literature — one dataset per research question plus one "
        "generalisation check."
    )

    doc.h2("9.2 System Configurations")
    doc.table(
        ["Config", "Retrieval", "Prompt", "Model"],
        [
            ["vanilla_rag", "Dense-only, top-5", "Plain Context + Question", "llama3.1:8b"],
            ["long_ctx_only", "Dense-only, top-15 (all concatenated)", "Long unstructured context", "llama3.1:8b"],
            ["self_route_base", "Hybrid RRF, top-8", "Expert role + basic prompt", "llama3.1:8b"],
            ["full_system", "Hybrid RRF, top-8", "Full 7-section system prompt with strict grounding rules", "llama3.1:8b (local, Ollama)"],
            ["mixtral_cluster", "Hybrid RRF, top-8", "Full 7-section system prompt with strict grounding rules", "Mixtral 8x7B, 4-bit NF4 (Warwick cluster)"],
        ],
        [32, 40, 65, 33],
    )

    doc.h2("9.3 Evaluation Metrics")
    doc.table(
        ["Metric", "Range", "Method", "What it measures"],
        [
            ["rouge_l", "0-1", "Pure Python LCS", "Lexical overlap vs. reference; supplementary on all 5 datasets"],
            ["accuracy", "0-1", "Exact label match", "3-way NLI label (contractnli), holding letter (casehold)"],
            ["macro_f1", "0-1", "F1 averaged per class, dataset-level", "Holding classification, primary metric (casehold)"],
            ["micro_f1", "0-1", "tp/fp/fn over span sets", "Evidence-span identification (contractnli)"],
            ["exact_match", "0-1", "Exact string match", "Span extraction correctness (cuad, timeqa)"],
            ["token_f1", "0-1", "Token overlap F1", "Partial-credit span match (cuad, timeqa)"],
            ["AUPR", "0-1", "Area under P-R curve, dataset-level", "Primary metric, confidence-ranked predictions (cuad)"],
            ["precision_at_recall", "0-1", "Precision at r=0.8/0.9, dataset-level", "cuad (r=0.8, r=0.9), contractnli (r=0.8)"],
            ["judge_score", "0/0.5/1", "LLM-as-judge, 6 category prompts", "Primary metric for conversational QA (locomoplus)"],
            ["constraint_consistency", "0-1", "LLM judge yes/no", "Does the answer still respect earlier constraints? (locomoplus)"],
            ["temporal_consistency", "0-1", "Exact ISO date match fraction", "Date-arithmetic accuracy, primary (timeqa)"],
            ["perturbation_consistency", "0-1", "Answer changes under shifted year", "Robustness check, primary (timeqa)"],
            ["retrieval_recall", "0-1", "Gold-span token overlap", "Did retrieval surface the needed passage? (cuad, contractnli)"],
        ],
        [38, 14, 55, 63],
    )
    doc.body(
        "Three further metrics (keyword_accuracy, contextual_accuracy, faithfulness) remain "
        "implemented in metrics.py but are retired from standard reporting — they were anchored "
        "to an earlier, now-removed benchmark suite."
    )

    doc.h2("9.4 Benchmark Datasets")
    doc.table(
        ["Dataset", "Type", "Source", "Size", "Research question"],
        [
            ["contractnli", "NDA NLI + evidence spans", "Stanford NLP Group", "2,091 pairs (test)", "RQ1"],
            ["timeqa", "Temporal reading comprehension", "wenhuchen/Time-Sensitive-QA (GitHub)", "1,978 (989 easy + 989 hard)", "RQ3"],
            ["locomoplus", "Long-context conversational QA, 6 categories", "xjtuleeyf/Locomo-Plus (GitHub)", "2,387", "RQ2"],
            ["casehold", "Multiple-choice holding selection", "coastalcph/lex_glue[case_hold]", "2,000", "Generalisation"],
            ["cuad", "Contract clause extraction", "chenghao/cuad_qa (HuggingFace)", "500 answerable", "RQ1"],
        ],
        [24, 40, 46, 32, 27],
    )
    doc.body(
        "An earlier suite (legalbench-rag, lexrag, chronoqa, lexglue) has been removed from the "
        "codebase entirely — two were Chinese-language (a cross-lingual confound unrelated to the "
        "architecture under test), one was superseded by contractnli's fuller evidence-span "
        "annotations, and one was a redundant generalisation check once casehold already provides one."
    )

    doc.h2("9.5 Running Evaluation")
    doc.body(
        "cuad requires a one-time corpus ingestion step before its first run: "
        "python backend/evaluation/ingest_cuad_docs.py, which embeds CUAD's contracts into a "
        "dedicated eval_docs Qdrant collection so retrieval has a real corpus to search. "
        "contractnli.json and timeqa.json are not fetched by any automated downloader in this "
        "codebase — there is no dataset-download feature in the app; those two files were obtained "
        "once, manually, from their original sources and placed in data/eval/ directly."
    )
    doc.body("Evaluation can be triggered in two ways:")
    doc.bullet("From the UI: click the 'Evaluation' tab, select a dataset and config, click 'Run Evaluation'")
    doc.bullet("CLI (local): python backend/evaluation/benchmarks.py --dataset contractnli --config full_system")
    doc.body(
        "Results are saved to the evaluation_results table in Postgres and appear automatically "
        "in the bar chart dashboard. The dashboard polls /evaluation/status every 5 seconds "
        "unconditionally while a run is in progress and refreshes results when complete."
    )

    doc.h2("9.6 Expected Results")
    doc.body(
        "The hypothesis is that full_system should outperform vanilla_rag and long_ctx_only across "
        "all thirteen metrics if the design decisions (hybrid retrieval, SAC chunking, adaptive "
        "routing, structured grounding, session memory) are sound. The gap between full_system "
        "(llama3.1:8b) and mixtral_cluster (Mixtral 8x7B, 4-bit NF4) quantifies how much of the "
        "quality gap between a small and large model can be closed by better prompt engineering and "
        "retrieval alone."
    )


def ch_frontend(doc: Doc):
    doc.add_page()
    doc.h1("Frontend and API")

    doc.h2("10.1 FastAPI SSE Streaming")
    doc.body(
        "All chat responses are delivered via Server-Sent Events (SSE) rather than a single "
        "blocking response. This allows the frontend to display sources, routing decisions, "
        "and tokens as they become available, giving immediate visual feedback."
    )
    doc.table(
        ["SSE Event Type", "Payload", "When Sent"],
        [
            ["constraint", "{id, type, value, is_permanent}", "Start of request -- existing constraints from DB"],
            ["source", "{filename, page, text, score, source_type}", "One per retrieved chunk"],
            ["route", "{path, expert_role, confidence}", "After routing decision"],
            ["temporal_facts", "{parallel, series, explicit_dates, ...}", "If temporal path triggered"],
            ["status", "{message}", "Status updates (e.g. 'Submitting to Mixtral cluster...')"],
            ["token", "{text}", "One per word during generation"],
            ["saved", "{session_id}", "After messages saved to DB"],
            ["done", "{session_id, message_id}", "End of response"],
            ["error", "{message}", "On any exception"],
        ],
        [35, 60, 75],
    )

    doc.h2("10.2 React Components")
    doc.table(
        ["Component", "File", "Responsibility"],
        [
            ["App", "App.tsx", "Top-level layout, tab switching (chat/eval), model selection forwarding"],
            ["Sidebar", "Sidebar.tsx", "Session list, new chat, delete session, documents panel, memory panel"],
            ["ChatWindow", "ChatWindow.tsx", "Message list, streaming bubble, source badges, route badge"],
            ["InputBar", "InputBar.tsx", "Textarea, model toggle (llama3.1:8b / Mixtral), upload button"],
            ["ConstraintStrip", "ConstraintStrip.tsx", "Active session constraints with remove button"],
            ["DocumentsPanel", "DocumentsPanel.tsx", "Ingested documents with two-phase delete confirmation"],
            ["UploadModal", "UploadModal.tsx", "File upload modal with source_type selector"],
            ["EvalDashboard", "EvalDashboard.tsx", "Run button, bar chart, filter selectors, raw results table"],
        ],
        [28, 40, 102],
    )

    doc.h2("10.3 Model Toggle in Chat")
    doc.body(
        "The InputBar shows two buttons: 'llama3.1:8b' (blue, default) and 'Mixtral 8x7B' "
        "(purple). Selecting Mixtral changes the textarea border to purple, shows a warning "
        "'Cluster - responses take 5-10 min', and sends use_mixtral=True in the ChatRequest. "
        "The backend then blocks on dispatch_to_cluster_sync() in a thread pool while streaming "
        "a 'Submitting to Mixtral...' status message to the user."
    )

    doc.h2("10.4 Session Management")
    doc.body(
        "Sessions are stored in Postgres. The useSession hook loads all sessions on mount and "
        "provides: selectSession (loads messages + constraints via two parallel API calls), "
        "newSession (creates empty session, clears chat state), removeSession (deletes session "
        "and cascades to messages and constraints)."
    )


def ch_datasets(doc: Doc):
    doc.add_page()
    doc.h1("Datasets -- Current and Future")

    doc.h2("11.1 Currently Supported Document Types")
    doc.body(
        "The ingestion pipeline supports any PDF, Markdown (.md), or plain text (.txt) file. "
        "The source_type parameter determines which Qdrant collection is used and whether SAC "
        "augmentation is applied:"
    )
    doc.table(
        ["source_type", "Collection", "SAC", "Example Use"],
        [
            ["legal", "legal_docs", "Yes (llama3.2:3b summary)", "Court orders, ICC decisions, case filings, statutes"],
            ["app_docs", "app_docs", "No", "Software manuals, API documentation, procedural guides"],
        ],
        [25, 28, 30, 87],
    )

    doc.h2("11.2 Test Document -- ICC Reparations Case")
    doc.body(
        "The primary test document ingested during development is an ICC amicus curiae submission "
        "in case ICC-01/14-01/18 (The Prosecutor v. Alfred Yekatom and Patrice-Edouard Ngaissona). "
        "This 22-page document was submitted on 13 April 2026 by ARC-EN-CIEL (Association des "
        "Reintegres en Centrafrique) on reparations issues."
    )
    doc.table(
        ["Field", "Value"],
        [
            ["Case number", "ICC-01/14-01/18"],
            ["Document number", "ICC-01/14-01/18-2906"],
            ["Filing date", "13 April 2026"],
            ["Submitting organisation", "ARC-EN-CIEL (Association des Reintegres en Centrafrique)"],
            ["Presiding Judge", "Judge Beti Hohler"],
            ["Other Judges", "Judge Joanna Korner, Judge Keebong Paek"],
            ["Legal basis for submission", "Paragraph 6(i) of the Order of 8 October 2025, confirmed by Decision of 16 December 2025"],
            ["Pages", "22"],
            ["Chunks produced (after fix)", "~110 (filtered from 22 pages, 80-char minimum)"],
        ],
        [60, 110],
    )

    doc.h2("11.3 CBR Case Library (22 Cases)")
    doc.body(
        "The case_library.json file contains 22 pre-seeded case templates embedded into Qdrant "
        "at startup. These are used for query augmentation (CBR). Case types include:"
    )
    case_types = [
        "contract_termination", "breach_of_contract", "confidentiality_clause",
        "jurisdiction_dispute", "force_majeure", "employment_dispute",
        "intellectual_property", "tort_claim", "property_dispute", "constitutional_challenge",
        "statutory_interpretation", "international_arbitration",
    ]
    for ct in case_types:
        doc.bullet(ct.replace("_", " ").title())

    doc.h2("11.4 Evaluation Datasets")
    doc.table(
        ["Dataset", "Size", "Source", "Acquisition"],
        [
            ["contractnli", "2,091 pairs", "Stanford NLP Group", "data/eval/contractnli.json (placed manually)"],
            ["timeqa", "1,978 (989 easy + 989 hard)", "wenhuchen/Time-Sensitive-QA (GitHub)", "data/eval/timeqa.json (placed manually)"],
            ["locomoplus", "2,387", "xjtuleeyf/Locomo-Plus (GitHub)", "download_datasets.py (direct download)"],
            ["casehold", "2,000", "coastalcph/lex_glue[case_hold]", "download_datasets.py (HuggingFace)"],
            ["cuad", "500 answerable", "chenghao/cuad_qa", "download_datasets.py (HuggingFace)"],
        ],
        [24, 30, 43, 53],
    )

    doc.h2("11.5 Recommended Future Documents")
    doc.body(
        "To make the system more generally useful for legal research, the following document "
        "categories are recommended for ingestion:"
    )
    doc.table(
        ["Category", "Examples", "Source"],
        [
            ["ICC Decisions", "All publicly available ICC decisions (reparations, arrest warrants, trial)", "icc-cpi.int"],
            ["Rome Statute", "Full text of the Rome Statute (128 articles)", "UN Treaty Collection"],
            ["ICC Rules of Procedure", "Rules of Procedure and Evidence, Regulations of the Court", "icc-cpi.int"],
            ["National statutes", "UK Criminal Justice Act, EU AI Act, GDPR", "legislation.gov.uk, EUR-Lex"],
            ["Case law", "ECHR decisions, ICJ advisory opinions", "echr.coe.int, icj-cij.org"],
            ["Academic papers", "Legal commentary on ICC jurisprudence", "HeinOnline, SSRN"],
        ],
        [35, 78, 57],
    )

    doc.h2("11.6 Recommended Future Evaluation Datasets")
    doc.table(
        ["Dataset", "Focus", "Why Useful"],
        [
            ["LegalBench (full)", "17 legal reasoning tasks", "Comprehensive coverage of legal NLP tasks"],
            ["CUAD", "Contract understanding, 41 clause types", "Tests clause-level retrieval precision"],
            ["LexGLUE", "6 legal classification benchmarks", "European legal text understanding"],
            ["ContractNLI", "NLI over contract clauses", "Tests entailment reasoning over legal text"],
            ["MAUD", "M&A agreement understanding", "Detailed document-level legal QA"],
        ],
        [30, 45, 95],
    )


def ch_future(doc: Doc):
    doc.add_page()
    doc.h1("Future Work and Limitations")

    doc.h2("12.1 Current Limitations")
    doc.table(
        ["Limitation", "Impact", "Proposed Fix"],
        [
            ["Single user only", "No multi-user isolation", "Add user-scoped Qdrant namespaces + auth layer"],
            ["Mixtral requires manual setup", "Cluster eval needs conda env + model download", "Automate setup via setup_cluster.sh script"],
            ["No re-ranking", "Retrieved chunks may not be optimally ordered for the LLM", "Add cross-encoder re-ranker (bge-reranker-v2)"],
            ["chunk_size=512 chars", "Long legal paragraphs may be cut mid-sentence", "Try 1024 chars with sentence-boundary splitting"],
            ["10-min Mixtral latency", "Interactive Mixtral mode is impractical for most queries", "Explore streaming from cluster via websocket"],
            ["No citation verification", "Model cites page numbers from metadata, not from answer text", "Add post-generation citation grounding step"],
        ],
        [38, 38, 94],
    )

    doc.h2("12.2 Planned Improvements")
    for item in [
        "Cross-encoder re-ranking: Add a second-stage re-ranker (e.g. bge-reranker-v2-m3) after RRF fusion to re-score the top-8 chunks before prompt assembly",
        "Streaming cluster responses: Replace the polling+SCP approach with a WebSocket tunnel so Mixtral tokens can stream back in real time",
        "Multi-document temporal reasoning: Extend the temporal circuits to reason across multiple documents simultaneously (e.g. statute + case law)",
        "Structured output mode: Use Ollama's JSON mode for constraint extraction to reduce parsing failures from malformed LLM output",
        "Automated cluster setup script: A single setup_cluster.sh that creates the conda env, downloads the model, and verifies the pipeline end-to-end",
        "Evaluation expansion: Run the full LegalBench suite (17 tasks, ~2000 questions) using the cluster to get statistically significant results",
        "Confidence calibration: Use the retrieval scores to calibrate a confidence estimate displayed alongside each answer in the UI",
    ]:
        doc.bullet(item)
        doc.ln(1)

    doc.h2("12.3 Research Questions")
    doc.body("The system is designed to answer these dissertation-level research questions:")
    for i, q in enumerate([
        "Does hybrid RRF retrieval significantly outperform dense-only retrieval on legal QA benchmarks?",
        "How much does SAC (Summary-Augmented Chunking) improve retrieval precision vs plain chunking?",
        "Can strict prompt grounding rules (full_system config) reduce hallucination to measurably lower levels than vanilla_rag?",
        "Does the performance gap between llama3.1:8b (full_system) and Mixtral 8x7B justify the 10-minute latency cost for interactive use?",
        "Are temporal logic circuits (pure Python) more reliable than asking the LLM to reason about dates directly?",
    ], 1):
        doc.bullet(f"RQ{i}: {q}")
        doc.ln(1)


def ch_glossary(doc: Doc):
    doc.add_page()
    doc.h1("Glossary of Technical Terms")
    doc.table(
        ["Term", "Definition"],
        [
            ["RAG", "Retrieval-Augmented Generation -- grounding LLM responses in retrieved documents"],
            ["SSE", "Server-Sent Events -- unidirectional HTTP streaming from server to browser"],
            ["RRF", "Reciprocal Rank Fusion -- score fusion formula: 1/(k+rank), k=60 by default"],
            ["SAC", "Summary-Augmented Chunking -- prefixing chunks with a document-level summary"],
            ["CBR", "Case-Based Reasoning -- augmenting queries with similar past cases"],
            ["BM25", "Best Match 25 -- classic IR ranking function based on term frequency and IDF"],
            ["MoE", "Mixture of Experts -- routing queries to different models based on type/complexity"],
            ["Slurm", "Simple Linux Utility for Resource Management -- HPC job scheduler"],
            ["Qdrant", "Open-source vector database supporting dense and sparse vectors"],
            ["nomic-embed-text", "768-dimensional open embedding model, run via Ollama"],
            ["llama3.2:3b", "Meta's 3B parameter language model, used for lightweight tasks"],
            ["llama3.1:8b", "Meta's 8B parameter language model, main generation model"],
            ["Mixtral 8x7B", "Mistral AI's 46.7B sparse MoE model (8 experts x 7B), run on cluster"],
            ["Obsidian", "Markdown-based knowledge management app, used as research vault"],
            ["LCS", "Longest Common Subsequence -- used in ROUGE-L metric calculation"],
            ["ISO-8601", "International date format standard: YYYY-MM-DD"],
            ["Amicus curiae", "Latin: 'friend of the court' -- a non-party allowed to submit observations"],
        ],
        [40, 130],
    )


# ── TOC page ──────────────────────────────────────────────────────────────────

def toc_page(doc: Doc):
    doc.add_page()
    doc.set_font("Helvetica", "B", 14)
    doc.set_text_color(*doc.TITLE_COLOR)
    doc.cell(0, 10, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    doc.set_draw_color(*doc.HEAD2_COLOR)
    doc.line(20, doc.get_y(), 190, doc.get_y())
    doc.ln(4)

    for level, title, page in doc._toc:
        indent = 0 if level == 1 else 8
        size = 10 if level == 1 else 9
        style = "B" if level == 1 else ""
        doc.set_font("Helvetica", style, size)
        doc.set_text_color(*doc.BODY_COLOR)
        doc.set_x(20 + indent)
        dots = "." * max(1, int((160 - indent - len(title) * 2.2 - 15)))
        doc.cell(0, 6,
                 clean(title) + "  " + dots + f"  {page}",
                 new_x="LMARGIN", new_y="NEXT")


# ── main ──────────────────────────────────────────────────────────────────────

def build_pdf():
    doc = Doc()

    # Pass 1: render all chapters (TOC entries collected)
    title_page(doc)
    doc.add_page()  # placeholder for TOC

    ch_introduction(doc)
    ch_architecture(doc)
    ch_ingestion(doc)
    ch_retrieval(doc)
    ch_routing(doc)
    ch_generation(doc)
    ch_memory(doc)
    ch_temporal(doc)
    ch_evaluation(doc)
    ch_frontend(doc)
    ch_datasets(doc)
    ch_future(doc)
    ch_glossary(doc)

    # Write TOC into page 2 (overwrite placeholder)
    doc.page = 2
    toc_page(doc)

    doc.output(str(OUT))
    print(f"PDF written: {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build_pdf()
