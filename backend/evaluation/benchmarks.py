"""Benchmark runner — compares RAG configurations across 6 benchmark datasets.

Datasets (ALL_DATASETS)
------------------------
cuad         CUAD (Contract Understanding Atticus Dataset, chenghao/cuad_qa) — 1,000
             answerable contract clause-extraction questions drawn from the test
             partition. Closed-book span extraction: the model is given a contract
             excerpt (up to 2,000 chars) directly with the question. RQ1.
             Metrics: exact_match, token_f1, rouge_l, AUPR (dataset-level aggregate,
             primary, Hendrycks et al. 2021), precision_at_recall (r=0.8, r=0.9,
             dataset-level aggregates, the paper's headline metrics)

contractnli  ContractNLI (Stanford NLP Group) — a fixed random sample of 1,000 of
             the 2,091 (document, hypothesis) NDA NLI test pairs (123 test documents
             x 17 fixed hypotheses), gold evidence-span annotations. 3-way NLI:
             entailed / contradicted / not mentioned, plus which NDA spans are
             the evidence. RQ1.
             Metrics: accuracy, precision_at_recall (r=0.8, dataset-level aggregate),
             micro_f1 (evidence set F1), rouge_l, retrieval_recall

locomo       LoCoMo (Snap Research, xjtuleeyf/Locomo-Plus's locomo10.json) — ten long
             (~600-turn) multi-session conversations; 1,000 of the 1,986 questions are
             sampled (fixed seed), spanning single_hop/multi_hop/temporal/open_domain/
             adversarial categories. Factual memory score. RQ2.
             Metrics: judge_score (category-specific LLM-judge prompts,
             correct=1/partial=0.5/wrong=0), rouge_l (supplementary)

locomoplus   LoCoMo-Plus (Li et al. 2026 extension, same GitHub repo, locomo_plus.json)
             — all 401 cue-trigger instances (category "cognitive"): a cue appears early
             in a conversation and a low-overlap trigger arrives much later; the model
             is asked what connection the earlier exchange has to the current one.
             Cognitive memory score. RQ2.
             Metrics: judge_score (LLM-judge, "cognitive" category prompt),
             constraint_consistency (LLM-judge — does the answer respect the earlier
             cue?), rouge_l and exact_match (supplementary)

timeqa       TimeQA (wenhuchen/Time-Sensitive-QA) — 1,000 questions per mode (2,000
             total) drawn from the real test partition, Easy/Hard (explicit vs.
             implicit time expression). RQ3.
             Metrics: exact_match, token_f1 (each mode scored separately, via the
             temporal_scope note field), rouge_l, perturbation_consistency (fraction
             of answers that change when the question's time expression is shifted)

casehold     CaseHOLD (US case law, via coastalcph/lex_glue[case_hold]) — 1,000
             multiple-choice holding-selection questions sampled from the ~53,000-item
             test partition. Held-out generalisation check and control: the full citing
             passage and all 5 candidates are given directly, so none of the system's
             architectural components (SAC, hybrid retrieval, routing, memory) has
             anything to act on — scores are expected to cluster across configs.
             Metrics: accuracy, macro_f1 (dataset-level aggregate, primary, across the
             5 holding classes)

Configurations (CONFIGS)
-------------------------
vanilla_rag      Dense-only retrieval, simple unstructured prompt, no routing.
long_ctx_only    Retrieve top-15 chunks, concatenate all as context, no routing.
self_route_base  Hybrid retrieval + routing, basic prompt (no strict grounding rules).
full_system      Hybrid retrieval + routing + full expert system prompt (current system).
mixtral_cluster  full_system, generation dispatched to Mixtral 8x7B on the Warwick GPU cluster
                 (via HuggingFace transformers, since Mixtral isn't an Ollama-gated pull).

llama_cluster runs the same full_system pipeline as mixtral_cluster but dispatches to
llama3.1:8b on the Warwick cluster instead of Mixtral (see CLUSTER_SBATCH). It's callable
directly via _generate("llama_cluster", ...) but isn't part of CONFIGS, so a standard
run across CONFIGS never reaches it.

Data acquisition
-----------------
data/eval/*.json holds the cached sample sets these loaders read from:
  contractnli.json  — Stanford NLP Group (stanfordnlp.github.io/contract-nli/),
                      test.json flattened into (document, hypothesis) pairs.
  timeqa.json       — wenhuchen/Time-Sensitive-QA (GitHub), dataset/test.easy.json +
                      test.hard.json, 1,000 sampled per split (fixed seed) and
                      flattened into {question, context, answer, split}.
  locomoplus_locomo10.json, locomoplus_plus.json
                    — xjtuleeyf/Locomo-Plus (GitHub); loaded as two separate
                      benchmarks, "locomo" and "locomoplus" (see _load_locomo/
                      _load_locomoplus).
  casehold.json     — coastalcph/lex_glue[case_hold] (HuggingFace), 1,000 samples.
  cuad.json         — chenghao/cuad_qa (HuggingFace), 1,000 answerable samples.

Usage
-----
  python backend/evaluation/benchmarks.py --dataset cuad
  python backend/evaluation/benchmarks.py --dataset locomoplus
  python backend/evaluation/benchmarks.py --dataset all --config full_system
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import asyncio
import csv
import json
import random
import uuid
from datetime import datetime, timezone

import ollama as _ollama
from qdrant_client import QdrantClient

from backend.config import settings
from backend.evaluation.metrics import (
    AUPR,
    accuracy,
    constraint_consistency,
    exact_match,
    locomoplus_judge,
    macro_f1,
    micro_f1,
    perturbation_consistency,
    precision_at_recall,
    retrieval_recall,
    rouge_l,
    temporal_consistency,
    token_f1,
)

import re  # used throughout scoring functions
from backend.retrieval.hybrid_retriever import retrieve as hybrid_retrieve
from backend.routing.moe_generator import build_system_prompt
from backend.routing.self_router import route
from backend.memory.session_store import SessionContext

# The 5 configs a standard benchmark run compares (see module docstring above
# for llama_cluster, which is not in this list but is still directly callable).
CONFIGS = ["vanilla_rag", "long_ctx_only", "self_route_base", "full_system", "mixtral_cluster"]

# eval config -> sbatch job used to run inference on the Warwick GPU cluster.
# Each config needs its own sbatch script because the two configs use different
# serving stacks: Mixtral runs via raw HuggingFace transformers (no Ollama pull
# exists for it), while llama_cluster reuses the same Ollama engine as local
# full_system — only the sbatch job differs, to isolate hardware from software.
CLUSTER_SBATCH = {
    "mixtral_cluster": "mixtral_inference.sbatch",
    "llama_cluster":   "llama_ollama_inference.sbatch",
}
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "eval"

_stop_requested: set[str] = set()


def request_stop(run_key: str) -> None:
    """Signal the benchmark runner to stop after the current sample."""
    _stop_requested.add(run_key)


def _llm_generate(prompt: str) -> str:
    """Call Ollama generate directly — no LangChain, no spurious options."""
    resp = _ollama.generate(
        model=settings.model_medium,
        prompt=prompt,
        options=None,
    )
    return resp.response


def _embed_query(text: str) -> list[float]:
    """Embed a single query string via Ollama — no LangChain, no spurious options."""
    resp = _ollama.embed(
        model=settings.model_embed,
        input=text,
        options=None,
    )
    return resp.embeddings[0]


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

_LOCOMO_CATEGORY_NAMES: dict[int, str] = {
    1: "single_hop",
    2: "temporal",
    3: "open_domain",
    4: "multi_hop",
    5: "adversarial",
}

# Fixed seed so "sample 1,000 of the 1,986 questions" is a reproducible
# subset across runs, not a different random draw each time.
_LOCOMO_SAMPLE_SEED = 42
_LOCOMO_SAMPLE_SIZE = 1000


def _load_locomo(limit: int | None = None) -> list[dict]:
    """Load LoCoMo (Snap Research, xjtuleeyf/Locomo-Plus's locomo10.json) —
    ten long multi-session conversations, 1,986 QA pairs total across 5
    categories (single_hop, temporal, open_domain, multi_hop, adversarial).
    Deterministically samples 1,000 of the 1,986 (fixed seed). `limit`
    truncates further, for quick sanity-check runs (e.g. n=10)."""
    path10 = DATA_DIR / "locomoplus_locomo10.json"
    if not path10.exists():
        path10 = DATA_DIR / "locomoplus_sample.json"
    if not path10.exists():
        print(f"[benchmarks] {path10} not found")
        return []

    raw10 = json.loads(path10.read_text())
    sessions = raw10 if isinstance(raw10, list) else [raw10]

    samples: list[dict] = []
    for session in sessions:
        # Build a short conversation context from the dialogue
        conv_lines: list[str] = []
        conv = session.get("conversation", {})
        for key in sorted(k for k in conv if k not in ("speaker_a", "speaker_b")):
            sess_turns = conv[key]
            if isinstance(sess_turns, list):
                for turn in sess_turns:
                    speaker = turn.get("speaker", "")
                    text = turn.get("text", "")
                    if text:
                        conv_lines.append(f"{speaker}: {text}")
        conv_context = "\n".join(conv_lines)[:2000]

        for qa in session.get("qa", []):
            cat = qa.get("category", 1)
            samples.append({
                "question":            qa.get("question", ""),
                "reference":           qa.get("answer", ""),
                "category":            cat,
                "category_name":       _LOCOMO_CATEGORY_NAMES.get(cat, "single_hop"),
                "adversarial_answer":  qa.get("adversarial_answer", ""),
                "conversation_ctx":    conv_context,
                "_benchmark":          "locomo",
            })

    if len(samples) > _LOCOMO_SAMPLE_SIZE:
        rng = random.Random(_LOCOMO_SAMPLE_SEED)
        chosen_idx = sorted(rng.sample(range(len(samples)), _LOCOMO_SAMPLE_SIZE))
        samples = [samples[i] for i in chosen_idx]

    if limit:
        samples = samples[:limit]
    return samples


def _load_locomoplus(limit: int | None = None) -> list[dict]:
    """Load LoCoMo-Plus (Li et al. 2026's extension, locomo_plus.json) — 401
    cue-trigger instances. A cue appears early in a conversation and a
    trigger arrives much later with low semantic overlap; the model is
    asked what connection the earlier exchange has to the current one (see
    _score_locomoplus). All 401 are used — it is the whole benchmark, no
    sampling."""
    path_plus = DATA_DIR / "locomoplus_plus.json"
    if not path_plus.exists():
        print(f"[benchmarks] {path_plus} not found")
        return []

    raw_plus = json.loads(path_plus.read_text())
    pairs = raw_plus if isinstance(raw_plus, list) else [raw_plus]

    samples: list[dict] = []
    for pair in pairs:
        cue = pair.get("cue_dialogue", "")
        trigger = pair.get("trigger_query", "")
        time_gap = pair.get("time_gap", "")
        if not (cue and trigger):
            continue
        question = (
            f"Earlier in our conversation ({time_gap} ago): {cue}\n"
            f"Now: {trigger}\n"
            f"What connection or implication does the earlier exchange have on the current situation?"
        )
        samples.append({
            "question":         question,
            "reference":        cue,  # judge evaluates based on evidence, not exact match
            "category":         6,
            "category_name":    "cognitive",
            "conversation_ctx": cue,
            "_benchmark":       "locomoplus",
        })

    if limit:
        samples = samples[:limit]
    return samples


def _load_casehold() -> list[dict]:
    """Load CaseHOLD multiple-choice holding-selection dataset.

    Loaded via LexGLUE (coastalcph/lex_glue, config=case_hold) which hosts the
    same data in a modern Parquet format that doesn't require deprecated scripts.

    Fields: context (str), endings (list[str] of 5 holdings), label (int 0-4)
    """
    path = DATA_DIR / "casehold.json"
    if path.exists():
        items = json.loads(path.read_text())
    else:
        try:
            from datasets import load_dataset as hf_load
            print("[benchmarks] Loading coastalcph/lex_glue[case_hold] from HuggingFace…")
            ds = hf_load("coastalcph/lex_glue", "case_hold", split="test[:1000]")
            items = [dict(ex) for ex in ds]
        except Exception as exc:
            print(f"[benchmarks] CaseHOLD load failed: {exc}")
            return []

    return [
        {
            "question":   ex.get("context", ex.get("citing_prompt", "")),
            "holdings":   ex.get("endings", [ex.get(f"holding_{i}", "") for i in range(5)]),
            "label":      int(ex.get("label", 0)),
            "reference":  (ex.get("endings") or [""])[int(ex.get("label", 0))],
            "_benchmark": "casehold",
        }
        for ex in items
        if ex.get("context") or ex.get("citing_prompt")
    ]


def _load_cuad() -> list[dict]:
    """Load CUAD contract clause QA dataset (SQuAD-style span extraction).

    Format (HuggingFace chenghao/cuad_qa):
      id       : str
      title    : str
      context  : str   — the contract text
      question : str   — the clause-type question
      answers  : dict  — {text: list[str], answer_start: list[int]}
    """
    path = DATA_DIR / "cuad.json"
    if path.exists():
        return json.loads(path.read_text())

    try:
        from datasets import load_dataset as hf_load
        # theatricusproject/cuad only hosts raw PDFs; chenghao/cuad_qa has
        # the pre-processed SQuAD-format version this loader expects.
        print("[benchmarks] Loading chenghao/cuad_qa from HuggingFace…")
        ds = hf_load("chenghao/cuad_qa", split="test")
        samples: list[dict] = []
        for ex in list(ds)[:1000]:
            answers = ex.get("answers", {})
            answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
            if not answer_texts:
                continue  # skip questions with no answer in this contract
            samples.append({
                "question":    ex.get("question", ""),
                "reference":   answer_texts[0],
                "all_answers": answer_texts,
                "context":     ex.get("context", "")[:2000],
                "_benchmark":  "cuad",
            })
        return samples
    except Exception as exc:
        print(f"[benchmarks] CUAD load failed: {exc}")
        return []


# Fixed seed so "a fixed random sample of 1,000 of these 2,091 test pairs"
# is reproducible across runs — same protocol as CUAD/LoCoMo/TimeQA's
# sampling (same seed value as _LOCOMO_SAMPLE_SEED, for consistency).
_CONTRACTNLI_SAMPLE_SEED = 42
_CONTRACTNLI_SAMPLE_SIZE = 1000


def _load_contractnli(limit: int | None = None) -> list[dict]:
    """Load ContractNLI (document, hypothesis) pairs from data/eval/contractnli.json.

    NOTE: this file is not produced by an automated downloader in this
    codebase — there is no in-app/CLI "download" feature for any dataset.
    It was fetched once, manually, from the Stanford NLP Group's direct
    download (stanfordnlp.github.io/contract-nli/resources/contract-nli.zip)
    and its test.json (123 documents x 17 fixed hypotheses = 2,091 pairs)
    was flattened into the schema below. A fixed random sample of 1,000 of
    the 2,091 test pairs is evaluated (fixed seed, reproducible across
    runs), the same sampling protocol as CUAD/LoCoMo/TimeQA. If this file
    is ever missing, redo the one-time fetch-and-flatten by hand — do not
    add an automated downloader for it. Expected schema, one row per
    (document, hypothesis) pair:
      document        : str        — full NDA text
      hypothesis      : str        — one of 17 fixed hypotheses
      label           : str        — "Entailment" | "Contradiction" | "NotMentioned"
      evidence_spans  : list[str]  — gold evidence span TEXT (resolved from
                                      span indices to text, so retrieval_recall/
                                      micro_f1 can compare directly against
                                      retrieved/predicted text; empty for
                                      NotMentioned)
      candidate_spans : list[str]  — ALL of the document's pre-segmented
                                      candidate spans (avg. ~77.8/doc), as
                                      TEXT, so _score_contractnli can present
                                      a numbered list to the model and
                                      resolve its answer back to span text.
    """
    path = DATA_DIR / "contractnli.json"
    if not path.exists():
        print(f"[benchmarks] {path} not found — place a contractnli.json file there "
              "(see this function's docstring for the expected schema) before running this benchmark")
        return []
    raw = json.loads(path.read_text())
    samples = [
        {
            "document":        s.get("document", ""),
            "hypothesis":      s.get("hypothesis", ""),
            "label":           s.get("label", "NotMentioned"),
            "evidence_spans":  s.get("evidence_spans", []),
            "candidate_spans": s.get("candidate_spans", []),
            "_benchmark":      "contractnli",
        }
        for s in raw
        if s.get("document") and s.get("hypothesis")
    ]

    if len(samples) > _CONTRACTNLI_SAMPLE_SIZE:
        rng = random.Random(_CONTRACTNLI_SAMPLE_SEED)
        chosen_idx = sorted(rng.sample(range(len(samples)), _CONTRACTNLI_SAMPLE_SIZE))
        samples = [samples[i] for i in chosen_idx]

    if limit:
        samples = samples[:limit]
    return samples


def _load_timeqa(limit: int | None = None) -> list[dict]:
    """Load TimeQA from data/eval/timeqa.json.

    NOTE: this file is not produced by an automated downloader in this
    codebase — there is no in-app/CLI "download" feature for any dataset.
    It was fetched once, manually, from wenhuchen/Time-Sensitive-QA on
    GitHub (dataset/test.easy.json + test.hard.json, the real template-
    generated test partition, ~3,000 samples per split), with 1,000 sampled
    per split (fixed seed, 2,000 total) and flattened into the schema below.
    If this file is ever missing, redo that same one-time fetch-sample-
    flatten by hand — do not add an automated downloader for it. Expected
    schema, one row per sample:
      question : str  — a time-sensitive question
      context  : str  — the relevant Wikipedia passage
      answer   : str  — gold answer span
      split    : str  — "easy" | "hard"
    """
    path = DATA_DIR / "timeqa.json"
    if not path.exists():
        print(f"[benchmarks] {path} not found — place a timeqa.json file there "
              "(see this function's docstring for the expected schema) before running this benchmark")
        return []
    raw = json.loads(path.read_text())
    samples = [
        {
            "question": s.get("question", ""),
            "context":  s.get("context", ""),
            "answer":   s.get("answer", ""),
            "split":    s.get("split", "easy"),
            # These two extra keys ride along so the generic _sample_notes()
            # helper tags every TimeQA row with its split, letting Easy/Hard
            # be reported separately without any TimeQA-specific code there.
            "task":            "timeqa",
            "temporal_scope":  s.get("split", "easy"),
            "_benchmark":      "timeqa",
        }
        for s in raw
        if s.get("question")
    ]
    if limit:
        samples = samples[:limit]
    return samples


def load_dataset(name: str) -> list[dict]:
    """Dispatch to the correct per-benchmark loader by dataset name."""
    if name == "locomo":
        return _load_locomo()
    if name == "locomoplus":
        return _load_locomoplus()
    if name == "casehold":
        return _load_casehold()
    if name == "cuad":
        return _load_cuad()
    if name == "contractnli":
        return _load_contractnli()
    if name == "timeqa":
        return _load_timeqa()
    raise ValueError(f"Unknown dataset: {name}")


# ---------------------------------------------------------------------------
# Per-config generation
# ---------------------------------------------------------------------------

def _dense_only_retrieve(
    query: str,
    top_k: int = 5,
    collections: tuple[str, ...] = ("legal_docs", "app_docs"),
) -> list[dict]:
    """Dense-only Qdrant query — used for vanilla_rag baseline."""
    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    vec = _embed_query(query)
    results: list[dict] = []
    for collection in collections:
        try:
            hits = qdrant.query_points(
                collection_name=collection,
                query=vec,
                using="dense",
                limit=top_k,
                with_payload=True,
            ).points
            for hit in hits:
                p = hit.payload or {}
                results.append({
                    "text":     p.get("text", ""),
                    "filename": p.get("filename", ""),
                    "page":     p.get("page", 0),
                    "score":    hit.score or 0.0,
                })
        except Exception:
            pass
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# eval dataset -> Qdrant collections it retrieves from. Datasets not listed
# here use the default production corpus (legal_docs, app_docs) — irrelevant
# for the closed-book datasets (contractnli, timeqa, cuad), which fold their
# own gold context directly into the prompt (see their _score_* functions)
# rather than depending on whatever the shared retrieval step happens to
# surface. ingest_cuad_docs.py builds a separate "eval_docs" collection for
# an open-book CUAD variant that isn't used by the standard scoring path.
DATASET_COLLECTIONS: dict[str, tuple[str, ...]] = {}


def _generate(config: str, question: str, dataset_name: str | None = None) -> tuple[str, list[dict]]:
    """Return (answer, sources) for the given configuration."""
    collections = DATASET_COLLECTIONS.get(dataset_name, ("legal_docs", "app_docs"))

    if config == "vanilla_rag":
        sources = _dense_only_retrieve(question, top_k=5, collections=collections)
        context = "\n\n".join(s["text"] for s in sources)
        prompt = f"Context:\n{context[:3000]}\n\nQuestion: {question}\n\nAnswer:"
        return _llm_generate(prompt), sources

    if config == "long_ctx_only":
        sources = _dense_only_retrieve(question, top_k=15, collections=collections)
        context = "\n\n".join(s["text"] for s in sources)
        prompt = (
            f"You have access to the following document excerpts:\n\n{context[:6000]}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        return _llm_generate(prompt), sources

    if config == "self_route_base":
        sources = hybrid_retrieve(question, top_k=8, collections=collections)
        decision = route(question)
        context = "\n\n".join(s.get("text", "") for s in sources)
        prompt = (
            f"You are a {decision.expert_role}.\n"
            f"Context:\n{context[:3000]}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        return _llm_generate(prompt), sources

    if config == "full_system":
        sources = hybrid_retrieve(question, top_k=8, collections=collections)
        decision = route(question)
        ctx = SessionContext(
            session_id=None, user_id="eval", title="eval",
            summary=None, permanent_constraints=[], session_constraints=[],
            recent_messages=[], turn_count=0,
        )
        system_prompt = build_system_prompt(ctx, decision, sources, None, question)
        full_prompt = f"{system_prompt}\n\nUser: {question}\n\nAssistant:"
        return _llm_generate(full_prompt), sources

    # mixtral_cluster | llama_cluster — same full_system pipeline, generation
    # dispatched to the Warwick GPU cluster instead of the local Ollama instance.
    from backend.routing.cluster_dispatcher import dispatch_to_cluster_sync
    sources = hybrid_retrieve(question, top_k=8, collections=collections)
    decision = route(question)
    ctx = SessionContext(
        session_id=None, user_id="eval", title="eval",
        summary=None, permanent_constraints=[], session_constraints=[],
        recent_messages=[], turn_count=0,
    )
    system_prompt = build_system_prompt(ctx, decision, sources, None, question)
    payload = {"query": question, "system_prompt": system_prompt}
    sbatch_file = CLUSTER_SBATCH.get(config, "mixtral_inference.sbatch")
    answer = dispatch_to_cluster_sync(payload, _llm_generate, timeout_seconds=600, sbatch_file=sbatch_file)
    return answer, sources


# ---------------------------------------------------------------------------
# Per-benchmark scoring
# ---------------------------------------------------------------------------


def _score_locomo(config: str, sample: dict) -> dict[str, float]:
    """Score LoCoMo (factual memory, RQ2): judge_score via category-specific
    LLM-judge prompts (single_hop/multi_hop/temporal/open_domain/adversarial
    — the dataset's own protocol), plus rouge_l as a supplementary
    lexical-overlap check."""
    question      = sample.get("question", "")
    reference     = sample.get("reference", "")
    category_name = sample.get("category_name", "single_hop")
    conv_ctx      = sample.get("conversation_ctx", "")

    # Prepend conversation context so the model has the memory it needs
    full_question = f"{conv_ctx}\n\nQuestion: {question}" if conv_ctx else question

    answer, sources = _generate(config, full_question)
    context = "\n".join(s.get("text", "") for s in sources)

    return {
        "judge_score": locomoplus_judge(question, reference, answer, context, category_name),
        "rouge_l":     rouge_l(answer, reference),
    }


def _score_locomoplus(config: str, sample: dict) -> dict[str, float]:
    """Score LoCoMo-Plus (cognitive memory, RQ2): judge_score (LLM-judged,
    via the "cognitive" category prompt — does the prediction demonstrate
    awareness of the connection between the earlier cue and the current
    query?), constraint_consistency (LLM-judged — does the answer respect
    the earlier cue as an implicit constraint?), rouge_l and exact_match
    (supplementary — cross-checked against the stricter judge scores, not
    reported alone)."""
    question      = sample.get("question", "")
    reference     = sample.get("reference", "")
    category_name = sample.get("category_name", "cognitive")
    conv_ctx      = sample.get("conversation_ctx", "")

    answer, sources = _generate(config, question)
    context = "\n".join(s.get("text", "") for s in sources)

    # constraint_consistency's real signature is (constraints: list[str],
    # answer: str) — the earlier cue (conv_ctx) is passed as the single
    # "constraint" source material, letting the judge LLM decide whether the
    # cue's implicit expectation is being respected by the answer.
    constraint_score = constraint_consistency([conv_ctx] if conv_ctx else [], answer)

    return {
        "judge_score":            locomoplus_judge(question, reference, answer, context, category_name),
        "constraint_consistency": constraint_score,
        "rouge_l":                rouge_l(answer, reference),
        "exact_match":            exact_match(answer, reference),
    }


def _score_casehold(config: str, sample: dict) -> dict[str, float]:
    """Score CaseHOLD: ask the model to pick A-E, compare to ground-truth label.

    Per-sample metrics: accuracy, rouge_l. macro_f1 is a DATASET-LEVEL
    AGGREGATE (see metrics.py) — the original paper's actual headline metric
    (Zheng et al., 2021) — computed once after the whole CaseHOLD run
    finishes, not here. This function only contributes its (gold, predicted)
    label pair via the "_pair" key below; run_benchmark() collects that key
    across every sample in the run and reduces it to a single macro_f1 row
    at the end (see run_benchmark()). "_pair" is not itself a metric score —
    it is filtered out of the per-sample rows automatically, since
    run_benchmark()'s row-building loop only persists (int, float) values.
    """
    question = sample.get("question", "")
    holdings = sample.get("holdings", [])
    correct  = int(sample.get("label", 0))

    choices = "\n".join(f"{chr(65 + i)}. {h}" for i, h in enumerate(holdings))
    prompt = (
        f"Legal holding selection.\n\n"
        f"Citing text:\n{question[:1200]}\n\n"
        f"Which holding correctly completes this judicial decision?\n{choices}\n\n"
        f"Answer with only the letter A, B, C, D, or E:"
    )
    answer, sources = _generate(config, prompt)
    answer_upper = answer.strip().upper()

    # Parse letter or digit from the first 20 chars of the response
    predicted = -1
    for i, letter in enumerate("ABCDE"):
        if answer_upper.startswith(letter) or re.search(rf'\b{letter}\b', answer_upper[:20]):
            predicted = i
            break
    if predicted == -1:
        m = re.search(r'\b([0-4])\b', answer.strip()[:20])
        if m:
            predicted = int(m.group(1))

    correct_holding = holdings[correct] if holdings and correct < len(holdings) else ""
    return {
        "accuracy": 1.0 if predicted == correct else 0.0,
        "rouge_l":  rouge_l(answer, correct_holding),
        "_pair":    (correct, predicted),
    }


def _score_cuad(config: str, sample: dict) -> dict[str, float]:
    """Score CUAD closed-book span extraction: the model is given the
    contract excerpt (up to 2,000 chars) directly alongside the clause-type
    question and must produce the exact span, the same closed-book protocol
    as ContractNLI/TimeQA.

    AUPR and precision_at_recall (r=0.8, r=0.9) are DATASET-LEVEL AGGREGATES
    (see metrics.py) — AUPR is CUAD's own headline metric (Hendrycks et al.,
    2021), computed once after the whole CUAD run finishes, not here. Since
    CUAD is generative span extraction rather than classification, there is
    no native model confidence score to rank predictions by; each sample's
    own continuous best_tf1 (token-F1 against the best-matching gold answer)
    is used as the confidence/ranking signal — a higher partial token
    overlap is treated as "the model was more confident it had the right
    span" — while the strict binary best_em (exact_match) is used as ground
    truth for "was this actually correct". This mirrors the standard
    AUPR/precision-recall pattern of ranking by a continuous score and
    checking against a stricter binary correctness criterion. This function
    only contributes that (confidence, is_correct) pair via the
    "_confidence_correct" key; run_benchmark() collects it across the whole
    run and reduces it to the 3 aggregate rows at the end.
    """
    question    = sample.get("question", "")
    context     = sample.get("context", "")
    all_answers = sample.get("all_answers", [sample.get("reference", "")])

    prompt = (
        f"Contract excerpt:\n{context[:2000]}\n\n"
        f"Question: {question}\n\n"
        f"Answer with the exact span from the excerpt:"
    )
    answer, _ = _generate(config, prompt, dataset_name="cuad")

    best_em    = max((exact_match(answer, a) for a in all_answers), default=0.0)
    best_tf1   = max((token_f1(answer, a)   for a in all_answers), default=0.0)
    best_rouge = max((rouge_l(answer, a)    for a in all_answers), default=0.0)
    return {
        "exact_match":         best_em,
        "token_f1":            best_tf1,
        "rouge_l":             best_rouge,
        "_confidence_correct": (best_tf1, best_em == 1.0),
    }


# ---------------------------------------------------------------------------
# ContractNLI scoring
# ---------------------------------------------------------------------------

_CONTRACTNLI_LABELS = ("Entailment", "Contradiction", "NotMentioned")

# Cap on how many of a document's (avg. ~77.8) candidate spans get numbered
# and shown to the model in the prompt — keeps the prompt bounded even for
# outlier documents with an unusually large span count.
_CONTRACTNLI_MAX_CANDIDATE_SPANS = 60


def _parse_contractnli_span_numbers(answer: str) -> set[int]:
    """Parse candidate-span reference numbers out of a ContractNLI model
    response. Prefers the strict "[N]" bracket format the prompt asks for;
    falls back to scanning a "Spans:" line for a bare comma/space-separated
    digit list if no brackets are found, since small local models don't
    always follow the exact requested format."""
    bracketed = re.findall(r'\[(\d+)\]', answer)
    if bracketed:
        return {int(n) for n in bracketed}
    m = re.search(r'spans?\s*:\s*(.*)', answer, re.IGNORECASE)
    if m:
        return {int(n) for n in re.findall(r'\d+', m.group(1))}
    return set()


def _score_contractnli(config: str, sample: dict) -> dict[str, float]:
    """Score ContractNLI: 3-way NLI label classification + evidence-span
    identification, mirroring the original paper's own evaluation protocol
    (Koreeda & Manning, 2021). The document and hypothesis are given
    directly to the model (closed-book classification).

    Per-sample metrics: accuracy, micro_f1 (evidence spans), rouge_l,
    retrieval_recall. precision_at_recall (r=0.8) is a DATASET-LEVEL
    AGGREGATE (see metrics.py) — the paper's own P@R0.8 headline
    evidence-identification metric — computed once after the whole
    ContractNLI run finishes; this function only contributes a
    (confidence, is_correct) pair via "_confidence_correct", using the same
    confidence-proxy pattern as _score_cuad: the continuous per-sample
    micro_f1 (partial credit for evidence-span overlap) as the ranking
    signal, and an exact predicted-vs-gold span-set match as the stricter
    binary "was this actually right" ground truth.
    """
    hypothesis      = sample.get("hypothesis", "")
    gold_label      = sample.get("label", "NotMentioned")
    evidence_spans  = sample.get("evidence_spans", [])
    candidate_spans = sample.get("candidate_spans", [])[:_CONTRACTNLI_MAX_CANDIDATE_SPANS]
    document        = sample.get("document", "")

    numbered = "\n".join(f"[{i + 1}] {sp[:300]}" for i, sp in enumerate(candidate_spans))
    prompt = (
        f"You are analysing a non-disclosure agreement (NDA).\n\n"
        f"Document excerpt:\n{document[:1500]}\n\n"
        f"Candidate spans:\n{numbered[:3000]}\n\n"
        f"Hypothesis: {hypothesis}\n\n"
        f"1) Classify the relationship as exactly one of: Entailment, Contradiction, NotMentioned.\n"
        f"2) List the numbers of any candidate spans (e.g. [2], [5]) that support your answer; "
        f"leave empty if NotMentioned.\n\n"
        f"Format your answer exactly as:\nLabel: <one of the three>\nSpans: <comma-separated numbers or none>"
    )
    answer, sources = _generate(config, prompt, dataset_name="contractnli")

    # Parse predicted label — first matching keyword found in the response.
    answer_lower = answer.lower()
    predicted_label = "NotMentioned"
    for lbl in _CONTRACTNLI_LABELS:
        if lbl.lower() in answer_lower:
            predicted_label = lbl
            break

    span_nums = _parse_contractnli_span_numbers(answer) if candidate_spans else set()
    predicted_spans = [candidate_spans[n - 1] for n in span_nums if 1 <= n <= len(candidate_spans)]

    retrieved_texts = [s.get("text", "") for s in sources]
    span_micro_f1 = micro_f1(set(predicted_spans), set(evidence_spans))
    is_exact_span_match = set(predicted_spans) == set(evidence_spans)

    return {
        "accuracy":         accuracy(predicted_label, gold_label),
        "micro_f1":         span_micro_f1,
        "rouge_l":          rouge_l(answer, gold_label),
        "retrieval_recall": retrieval_recall(evidence_spans, retrieved_texts),
        "_confidence_correct": (span_micro_f1, is_exact_span_match),
    }


# ---------------------------------------------------------------------------
# TimeQA scoring
# ---------------------------------------------------------------------------

# Local, regex-only date/year extraction — backend/temporal/parser.py uses
# an LLM call for this, which would be overkill and slow for scoring/
# perturbing thousands of eval samples. Prefers full ISO-8601 dates if
# present; falls back to bare 4-digit years, since TimeQA answers/context
# more often reference a bare year (e.g. "2004") than a full calendar date.
_ISO_DATE_RE = re.compile(r'\b\d{4}-\d{2}-\d{2}\b')
_YEAR_RE = re.compile(r'\b(1[0-9]{3}|20[0-9]{2})\b')

# Fixed year-shift used to build TimeQA's perturbed question — a clearly
# different but still plausible shift.
_TIMEQA_YEAR_SHIFT = 5


def _extract_dates(text: str) -> list[str]:
    """Extract date-shaped substrings from `text` for temporal_consistency:
    full ISO-8601 dates if any are present, else bare 4-digit years."""
    iso_dates = _ISO_DATE_RE.findall(text)
    if iso_dates:
        return iso_dates
    return _YEAR_RE.findall(text)


def _perturb_timeqa_year(question: str, offset: int = _TIMEQA_YEAR_SHIFT) -> str | None:
    """Shift the first 4-digit year found in `question` by `offset` years and
    return the rewritten question, or None if no year pattern is found."""
    m = _YEAR_RE.search(question)
    if not m:
        return None
    shifted_year = int(m.group(0)) + offset
    start, end = m.span()
    return f"{question[:start]}{shifted_year}{question[end:]}"


def _score_timeqa(config: str, sample: dict) -> dict[str, float]:
    """Score TimeQA: exact_match/token_f1/rouge_l against the gold answer,
    temporal_consistency (date/year overlap), and perturbation_consistency
    (answer stability under a shifted-year question — the source paper's own
    robustness check).

    Folds the sample's fixed Wikipedia passage directly into the question
    text passed to _generate() rather than relying on retrieval, since
    TimeQA's passages aren't part of the indexed Qdrant corpus and every
    config's retrieval step is otherwise generic/shared.

    perturbation_consistency requires a SECOND full generation call per
    sample (the perturbed-question answer) whenever a year is found in the
    question — TimeQA is therefore ~2x the per-sample generation cost of a
    single-call dataset.
    """
    question  = sample.get("question", "")
    context   = sample.get("context", "")
    reference = sample.get("answer", sample.get("reference", ""))

    augmented_q = (
        f"Passage:\n{context[:3000]}\n\nQuestion: {question}\n\n"
        f"Answer with a short span from the passage:"
    )
    answer, _ = _generate(config, augmented_q, dataset_name="timeqa")

    scores: dict[str, float] = {
        "exact_match": exact_match(answer, reference),
        "token_f1":    token_f1(answer, reference),
        "rouge_l":     rouge_l(answer, reference),
        "temporal_consistency": temporal_consistency(_extract_dates(answer), _extract_dates(reference)),
    }

    perturbed_question = _perturb_timeqa_year(question)
    if perturbed_question is not None:
        perturbed_augmented_q = (
            f"Passage:\n{context[:3000]}\n\nQuestion: {perturbed_question}\n\n"
            f"Answer with a short span from the passage:"
        )
        perturbed_answer, _ = _generate(config, perturbed_augmented_q, dataset_name="timeqa")
        scores["perturbation_consistency"] = perturbation_consistency(answer, perturbed_answer)
    # else: no year detected in the question — perturbation_consistency is
    # omitted for this sample entirely.

    return scores


def _score_sample(config: str, sample: dict, dataset_name: str) -> dict[str, float]:
    """Dispatch to the correct per-benchmark scoring function."""
    benchmark = sample.get("_benchmark", dataset_name)
    if benchmark == "locomo":
        return _score_locomo(config, sample)
    if benchmark == "locomoplus":
        return _score_locomoplus(config, sample)
    if benchmark == "casehold":
        return _score_casehold(config, sample)
    if benchmark == "cuad":
        return _score_cuad(config, sample)
    if benchmark == "contractnli":
        return _score_contractnli(config, sample)
    if benchmark == "timeqa":
        return _score_timeqa(config, sample)
    raise ValueError(f"Unknown benchmark: {benchmark}")


def _sample_notes(sample: dict) -> str:
    """Encode per-sample metadata in the notes field."""
    meta = {
        k: sample[k]
        for k in ("task", "temporal_type", "temporal_scope", "category_name", "conv_type")
        if k in sample
    }
    return json.dumps(meta, ensure_ascii=False) if meta else ""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _make_row(dataset_name: str, config: str, metric: str, score: float, notes: str = "") -> dict:
    """Build one evaluation_results row dict — the single row-shape used for
    both ordinary per-sample rows and the post-loop dataset-level-aggregate
    rows below, so there is exactly one place that defines the row schema."""
    return {
        "dataset": dataset_name,
        "config":  config,
        "metric":  metric,
        "score":   round(float(score), 4),
        "run_at":  datetime.now(timezone.utc).isoformat(),
        "notes":   notes,
    }


def run_benchmark(
    dataset_name: str,
    configs: list[str] | None = None,
    run_key: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Run every config against a dataset's samples, scoring each, and return flat metric rows.

    Beyond the ordinary per-sample rows, three datasets need a DATASET-LEVEL
    AGGREGATE metric computed once per config over the *entire* run rather
    than per sample (see metrics.py's macro_f1/AUPR/precision_at_recall
    docstrings): casehold (macro_f1), cuad (AUPR + precision_at_recall r=0.8
    and r=0.9), contractnli (precision_at_recall r=0.8). The corresponding
    _score_* functions smuggle the extra data those aggregates need through
    special "_pair" / "_confidence_correct" dict keys, which are collected
    below across the whole per-config sample loop and never themselves
    written as ordinary rows (they aren't (int, float) values, so the
    existing per-sample row-building loop already skips them unmodified).
    Once the sample loop for a config finishes (or is stopped early — see
    below), those collected pairs are reduced to 1-4 extra rows via the same
    _make_row() helper used for every other row in this file, so there is a
    single row-building path rather than a parallel one.
    """
    samples = load_dataset(dataset_name)
    if not samples:
        print(f"[benchmarks] No samples for {dataset_name}")
        return []

    if limit and limit > 0:
        samples = samples[:limit]

    active_configs = configs or CONFIGS
    rows: list[dict] = []

    for config in active_configs:
        print(f"\n[benchmarks] config={config}  dataset={dataset_name}  n={len(samples)}")

        # Per-config aggregate collectors — reset for every config, since
        # each config's macro_f1/AUPR/precision_at_recall is computed
        # independently (a config's aggregate must only reflect that
        # config's own predictions).
        casehold_pairs: list[tuple[int, int]] = []
        confidence_correct: list[tuple[float, bool]] = []

        stopped = False
        for i, sample in enumerate(samples):
            # Checked only between samples, not mid-generation: an in-flight LLM
            # call can't be safely interrupted, so cancellation takes effect at
            # the next sample boundary rather than instantly.
            if run_key and run_key in _stop_requested:
                _stop_requested.discard(run_key)
                print(f"[benchmarks] Stop requested for {run_key} — halting after {i} samples.")
                stopped = True
                break
            question = sample.get("question", "")
            print(f"  [{i+1}/{len(samples)}] {question[:70]}…")
            try:
                scores = _score_sample(config, sample, dataset_name)
                notes  = _sample_notes(sample)

                # Pull out the dataset-level-aggregate carrier keys (if any)
                # before building ordinary rows — they aren't scores
                # themselves, just data smuggled through for the post-loop
                # aggregate below.
                pair = scores.pop("_pair", None)
                if pair is not None:
                    casehold_pairs.append(pair)
                conf_correct = scores.pop("_confidence_correct", None)
                if conf_correct is not None:
                    confidence_correct.append(conf_correct)

                for metric, score in scores.items():
                    if not isinstance(score, (int, float)):
                        continue
                    rows.append(_make_row(dataset_name, config, metric, score, notes))
            except Exception as exc:
                print(f"  [error] sample {i+1}: {exc}")

        # --- Post-loop dataset-level aggregates for this config ---
        # Computed from whatever was collected even if the run was stopped
        # early — a macro_f1/AUPR over the samples actually completed is more
        # informative than dropping the aggregate just because the run
        # didn't reach the last sample.
        if dataset_name == "casehold" and casehold_pairs:
            gold_labels = [p[0] for p in casehold_pairs]
            pred_labels = [p[1] for p in casehold_pairs]
            agg = macro_f1(gold_labels, pred_labels, labels=[0, 1, 2, 3, 4])
            rows.append(_make_row(dataset_name, config, "macro_f1", agg))

        if dataset_name == "cuad" and confidence_correct:
            rows.append(_make_row(dataset_name, config, "AUPR", AUPR(confidence_correct)))
            rows.append(_make_row(dataset_name, config, "precision_at_recall_r0.8",
                                   precision_at_recall(confidence_correct, 0.8)))
            rows.append(_make_row(dataset_name, config, "precision_at_recall_r0.9",
                                   precision_at_recall(confidence_correct, 0.9)))

        if dataset_name == "contractnli" and confidence_correct:
            rows.append(_make_row(dataset_name, config, "precision_at_recall_r0.8",
                                   precision_at_recall(confidence_correct, 0.8)))

        if stopped:
            # A stop request halts every remaining config in this run, not
            # just the current one.
            break

    return rows


# ---------------------------------------------------------------------------
# Output: CSV + Postgres
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], label: str) -> Path:
    """Write per-sample metric rows to a timestamped CSV file and return its path."""
    output_dir = Path(settings.cluster_eval_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{label}_{ts}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "config", "metric", "score", "run_at", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[benchmarks] CSV → {csv_path}")
    return csv_path


async def import_to_postgres(rows: list[dict], run_id: str | None = None) -> None:
    """Save per-sample rows to evaluation_results, and — if run_id is given — also
    save one aggregated (mean) row per (dataset, config, metric) to evaluation_runs,
    representing the final score for this specific execution."""
    try:
        from backend.database import AsyncSessionLocal
        from backend.models import EvaluationResult, EvaluationRun
        run_uuid = uuid.UUID(run_id) if run_id else None
        async with AsyncSessionLocal() as db:
            for row in rows:
                db.add(EvaluationResult(
                    dataset=row["dataset"],
                    config=row["config"],
                    metric=row["metric"],
                    score=row["score"],
                    run_at=datetime.fromisoformat(row["run_at"]),
                    notes=row.get("notes"),
                    run_id=run_uuid,
                ))

            if run_uuid is not None:
                grouped: dict[tuple[str, str, str], list[float]] = {}
                for row in rows:
                    key = (row["dataset"], row["config"], row["metric"])
                    grouped.setdefault(key, []).append(row["score"])
                run_at = datetime.now(timezone.utc)
                for (dataset, config, metric), scores in grouped.items():
                    db.add(EvaluationRun(
                        run_id=run_uuid,
                        dataset=dataset,
                        config=config,
                        metric=metric,
                        score=round(sum(scores) / len(scores), 4),
                        n_samples=len(scores),
                        run_at=run_at,
                    ))

            await db.commit()
        print(f"[benchmarks] Saved {len(rows)} rows to Postgres" + (f" (run {run_id})" if run_id else ""))
    except Exception as exc:
        print(f"[benchmarks] Postgres import skipped: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# The full dataset set — see module docstring for details on each.
ALL_DATASETS = ["contractnli", "timeqa", "locomo", "locomoplus", "casehold", "cuad"]


async def main() -> None:
    """CLI entry point: parse args, run the selected dataset(s)/config(s), then write CSV + Postgres output."""
    parser = argparse.ArgumentParser(description="Run benchmark evaluations")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=ALL_DATASETS + ["all"],
    )
    parser.add_argument(
        "--config",
        choices=CONFIGS + ["all"],
        default="all",
        help="Which RAG configuration to evaluate (default: all)",
    )
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]
    configs  = CONFIGS if args.config == "all" else [args.config]

    run_id = str(uuid.uuid4())
    all_rows: list[dict] = []
    for ds in datasets:
        rows = run_benchmark(ds, configs=configs)
        all_rows.extend(rows)

    if all_rows:
        write_csv(all_rows, args.dataset)
        await import_to_postgres(all_rows, run_id=run_id)

    print(f"\n[benchmarks] Done — {len(all_rows)} metric rows produced.")


if __name__ == "__main__":
    asyncio.run(main())
