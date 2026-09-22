"""Evaluation metrics for the legal RAG system.

13 metrics, used across the 6 benchmark datasets (contractnli, timeqa,
locomo, locomoplus, casehold, cuad — see benchmarks.py's module docstring):
  rouge_l, accuracy, macro_f1, micro_f1, exact_match, token_f1, AUPR,
  precision_at_recall, judge_score, constraint_consistency,
  temporal_consistency, perturbation_consistency, retrieval_recall.

`judge_score` is implemented below as `locomoplus_judge`, used by both
"locomo" (categories 1-5: single_hop, multi_hop, temporal, open_domain,
adversarial — via benchmarks.py's _score_locomo) and "locomoplus" (category
6: cognitive, via _score_locomoplus), each with its own category prompt.
The metric key written to evaluation_results is the string "judge_score",
not the function name.

IMPORTANT — two of the thirteen are DATASET-LEVEL AGGREGATES, not per-sample
metrics: `macro_f1` and `AUPR`/`precision_at_recall`. Each must be called
exactly ONCE per (dataset, config) benchmark run, over the complete list of
predictions collected across every sample in that run — never once per
sample. Every other metric in this module (rouge_l, accuracy, micro_f1,
exact_match, token_f1, judge_score, constraint_consistency,
temporal_consistency, perturbation_consistency, retrieval_recall) is an
ordinary per-sample metric, computed independently for each sample the same
way rouge_l always has been. See each aggregate function's own docstring
below for exactly what to collect and when to call it.

Retired metrics (implementation retained here, not part of standard
reporting): keyword_accuracy, contextual_accuracy, faithfulness.

Also present but unused by any current dataset: recall_at_k, precision_at_k,
mrr, ndcg_at_k — general retrieval-ranking helpers not currently wired into
any benchmark's scoring path.
"""
import json
import re
import ollama as _ollama
from backend.config import settings


def _judge_invoke(prompt: str) -> str:
    """Call Ollama directly — no LangChain, no spurious options that cause decode errors."""
    return _ollama.generate(
        model=settings.model_small,
        prompt=prompt,
        options=None,
    ).response


def _extract_score(text: str, scale: int = 5) -> float:
    """Parse first integer 1-scale from judge output."""
    m = re.search(r"\b([1-" + str(scale) + r"])\b", str(text))
    return float(m.group(1)) / scale if m else 0.0


# ---------------------------------------------------------------------------
# 1. Recall@k
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant docs found in top-k retrieved results."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / len(relevant_ids)


# ---------------------------------------------------------------------------
# 2. Contextual Accuracy
# ---------------------------------------------------------------------------

_ACCURACY_PROMPT = """\
You are a legal expert evaluator.
Rate the accuracy of this answer given the question and retrieved context, on a scale of 1-5:
  1 = completely wrong or irrelevant
  5 = fully accurate and well-supported by context

Question: {question}
Context: {context}
Answer: {answer}

Output only the integer score (1-5):"""


def contextual_accuracy(question: str, context: str, answer: str) -> float:
    """LLM-judge metric: asks the small local model to rate answer accuracy 1-5, normalised to 0-1."""
    prompt = _ACCURACY_PROMPT.format(question=question, context=context[:2000], answer=answer[:1000])
    return _extract_score(_judge_invoke(prompt), 5)


# ---------------------------------------------------------------------------
# 3. Faithfulness
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """\
You are a legal expert evaluator.
Rate how faithfully the answer sticks to the retrieved context (no hallucinations), on a scale of 1-5:
  1 = heavily hallucinates / contradicts context
  5 = every claim is supported by the context

Context: {context}
Answer: {answer}

Output only the integer score (1-5):"""


def faithfulness(context: str, answer: str) -> float:
    """LLM-judge metric: asks the small local model to rate hallucination-free groundedness 1-5, normalised to 0-1."""
    prompt = _FAITHFULNESS_PROMPT.format(context=context[:2000], answer=answer[:1000])
    return _extract_score(_judge_invoke(prompt), 5)


# ---------------------------------------------------------------------------
# 4. Temporal Consistency
# ---------------------------------------------------------------------------

def temporal_consistency(predicted_dates: list[str], ground_truth_dates: list[str]) -> float:
    """Exact ISO date match fraction."""
    if not ground_truth_dates:
        return 1.0
    matches = sum(1 for pd in predicted_dates if pd in ground_truth_dates)
    return matches / len(ground_truth_dates)


# ---------------------------------------------------------------------------
# 5. Constraint Consistency
# ---------------------------------------------------------------------------

_CONSTRAINT_PROMPT = """\
You are a legal expert evaluator.
Does the answer respect all the stated constraints below?

Constraints: {constraints}
Answer: {answer}

Output only: yes or no"""


def constraint_consistency(constraints: list[str], answer: str) -> float:
    """LLM-judge metric: binary yes/no check that the answer respects all stated constraints."""
    if not constraints:
        return 1.0
    prompt = _CONSTRAINT_PROMPT.format(
        constraints="; ".join(constraints),
        answer=answer[:1000],
    )
    return 1.0 if _judge_invoke(prompt).strip().lower().startswith("yes") else 0.0


# ---------------------------------------------------------------------------
# 6. ROUGE-L
# ---------------------------------------------------------------------------

def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 via longest common subsequence (no extra dependencies)."""
    pred_tokens = prediction.lower().split()
    ref_tokens  = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(ref_tokens), len(pred_tokens)
    # Standard LCS dynamic-programming table, kept to two rolling rows (prev/curr) instead of a
    # full m*n matrix since only the LCS length is needed, not the alignment itself.
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[n]
    precision = lcs / n
    recall    = lcs / m
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# 7. Precision@k
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / min(k, len(retrieved_ids))


# ---------------------------------------------------------------------------
# 8. MRR
# ---------------------------------------------------------------------------

def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank — rank of the first relevant document."""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# 9. NDCG@k
# ---------------------------------------------------------------------------

def ndcg_at_k(retrieved_ids: list[str], relevance_map: dict[str, float], k: int) -> float:
    """NDCG@k given a mapping of doc_id → relevance score."""
    try:
        import numpy as np
        from sklearn.metrics import ndcg_score as sklearn_ndcg
    except ImportError:
        return 0.0
    if not relevance_map:
        return 0.0
    all_ids = list(dict.fromkeys(list(relevance_map) + retrieved_ids))
    true_rel = np.array([[relevance_map.get(d, 0.0) for d in all_ids]])
    # sklearn's ndcg_score ranks docs by predicted *score*, not by an explicit rank list, so we
    # synthesise a score per doc from its retrieval position (1st place scores highest) —
    # docs not retrieved at all get 0 and sort last.
    pred_scores = np.array([[
        1.0 / (retrieved_ids.index(d) + 1) if d in retrieved_ids else 0.0
        for d in all_ids
    ]])
    try:
        return float(sklearn_ndcg(true_rel, pred_scores, k=k))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 10. Keyword Accuracy (LexRAG)
# ---------------------------------------------------------------------------

def keyword_accuracy(keywords: list[str], answer: str) -> float:
    """Fraction of expected keywords present in the answer."""
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


# ---------------------------------------------------------------------------
# 11. Exact Match (ChronoQA / LegalBench)
# ---------------------------------------------------------------------------

def exact_match(prediction: str, reference: str) -> float:
    """Normalised exact string match."""
    def _norm(s: str) -> str:
        return " ".join(s.lower().split())
    return 1.0 if _norm(prediction) == _norm(reference) else 0.0


# ---------------------------------------------------------------------------
# 12. Token F1 (ChronoQA)
# ---------------------------------------------------------------------------

def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-words token-level F1."""
    pred_toks = set(prediction.lower().split())
    ref_toks  = set(reference.lower().split())
    if not pred_toks or not ref_toks:
        return 0.0
    common = pred_toks & ref_toks
    if not common:
        return 0.0
    p = len(common) / len(pred_toks)
    r = len(common) / len(ref_toks)
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# 13. LoCoMo / LoCoMo-Plus LLM-as-Judge (category-specific prompts)
# ---------------------------------------------------------------------------

_LOCOMOPLUS_PROMPTS: dict[str, str] = {
    "single_hop": """\
Evaluate whether the prediction correctly answers the question.
Question: {question}
Ground Truth: {reference}
Prediction: {prediction}
Score — correct: exact or semantically equivalent; partial: mostly right, minor gap; wrong: factually incorrect.
Return JSON only: {{"label": "correct"|"partial"|"wrong", "reason": "one sentence"}}""",

    "multi_hop": """\
Evaluate this multi-step reasoning answer.
Question: {question}
Ground Truth: {reference}
Prediction: {prediction}
Score — correct: all reasoning steps right; partial: main entity correct but gaps; wrong: factually incorrect.
Return JSON only: {{"label": "correct"|"partial"|"wrong", "reason": "one sentence"}}""",

    "temporal": """\
Evaluate whether the prediction correctly identifies the time, duration, or sequence.
Question: {question}
Ground Truth: {reference}
Prediction: {prediction}
Score — correct: exact or equivalent timing; wrong: incorrect time or reversed sequence.
Return JSON only: {{"label": "correct"|"wrong", "reason": "one sentence"}}""",

    "open_domain": """\
Evaluate this open-domain/commonsense reasoning answer.
Question: {question}
Ground Truth: {reference}
Prediction: {prediction}
Score — correct: sound logic; partial: mostly right but vague conclusion; wrong: contradicts commonsense.
Return JSON only: {{"label": "correct"|"partial"|"wrong", "reason": "one sentence"}}""",

    "adversarial": """\
The question is adversarial — the correct answer is that the information was NOT present in context.
Question: {question}
Ground Truth: {reference}
Prediction: {prediction}
Score — correct: prediction says info not available/not mentioned; wrong: prediction gives a concrete answer.
Return JSON only: {{"label": "correct"|"wrong", "reason": "one sentence"}}""",

    "cognitive": """\
Evaluate if the prediction demonstrates awareness of the connection between earlier context and the query.
Question: {question}
Ground Truth: {reference}
Prediction: {prediction}
Evidence: {context}
Score — correct: prediction reflects the evidence link; wrong: no demonstrable connection to evidence.
Return JSON only: {{"label": "correct"|"wrong", "reason": "one sentence"}}""",
}
# "cognitive" is LoCoMo-Plus's category (401 cue-trigger instances, its own
# dataset — see benchmarks.py's _load_locomoplus/_score_locomoplus); the
# other five are LoCoMo's (see _load_locomo/_score_locomo).

_LABEL_TO_SCORE: dict[str, float] = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


def locomoplus_judge(
    question: str,
    reference: str,
    prediction: str,
    context: str,
    category: str,
) -> float:
    """LLM-as-judge for LoCoMo/LoCoMo-Plus (category-specific prompts). Returns 0.0, 0.5, or 1.0."""
    template = _LOCOMOPLUS_PROMPTS.get(category, _LOCOMOPLUS_PROMPTS["single_hop"])
    prompt = template.format(
        question=question[:400],
        reference=reference[:300],
        prediction=prediction[:500],
        context=context[:400],
    )
    raw = _judge_invoke(prompt).strip()

    # Try JSON extraction first, then keyword fallback
    m = re.search(r'\{[^}]+\}', raw, re.DOTALL)
    if m:
        try:
            label = json.loads(m.group()).get("label", "wrong").lower()
            return _LABEL_TO_SCORE.get(label, 0.0)
        except Exception:
            pass
    raw_lower = raw.lower()
    if "partial" in raw_lower:
        return 0.5
    if "correct" in raw_lower:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Classification / ranking metrics. All are PER-SAMPLE metrics unless the
# docstring explicitly says "DATASET-LEVEL AGGREGATE".
# ---------------------------------------------------------------------------

def accuracy(pred_label, gold_label) -> float:
    """Binary 0/1 exact match between a predicted class label and the gold
    label. Deliberately generic (no type annotation on the label params
    beyond "any comparable value") so it works equally for a string label
    (ContractNLI's "Entailment"/"Contradiction"/"NotMentioned") or an integer
    class id (CaseHOLD's 0-4 holding index). Per-sample metric.
    """
    return 1.0 if pred_label == gold_label else 0.0


def macro_f1(y_true: list, y_pred: list, labels: list) -> float:
    """Macro-averaged F1 across `labels`.

    *** DATASET-LEVEL AGGREGATE — NOT a per-sample metric. ***
    `y_true`/`y_pred` must be the COMPLETE, parallel lists of gold and
    predicted labels collected across every sample of a finished benchmark
    run for one (dataset, config) pair. Call this exactly ONCE, after the
    run's sample loop has finished — never once per sample — since a
    per-class precision/recall/F1 can only be computed once every sample's
    verdict is known. This is structurally different from every other metric
    in this module (rouge_l, accuracy, exact_match, token_f1, ...), which
    benchmarks.py's scoring loop computes and persists per-sample.

    For each class c in `labels`:
      TP = count(true == c and pred == c)
      FP = count(true != c and pred == c)
      FN = count(true == c and pred != c)
      precision = TP / (TP + FP) if TP + FP > 0 else 0.0
      recall    = TP / (TP + FN) if TP + FN > 0 else 0.0
      f1        = 2 * precision * recall / (precision + recall)
                  if precision + recall > 0 else 0.0
    Returns the mean of per-class f1 across all classes in `labels`, so a
    class the model gets systematically wrong isn't masked by good
    performance on the others (Zheng et al., 2021 — CaseHOLD's own headline
    metric).
    """
    f1_scores: list[float] = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def micro_f1(pred_set: set, gold_set: set) -> float:
    """Per-sample micro-averaged F1 between a predicted set and a gold set
    (e.g. ContractNLI's predicted vs. gold evidence-span text, treating
    evidence identification as multi-label classification over candidate
    spans). Unlike `macro_f1` above, this IS an ordinary per-sample metric —
    call it once per sample, the same way rouge_l/exact_match/token_f1 are
    called.

    tp = |pred_set ∩ gold_set|
    precision = tp / |pred_set|  (0.0 if pred_set is empty)
    recall    = tp / |gold_set|  (0.0 if gold_set is empty)
    F1        = 2 * precision * recall / (precision + recall), else 0.0

    Used by benchmarks.py's _score_contractnli to score predicted evidence
    spans against gold evidence_spans (ContractNLI's multi-label
    evidence-identification task).
    """
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def AUPR(scored: list[tuple[float, bool]]) -> float:
    """Area under the precision-recall curve (Average-Precision formulation),
    over confidence-ranked predictions.

    *** DATASET-LEVEL AGGREGATE — NOT a per-sample metric. ***
    `scored` must be the COMPLETE list of (confidence, is_correct) pairs
    collected across every sample of a finished benchmark run for one
    (dataset, config) pair — one entry per sample. Call this exactly ONCE,
    after the run finishes, not accumulated/returned per sample. This is
    CUAD's own headline metric (Hendrycks et al., 2021), not raw EM/F1.

    Algorithm: sort `scored` by confidence descending. Let total_correct be
    the number of correct entries overall (return 0.0 if there are none).
    Walk the sorted list tracking a running correct-count; at each
    1-indexed position i whose entry is itself correct, accumulate
    precision_at_i = running_correct / i. Return the sum of those
    precision_at_i values divided by total_correct.
    """
    total_correct = sum(1 for _, is_correct in scored if is_correct)
    if total_correct == 0:
        return 0.0
    ranked = sorted(scored, key=lambda pair: pair[0], reverse=True)
    running_correct = 0
    precision_sum = 0.0
    for i, (_, is_correct) in enumerate(ranked, start=1):
        if is_correct:
            running_correct += 1
            precision_sum += running_correct / i
    return precision_sum / total_correct


def precision_at_recall(scored: list[tuple[float, bool]], r: float) -> float:
    """Precision once recall first reaches threshold `r`, over
    confidence-ranked predictions (CUAD's P@80%R / P@90%R, ContractNLI's own
    P@R0.8 evidence-identification metric — one implementation, parametrised
    by `r`, covers every paper's own reporting convention).

    *** DATASET-LEVEL AGGREGATE — NOT a per-sample metric. *** Same calling
    convention as `AUPR` above: `scored` must be the complete list of
    (confidence, is_correct) pairs from a finished run, passed in and
    reduced exactly ONCE, not accumulated per sample.

    Sorts `scored` by confidence descending, then walks the list tracking a
    running correct-count and 1-indexed position i, computing
    recall_i = running_correct / total_correct and
    precision_i = running_correct / i at each step. Returns the precision_i
    at the FIRST position where recall_i >= r. If recall never reaches r
    (e.g. r is unreachably high, or too few correct entries exist to ever
    satisfy it), returns the precision computed over the entire list rather
    than raising or returning a sentinel value.
    """
    total_correct = sum(1 for _, is_correct in scored if is_correct)
    if total_correct == 0:
        return 0.0
    ranked = sorted(scored, key=lambda pair: pair[0], reverse=True)
    running_correct = 0
    last_precision = 0.0
    for i, (_, is_correct) in enumerate(ranked, start=1):
        if is_correct:
            running_correct += 1
        recall_i = running_correct / total_correct
        precision_i = running_correct / i
        last_precision = precision_i
        if recall_i >= r:
            return precision_i
    return last_precision


def perturbation_consistency(original_answer: str, perturbed_answer: str) -> float:
    """The source paper's own robustness check (TimeQA): the question's time
    expression is perturbed (e.g. its year shifted by a few years) and this
    metric checks whether the model's answer actually changed in response,
    rather than staying fixed regardless of the (now-different) time window
    being asked about.

    Returns 1.0 if the two answers differ after normalisation (lowercase,
    whitespace-collapsed) — i.e. the model's answer tracked the perturbation
    — else 0.0. Per-sample metric; skip calling it entirely for a sample if
    no perturbation could be constructed (e.g. no year found in the
    question) rather than passing in two identical un-perturbed answers,
    since that would always incorrectly score 0.0.
    """
    def _norm(s: str) -> str:
        return " ".join(s.lower().split())
    return 1.0 if _norm(original_answer) != _norm(perturbed_answer) else 0.0


def retrieval_recall(gold_spans: list[str], retrieved_texts: list[str]) -> float:
    """Approximate recall: fraction of `gold_spans` whose first 5 non-trivial
    tokens (length > 2, case-insensitive) appear anywhere in the
    concatenated `retrieved_texts`. Per-sample metric, used by ContractNLI
    to score retrieval against the gold evidence spans.

    Returns 0.0 for an empty `gold_spans` list rather than a vacuous 1.0 —
    important for ContractNLI's "NotMentioned" samples, whose evidence_spans
    list is legitimately empty and should contribute no signal rather than
    inflate the aggregate score.
    """
    if not gold_spans:
        return 0.0
    retrieved_combined = " ".join(retrieved_texts).lower()
    hits = 0
    for span in gold_spans:
        tokens = span.split()[:5]
        probe_tokens = [t for t in tokens if len(t) > 2]
        if probe_tokens and any(t.lower() in retrieved_combined for t in probe_tokens):
            hits += 1
    return hits / len(gold_spans)
