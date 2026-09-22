"""Adaptive query router.

No LLM call is made anywhere in this module — both stages are pure,
zero/low-cost string scans over the query text.

Stage 1 — temporal keyword scan (zero cost): the lowercased query is
          checked as a substring match against 18 hardcoded temporal
          keywords. Any match → route="temporal", bypassing Stage 2
          entirely.

Stage 2 — complexity scoring (O(n) keyword scan), for non-temporal
          queries only: word_count > 40 OR analysis-signal count >= 2
          → confidence=0.50 → route="cluster"; otherwise
          → confidence=0.85 → route="local".
          confidence_threshold = 0.75 (settings.confidence_threshold).

This deliberately avoids the wasteful pattern of generating a speculative
draft answer and scoring its log-probabilities (as in the original
Self-RAG and speculative-decoding papers): generating a draft purely to
decide how to route is expensive, and the draft would be discarded
immediately afterwards. Complexity signals from the query text alone are
a sufficient proxy for whether the request needs multi-hop reasoning
(→ cluster) or straightforward factual retrieval (→ local).
"""
from dataclasses import dataclass

from backend.config import settings

TEMPORAL_KEYWORDS = [
    "deadline", "prior to", "effective date", "within", "days of",
    "statute of limitations", "before", "after", "calculated from",
    "expiry", "expires", "signed on", "commencing", "termination date",
    "notice period", "limitation period", "time-bar", "accrual",
]

EXPERT_ROLES = {
    frozenset(["contract", "clause"]): "contract law expert",
    frozenset(["case", "precedent"]): "case law expert",
    frozenset(["deadline", "date", "termination date", "expiry"]): "temporal reasoning expert",
}


def select_expert_role(query: str, has_app_docs: bool = False) -> str:
    """Pick an expert persona: app-doc context wins outright, else keyword overlap with the query."""
    if has_app_docs:
        return "software documentation expert"
    q = query.lower()
    for keywords, role in EXPERT_ROLES.items():
        if any(kw in q for kw in keywords):
            return role
    return "legal assistant"


@dataclass
class RouteDecision:
    path: str           # "temporal" | "local" | "cluster"
    expert_role: str
    confidence: float
    draft: str | None = None


def _score_confidence(query: str) -> float:
    """
    Route complex/vague queries to cluster, simple queries to local.
    Based on query complexity signals only — never pre-generates an answer.
    """
    q = query.lower()
    complex_signals = [
        "analyse", "analyze", "compare", "explain in detail", "what are all",
        "summarise", "summarize", "critically", "implications", "comprehensive",
        "what is the relationship", "how does", "argue",
    ]
    n_signals = sum(1 for s in complex_signals if s in q)
    # Long queries tend to be more complex
    word_count = len(query.split())
    if word_count > 40 or n_signals >= 2:
        return 0.5   # → cluster
    return 0.85      # → local


def route(query: str, has_app_docs: bool = False) -> RouteDecision:
    """
    Determine routing path and expert role for this query.
    Returns RouteDecision with path, expert_role, confidence.
    """
    q_lower = query.lower()
    # Persona (expert_role) and path are two independent signals: persona comes
    # from app-doc context / keyword overlap here, while path below comes from a
    # separate temporal-keyword scan followed by a complexity/confidence score.
    expert = select_expert_role(query, has_app_docs)

    # Step 1: temporal keyword scan (zero cost)
    if any(kw in q_lower for kw in TEMPORAL_KEYWORDS):
        return RouteDecision(path="temporal", expert_role="temporal reasoning expert", confidence=1.0)

    # Step 2: score query complexity to decide local vs cluster
    # Never draft without context — complexity is judged from query length/hedge words alone
    confidence = _score_confidence(query)

    if confidence >= settings.confidence_threshold:
        return RouteDecision(path="local", expert_role=expert, confidence=confidence, draft=None)
    else:
        return RouteDecision(path="cluster", expert_role=expert, confidence=confidence, draft=None)
