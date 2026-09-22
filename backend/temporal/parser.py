"""Temporal parser — uses llama3.2:3b structured output to extract dates/events."""
import json

from dateutil import parser as dateutil_parser
from langchain_ollama import OllamaLLM
from pydantic import BaseModel

from backend.config import settings

_llm_small: OllamaLLM | None = None


def _get_llm() -> OllamaLLM:
    """Lazily construct and cache a single shared OllamaLLM client (singleton)."""
    global _llm_small
    if _llm_small is None:
        _llm_small = OllamaLLM(
            model=settings.model_small,
            base_url=settings.ollama_base_url,
        )
    return _llm_small


class TemporalData(BaseModel):
    explicit_dates: list[dict] = []   # [{date: ISO str, context: str}]
    trigger_events: list[dict] = []   # [{event: str, date: str|None}]
    relative_refs: list[str] = []     # ["within 30 days", ...]


_TEMPORAL_PROMPT = """\
You are a temporal fact extractor for legal documents.
Extract all date-related information from the text below.
Output ONLY valid JSON, no commentary:

{{
  "explicit_dates": [
    {{"date": "<ISO-8601 date string>", "context": "<what this date refers to>"}}
  ],
  "trigger_events": [
    {{"event": "<event description>", "date": "<ISO-8601 or null>"}}
  ],
  "relative_refs": ["<e.g. within 30 days>", ...]
}}

Text:
{text}
"""


def _validate_date(date_str: str) -> str | None:
    """Return ISO date string or None if invalid."""
    try:
        parsed = dateutil_parser.parse(date_str)
        return parsed.date().isoformat()
    except Exception:
        return None


def _parse_temporal_json(raw: str) -> TemporalData:
    """Extract and validate the JSON object from the LLM's raw text response."""
    try:
        # The LLM is instructed to output ONLY JSON, but small local models
        # sometimes still wrap it in markdown fences or add stray commentary.
        # Slicing between the first "{" and the last "}" strips that noise
        # without needing a stricter (and more fragile) output parser.
        start = raw.index("{")
        end = raw.rindex("}") + 1
        data = json.loads(raw[start:end])

        # Validate all explicit dates; entries with unparseable dates are
        # silently dropped rather than failing the whole extraction, since
        # a partial result is more useful than none for downstream circuits
        valid_dates = []
        for entry in data.get("explicit_dates", []):
            iso = _validate_date(str(entry.get("date", "")))
            if iso:
                valid_dates.append({"date": iso, "context": entry.get("context", "")})

        # Validate trigger event dates if present
        valid_triggers = []
        for entry in data.get("trigger_events", []):
            d = entry.get("date")
            iso = _validate_date(str(d)) if d else None
            valid_triggers.append({"event": entry.get("event", ""), "date": iso})

        return TemporalData(
            explicit_dates=valid_dates,
            trigger_events=valid_triggers,
            relative_refs=data.get("relative_refs", []),
        )
    except Exception:
        # Malformed/missing JSON from the LLM should not crash the pipeline —
        # fall back to an empty result so callers can proceed gracefully.
        return TemporalData()


def extract_temporal_data(text: str) -> TemporalData:
    """Extract temporal facts from query + context using llama3.2:3b."""
    llm = _get_llm()
    # Truncate to keep the prompt within the small model's context window
    # and bound latency/cost — temporal facts are typically stated early
    # in a document, so the head of the text is sufficient in practice.
    prompt = _TEMPORAL_PROMPT.format(text=text[:4000])
    raw = str(llm.invoke(prompt))
    return _parse_temporal_json(raw)
