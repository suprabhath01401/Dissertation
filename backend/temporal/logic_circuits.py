"""Pure Python temporal logic circuits — zero LLM calls.

Two circuits:
  parallel_circuit — find earliest/latest date, detect conflicts
  series_circuit   — trigger event + rule_days → deadline
"""
from datetime import date, timedelta
from typing import Any

from dateutil import parser as dateutil_parser

from backend.temporal.parser import TemporalData


def _parse_date(s: str) -> date:
    """Parse a free-form date string into a `date` object."""
    return dateutil_parser.parse(s).date()


def _detect_conflicts(dates: list[date], temporal_data: TemporalData) -> list[str]:
    """Flag date pairs that appear to contradict relative_refs."""
    conflicts: list[str] = []
    if len(dates) < 2:
        return conflicts

    for ref in temporal_data.relative_refs:
        # Look for "within N days" pattern and check if date range violates it
        import re
        m = re.search(r"within\s+(\d+)\s+day", ref, re.IGNORECASE)
        if m:
            allowed_days = int(m.group(1))
            span = (max(dates) - min(dates)).days
            if span > allowed_days:
                conflicts.append(
                    f"Date span ({span} days) exceeds reference '{ref}'"
                )
    return conflicts


def parallel_circuit(temporal_data: TemporalData) -> dict[str, Any]:
    """
    Find earliest and latest explicit date, detect any conflicts.

    Returns:
      {earliest, latest, span_days, conflicts: list[str]}
    """
    if not temporal_data.explicit_dates:
        return {"earliest": None, "latest": None, "span_days": 0, "conflicts": []}

    dates = [_parse_date(d["date"]) for d in temporal_data.explicit_dates]
    earliest = min(dates)
    latest = max(dates)
    span = (latest - earliest).days
    conflicts = _detect_conflicts(dates, temporal_data)

    return {
        "earliest": earliest.isoformat(),
        "latest": latest.isoformat(),
        "span_days": span,
        "conflicts": conflicts,
    }


def _find_earliest_trigger(temporal_data: TemporalData) -> dict | None:
    """Return the trigger event with the earliest known date, or the first event."""
    triggers_with_dates = [
        t for t in temporal_data.trigger_events if t.get("date")
    ]
    if triggers_with_dates:
        return min(triggers_with_dates, key=lambda t: _parse_date(t["date"]))
    if temporal_data.trigger_events:
        return temporal_data.trigger_events[0]
    # Fall back to earliest explicit date as anchor
    if temporal_data.explicit_dates:
        earliest = min(temporal_data.explicit_dates, key=lambda d: _parse_date(d["date"]))
        return {"event": earliest.get("context", "anchor date"), "date": earliest["date"]}
    return None


def series_circuit(temporal_data: TemporalData, rule_days: int) -> dict[str, Any]:
    """
    Calculate a deadline from a trigger event.

    rule_days: number of days from the trigger event.
    Returns:
      {trigger_event, trigger_date, deadline, rule_applied}
    """
    trigger = _find_earliest_trigger(temporal_data)
    if not trigger or not trigger.get("date"):
        return {
            "trigger_event": None,
            "trigger_date": None,
            "deadline": None,
            "rule_applied": f"+{rule_days} days",
            "error": "No trigger event with a known date found",
        }

    trigger_date = _parse_date(trigger["date"])
    deadline = trigger_date + timedelta(days=rule_days)

    return {
        "trigger_event": trigger.get("event", ""),
        "trigger_date": trigger_date.isoformat(),
        "deadline": deadline.isoformat(),
        "rule_applied": f"+{rule_days} days",
    }


def run_temporal_pipeline(temporal_data: TemporalData) -> dict[str, Any]:
    """
    Run both circuits and return combined output.
    Extracts rule_days from relative_refs if present.
    """
    import re

    # Legal rule: use the first explicit "N day(s)" reference found in the
    # text (e.g. "must respond within 30 days") as the governing deadline
    # period for the series circuit. If no such reference was extracted,
    # fall back to 30 days — a common statutory/contractual default period
    # — as a conservative estimate rather than leaving the deadline unset.
    rule_days = 30  # default
    for ref in temporal_data.relative_refs:
        m = re.search(r"(\d+)\s+day", ref, re.IGNORECASE)
        if m:
            rule_days = int(m.group(1))
            break

    parallel = parallel_circuit(temporal_data)
    series = series_circuit(temporal_data, rule_days)

    return {
        "parallel": parallel,
        "series": series,
        "relative_refs": temporal_data.relative_refs,
        "explicit_dates": temporal_data.explicit_dates,
        "trigger_events": temporal_data.trigger_events,
    }
