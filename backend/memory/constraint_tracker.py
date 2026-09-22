"""Extract and persist user constraints via llama3.2:3b structured output."""
import json
import uuid
from datetime import datetime, timezone

from langchain_ollama import OllamaLLM
from pydantic import BaseModel
from sqlalchemy import select

from backend.config import settings
from backend.database import AsyncSessionLocal
from backend.models import Constraint, Session

_llm_small: OllamaLLM | None = None


def _get_llm() -> OllamaLLM:
    """Lazily instantiate and cache the small Ollama LLM used for constraint extraction."""
    global _llm_small
    if _llm_small is None:
        _llm_small = OllamaLLM(
            model=settings.model_small,
            base_url=settings.ollama_base_url,
        )
    return _llm_small


class ExtractedConstraints(BaseModel):
    jurisdiction: str | None = None
    financial_threshold: float | None = None
    applicable_law: str | None = None
    excluded_clauses: list[str] = []
    custom_rules: list[str] = []
    is_permanent: bool = False


_EXTRACT_PROMPT = """\
You are a constraint extraction engine for a legal RAG system.
Read the user message and extract any explicit constraints the user is setting.

Output ONLY valid JSON matching this schema (no commentary):
{{
  "jurisdiction": <string or null>,
  "financial_threshold": <float or null>,
  "applicable_law": <string or null>,
  "excluded_clauses": [<string>, ...],
  "custom_rules": [<string>, ...],
  "is_permanent": <true if user says "always", "permanently", "in all sessions", else false>
}}

User message:
{message}
"""


def _parse_constraints_json(raw: str) -> ExtractedConstraints:
    """Parse LLM output, tolerating minor formatting issues."""
    try:
        # Find the first { and last }
        start = raw.index("{")
        end = raw.rindex("}") + 1
        data = json.loads(raw[start:end])
        return ExtractedConstraints(**data)
    except Exception:
        return ExtractedConstraints()


async def extract_and_save_constraints(
    session_id: uuid.UUID,
    user_message: str,
    user_id: str = settings.default_user_id,
) -> list[Constraint]:
    """Run llama3.2:3b to extract constraints, save new ones to Postgres."""
    llm = _get_llm()
    prompt = _EXTRACT_PROMPT.format(message=user_message)
    raw = llm.invoke(prompt)
    extracted = _parse_constraints_json(str(raw))

    rows_added: list[Constraint] = []

    async with AsyncSessionLocal() as db:
        to_save: list[tuple[str, str]] = []

        if extracted.jurisdiction:
            to_save.append(("jurisdiction", extracted.jurisdiction))
        if extracted.financial_threshold is not None:
            to_save.append(("financial_threshold", str(extracted.financial_threshold)))
        if extracted.applicable_law:
            to_save.append(("applicable_law", extracted.applicable_law))
        for clause in extracted.excluded_clauses:
            to_save.append(("excluded_clause", clause))
        for rule in extracted.custom_rules:
            to_save.append(("custom_rule", rule))

        for ctype, value in to_save:
            row = Constraint(
                session_id=session_id,
                constraint_type=ctype,
                value=value,
                is_active=True,
                # is_permanent is a single flag decided by the LLM for the whole
                # message (based on cues like "always"/"permanently"), so every
                # constraint extracted from this message shares the same scope.
                is_permanent=extracted.is_permanent,
                created_at=datetime.now(timezone.utc),
            )
            db.add(row)
            rows_added.append(row)

        await db.commit()

    return rows_added


async def get_permanent_constraints(user_id: str) -> list[Constraint]:
    """Load is_permanent=True constraints for this user across all sessions."""
    async with AsyncSessionLocal() as db:
        # Permanent constraints are user-scoped, not session-scoped, so we join
        # through Session to filter by user_id rather than a single session_id
        # (that's what distinguishes this from get_session_constraints below).
        result = await db.execute(
            select(Constraint)
            .join(Session, Constraint.session_id == Session.id)
            .where(
                Session.user_id == user_id,
                Constraint.is_permanent == True,
                Constraint.is_active == True,
            )
        )
        return list(result.scalars().all())


async def get_session_constraints(session_id: uuid.UUID) -> list[Constraint]:
    """Load active constraints for this session."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Constraint).where(
                Constraint.session_id == session_id,
                Constraint.is_active == True,
            )
        )
        return list(result.scalars().all())


async def deactivate_constraint(constraint_id: uuid.UUID) -> None:
    """Mark a constraint as inactive so it's excluded from future context loads."""
    async with AsyncSessionLocal() as db:
        row = await db.get(Constraint, constraint_id)
        if row:
            row.is_active = False
            await db.commit()
