"""Compress old messages into a factual summary using llama3.1:8b.

Triggered after every 10 turns or at the start of a new session.
Writes compressed summary to sessions.summary in Postgres.
"""
import uuid

from langchain_ollama import OllamaLLM

from backend.config import settings
from backend.memory.session_store import get_session_messages, update_session_summary

_llm_medium: OllamaLLM | None = None

_COMPRESS_PROMPT = """\
You are a factual summariser for a legal research conversation.
Compress the following conversation history into a concise factual summary.
Preserve: legal conclusions reached, constraints set, key facts, document references.
Omit: filler words, pleasantries, repeated questions.
Output only the summary paragraph.

Conversation:
{history}
"""


def _get_llm() -> OllamaLLM:
    """Lazily instantiate and cache the medium Ollama LLM used for summarisation."""
    global _llm_medium
    if _llm_medium is None:
        _llm_medium = OllamaLLM(
            model=settings.model_medium,
            base_url=settings.ollama_base_url,
        )
    return _llm_medium


def _format_history(messages) -> str:
    """Render messages as 'ROLE: content' lines, truncating each to 500 chars."""
    lines = []
    for m in messages:
        role = m.role.upper()
        lines.append(f"{role}: {m.content[:500]}")
    return "\n".join(lines)


async def maybe_compress(session_id: uuid.UUID, turn_count: int) -> str | None:
    """
    If turn_count is a multiple of 10, compress old messages and save summary.
    Returns new summary string, or None if compression was not triggered.
    """
    # Only fire on every 10th turn (and never on turn 0, before any messages
    # exist) so compression runs periodically instead of on every message.
    if turn_count == 0 or turn_count % 10 != 0:
        return None

    messages = await get_session_messages(session_id)
    if len(messages) < 4:
        return None

    # Compress all but the last 4 messages
    to_compress = messages[:-4]
    if not to_compress:
        return None

    history_text = _format_history(to_compress)
    llm = _get_llm()
    prompt = _COMPRESS_PROMPT.format(history=history_text)
    summary = str(llm.invoke(prompt)).strip()

    await update_session_summary(session_id, summary)
    return summary
