"""Load and save session state from Postgres.

Context loading order (highest priority first):
  1. Permanent constraints (is_permanent=True, this user)
  2. Active session constraints (this session_id)
  3. Session summary
  4. Last N raw messages
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select, desc

from backend.config import settings
from backend.database import AsyncSessionLocal
from backend.models import Constraint, Message, Session
from backend.memory.constraint_tracker import get_permanent_constraints, get_session_constraints


@dataclass
class SessionContext:
    session_id: uuid.UUID
    user_id: str
    title: str
    summary: str | None
    permanent_constraints: list[Constraint]
    session_constraints: list[Constraint]
    recent_messages: list[Message]
    turn_count: int = 0


async def get_or_create_session(
    session_id: uuid.UUID | None,
    user_id: str = settings.default_user_id,
    title: str = "New Chat",
) -> uuid.UUID:
    """Return the given session's id if it exists, else create a new session."""
    async with AsyncSessionLocal() as db:
        if session_id:
            row = await db.get(Session, session_id)
            if row:
                return row.id

        new_session = Session(
            user_id=user_id,
            title=title,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            is_active=True,
        )
        db.add(new_session)
        await db.commit()
        await db.refresh(new_session)
        return new_session.id


async def load_session_context(
    session_id: uuid.UUID,
    user_id: str = settings.default_user_id,
) -> SessionContext:
    """Load full session context in priority order."""
    permanent = await get_permanent_constraints(user_id)
    session_cons = await get_session_constraints(session_id)

    async with AsyncSessionLocal() as db:
        sess = await db.get(Session, session_id)
        summary = sess.summary if sess else None
        title = sess.title if sess else "Chat"

        # Order newest-first so LIMIT keeps the most recent N messages, then
        # reverse back to chronological order for use in prompts/UI.
        result = await db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(desc(Message.created_at))
            .limit(settings.max_context_messages)
        )
        messages_desc = list(result.scalars().all())
        messages = list(reversed(messages_desc))

        # Total turn count is queried separately (unbounded) because
        # `messages` above is capped by max_context_messages and can't be used
        # to derive the true count needed for the compression trigger.
        count_result = await db.execute(
            select(Message).where(Message.session_id == session_id)
        )
        turn_count = len(list(count_result.scalars().all()))

    return SessionContext(
        session_id=session_id,
        user_id=user_id,
        title=title,
        summary=summary,
        permanent_constraints=permanent,
        session_constraints=session_cons,
        recent_messages=messages,
        turn_count=turn_count,
    )


async def save_message(
    session_id: uuid.UUID,
    role: str,
    content: str,
    model_used: str | None = None,
    route_path: str | None = None,
    sources: list[dict] | None = None,
    temporal_facts: dict | None = None,
) -> Message:
    """Persist a chat message and bump the parent session's updated_at."""
    async with AsyncSessionLocal() as db:
        msg = Message(
            session_id=session_id,
            role=role,
            content=content,
            model_used=model_used,
            route_path=route_path,
            sources=sources,
            temporal_facts=temporal_facts,
            created_at=datetime.now(timezone.utc),
        )
        db.add(msg)

        # Update session updated_at
        sess = await db.get(Session, session_id)
        if sess:
            sess.updated_at = datetime.now(timezone.utc)

        await db.commit()
        await db.refresh(msg)
        return msg


async def list_sessions(user_id: str) -> list[Session]:
    """List this user's active sessions, most recently updated first."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Session)
            .where(Session.user_id == user_id, Session.is_active == True)
            .order_by(desc(Session.updated_at))
        )
        return list(result.scalars().all())


async def get_session_messages(session_id: uuid.UUID) -> list[Message]:
    """Load all messages for a session in chronological order."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at)
        )
        return list(result.scalars().all())


async def delete_session(session_id: uuid.UUID) -> None:
    """Soft-delete a session by marking it inactive (does not remove rows)."""
    async with AsyncSessionLocal() as db:
        sess = await db.get(Session, session_id)
        if sess:
            sess.is_active = False
            await db.commit()


async def update_session_summary(session_id: uuid.UUID, summary: str) -> None:
    """Overwrite the session's stored factual summary."""
    async with AsyncSessionLocal() as db:
        sess = await db.get(Session, session_id)
        if sess:
            sess.summary = summary
            sess.updated_at = datetime.now(timezone.utc)
            await db.commit()
