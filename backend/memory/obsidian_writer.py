"""Write each response as a markdown note to the Obsidian vault.

Runs as a FastAPI BackgroundTask — never blocks the SSE stream.
Path: {VAULT}/{SUBFOLDER}/{YYYY-MM-DD}/{session_title}/{HH-MM-SS}.md
"""
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend.config import settings


def _sanitise(name: str) -> str:
    """Remove filesystem-unsafe characters from a filename segment."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    # Cap length to stay well under filesystem path-component limits, and strip
    # leading/trailing dots/underscores so we never emit a hidden (dotfile) or
    # trailing-junk folder/file name.
    return name[:80].strip("._")


def write_note(
    session_id: uuid.UUID,
    session_title: str,
    query: str,
    response: str,
    model_used: str,
    route_path: str,
    sources: list[dict],
    temporal_facts: dict | None,
    active_constraints: list[dict],
) -> str | None:
    """
    Write the markdown note. Returns note path on success, None on failure.
    Silently catches all exceptions so the chat stream is never disrupted.
    """
    if not settings.obsidian_enabled:
        return None

    try:
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")

        # Fall back to a short session-id prefix if the title sanitises to
        # empty, so we never collide on / write into a nameless folder.
        safe_title = _sanitise(session_title) or _sanitise(str(session_id)[:8])
        # Naming scheme: {vault}/{subfolder}/{date}/{session_title}/{time}.md
        # (e.g. LegalChat/2026-07-21/My Session/14-30-00.md) — one file per
        # turn, grouped by day and then by session, so a session's notes read
        # as a chronological thread within its own folder.
        note_dir = (
            Path(settings.obsidian_vault_path)
            / settings.obsidian_subfolder
            / date_str
            / safe_title
        )
        note_dir.mkdir(parents=True, exist_ok=True)
        note_path = note_dir / f"{time_str}.md"

        # Build tags from source types
        source_types = list({s.get("source_type", "unknown") for s in sources})
        tags = ["legal-rag"] + source_types

        # Simple "filename p.N" representation of each retrieved source, reused
        # for the frontmatter's `sources` list (same data already rendered in
        # the "## Sources" body section below).
        source_labels = [
            f"{s.get('filename', '')} p.{s.get('page', '')}" for s in sources
        ]

        # Frontmatter
        lines = [
            "---",
            f"date: {now.isoformat()}",
            f"session: {safe_title}",
            f"session_id: {session_id}",
            f"model: {model_used}",
            f"route_path: {route_path}",
            f"tags: [{', '.join(tags)}]",
        ]
        if source_labels:
            lines.append("sources:")
            for label in source_labels:
                lines.append(f'  - "{label}"')
        else:
            lines.append("sources: []")
        if active_constraints:
            cstr = ", ".join(f"{c['type']}: {c['value']}" for c in active_constraints)
            lines.append(f"constraints: \"{cstr}\"")
        lines += ["---", ""]

        # Body
        lines += [f"## Query\n\n{query}\n", f"## Response\n\n{response}\n"]

        if sources:
            lines.append("## Sources\n")
            for s in sources:
                src_type = s.get("source_type", "")
                fname = s.get("filename", "")
                page = s.get("page", "")
                lines.append(f"- **[{src_type}]** {fname} (page {page})")
            lines.append("")

        if temporal_facts:
            lines.append("## Temporal facts\n")
            for k, v in temporal_facts.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        if active_constraints:
            lines.append("## Active constraints\n")
            for c in active_constraints:
                perm = " *(permanent)*" if c.get("is_permanent") else ""
                lines.append(f"- **{c['type']}**: {c['value']}{perm}")
            lines.append("")

        note_path.write_text("\n".join(lines), encoding="utf-8")
        return str(note_path)

    except Exception as exc:
        # Never propagate — log silently
        print(f"[obsidian_writer] Failed to write note: {exc}")
        return None
