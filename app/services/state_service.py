from __future__ import annotations

import json
from pathlib import Path

from app.config import STATE_FILE


def save_project_state(state: dict) -> None:
    """Persist the current project state to disk."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    temp_file = STATE_FILE.with_suffix(".tmp")

    temp_file.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8",
    )

    temp_file.replace(STATE_FILE)


def load_project_state() -> dict | None:
    """Load the persisted project state if it exists."""
    if not STATE_FILE.exists():
        return None

    try:
        return json.loads(
            STATE_FILE.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError):
        return None


def clear_project_state() -> None:
    """Remove persisted project state."""
    STATE_FILE.unlink(missing_ok=True)