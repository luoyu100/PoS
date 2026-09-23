"""Append-only structured event logging."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Logger:
    """Logger."""

    def __init__(self, path: Path):
        """Initialize configuration and episode-local runtime state."""

        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", encoding="utf-8")

    def log(self, event: str, **data: Any) -> None:
        """Log."""

        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **data,
        }
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self) -> None:
        """Close."""

        self.file.close()
