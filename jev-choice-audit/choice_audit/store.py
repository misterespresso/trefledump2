"""Append-only JSONL store that makes a run resumable.

One line per HTTP response, holding the exact request body sent and the exact
response body received. Nothing is ever rewritten, so a crashed run resumes by
skipping the record_ids already present.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterator


class JsonlStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._done: set[str] = set()
        for rec in self.read():
            rid = rec.get("record_id")
            if rid and rec.get("ok"):
                self._done.add(rid)

    def read(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open() as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # A partial final line from a killed process; ignore it and move on.
                    if n > 0:
                        continue

    def has(self, record_id: str) -> bool:
        return record_id in self._done

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True)
        with self._lock:
            with self.path.open("a") as f:
                f.write(line + "\n")
                f.flush()
            if record.get("ok"):
                self._done.add(record["record_id"])

    def __len__(self) -> int:
        return len(self._done)
