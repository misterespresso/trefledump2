"""Small Jev client: rate limited, retried, appends every exchange to JSONL."""

from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

URL, MODEL, KEY_ENV = "https://api.typesafe.ai/v1/systemone", "jev-latest", "TYPESAFE_API_KEY"
RETRY = frozenset({408, 429, 500, 502, 503, 504})


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.done = {r["record_id"]: r for r in self.read() if r.get("ok")}

    def read(self) -> Iterator[dict[str, Any]]:
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def append(self, rec: dict[str, Any]) -> None:
        with self._lock:
            with self.path.open("a") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
            if rec.get("ok"):
                self.done[rec["record_id"]] = rec


class Jev:
    def __init__(self, per_second: float = 10.0):
        key = os.environ.get(KEY_ENV, "").strip()
        if not key:
            raise SystemExit(f"{KEY_ENV} is not set.")
        self._key, self._gap = key, 1.0 / per_second
        self._lock, self._next = threading.Lock(), 0.0
        self.requests = self.input_tokens = 0

    def ask(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        body = {"state": state, "model": MODEL, "questions": questions}
        wait, last = 1.0, ""
        for attempt in range(1, 6):
            with self._lock:
                now = time.monotonic()
                delay = max(0.0, self._next - now)
                self._next = max(now, self._next) + self._gap
            if delay:
                time.sleep(delay)
            req = urllib.request.Request(
                URL, data=json.dumps(body).encode(), method="POST",
                headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
                         "Accept": "application/json", "User-Agent": "semantic-microscope/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=40) as r:
                    payload = json.load(r)
                with self._lock:
                    self.requests += 1
                    self.input_tokens += (payload.get("usage") or {}).get("input_tokens") or 0
                return {"ok": True, "response": payload, "attempts": attempt}
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"
                if e.code not in RETRY or attempt == 5:
                    return {"ok": False, "error": last, "attempts": attempt}
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
                if attempt == 5:
                    return {"ok": False, "error": last, "attempts": attempt}
            time.sleep(wait * (1 + random.random() * 0.25))
            wait = min(wait * 2, 30.0)
        return {"ok": False, "error": last, "attempts": 5}
