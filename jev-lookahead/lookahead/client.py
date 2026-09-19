"""Minimal, self-contained Jev client so this folder stands on its own.

Rate limited, retried with exponential backoff, and every exchange appended to
JSONL so a run is resumable. The API key is read from the environment and is
never logged or stored.
"""

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

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
KEY_ENV = "TYPESAFE_API_KEY"
TIMEOUT = 30.0
MAX_ATTEMPTS = 5
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.done: dict[str, dict[str, Any]] = {}
        for rec in self.read():
            if rec.get("ok"):
                self.done[rec["record_id"]] = rec

    def read(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
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
            raise SystemExit(f"{KEY_ENV} is not set in the environment.")
        self._key = key
        self._interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next = 0.0
        self.requests = 0
        self.input_tokens = 0

    def _wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._interval
        if delay:
            time.sleep(delay)

    def ask(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        body = {"state": state, "model": MODEL, "questions": questions}
        wait = 1.0
        last = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._wait()
            req = urllib.request.Request(
                URL, data=json.dumps(body).encode(), method="POST",
                headers={"Authorization": f"Bearer {self._key}",
                         "Content-Type": "application/json", "Accept": "application/json",
                         "User-Agent": "jev-lookahead/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                    payload = json.loads(r.read().decode())
                with self._lock:
                    self.requests += 1
                    self.input_tokens += (payload.get("usage") or {}).get("input_tokens") or 0
                return {"ok": True, "request": body, "response": payload, "attempts": attempt}
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
                if e.code not in RETRY_STATUS or attempt == MAX_ATTEMPTS:
                    return {"ok": False, "request": body, "error": last, "attempts": attempt}
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
                if attempt == MAX_ATTEMPTS:
                    return {"ok": False, "request": body, "error": last, "attempts": attempt}
            time.sleep(wait * (1 + random.random() * 0.25))
            wait = min(wait * 2, 30.0)
        return {"ok": False, "request": body, "error": last, "attempts": MAX_ATTEMPTS}
