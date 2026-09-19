"""HTTP client for POST /v1/systemone: rate limit, retries with exponential backoff,
and a request counter. Uses only the standard library so the harness has no hidden
dependency on an SDK version.

The API key is read from the environment and never logged, printed or stored.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from . import config


class RateLimiter:
    """Coarse limiter: no two request starts closer than 1/rate seconds."""

    def __init__(self, per_second: float):
        self._min_interval = 1.0 / per_second if per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self._min_interval
        if wait:
            time.sleep(wait)


@dataclass
class Counter:
    requests: int = 0
    retries: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, *, retries: int, ok: bool, usage: dict[str, Any] | None) -> None:
        with self._lock:
            self.requests += 1
            self.retries += retries
            if not ok:
                self.failures += 1
            if usage:
                self.input_tokens += usage.get("input_tokens") or 0
                self.output_tokens += usage.get("output_tokens") or 0


class TransientError(RuntimeError):
    """Worth retrying."""


class PermanentError(RuntimeError):
    """Not worth retrying; the request itself is wrong."""


class JevClient:
    def __init__(self, rate: float = config.MAX_REQUESTS_PER_SECOND, counter: Counter | None = None):
        key = os.environ.get(config.API_KEY_ENV, "").strip()
        if not key:
            raise SystemExit(
                f"{config.API_KEY_ENV} is not set in the environment. "
                "Export the existing key; do not pass it on the command line."
            )
        self._key = key
        self._url = config.BASE_URL.rstrip("/") + config.ENDPOINT
        self.limiter = RateLimiter(rate)
        self.counter = counter or Counter()

    # -- one attempt ---------------------------------------------------------
    def _attempt(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        req = urllib.request.Request(
            self._url,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "jev-choice-audit/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=config.TIMEOUT_S) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")[:800]
            if e.code in config.RETRY_STATUSES:
                raise TransientError(f"HTTP {e.code}: {raw}") from e
            raise PermanentError(f"HTTP {e.code}: {raw}") from e
        except urllib.error.URLError as e:
            raise TransientError(f"connection error: {e.reason}") from e
        except TimeoutError as e:
            raise TransientError("timeout") from e
        except json.JSONDecodeError as e:
            raise TransientError(f"malformed JSON body: {e}") from e

    # -- with retries --------------------------------------------------------
    def post(self, body: dict[str, Any]) -> dict[str, Any]:
        """Return a record describing the call: never raises for transport failure."""
        delay = config.BACKOFF_INITIAL_S
        last = ""
        t0 = time.time()
        for attempt in range(1, config.MAX_ATTEMPTS + 1):
            self.limiter.acquire()
            started = time.monotonic()
            try:
                status, payload = self._attempt(body)
            except PermanentError as e:
                last = str(e)
                self.counter.record(retries=attempt - 1, ok=False, usage=None)
                return {"ok": False, "attempts": attempt, "error": last, "latency_s": round(time.monotonic() - started, 4), "started_at": t0}
            except TransientError as e:
                last = str(e)
                if attempt == config.MAX_ATTEMPTS:
                    break
                time.sleep(min(delay, config.BACKOFF_MAX_S) * (1 + random.random() * 0.25))
                delay = min(delay * 2, config.BACKOFF_MAX_S)
                continue
            self.counter.record(retries=attempt - 1, ok=True, usage=payload.get("usage"))
            return {
                "ok": True,
                "attempts": attempt,
                "http_status": status,
                "response": payload,
                "latency_s": round(time.monotonic() - started, 4),
                "started_at": t0,
            }
        self.counter.record(retries=config.MAX_ATTEMPTS - 1, ok=False, usage=None)
        return {"ok": False, "attempts": config.MAX_ATTEMPTS, "error": last, "latency_s": round(time.time() - t0, 4), "started_at": t0}
