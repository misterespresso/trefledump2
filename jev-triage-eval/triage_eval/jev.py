"""Calling Jev: one request per patient, cached, with a keyless mock for dry runs."""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Optional

from .buckets import danger_zone
from .datasets.base import TriageRecord
from .esi import DEFAULT_PROMPT_VERSION, NOUL_KEYS, JevJudgment, build_questions, build_state


def _questions_hash(version: str = DEFAULT_PROMPT_VERSION) -> str:
    qs = {k: q.model_dump() for k, q in build_questions(version).items()}
    return hashlib.sha256(json.dumps(qs, sort_keys=True).encode()).hexdigest()[:12]


class JudgmentCache:
    """Append-only JSONL keyed by record id + prompt version so re-runs cost nothing."""

    def __init__(self, path: Path, backend: str, version: str = DEFAULT_PROMPT_VERSION):
        self.path = Path(path)
        self.version = version
        self.key = f"{version}:{_questions_hash(version)}:{backend}"
        self._lock = threading.Lock()
        self._items: dict[str, JevJudgment] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("cache_key") == self.key:
                    self._items[d["record_id"]] = JevJudgment.from_dict(d)

    def get(self, record_id: str) -> Optional[JevJudgment]:
        return self._items.get(record_id)

    def put(self, j: JevJudgment) -> None:
        d = j.to_dict()
        d["cache_key"] = self.key
        with self._lock:
            self._items[j.record_id] = j
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(d) + "\n")

    def __len__(self) -> int:
        return len(self._items)


class TypeSafeBackend:
    """Thin wrapper over the official SDK. Reads TYPESAFE_API_KEY from the environment."""

    name = "typesafe"

    def __init__(self, model: Optional[str] = None, timeout: float = 30.0, version: str = DEFAULT_PROMPT_VERSION):
        from typesafe_sdk import RetryPolicy, TypeSafeClient

        if not os.environ.get("TYPESAFE_API_KEY"):
            raise RuntimeError("TYPESAFE_API_KEY is not set. Export it, or use --jev-backend mock (dry run) or cached (no new requests).")
        self.client = TypeSafeClient(model=model, timeout=timeout, retry=RetryPolicy(max_retries=4, timeout=120.0))
        self.questions = build_questions(version)

    def judge(self, rec: TriageRecord) -> JevJudgment:
        t0 = time.perf_counter()
        resp = self.client.system_one(state=build_state(rec), questions=self.questions)
        dt = time.perf_counter() - t0
        acuity = resp.scores["acuity"]
        return JevJudgment(
            record_id=rec.record_id,
            nouls={k: float(resp.nouls[k].noul) for k in NOUL_KEYS},
            score_probs={int(k): float(v) for k, v in acuity.probabilities.items()},
            score_expected=float(acuity.score),
            score_confidence=float(acuity.confidence),
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_s=round(dt, 4),
            backend=self.name,
        )

    def close(self) -> None:
        self.client.close()


class MockBackend:
    """Deterministic, vitals-driven stand-in so the whole pipeline runs without a key.

    It is intentionally crude (it barely reads the complaint) and exists to exercise
    the plumbing, the cache, the combination policy and the plots. Never report its
    numbers as Jev results; every output it produces is tagged backend="mock".
    """

    name = "mock"

    RED_FLAGS = ("chest pain", "dyspnea", "hematemesis", "melena", "syncope", "mental change", "dysarthria", "seizure", "hemoptysis", "stroke", "palsy", "unconscious")
    LOW_FLAGS = ("rash", "refill", "wound", "abrasion", "sprain", "cold", "sore throat", "cough", "constipation", "tooth", "medication")

    def judge(self, rec: TriageRecord) -> JevJudgment:
        rng = random.Random(rec.record_id)
        cc = (rec.chief_complaint or "").lower()
        ms = rec.mental_status or "alert"
        lifesaving = 0.05
        if ms == "unresponsive" or (rec.spo2 is not None and rec.spo2 < 85) or (rec.sbp is not None and rec.sbp < 80):
            lifesaving = 0.85
        elif ms == "responds to pain":
            lifesaving = 0.45
        high_risk = 0.15 + 0.5 * any(f in cc for f in self.RED_FLAGS) + 0.2 * danger_zone(rec) + 0.1 * ((rec.age or 0) >= 65)
        altered = {"alert": 0.05, "responds to voice": 0.7, "responds to pain": 0.9, "unresponsive": 0.98}[ms]
        if "mental" in cc or "drowsy" in cc:
            altered = max(altered, 0.7)
        distress = 0.1 + (0.6 if (rec.pain_score or 0) >= 7 else 0.15 if (rec.pain_score or 0) >= 4 else 0.0)
        low = any(f in cc for f in self.LOW_FLAGS)
        many = 0.75 - 0.5 * low + 0.15 * danger_zone(rec) + 0.1 * bool(rec.arrival_mode and "ambulance" in rec.arrival_mode)
        anyr = 0.9 - 0.35 * low

        def clip(x):
            return min(0.99, max(0.01, x + rng.uniform(-0.08, 0.08)))

        nouls = {
            "lifesaving": clip(lifesaving),
            "high_risk": clip(high_risk),
            "altered_mental": clip(altered),
            "severe_distress": clip(distress),
            "many_resources": clip(many),
            "any_resources": clip(anyr),
        }
        centre = 1 if nouls["lifesaving"] > 0.5 else 2 if max(nouls["high_risk"], nouls["altered_mental"], nouls["severe_distress"]) > 0.5 else 3 if nouls["many_resources"] > 0.5 else 4 if nouls["any_resources"] > 0.5 else 5
        raw = [0.0] * 5
        for k in range(5):
            raw[k] = 1.0 / (1 + abs((k + 1) - centre)) ** 2 + rng.uniform(0, 0.15)
        s = sum(raw)
        probs = {k: raw[k] / s for k in range(5)}
        expected = sum(k * p for k, p in probs.items())
        return JevJudgment(
            record_id=rec.record_id,
            nouls=nouls,
            score_probs=probs,
            score_expected=expected,
            score_confidence=max(probs.values()),
            model="mock-heuristic",
            latency_s=0.0,
            backend=self.name,
        )

    def close(self) -> None:
        pass


class CachedBackend:
    """Serves only what is already in the cache; any miss is an error. Lets analyses run without a key."""

    name = "typesafe"  # shares the real backend's cache entries

    def judge(self, rec: TriageRecord) -> JevJudgment:
        raise RuntimeError(f"{rec.record_id} is not in the cache and --jev-backend cached makes no requests")

    def close(self) -> None:
        pass


def make_backend(name: str, model: Optional[str] = None, version: str = DEFAULT_PROMPT_VERSION):
    if name == "typesafe":
        return TypeSafeBackend(model=model, version=version)
    if name == "cached":
        return CachedBackend()
    if name == "mock":
        return MockBackend()
    raise ValueError(f"unknown backend {name!r}")


def judge_all(
    records: Iterable[TriageRecord],
    backend,
    cache: JudgmentCache,
    workers: int = 8,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict[str, JevJudgment]:
    records = list(records)
    out: dict[str, JevJudgment] = {}
    todo = []
    for r in records:
        hit = cache.get(r.record_id)
        if hit is not None:
            out[r.record_id] = hit
        else:
            todo.append(r)
    done = len(out)
    if progress:
        progress(done, len(records))
    if not todo:
        return out
    errors: list[tuple[str, Exception]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(backend.judge, r): r for r in todo}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                j = fut.result()
            except Exception as e:  # keep going; report at the end
                errors.append((r.record_id, e))
                continue
            cache.put(j)
            out[r.record_id] = j
            done += 1
            if progress:
                progress(done, len(records))
    if errors:
        sample = "; ".join(f"{rid}: {type(e).__name__}: {e}" for rid, e in errors[:3])
        raise RuntimeError(f"{len(errors)} of {len(todo)} Jev requests failed. First: {sample}")
    return out
