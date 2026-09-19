"""Turn raw JSONL records into one flat row per question. No API calls, no stats."""

from __future__ import annotations

import datetime as dt
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config
from .store import JsonlStore


@dataclass
class Row:
    record_id: str
    experiment: str
    repeat: int
    batch_size: int
    qname: str
    date: str
    truth: str
    served_model: str
    choice: str
    confidence: float
    probabilities: dict[str, float]
    latency_s: float = 0.0

    # -- derived, all computed from `probabilities` with an explicit tolerance --
    @property
    def p_choice(self) -> float:
        return self.probabilities.get(self.choice, float("nan"))

    @property
    def p_max(self) -> float:
        return max(self.probabilities.values())

    @property
    def argmax_options(self) -> list[str]:
        """Every option within TOL of the maximum, in the fixed option order."""
        m = self.p_max
        return [o for o in config.OPTIONS if o in self.probabilities and self.probabilities[o] >= m - config.TOL]

    @property
    def is_tie(self) -> bool:
        return len(self.argmax_options) > 1

    @property
    def mismatch(self) -> bool:
        """`choice` did not take a maximum-probability option, beyond float tolerance."""
        return self.p_choice < self.p_max - config.TOL

    @property
    def deficit(self) -> float:
        """How far below the maximum the selected option sits."""
        return self.p_max - self.p_choice

    @property
    def top2_gap(self) -> float:
        """Top-1 minus top-2 probability. Zero when the top is tied."""
        vals = sorted(self.probabilities.values(), reverse=True)
        return vals[0] - vals[1] if len(vals) > 1 else vals[0]

    @property
    def gap_bucket(self) -> str:
        g = self.top2_gap
        if g <= config.TOL:
            return "0 (exact tie)"
        if g <= config.NEAR_TIE + config.TOL:
            return "<=0.01"
        return ">0.01"

    @property
    def prob_sum(self) -> float:
        return sum(self.probabilities.values())

    @property
    def p_unknown(self) -> float:
        return self.probabilities.get("Unknown", float("nan"))

    @property
    def choice_correct(self) -> bool:
        return self.choice == self.truth

    def argmax_correct(self, rng: random.Random) -> bool:
        """Accuracy of taking the highest probability, ties broken at random (seeded)."""
        opts = self.argmax_options
        return rng.choice(opts) == self.truth

    @property
    def choice_in_criteria(self) -> bool:
        return self.choice in config.OPTIONS


def load_rows(path: Path | str = config.RAW_JSONL) -> list[Row]:
    store = JsonlStore(Path(path))
    rows: list[Row] = []
    for rec in store.read():
        if not rec.get("ok"):
            continue
        answers = rec.get("response", {}).get("answers", {})
        meta = {m["qname"]: m for m in rec.get("items", [])}
        for qname, ans in answers.items():
            if ans.get("type") != "choice":
                continue
            m = meta.get(qname, {})
            date = m.get("date", "")
            # Recompute ground truth here rather than trusting what collection stored.
            truth = config.truth(dt.date.fromisoformat(date)) if date else m.get("truth", "")
            rows.append(
                Row(
                    record_id=rec["record_id"],
                    experiment=rec.get("experiment", ""),
                    repeat=int(rec.get("repeat", 0)),
                    batch_size=int(rec.get("batch_size", 1)),
                    qname=qname,
                    date=date,
                    truth=truth,
                    served_model=rec.get("served_model", ""),
                    choice=ans["choice"],
                    confidence=float(ans["confidence"]),
                    probabilities={k: float(v) for k, v in ans["probabilities"].items()},
                    latency_s=float(rec.get("latency_s", 0.0)),
                )
            )
    rows.sort(key=lambda r: (r.experiment, r.record_id, r.qname))
    return rows


def decimals_used(rows: list[Row]) -> dict[str, Any]:
    """Is the reported probability vector quantised? A 0.01 grid would make a ~0.01
    mismatch explainable by rounding alone, so this has to be established first."""
    vals = [p for r in rows for p in r.probabilities.values()]
    if not vals:
        return {}
    def on_grid(step: float) -> float:
        return sum(1 for v in vals if abs(v / step - round(v / step)) < 1e-6) / len(vals)
    return {
        "n_values": len(vals),
        "on_0.01_grid": on_grid(0.01),
        "on_0.001_grid": on_grid(0.001),
        "distinct_values": len({round(v, 10) for v in vals}),
        "min": min(vals),
        "max": max(vals),
        "max_decimals": max(len(f"{v:.10f}".rstrip("0").split(".")[1]) for v in vals),
        # The wire carries full float64 repr, so a value can sit an ULP off the grid
        # (e.g. 0.13999999999999999). Distinguish exact equality from grid membership.
        "exactly_equal_to_own_2dp": sum(1 for v in vals if v == round(v, 2)) / len(vals),
        "max_deviation_from_grid": max(abs(v - round(v, 2)) for v in vals),
    }
