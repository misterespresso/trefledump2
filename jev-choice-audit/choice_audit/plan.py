"""The deterministic work plan.

Everything is derived from config.SEED, so two people running this build exactly
the same request list and can compare results line for line.
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass
from typing import Any

from . import config


@dataclass(frozen=True)
class Item:
    """One question inside one request."""

    qname: str
    date: dt.date

    def as_meta(self) -> dict[str, Any]:
        return {"qname": self.qname, "date": self.date.isoformat(), "truth": config.truth(self.date)}


@dataclass(frozen=True)
class Unit:
    """One HTTP request: a record_id, the experiment it belongs to, and its questions."""

    record_id: str
    experiment: str
    items: tuple[Item, ...]
    repeat: int = 0
    state: Any = None  # None keeps the constant neutral greeting; the probe overrides it

    def body(self) -> dict[str, Any]:
        return {
            "state": config.STATE if self.state is None else self.state,
            "model": config.MODEL,
            "questions": {
                it.qname: {
                    "type": "choice",
                    "instructions": config.question_text(it.date),
                    "criteria": {opt: None for opt in config.OPTIONS},
                }
                for it in self.items
            },
        }


def all_dates() -> list[dt.date]:
    span = (config.DATE_END - config.DATE_START).days + 1
    return [config.DATE_START + dt.timedelta(days=i) for i in range(span)]


def main_dates() -> list[dt.date]:
    """N_MAIN unique dates, uniform over the range, sorted for a stable order."""
    rng = random.Random(config.SEED)
    return sorted(rng.sample(all_dates(), config.N_MAIN))


def determinism_dates(main: list[dt.date] | None = None) -> list[dt.date]:
    """N_DETERMINISM dates drawn from the main sample."""
    main = main if main is not None else main_dates()
    rng = random.Random(config.SEED + 1)
    return sorted(rng.sample(main, config.N_DETERMINISM))


def build_units() -> list[Unit]:
    main = main_dates()
    det = determinism_dates(main)
    units: list[Unit] = []

    for d in main:
        units.append(Unit(f"main:{d.isoformat()}", "main", (Item("q0", d),)))

    for d in det:
        for r in range(config.N_REPEATS):
            units.append(Unit(f"repeat:{d.isoformat()}:{r}", "determinism", (Item("q0", d),), repeat=r))

    for i in range(0, len(det), config.BATCH_SIZE):
        chunk = det[i : i + config.BATCH_SIZE]
        items = tuple(Item(f"q{j}", d) for j, d in enumerate(chunk))
        units.append(Unit(f"batch:{i // config.BATCH_SIZE:03d}", "batch", items))

    return units
