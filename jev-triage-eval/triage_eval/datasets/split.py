"""Fixed, stratified dev/test split so prompt iteration can be scored on untouched patients.

Rule: iterate prompt wording on `dev` only. `test` is looked at once per prompt
version. The split is deterministic (seed 0) and identical across runs.
"""

from __future__ import annotations

from sklearn.model_selection import train_test_split

from .base import TriageRecord

SUBSETS = ("all", "dev", "test")


def subset(records: list[TriageRecord], name: str = "all", seed: int = 0) -> list[TriageRecord]:
    if name == "all":
        return list(records)
    if name not in SUBSETS:
        raise ValueError(f"unknown subset {name!r}")
    y = [r.true_acuity for r in records]
    dev, test = train_test_split(records, test_size=0.5, random_state=seed, stratify=y)
    return list(dev if name == "dev" else test)
