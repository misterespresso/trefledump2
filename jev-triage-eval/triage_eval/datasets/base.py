"""Common record schema shared by every dataset loader."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import pandas as pd

ACUITY_LEVELS = [1, 2, 3, 4, 5]

MENTAL_LABELS = ("alert", "responds to voice", "responds to pain", "unresponsive")


@dataclass
class TriageRecord:
    """One ED visit as seen at nursing triage. Outcome fields are deliberately absent."""

    record_id: str
    dataset: str
    true_acuity: int
    chief_complaint: str
    age: Optional[float] = None
    sex: Optional[str] = None  # "female" | "male"
    arrival_mode: Optional[str] = None  # short free text
    injury: Optional[bool] = None
    mental_status: Optional[str] = None  # one of MENTAL_LABELS
    pain_present: Optional[bool] = None
    pain_score: Optional[float] = None  # 0-10 NRS
    sbp: Optional[float] = None
    dbp: Optional[float] = None
    hr: Optional[float] = None
    rr: Optional[float] = None
    temp_c: Optional[float] = None
    spo2: Optional[float] = None
    nurse_acuity: Optional[int] = None  # human reference, never a model input
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra")
        for k, v in extra.items():
            d[f"extra_{k}"] = v
        return d


def records_to_frame(records: list[TriageRecord]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in records])


def _num(x: Any) -> Optional[float]:
    """Coerce a cell to float, returning None for blanks, junk and sentinel negatives."""
    if x is None:
        return None
    try:
        v = float(str(x).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def plausible(v: Optional[float], lo: float, hi: float) -> Optional[float]:
    if v is None or v < lo or v > hi:
        return None
    return v
