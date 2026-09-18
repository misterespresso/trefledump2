"""Deterministic vital-sign bucketing. Runs in code, never in the model.

Buckets are attached to every number sent to Jev and also feed the ML feature
matrix, so both approaches see the same clinical categories. Thresholds are
adult ranges except heart rate and respiratory rate, which follow the ESI
paediatric danger-zone table by age.
"""

from __future__ import annotations

from typing import Optional

from .datasets.base import TriageRecord

NOT_RECORDED = "not recorded"


def _band(v: Optional[float], edges: list[tuple[float, str]], top: str) -> str:
    """edges: ascending (upper_bound_exclusive, label). Value >= last bound -> top."""
    if v is None:
        return NOT_RECORDED
    for bound, label in edges:
        if v < bound:
            return label
    return top


def bucket_sbp(v):
    return _band(v, [(90, "hypotensive"), (100, "low"), (140, "normal"), (180, "elevated")], "severely elevated")


def bucket_dbp(v):
    return _band(v, [(60, "low"), (90, "normal"), (110, "elevated")], "severely elevated")


def hr_limits(age: Optional[float]) -> tuple[float, float]:
    """(bradycardia below, tachycardia above) by age, ESI danger-zone style."""
    if age is None or age >= 8:
        return 60, 100
    if age < 0.25:
        return 100, 180
    if age < 3:
        return 90, 160
    return 70, 140


def rr_limits(age: Optional[float]) -> tuple[float, float]:
    if age is None or age >= 8:
        return 12, 20
    if age < 0.25:
        return 30, 50
    if age < 3:
        return 20, 40
    return 16, 30


def bucket_hr(v, age=None):
    lo, hi = hr_limits(age)
    return _band(
        v,
        [(lo - 20, "severe bradycardia"), (lo, "bradycardia"), (hi + 1, "normal"), (hi + 21, "tachycardia"), (hi + 51, "marked tachycardia")],
        "severe tachycardia",
    )


def bucket_rr(v, age=None):
    lo, hi = rr_limits(age)
    return _band(
        v,
        [(lo - 4, "severely low"), (lo, "low"), (hi + 1, "normal"), (hi + 5, "elevated"), (hi + 11, "high")],
        "severely high",
    )


def bucket_spo2(v):
    return _band(v, [(85, "critical hypoxia"), (90, "severe hypoxia"), (92, "hypoxia"), (95, "borderline")], "normal")


def bucket_temp(v):
    return _band(v, [(35.0, "hypothermia"), (37.6, "normal"), (38.5, "low-grade fever"), (40.0, "fever")], "high fever")


def bucket_pain(v):
    return _band(v, [(1, "none"), (4, "mild"), (7, "moderate")], "severe")


def bucket_age(v):
    return _band(v, [(1, "infant"), (12, "child"), (18, "adolescent"), (65, "adult"), (80, "older adult")], "very old")


ABNORMAL = {"normal", NOT_RECORDED, "none", "mild", "borderline", "low-grade fever"}


def vitals_summary(rec: TriageRecord) -> dict:
    """Numbers plus their bucket, in the shape sent to Jev."""

    def item(value, unit, bucket):
        return {"value": value, "unit": unit, "bucket": bucket} if value is not None else {"value": None, "bucket": NOT_RECORDED}

    return {
        "systolic_bp": item(rec.sbp, "mmHg", bucket_sbp(rec.sbp)),
        "diastolic_bp": item(rec.dbp, "mmHg", bucket_dbp(rec.dbp)),
        "heart_rate": item(rec.hr, "bpm", bucket_hr(rec.hr, rec.age)),
        "respiratory_rate": item(rec.rr, "breaths/min", bucket_rr(rec.rr, rec.age)),
        "spo2": item(rec.spo2, "%", bucket_spo2(rec.spo2)),
        "temperature": item(rec.temp_c, "C", bucket_temp(rec.temp_c)),
    }


def abnormal_vitals(rec: TriageRecord) -> list[str]:
    out = []
    for name, v in vitals_summary(rec).items():
        if v["bucket"] not in ABNORMAL:
            out.append(f"{name}: {v['bucket']}")
    return out


def danger_zone(rec: TriageRecord) -> bool:
    """ESI decision point D: HR, RR or SpO2 outside the danger-zone limits for age."""
    _, hr_hi = hr_limits(rec.age)
    _, rr_hi = rr_limits(rec.age)
    return bool(
        (rec.hr is not None and rec.hr > hr_hi)
        or (rec.rr is not None and rec.rr > rr_hi)
        or (rec.spo2 is not None and rec.spo2 < 92)
    )


def shock_index(rec: TriageRecord) -> Optional[float]:
    if rec.hr is None or rec.sbp is None or rec.sbp <= 0:
        return None
    return round(rec.hr / rec.sbp, 2)
