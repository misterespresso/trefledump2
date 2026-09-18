from triage_eval.buckets import bucket_hr, bucket_rr, bucket_sbp, bucket_spo2, bucket_temp, danger_zone, NOT_RECORDED
from triage_eval.datasets.base import TriageRecord


def rec(**kw):
    base = dict(record_id="t", dataset="test", true_acuity=3, chief_complaint="x")
    base.update(kw)
    return TriageRecord(**base)


def test_adult_buckets():
    assert bucket_sbp(85) == "hypotensive"
    assert bucket_sbp(120) == "normal"
    assert bucket_sbp(185) == "severely elevated"
    assert bucket_hr(84) == "normal"
    assert bucket_hr(110) == "tachycardia"
    assert bucket_hr(35) == "severe bradycardia"
    assert bucket_rr(18) == "normal"
    assert bucket_rr(26) == "high"
    assert bucket_spo2(88) == "severe hypoxia"
    assert bucket_spo2(98) == "normal"
    assert bucket_temp(38.0) == "low-grade fever"
    assert bucket_temp(None) == NOT_RECORDED


def test_paediatric_limits_shift_with_age():
    assert bucket_hr(150, age=1) == "normal"
    assert bucket_hr(150, age=30) == "marked tachycardia"
    assert bucket_rr(35, age=1) == "normal"
    assert bucket_rr(35, age=30) == "severely high"


def test_danger_zone():
    assert not danger_zone(rec(hr=90, rr=18, spo2=97))
    assert danger_zone(rec(hr=105))
    assert danger_zone(rec(spo2=91))
    assert not danger_zone(rec(age=2, hr=150))
    assert danger_zone(rec(age=2, hr=170))
