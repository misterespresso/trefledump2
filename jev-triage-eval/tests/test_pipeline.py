import json
from pathlib import Path

import numpy as np

from triage_eval.datasets import load_ktas
from triage_eval.jev import JudgmentCache, MockBackend, judge_all
from triage_eval.metrics import evaluate
from triage_eval.run import main


def test_ktas_loader():
    recs = load_ktas()
    assert len(recs) == 1267
    assert {r.true_acuity for r in recs} == {1, 2, 3, 4, 5}
    assert all(r.nurse_acuity in {1, 2, 3, 4, 5} for r in recs)
    assert all(r.chief_complaint for r in recs)
    assert recs[0].temp_c == 36.6 and recs[0].sbp == 160


def test_cache_roundtrip(tmp_path: Path):
    recs = load_ktas(limit=7)
    cache = JudgmentCache(tmp_path / "c.jsonl", backend="mock")
    first = judge_all(recs, MockBackend(), cache, workers=3)
    assert len(first) == 7 and len(cache) == 7
    reloaded = JudgmentCache(tmp_path / "c.jsonl", backend="mock")
    assert len(reloaded) == 7
    calls = []

    class Exploding:
        def judge(self, r):
            calls.append(r.record_id)
            raise AssertionError("should be served from cache")

    second = judge_all(recs, Exploding(), reloaded, workers=2)
    assert not calls and second[recs[0].record_id].nouls == first[recs[0].record_id].nouls


def test_evaluate_perfect_and_off_by_one():
    y = np.array([1, 2, 3, 4, 5, 3, 3])
    m = evaluate(y, y, np.eye(5)[y - 1])
    assert m["accuracy"] == 1.0 and m["under_triage_rate"] == 0.0 and m["ece_top1"] == 0.0
    m2 = evaluate(y, np.minimum(y + 1, 5), None)
    assert m2["under_triage_rate"] > 0.8 and m2["within_one_level"] == 1.0


def test_end_to_end_mock(tmp_path: Path):
    out = tmp_path / "run"
    rc = main(["--dataset", "ktas", "--limit", "120", "--jev-backend", "mock", "--ml-models", "logreg", "--folds", "3", "--out", str(out), "--workers", "2"])
    assert rc == 0
    m = json.loads((out / "metrics.json").read_text())
    assert "jev_combined[mock]" in m["results"] and "logreg" in m["results"]
    assert (out / "calibration.png").exists() and (out / "confusion_logreg.png").exists()
    assert (out / "predictions.csv").read_text().count("\n") == 121
