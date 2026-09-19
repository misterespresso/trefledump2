import numpy as np
from sklearn.metrics import cohen_kappa_score

from triage_eval.datasets import load_ktas
from triage_eval.jev import JudgmentCache, MockBackend, judge_all
from triage_eval.tune import cuts_predict, fit_cuts, load_judgments, make_objectives, qwk, run


def test_qwk_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(1, 6, 300)
    p = np.clip(y + rng.integers(-1, 2, 300), 1, 5)
    assert abs(qwk(y, p) - cohen_kappa_score(y, p, weights="quadratic")) < 1e-9


def test_cuts_are_monotone_and_fit():
    expected = np.array([1.0, 1.4, 2.2, 2.9, 3.6, 4.4, 4.9])
    y = np.array([1, 1, 2, 3, 4, 5, 5])
    cuts = fit_cuts(expected, y, make_objectives(0.1)["qwk"])
    assert cuts == sorted(cuts) and len(cuts) == 4
    assert (cuts_predict(expected, cuts) == y).all()


def test_safe_objective_penalises_under_triage():
    y = np.array([1, 2, 3, 4, 5] * 20)
    obj = make_objectives(0.05)["qwk_safe"]
    assert obj(y, y) == 1.0
    assert obj(y, np.minimum(y + 1, 5)) < obj(y, np.maximum(y - 1, 1))


def test_run_is_out_of_fold(tmp_path):
    recs = load_ktas(limit=300)
    cache = JudgmentCache(tmp_path / "c.jsonl", backend="mock")
    js = judge_all(recs, MockBackend(), cache, workers=2)
    judgments = [js[r.record_id] for r in recs]
    results, chosen, full, pred_df = run(recs, judgments, folds=3, objective="qwk")
    assert len(chosen) == 3 and len(pred_df) == 300
    for k in ("jev_rules_tuned", "jev_score_shift", "jev_score_cuts", "jev_stacked_lr"):
        assert 1 <= pred_df[f"pred_{k}"].min() and pred_df[f"pred_{k}"].max() <= 5
        assert 0 <= results[k]["accuracy"] <= 1


def test_cache_error_names_the_version_mismatch(tmp_path):
    """A cache written for one prompt version must not look like an empty cache to the next."""
    import pytest

    recs = load_ktas(limit=5)
    cache = JudgmentCache(tmp_path / "c.jsonl", backend="mock", version="esi-v1")
    judge_all(recs, MockBackend(), cache, workers=1)

    with pytest.raises(SystemExit) as e:
        load_judgments(tmp_path / "c.jsonl", recs, backend="mock", version="esi-v2")
    msg = str(e.value)
    assert "esi-v1" in msg and "--prompt-version" in msg

    with pytest.raises(SystemExit) as e:
        load_judgments(tmp_path / "missing.jsonl", recs, backend="mock", version="esi-v1")
    assert "No readable cache" in str(e.value)
