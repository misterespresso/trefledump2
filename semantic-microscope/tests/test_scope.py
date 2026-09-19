"""Offline checks. Nothing here touches the network or the API key."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scope import analyse, lenses  # noqa: E402

DATA = ROOT / "data"


# ---- the instrument itself ------------------------------------------------

def test_every_lens_is_a_distinct_readable_sentence():
    seen = set()
    for key, (group, text) in lenses.LENSES.items():
        assert group in lenses.GROUPS, key
        assert text.endswith("."), key
        assert text[0].isupper(), key
        assert len(text) > 25, key
        assert text not in seen, f"duplicate wording: {key}"
        seen.add(text)
    assert len(lenses.LENSES) == 40


def test_questions_are_nouls_carrying_the_lens_text():
    qs = lenses.questions()
    assert set(qs) == set(lenses.LENSES)
    for key, q in qs.items():
        assert q["type"] == "noul"
        assert q["instructions"] == lenses.LENSES[key][1]


def test_state_withholds_everything_used_as_ground_truth():
    pkg = {"name": "fastapi", "summary": "FastAPI framework, high performance",
           "topics": ["Internet"], "maturity": "4 - Beta", "rank": 12, "n_releases": 90}
    blob = json.dumps(lenses.state_for(pkg)).lower()
    assert "fastapi framework" in blob
    for withheld in ("internet", "beta", "4 -", "n_releases"):
        assert withheld not in blob, withheld
    assert "name" not in lenses.state_for(pkg)


# ---- the scoring ----------------------------------------------------------

def test_auroc_is_half_on_noise_and_one_on_a_giveaway():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 60)
    noise = rng.normal(size=(120, 4))
    assert analyse.auroc_cv(noise, y) == pytest.approx(0.5, abs=0.15)
    giveaway = np.column_stack([y + rng.normal(scale=0.01, size=120), noise])
    assert analyse.auroc_cv(giveaway, y) > 0.99


def test_auroc_refuses_a_target_too_small_to_fold():
    y = np.array([1] + [0] * 60)
    assert np.isnan(analyse.auroc_cv(np.zeros((61, 3)), y))


# ---- the measured data ----------------------------------------------------

@pytest.mark.skipif(not (DATA / "matrix.json").exists(), reason="no run recorded")
def test_readings_are_probabilities_on_the_hundredth_grid():
    matrix = json.loads((DATA / "matrix.json").read_text())
    keys = set(matrix["lenses"])
    assert keys == set(lenses.LENSES)
    for pkg in matrix["packages"]:
        assert set(pkg["readings"]) == keys, pkg["name"]
        for key, v in pkg["readings"].items():
            assert 0.0 <= v <= 1.0, (pkg["name"], key, v)
            assert abs(v * 100 - round(v * 100)) < 1e-6, (pkg["name"], key, v)


@pytest.mark.skipif(not (DATA / "analysis.json").exists(), reason="no analysis recorded")
def test_the_page_payload_matches_the_analysis():
    import scope.page as page

    page.main()
    text = (ROOT / "page" / "data.js").read_text()
    payload = json.loads(text.split("=", 1)[1].rstrip().rstrip(";\n").rstrip(";"))
    analysis = json.loads((DATA / "analysis.json").read_text())
    assert len(payload["packages"]) == analysis["n_packages"]
    assert len(payload["lenses"]) == analysis["n_lenses"]
    assert all(len(p["r"]) == 40 for p in payload["packages"])
    assert payload["meanAurocLenses"] == analysis["mean_auroc_lenses"]
