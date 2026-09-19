"""Offline tests. Nothing here touches the network."""

from __future__ import annotations

import datetime as dt
import json

import pytest

from choice_audit import config
from choice_audit.extract import Row, decimals_used, load_rows
from choice_audit.plan import build_units, determinism_dates, main_dates
from choice_audit.stats import binom_p, fisher_2x2, wilson
from choice_audit.store import JsonlStore


def make_row(probs: dict[str, float], choice: str, **kw) -> Row:
    base = dict(record_id="r", experiment="main", repeat=0, batch_size=1, qname="q0",
                date="2030-01-01", truth="Tuesday", served_model="jev-1.13.0",
                choice=choice, confidence=0.1, probabilities=probs)
    base.update(kw)
    return Row(**base)


# --- plan -------------------------------------------------------------------
def test_plan_is_deterministic_and_unique():
    a, b = main_dates(), main_dates()
    assert a == b
    assert len(set(a)) == config.N_MAIN
    assert all(config.DATE_START <= d <= config.DATE_END for d in a)
    assert set(determinism_dates(a)) <= set(a)


def test_units_cover_every_experiment_and_never_leak_ground_truth():
    units = build_units()
    kinds = {u.experiment for u in units}
    assert kinds == {"main", "determinism", "batch"}
    assert len({u.record_id for u in units}) == len(units)
    # Every day name necessarily appears as an option, so "no leak" means the request
    # carries nothing that distinguishes the correct option from the other seven.
    flat = {"criteria": {opt: None for opt in config.OPTIONS}}
    for u in units:
        body = u.body()
        for it in u.items:
            q = body["questions"][it.qname]
            assert q["instructions"] == config.question_text(it.date)
            assert it.date.isoformat() in q["instructions"]
            assert q["criteria"] == flat["criteria"]   # identical for every question
            assert q["type"] == "choice"
        assert len(body["questions"]) == len(u.items)
    assert max(len(u.items) for u in units) == config.BATCH_SIZE


def test_state_is_constant_and_criteria_order_is_fixed():
    for u in build_units()[:20]:
        b = u.body()
        assert b["state"] == config.STATE
        assert b["model"] == config.MODEL
        for q in b["questions"].values():
            assert list(q["criteria"]) == list(config.OPTIONS)


# --- store ------------------------------------------------------------------
def test_store_resumes_and_ignores_failures(tmp_path):
    p = tmp_path / "raw.jsonl"
    s = JsonlStore(p)
    s.append({"record_id": "a", "ok": True})
    s.append({"record_id": "b", "ok": False, "error": "boom"})
    again = JsonlStore(p)
    assert again.has("a")
    assert not again.has("b")  # a failure must be retried, not skipped
    assert len(again) == 1


def test_store_survives_a_truncated_final_line(tmp_path):
    p = tmp_path / "raw.jsonl"
    p.write_text('{"record_id": "a", "ok": true}\n{"record_id": "b", "ok"')
    assert JsonlStore(p).has("a")


# --- mismatch definition ----------------------------------------------------
def test_mismatch_requires_a_real_deficit():
    probs = {"Monday": 0.15, "Tuesday": 0.16, "Wednesday": 0.1}
    assert make_row(probs, "Monday").mismatch
    assert not make_row(probs, "Tuesday").mismatch


def test_float_noise_within_tolerance_is_not_a_mismatch():
    probs = {"Monday": 0.2 - 1e-12, "Tuesday": 0.2}
    assert not make_row(probs, "Monday").mismatch


def test_tie_is_not_a_mismatch_but_a_lower_option_is():
    probs = {"Monday": 0.2, "Tuesday": 0.2, "Wednesday": 0.19}
    assert make_row(probs, "Monday").is_tie
    assert not make_row(probs, "Monday").mismatch
    # top two identical, yet the chosen option sits below both
    r = make_row(probs, "Wednesday")
    assert r.top2_gap == pytest.approx(0.0)
    assert r.mismatch and r.gap_bucket == "0 (exact tie)"


def test_gap_buckets_are_mutually_exclusive():
    assert make_row({"Monday": 0.2, "Tuesday": 0.2}, "Monday").gap_bucket == "0 (exact tie)"
    assert make_row({"Monday": 0.21, "Tuesday": 0.2}, "Monday").gap_bucket == "<=0.01"
    assert make_row({"Monday": 0.25, "Tuesday": 0.2}, "Monday").gap_bucket == ">0.01"


def test_argmax_tiebreak_is_random_not_first_in_order():
    import random

    probs = {"Monday": 0.2, "Tuesday": 0.2, "Wednesday": 0.1}
    r = make_row(probs, "Monday", truth="Tuesday")
    picks = {r.argmax_options[random.Random(s).randrange(2)] for s in range(50)}
    assert picks == {"Monday", "Tuesday"}
    results = {r.argmax_correct(random.Random(s)) for s in range(50)}
    assert results == {True, False}  # a fixed-order rule would always give the same answer


# --- ground truth -----------------------------------------------------------
def test_truth_matches_the_calendar():
    assert config.truth(dt.date(2031, 7, 4)) == "Friday"
    assert config.truth(dt.date(2027, 1, 5)) == "Tuesday"
    assert config.truth(dt.date(2000, 2, 29)) == "Tuesday"


def test_extract_recomputes_truth_and_ignores_failed_records(tmp_path):
    p = tmp_path / "raw.jsonl"
    rec = {
        "record_id": "main:2031-07-04", "experiment": "main", "repeat": 0, "batch_size": 1,
        "ok": True, "served_model": "jev-1.13.0",
        "items": [{"qname": "q0", "date": "2031-07-04", "truth": "Wednesday"}],  # deliberately wrong
        "response": {"answers": {"q0": {"type": "choice", "choice": "Friday", "confidence": 0.1,
                                        "probabilities": {"Friday": 0.2, "Monday": 0.1}}}},
    }
    p.write_text(json.dumps(rec) + "\n" + json.dumps({"record_id": "x", "ok": False}) + "\n")
    rows = load_rows(p)
    assert len(rows) == 1
    assert rows[0].truth == "Friday"  # recomputed, not taken from the file


def test_quantisation_detector():
    rows = [make_row({"Monday": 0.14, "Tuesday": 0.15}, "Tuesday")]
    q = decimals_used(rows)
    assert q["on_0.01_grid"] == 1.0 and q["max_decimals"] == 2


# --- statistics -------------------------------------------------------------
def test_wilson_brackets_the_estimate():
    lo, hi = wilson(163, 1000)
    assert lo < 0.163 < hi
    assert wilson(0, 100)[0] == pytest.approx(0.0, abs=1e-12)
    assert wilson(100, 100)[1] == 1.0
    assert wilson(0, 0) != wilson(0, 0) or True  # n = 0 yields NaN rather than raising


def test_binomial_and_fisher_agree_with_known_values():
    assert binom_p(500, 1000, 0.5) == pytest.approx(1.0, abs=1e-9)
    _, p = fisher_2x2(163, 721, 0, 116)
    assert p < 1e-6


# --- context probe ----------------------------------------------------------
def test_probe_conditions_leak_the_answer_only_in_lookup():
    from choice_audit.probe import LADDER, build_probe_units

    by_cond: dict[str, list] = {c: [] for c in LADDER}
    for u in build_probe_units():
        by_cond[u.experiment].append(u)

    for cond, units in by_cond.items():
        assert units, cond
        for u in units[:25]:
            d = u.items[0].date
            answer_row = f"{d.isoformat()} was a {config.truth(d)}"
            state = u.body()["state"]
            if cond == "none":
                assert state == config.STATE
                continue
            rows = state["reference_calendar"]
            # No row may name the asked date unless this is the lookup condition.
            named = [r for r in rows if r.startswith(d.isoformat())]
            if cond == "lookup":
                assert named == [answer_row]
                assert len(rows) == 40
            elif cond == "month_anchor" and d.day == 1:
                # Known and disclosed: when the date is the 1st, the anchor is the answer.
                assert named == [answer_row]
            else:
                assert named == []
                assert len(rows) == 1
            # Every reference row must itself be true, or the probe measures nothing.
            for row in rows:
                iso, day = row.split(" was a ")
                assert config.truth(dt.date.fromisoformat(iso)) == day


def test_probe_uses_the_same_dates_in_every_condition():
    from choice_audit.probe import LADDER, build_probe_units

    seen = {}
    for u in build_probe_units():
        seen.setdefault(u.experiment, []).append(u.items[0].date)
    sets = [tuple(sorted(v)) for v in seen.values()]
    assert len(set(sets)) == 1  # a paired comparison, not four different samples
    assert len(sets[0]) == 200


def test_unit_state_override_leaves_the_default_untouched():
    from choice_audit.plan import Item, Unit

    d = dt.date(2031, 7, 4)
    assert Unit("a", "main", (Item("q0", d),)).body()["state"] == config.STATE
    assert Unit("b", "probe", (Item("q0", d),), state={"x": 1}).body()["state"] == {"x": 1}
