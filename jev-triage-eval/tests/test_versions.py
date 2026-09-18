from triage_eval.datasets import load_ktas, subset
from triage_eval.esi import HIGH_RISK, build_questions
from triage_eval.jev import JudgmentCache, _questions_hash


def test_versions_differ_only_in_high_risk():
    v1, v2 = build_questions("esi-v1"), build_questions("esi-v2")
    assert set(v1) == set(v2)
    for k in v1:
        if k == "high_risk":
            assert v1[k].model_dump() != v2[k].model_dump()
        else:
            assert v1[k].model_dump() == v2[k].model_dump()
    assert _questions_hash("esi-v1") != _questions_hash("esi-v2")
    assert set(HIGH_RISK) == {"esi-v1", "esi-v2"}


def test_cache_keys_are_version_specific(tmp_path):
    a = JudgmentCache(tmp_path / "c.jsonl", backend="typesafe", version="esi-v1")
    b = JudgmentCache(tmp_path / "c.jsonl", backend="typesafe", version="esi-v2")
    assert a.key != b.key


def test_split_is_fixed_and_stratified():
    recs = load_ktas()
    dev, test = subset(recs, "dev"), subset(recs, "test")
    assert len(dev) + len(test) == len(recs)
    assert {r.record_id for r in dev}.isdisjoint({r.record_id for r in test})
    assert [r.record_id for r in subset(recs, "dev")] == [r.record_id for r in dev]
    for lvl in range(1, 6):
        nd = sum(r.true_acuity == lvl for r in dev)
        nt = sum(r.true_acuity == lvl for r in test)
        assert abs(nd - nt) <= 1
