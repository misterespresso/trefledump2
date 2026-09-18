import math

from triage_eval.esi import JevJudgment, Policy, build_questions, build_state, combined_distribution, predict, rules_level
from triage_eval.datasets.base import TriageRecord


def judgment(**nouls):
    base = dict(lifesaving=0.02, high_risk=0.1, altered_mental=0.02, severe_distress=0.05, many_resources=0.5, any_resources=0.9)
    base.update(nouls)
    return JevJudgment("r", base, {0: 0.05, 1: 0.1, 2: 0.4, 3: 0.35, 4: 0.1}, 2.35, 0.4)


def test_rules_ladder():
    assert rules_level(judgment(lifesaving=0.9), dz=False) == 1
    assert rules_level(judgment(high_risk=0.8), dz=False) == 2
    assert rules_level(judgment(altered_mental=0.7), dz=False) == 2
    assert rules_level(judgment(many_resources=0.8), dz=False) == 3
    assert rules_level(judgment(many_resources=0.8), dz=True) == 2
    assert rules_level(judgment(many_resources=0.2, any_resources=0.8), dz=False) == 4
    assert rules_level(judgment(many_resources=0.2, any_resources=0.2), dz=False) == 5


def test_distributions_are_normalised_and_gated():
    j = judgment(lifesaving=0.0, high_risk=0.0, altered_mental=0.0, severe_distress=0.0)
    p = combined_distribution(j, dz=False)
    assert math.isclose(sum(p), 1.0, abs_tol=1e-9)
    assert p[0] == 0.0 and p[1] == 0.0
    # tail follows the Score over levels 3-5
    assert p[2] > p[3] > p[4]
    for m in ("jev_score", "jev_rules", "jev_combined"):
        lvl, pr = predict(j, False, m)
        assert 1 <= lvl <= 5 and math.isclose(sum(pr), 1.0, abs_tol=1e-9)


def test_policy_thresholds():
    j = judgment(high_risk=0.45)
    assert rules_level(j, dz=False, pol=Policy()) == 3
    assert rules_level(j, dz=False, pol=Policy(t_high_risk=0.4)) == 2


def test_state_never_leaks_outcomes():
    r = TriageRecord("id", "ktas", 2, "chest pain", nurse_acuity=1, extra={"diagnosis": "acute MI", "disposition": "ICU"})
    s = build_state(r)

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from walk(v)
        elif isinstance(node, list):
            for v in node:
                yield from walk(v)
        else:
            yield node

    leaves = {str(x) for x in walk(s)}
    for leak in ("nurse_acuity", "true_acuity", "diagnosis", "disposition", "acute MI", "ICU", "extra"):
        assert leak not in leaves


def test_questions_shape():
    qs = build_questions()
    kinds = {k: q.model_dump()["type"] for k, q in qs.items()}
    assert kinds["acuity"] == "score"
    assert sum(v == "noul" for v in kinds.values()) == 6
    assert len(qs["acuity"].criteria) == 5
