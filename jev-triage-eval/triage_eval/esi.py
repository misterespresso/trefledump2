"""ESI-style decision points as TypeSafe questions, and the code that combines them.

One request per patient carries six Nouls (the ESI decision points A-C, split so
each is a single yes/no judgment) and one Score over the five acuity levels.
Decision point D (danger-zone vitals) is computed in code from the buckets.
Everything below is policy that lives in code: thresholds, the ladder, and the
way the Score distribution is blended in. Raw answers are kept so the policy
can be re-run without new inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from typesafe_sdk import Noul, Score

from .buckets import abnormal_vitals, danger_zone, shock_index, vitals_summary, bucket_pain, bucket_age
from .datasets.base import TriageRecord

PROMPT_VERSION = "esi-v1"

CONTEXT = (
    "Emergency department nursing triage at the moment of arrival. Only the chief complaint, "
    "arrival details, mental status, pain and the first set of vital signs are known. No "
    "examination findings, labs, imaging or diagnosis exist yet. The triage scale has five "
    "levels (1 resuscitation, 2 emergent, 3 urgent, 4 less urgent, 5 non-urgent), as in the "
    "Canadian and Korean triage scales and the Emergency Severity Index."
)

LEVEL_RUBRIC = [
    "Level 1, resuscitation: needs an immediate life-saving intervention now. Examples: cardiac "
    "or respiratory arrest, unresponsive, severe respiratory distress, SpO2 below 85 percent, "
    "shock with hypotension, active major haemorrhage, anaphylaxis, major trauma with unstable vitals.",
    "Level 2, emergent: high-risk situation that could deteriorate quickly, or new confusion or "
    "lethargy, or severe pain or distress. Examples: chest pain suspicious for a heart attack, "
    "stroke symptoms, severe dyspnoea with hypoxia, sepsis picture with abnormal vitals, active GI "
    "bleeding with tachycardia, suicidal ideation, severe abdominal pain in an older adult.",
    "Level 3, urgent: stable vital signs but likely to need two or more ED resources such as blood "
    "tests, ECG, imaging, IV fluids or IV medication, or a specialist consultation. Examples: "
    "abdominal pain, moderate dyspnoea with normal SpO2, fever in an adult without red flags, "
    "dizziness needing work-up, headache without neurological signs.",
    "Level 4, less urgent: needs one ED resource. Examples: simple laceration needing sutures, a "
    "sprain needing an X-ray, urinary symptoms needing a urine test, mild allergic rash needing one medication.",
    "Level 5, non-urgent: needs no ED resources beyond an examination or a prescription. Examples: "
    "medication refill, minor scrape, cold symptoms with normal vitals, suture removal.",
]


def build_state(rec: TriageRecord) -> dict[str, Any]:
    """Everything the model sees for one patient. Outcomes and nurse level are never included."""
    vit = vitals_summary(rec)
    return {
        "context": CONTEXT,
        "patient": {
            "age_years": rec.age,
            "age_group": bucket_age(rec.age),
            "sex": rec.sex or "not recorded",
            "arrival_mode": rec.arrival_mode or "not recorded",
            "injury_related_visit": rec.injury if rec.injury is not None else "not recorded",
        },
        "presentation": {
            "chief_complaint": rec.chief_complaint or "not recorded",
            "mental_status": rec.mental_status or "not recorded",
            "pain": {
                "present": rec.pain_present if rec.pain_present is not None else "not recorded",
                "numeric_rating_0_to_10": rec.pain_score,
                "bucket": bucket_pain(rec.pain_score),
            },
        },
        "vital_signs": vit,
        "derived_in_code": {
            "abnormal_vitals": abnormal_vitals(rec),
            "danger_zone_vitals": danger_zone(rec),
            "shock_index": shock_index(rec),
        },
    }


def build_questions() -> dict[str, Noul | Score]:
    return {
        "lifesaving": Noul(
            instructions=(
                "Decision point A. Based on `presentation` and `vital_signs`, does this patient need an "
                "immediate life-saving intervention right now (airway or ventilation support, CPR, "
                "defibrillation, emergency haemorrhage control, fluids or vasopressors for shock, "
                "naloxone or other emergency drugs for a threatened airway or circulation)?"
            ),
            criteria={
                "true": "Unresponsive, in arrest, in severe respiratory distress, profoundly hypoxic or hypotensive, "
                "or bleeding uncontrollably. Level 1 territory.",
                "false": "Breathing and circulation are adequate for the next several minutes even if the patient is very sick.",
            },
        ),
        "high_risk": Noul(
            instructions=(
                "Decision point B, part 1. Is this a high-risk situation where a serious time-critical "
                "condition is plausible from the chief complaint, age and vital signs and the patient "
                "could deteriorate quickly if they waited? Consider acute coronary syndrome, stroke, sepsis, "
                "pulmonary embolism, ectopic pregnancy, GI bleeding, testicular or ovarian torsion, acute "
                "abdomen in the elderly, overdose, and psychiatric emergency."
            ),
            criteria={
                "true": "A physician would want to see this patient within about ten minutes.",
                "false": "The presentation can safely wait for routine assessment.",
            },
        ),
        "altered_mental": Noul(
            instructions=(
                "Decision point B, part 2. Is the patient newly confused, lethargic, disoriented or otherwise "
                "not fully alert? Use `presentation.mental_status` and the chief complaint (for example "
                "'mental change', 'drowsy', 'syncope with confusion')."
            ),
            criteria={"true": "Any acute change from alert.", "false": "Alert and oriented."},
        ),
        "severe_distress": Noul(
            instructions=(
                "Decision point B, part 3. Is the patient in severe pain or severe distress that warrants "
                "immediate intervention? Severe means a pain rating of 7 or more that is consistent with the "
                "complaint and vital signs, or marked physiological or psychological distress."
            ),
            criteria={
                "true": "Severe, credible pain or distress needing treatment now.",
                "false": "Pain is absent, mild, moderate, or a high rating is not supported by the rest of the picture.",
            },
        ),
        "many_resources": Noul(
            instructions=(
                "Decision point C. Assuming the patient is stable, would an experienced ED nurse expect this "
                "visit to need two or more distinct resources? Resources: blood or urine tests, ECG, X-ray, "
                "CT, ultrasound, IV fluids, IV or IM medication, nebuliser, specialist consultation, a "
                "procedure such as laceration repair or fracture reduction. A physical examination, an oral "
                "medication or a prescription do not count."
            ),
            criteria={"true": "Two or more resources.", "false": "One resource or none."},
        ),
        "any_resources": Noul(
            instructions=(
                "Decision point C, continued. Would this visit need at least one ED resource as defined above "
                "(tests, imaging, IV treatment, procedure or consultation)?"
            ),
            criteria={"true": "At least one resource.", "false": "No resources; examination and prescription only."},
        ),
        "acuity": Score(
            instructions=(
                "Rate the triage acuity of this patient on the five-level scale described in `context`, "
                "using all fields. Level 1 is the most urgent."
            ),
            criteria=LEVEL_RUBRIC,
        ),
    }


NOUL_KEYS = ("lifesaving", "high_risk", "altered_mental", "severe_distress", "many_resources", "any_resources")


@dataclass
class JevJudgment:
    """The raw, reusable answers for one patient."""

    record_id: str
    nouls: dict[str, float]
    score_probs: dict[int, float]  # keys 0..4, meaning level = key + 1
    score_expected: float
    score_confidence: float
    model: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_s: float | None = None
    backend: str = "typesafe"

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "nouls": self.nouls,
            "score_probs": {str(k): v for k, v in self.score_probs.items()},
            "score_expected": self.score_expected,
            "score_confidence": self.score_confidence,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_s": self.latency_s,
            "backend": self.backend,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JevJudgment":
        return cls(
            record_id=d["record_id"],
            nouls=dict(d["nouls"]),
            score_probs={int(k): float(v) for k, v in d["score_probs"].items()},
            score_expected=float(d["score_expected"]),
            score_confidence=float(d["score_confidence"]),
            model=d.get("model", ""),
            input_tokens=d.get("input_tokens"),
            output_tokens=d.get("output_tokens"),
            latency_s=d.get("latency_s"),
            backend=d.get("backend", "typesafe"),
        )


@dataclass
class Policy:
    """Thresholds for the rule ladder. Tune on a held-out slice, not on the test set."""

    t_lifesaving: float = 0.5
    t_high_risk: float = 0.5
    t_altered: float = 0.5
    t_distress: float = 0.5
    t_many: float = 0.5
    t_any: float = 0.5
    danger_zone_weight: float = 0.6  # ESI says "consider" level 2; weight on many_resources when D fires


def _norm(p: list[float]) -> list[float]:
    s = sum(p)
    return [x / s for x in p] if s > 0 else [0.2] * 5


def score_distribution(j: JevJudgment) -> list[float]:
    """P(level 1..5) straight from the Score question."""
    return _norm([j.score_probs.get(k, 0.0) for k in range(5)])


def rules_level(j: JevJudgment, dz: bool, pol: Policy = Policy()) -> int:
    """The classic ESI ladder, hard-thresholded."""
    n = j.nouls
    if n["lifesaving"] >= pol.t_lifesaving:
        return 1
    if n["high_risk"] >= pol.t_high_risk or n["altered_mental"] >= pol.t_altered or n["severe_distress"] >= pol.t_distress:
        return 2
    if n["many_resources"] >= pol.t_many:
        return 2 if dz else 3
    if n["any_resources"] >= pol.t_any:
        return 4
    return 5


def rules_distribution(j: JevJudgment, dz: bool, pol: Policy = Policy()) -> list[float]:
    """Turn the ladder into a probability over levels using the Noul probabilities.

    P1 = A;  P2 = (1-A) * max(B-branch, dz * many);  the remainder is split over 3-5
    by the resource questions: P3 ~ many, P4 ~ any & not many, P5 ~ not any.
    """
    n = j.nouls
    p1 = n["lifesaving"]
    b = max(n["high_risk"], n["altered_mental"], n["severe_distress"], (pol.danger_zone_weight * n["many_resources"]) if dz else 0.0)
    p2 = (1 - p1) * b
    rest = 1 - p1 - p2
    p3 = rest * n["many_resources"]
    p4 = rest * (1 - n["many_resources"]) * n["any_resources"]
    p5 = rest * (1 - n["many_resources"]) * (1 - n["any_resources"])
    return _norm([p1, p2, p3, p4, p5])


def combined_distribution(j: JevJudgment, dz: bool, pol: Policy = Policy()) -> list[float]:
    """Gates for levels 1-2 come from the Nouls; the mass left over is shaped by the Score over 3-5."""
    r = rules_distribution(j, dz, pol)
    s = score_distribution(j)
    rest = r[2] + r[3] + r[4]
    tail = _norm(s[2:5])
    return _norm([r[0], r[1], rest * tail[0], rest * tail[1], rest * tail[2]])


METHODS = {
    "jev_score": lambda j, dz, pol: score_distribution(j),
    "jev_rules": lambda j, dz, pol: rules_distribution(j, dz, pol),
    "jev_combined": lambda j, dz, pol: combined_distribution(j, dz, pol),
}


def predict(j: JevJudgment, dz: bool, method: str = "jev_combined", pol: Policy = Policy()) -> tuple[int, list[float]]:
    probs = METHODS[method](j, dz, pol)
    if method == "jev_rules":
        return rules_level(j, dz, pol), probs
    return int(max(range(5), key=lambda k: probs[k])) + 1, probs
