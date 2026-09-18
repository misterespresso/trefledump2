"""Smoke-test the TypeSafe key: list models, judge one KTAS patient, print the raw answers.

    export TYPESAFE_API_KEY=...   # never paste the key into a prompt or commit it
    python scripts/check_key.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triage_eval.datasets import load_ktas  # noqa: E402
from triage_eval.esi import build_questions, build_state  # noqa: E402


def main() -> int:
    if not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set", file=sys.stderr)
        return 2
    from typesafe_sdk import TypeSafeClient

    with TypeSafeClient(timeout=30.0) as client:
        models = client.models.list()
        print("models:", [m.name for m in models.models])
        rec = load_ktas(limit=1)[0]
        resp = client.system_one(state=build_state(rec), questions=build_questions())
        print("model used:", resp.model, "| usage:", resp.usage.input_tokens, "in /", resp.usage.output_tokens, "out")
        print("nouls:", json.dumps({k: round(v.noul, 3) for k, v in resp.nouls.items()}, indent=1))
        a = resp.scores["acuity"]
        print("acuity: expected level", round(a.score + 1, 2), "confidence", round(a.confidence, 3))
        print("probabilities by level:", {k + 1: round(v, 3) for k, v in a.probabilities.items()})
        print("true expert level:", rec.true_acuity, "| nurse:", rec.nurse_acuity)
    return 0


if __name__ == "__main__":
    sys.exit(main())
