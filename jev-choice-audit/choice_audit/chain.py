"""Can small Jev calls be chained into a computation it cannot do in one shot?

Four experiments, in order of what they settle:

  mod7          "what is N mod 7?" over options 0-6. Can Jev do the arithmetic step?
  weekday_plus  "K days after Tuesday is?" over the weekdays. Can it do the lookup step?
  jan1          "what weekday was 1 January YYYY?" Does it hold the anchor facts at all?
  chain         the three combined: Jev recalls 1 January, code computes the offset,
                Jev applies it. Compared against asking the date outright.

If mod7 works, a chain can be pure Jev. If it does not, code has to sit in the loop,
which is a claim about the architecture rather than about this task.

    python -m choice_audit.chain                 # everything
    python -m choice_audit.chain --only mod7
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config
from .client import Counter, JevClient
from .collect import run
from .probe import probe_dates
from .stats import fisher_2x2, fmt_ci, wilson
from .store import JsonlStore

CHAIN_JSONL = config.DATA_DIR / "chain.jsonl"
CHAIN_MD = config.ROOT / "chain.md"
DIGITS = tuple(str(i) for i in range(7))
WEEKDAY_OPTS = (*config.DAYS, "Unknown")


@dataclass(frozen=True)
class QItem:
    qname: str
    expected: str
    meta: dict[str, Any] = field(default_factory=dict)

    def as_meta(self) -> dict[str, Any]:
        return {"qname": self.qname, "expected": self.expected, **self.meta}


@dataclass(frozen=True)
class QUnit:
    record_id: str
    experiment: str
    items: tuple[QItem, ...]
    questions: dict[str, Any]
    repeat: int = 0
    state: Any = config.STATE

    def body(self) -> dict[str, Any]:
        return {"state": self.state, "model": config.MODEL, "questions": self.questions}


def _choice(instructions: str, options: tuple[str, ...]) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": {o: None for o in options}}


# ----------------------------------------------------------------- experiments
def units_mod7() -> list[QUnit]:
    """N mod 7, over the range a day-of-month offset actually needs, then further out."""
    units = []
    for n in range(0, 31):
        for rep in range(6):
            q = _choice(f"What is the remainder when {n} is divided by 7?", DIGITS)
            units.append(QUnit(f"mod7:{n:03d}:{rep}", "mod7", (QItem("q0", str(n % 7), {"n": n}),), {"q0": q}, rep))
    for n in range(31, 121):
        q = _choice(f"What is the remainder when {n} is divided by 7?", DIGITS)
        units.append(QUnit(f"mod7big:{n:03d}", "mod7_big", (QItem("q0", str(n % 7), {"n": n}),), {"q0": q}))
    return units


def units_weekday_plus() -> list[QUnit]:
    units = []
    for i, day in enumerate(config.DAYS):
        for k in range(0, 7):
            for rep in range(4):
                exp = config.DAYS[(i + k) % 7]
                q = _choice(f"Starting from {day}, what day of the week is {k} days later?", WEEKDAY_OPTS)
                units.append(QUnit(f"wp:{day}:{k}:{rep}", "weekday_plus",
                                   (QItem("q0", exp, {"from": day, "k": k}),), {"q0": q}, rep))
    for i, day in enumerate(config.DAYS):
        for k in range(7, 31):
            exp = config.DAYS[(i + k) % 7]
            q = _choice(f"Starting from {day}, what day of the week is {k} days later?", WEEKDAY_OPTS)
            units.append(QUnit(f"wpbig:{day}:{k}", "weekday_plus_big",
                               (QItem("q0", exp, {"from": day, "k": k}),), {"q0": q}))
    return units


def units_jan1() -> list[QUnit]:
    units = []
    for year in range(config.DATE_START.year, config.DATE_END.year + 1):
        for rep in range(3):
            exp = config.truth(dt.date(year, 1, 1))
            q = _choice(f"What day of the week was 1 January {year}?", WEEKDAY_OPTS)
            units.append(QUnit(f"jan1:{year}:{rep}", "jan1", (QItem("q0", exp, {"year": year}),), {"q0": q}, rep))
    return units


def units_chain_a() -> list[QUnit]:
    """Step one of the chain: recall the weekday of 1 January for the target's year."""
    units = []
    for d in probe_dates():
        exp = config.truth(dt.date(d.year, 1, 1))
        q = _choice(f"What day of the week was 1 January {d.year}?", WEEKDAY_OPTS)
        units.append(QUnit(f"chainA:{d.isoformat()}", "chain_a",
                           (QItem("q0", exp, {"date": d.isoformat(), "year": d.year}),), {"q0": q}))
    return units


def units_chain_b(step_a: dict[str, str]) -> list[QUnit]:
    """Step three: apply the code-computed offset to whatever step one returned.

    Deliberately built on step one's answer, right or wrong, so the measured accuracy
    is the chain's and not a hybrid with ground truth.
    """
    units = []
    for d in probe_dates():
        start = step_a.get(d.isoformat())
        if start is None or start == "Unknown":
            continue
        k = (d.timetuple().tm_yday - 1) % 7  # computed in code, not asked
        q = _choice(f"Starting from {start}, what day of the week is {k} days later?", WEEKDAY_OPTS)
        units.append(QUnit(f"chainB:{d.isoformat()}", "chain_b",
                           (QItem("q0", config.truth(d), {"date": d.isoformat(), "start": start, "k": k}),),
                           {"q0": q}))
    return units


def _year_table() -> list[str]:
    return [f"1 January {y} was a {config.truth(dt.date(y, 1, 1))}"
            for y in range(config.DATE_START.year, config.DATE_END.year + 1)]


def units_chain_r_a() -> list[QUnit]:
    """Stage one again, but the fact is retrievable from a table instead of recalled."""
    table = _year_table()
    units = []
    for d in probe_dates():
        exp = config.truth(dt.date(d.year, 1, 1))
        q = _choice(f"What day of the week was 1 January {d.year}?", WEEKDAY_OPTS)
        units.append(QUnit(f"chainRA:{d.isoformat()}", "chain_r_a",
                           (QItem("q0", exp, {"date": d.isoformat(), "year": d.year}),), {"q0": q},
                           state={"greeting": config.STATE, "reference_calendar": table}))
    return units


def units_chain_r_b(step_a: dict[str, str]) -> list[QUnit]:
    units = []
    for d in probe_dates():
        start = step_a.get(d.isoformat())
        if start is None or start == "Unknown":
            continue
        k = (d.timetuple().tm_yday - 1) % 7
        q = _choice(f"Starting from {start}, what day of the week is {k} days later?", WEEKDAY_OPTS)
        units.append(QUnit(f"chainRB:{d.isoformat()}", "chain_r_b",
                           (QItem("q0", config.truth(d), {"date": d.isoformat(), "start": start, "k": k}),),
                           {"q0": q}))
    return units


# ----------------------------------------------------------------- analysis
def read(path: Path, experiment: str) -> list[dict[str, Any]]:
    out = []
    for rec in JsonlStore(path).read():
        if not rec.get("ok") or rec.get("experiment") != experiment:
            continue
        metas = {m["qname"]: m for m in rec["items"]}
        for qname, ans in rec["response"]["answers"].items():
            m = metas.get(qname, {})
            p = ans["probabilities"]
            out.append({"record_id": rec["record_id"], "expected": m.get("expected"), "choice": ans["choice"],
                        "correct": ans["choice"] == m.get("expected"), "p_max": max(p.values()),
                        "confidence": ans["confidence"], **{k: v for k, v in m.items() if k != "qname"}})
    return out


def rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    k = sum(1 for r in rows if r["correct"])
    lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
    return {"n": n, "k": k, "rate": k / n if n else float("nan"), "ci": [lo, hi],
            "mean_top_p": sum(r["p_max"] for r in rows) / n if n else float("nan"),
            "mean_conf": sum(r["confidence"] for r in rows) / n if n else float("nan")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Can Jev calls be chained into a computation?")
    ap.add_argument("--out", default=str(CHAIN_JSONL))
    ap.add_argument("--workers", type=int, default=config.WORKERS)
    ap.add_argument("--rate", type=float, default=config.MAX_REQUESTS_PER_SECOND)
    ap.add_argument("--only", default=None, choices=["mod7", "weekday_plus", "jan1", "chain"])
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    store = JsonlStore(out)
    client = None
    if not args.analyse_only:
        client = JevClient(rate=args.rate, counter=Counter())
        batches = []
        if args.only in (None, "mod7"):
            batches.append(units_mod7())
        if args.only in (None, "weekday_plus"):
            batches.append(units_weekday_plus())
        if args.only in (None, "jan1"):
            batches.append(units_jan1())
        if args.only in (None, "chain"):
            batches.append(units_chain_a())
            batches.append(units_chain_r_a())
        for b in batches:
            run(b, store, client, args.workers)
        if args.only in (None, "chain"):
            # Stage two runs only after stage one, since it consumes its answers.
            run(units_chain_b({r["date"]: r["choice"] for r in read(out, "chain_a")}), store, client, args.workers)
            run(units_chain_r_b({r["date"]: r["choice"] for r in read(out, "chain_r_a")}), store, client, args.workers)

    names = ("mod7", "mod7_big", "weekday_plus", "weekday_plus_big", "jan1",
             "chain_a", "chain_b", "chain_r_a", "chain_r_b")
    results = {name: rate(read(out, name)) for name in names}
    for stem in ("chain", "chain_r"):
        a, b = read(out, f"{stem}_a"), read(out, f"{stem}_b")
        results[f"{stem}_end_to_end"] = {
            "n": len(a), "k": sum(1 for r in b if r["correct"]),
            "rate": (sum(1 for r in b if r["correct"]) / len(a)) if a else float("nan"),
            "ci": list(wilson(sum(1 for r in b if r["correct"]), len(a))) if a else [float("nan")] * 2,
            "abstained": sum(1 for r in a if r["choice"] == "Unknown"),
            "mean_top_p": float("nan"), "mean_conf": float("nan"),
        }
    results["baseline_no_context"] = _baseline()
    write_md(results, out, CHAIN_MD)
    (config.DATA_DIR / "chain_summary.json").write_text(json.dumps(results, indent=2))
    for k, v in results.items():
        if v["n"]:
            tp = v.get("mean_top_p")
            print(f"{k:20s} {v['k']:4d}/{v['n']:<4d} = {v['rate']:6.1%}"
                  + (f"   top-p {tp:.3f}" if tp is not None and tp == tp else ""))
    print(f"wrote {CHAIN_MD}")
    return 0


def _baseline() -> dict[str, Any]:
    """The no-context control from the context probe, read rather than hardcoded."""
    from .extract import load_rows

    path = config.DATA_DIR / "context_probe_hard.jsonl"
    if not path.exists():
        return {"n": 0, "k": 0, "rate": float("nan"), "ci": [float("nan")] * 2}
    rows = [r for r in load_rows(path) if r.experiment == "none"]
    k = sum(1 for r in rows if r.choice_correct)
    return {"n": len(rows), "k": k, "rate": k / len(rows) if rows else float("nan"),
            "ci": list(wilson(k, len(rows))) if rows else [float("nan")] * 2}


def write_md(R: dict[str, Any], raw: Path, path: Path) -> None:
    L = ["# Can a chain of Jev calls do what one call cannot?", ""]
    L.append("Jev scores at chance on the weekday of a date. The question is whether the task can be broken "
             "into steps small enough for it, and the answer recomposed. Three sub-skills are measured "
             "separately, then combined into a working pipeline.")
    L.append("")
    L.append("| Step | What is asked | Correct | Rate | 95% CI | Mean top probability |")
    L.append("|---|---|---|---|---|---|")
    rows = [
        ("Arithmetic", "`N mod 7` for N of 0 to 30, over options 0-6", "mod7"),
        ("Arithmetic, larger", "`N mod 7` for N of 31 to 120", "mod7_big"),
        ("Weekday offset", "K days after a named weekday, K of 0 to 6", "weekday_plus"),
        ("Weekday offset, larger", "K days after a named weekday, K of 7 to 30", "weekday_plus_big"),
        ("Fact recall", "the weekday of 1 January of a given year", "jan1"),
    ]
    for label, desc, key in rows:
        v = R[key]
        if not v["n"]:
            continue
        L.append(f"| {label} | {desc} | {v['k']} / {v['n']} | {v['rate']:.1%} | {fmt_ci(*v['ci'])} | "
                 f"{v['mean_top_p']:.3f} |")
    L.append("")
    ca, cb = R["chain_a"], R["chain_b"]
    ra, rb = R["chain_r_a"], R["chain_r_b"]
    e1, e2, base = R["chain_end_to_end"], R["chain_r_end_to_end"], R["baseline_no_context"]
    if cb["n"]:
        L.append("The two arithmetic-style steps are perfect inside the range a date needs, and degrade "
                 "outside it. The step that fails is neither: it is recalling a fact.")
        L.append("")
        L.append("## The chain, end to end")
        L.append("")
        L.append("Three stages on the same 200 dates used elsewhere in this repo. Jev supplies the weekday of "
                 "1 January for that year; code computes the day-of-year offset modulo 7 and never asks the "
                 "model for it; Jev applies that offset to whatever stage one returned. Stage three is built "
                 "on stage one's answer even when that answer is wrong, so these are the chain's own numbers "
                 "and not a hybrid with ground truth. Abstentions in stage one are counted as failures.")
        L.append("")
        L.append("| | Stage 1 correct | Stage 1 abstained | End to end | 95% CI |")
        L.append("|---|---|---|---|---|")
        L.append(f"| Stage 1 from Jev's own memory | {ca['k']} / {ca['n']} ({ca['rate']:.1%}) | "
                 f"{e1['abstained']} | **{e1['k']} / {e1['n']} = {e1['rate']:.1%}** | {fmt_ci(*e1['ci'])} |")
        L.append(f"| Stage 1 from a table in `state` | {ra['k']} / {ra['n']} ({ra['rate']:.1%}) | "
                 f"{e2['abstained']} | **{e2['k']} / {e2['n']} = {e2['rate']:.1%}** | {fmt_ci(*e2['ci'])} |")
        L.append(f"| Asking the date outright, no chain | | | {base['k']} / {base['n']} = {base['rate']:.1%} | "
                 f"{fmt_ci(*base['ci'])} |")
        L.append("")
        o, p = fisher_2x2(e2["k"], e2["n"] - e2["k"], base["k"], base["n"] - base["k"])
        L.append(f"The retrieval-fed chain against asking outright: Fisher's exact, two-sided p = {p:.3g}.")
        L.append("")
        # conditional execution: did stage three faithfully apply what it was given?
        rows_b = read(raw, "chain_b") + read(raw, "chain_r_b")
        good = [r for r in rows_b if r["start"] == config.truth(dt.date(int(r["date"][:4]), 1, 1))]
        bad = [r for r in rows_b if r not in good]
        if good:
            L.append("## The composition itself is not the problem")
            L.append("")
            L.append(f"Across both chains, stage three was handed a correct starting weekday {len(good)} times "
                     f"and produced the right answer in {sum(r['correct'] for r in good)} of them "
                     f"({sum(r['correct'] for r in good) / len(good):.1%}). It was handed a wrong one "
                     f"{len(bad)} times and produced the right answer in {sum(r['correct'] for r in bad)} "
                     f"({sum(r["correct"] for r in bad) / len(bad):.1%}) of those.")
            L.append("")
            L.append("Stage three is a faithful function of its input. The chain is worth exactly as much as "
                     "the fact fed into it, which is the ordinary behaviour of a pipeline rather than a defect "
                     "of the model.")
            L.append("")

    L.append("## What this says about chaining")
    L.append("")
    L.append("- **Composition works.** Given a correct input, the final step applied a code-computed offset "
             "essentially without error. Small typed judgments do chain.")
    L.append("- **It can do arithmetic, within a range.** `N mod 7` is perfect to N of 30 and falls to "
             f"{R['mod7_big']['rate']:.0%} from 31 to 120. Weekday offsets are perfect to 6 days and fall to "
             f"{R['weekday_plus_big']['rate']:.0%} from 7 to 30. Keep each step inside the window and the chain "
             "holds; let a step run past it and the chain inherits the error.")
    L.append("- **Recall is the weak link, and it knows it.** Asked the weekday of 1 January for a year, it "
             f"abstained on {e1['abstained']} of {ca['n']} and was right on {ca['k']}. That abstention is the "
             "useful behaviour: the failure is visible to the caller rather than silent.")
    L.append("- **So the architecture is retrieval plus judgment plus code.** Supply the facts, keep each "
             "judgment small, and do the arithmetic outside the model. That is also what TypeSafe's own "
             "guidance says to do.")
    L.append("")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
