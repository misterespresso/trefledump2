"""Analysis: reads the JSONL only, makes zero API calls.

    python -m choice_audit.analyze                    # csv + results.json + report.md + charts
    python -m choice_audit.analyze --no-charts

Every number in report.md is produced here from the raw records.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import config
from .extract import Row, decimals_used, load_rows
from .stats import binom_p, fisher_2x2, fmt_ci, fmt_p, normalised_entropy, wilson


# --------------------------------------------------------------------------- blocks
def proportion(successes: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(successes, n)
    return {"k": successes, "n": n, "rate": (successes / n) if n else float("nan"), "ci_low": lo, "ci_high": hi}


def mismatch_block(rows: list[Row]) -> dict[str, Any]:
    n = len(rows)
    mm = [r for r in rows if r.mismatch]
    out = proportion(len(mm), n)
    out["deficits"] = sorted(round(r.deficit, 6) for r in mm)
    out["deficit_counts"] = dict(Counter(round(r.deficit, 4) for r in mm))
    out["max_deficit"] = max((r.deficit for r in mm), default=0.0)
    out["beyond_rounding"] = proportion(sum(1 for r in mm if r.deficit > 0.01 + config.TOL), n)
    return out


def gap_buckets(rows: list[Row]) -> dict[str, Any]:
    buckets = ["0 (exact tie)", "<=0.01", ">0.01"]
    out = {}
    for b in buckets:
        sub = [r for r in rows if r.gap_bucket == b]
        out[b] = proportion(sum(1 for r in sub if r.mismatch), len(sub))
    near = [r for r in rows if r.top2_gap <= config.NEAR_TIE + config.TOL]
    far = [r for r in rows if r.top2_gap > config.NEAR_TIE + config.TOL]
    a = sum(1 for r in near if r.mismatch)
    c = sum(1 for r in far if r.mismatch)
    odds, p = fisher_2x2(a, len(near) - a, c, len(far) - c)
    out["fisher_near_vs_far"] = {
        "near_tie_mismatch": a, "near_tie_n": len(near),
        "non_near_tie_mismatch": c, "non_near_tie_n": len(far),
        "odds_ratio": odds, "p_value": p,
    }
    return out


def accuracy_block(rows: list[Row], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(rows)
    ch = sum(1 for r in rows if r.choice_correct)
    # One seeded RNG walked over rows in their sorted order: reproducible, and ties
    # are broken at random rather than by the fixed option order.
    am = sum(1 for r in rows if r.argmax_correct(rng))
    out = {
        "choice": {**proportion(ch, n), "binom_p_vs_chance": binom_p(ch, n, config.CHANCE)},
        "argmax_random_tiebreak": {**proportion(am, n), "binom_p_vs_chance": binom_p(am, n, config.CHANCE)},
        "chance": config.CHANCE,
    }
    return out


def confidence_block(rows: list[Row]) -> dict[str, Any]:
    """What is `confidence`? Test candidate closed forms against the reported value."""
    k = len(config.OPTIONS)
    uni = 1.0 / k

    def candidates(r: Row) -> dict[str, float]:
        ps = sorted(r.probabilities.values(), reverse=True)
        p1, p2 = ps[0], (ps[1] if len(ps) > 1 else 0.0)
        return {
            "p_max": p1,
            "p_choice": r.p_choice,
            "margin_top1_top2": p1 - p2,
            "excess_uniform_on_max": (p1 - uni) / (1 - uni),
            "excess_uniform_on_choice": (r.p_choice - uni) / (1 - uni),
            "one_minus_norm_entropy": 1 - normalised_entropy(list(r.probabilities.values())),
            "gini": sum(p * p for p in r.probabilities.values()),
        }

    names = list(candidates(rows[0]))
    conf = [r.confidence for r in rows]
    out: dict[str, Any] = {"n": len(rows), "confidence_min": min(conf), "confidence_max": max(conf),
                           "confidence_mean": statistics.fmean(conf), "candidates": {}}
    for name in names:
        vals = [candidates(r)[name] for r in rows]
        exact2 = sum(1 for c, v in zip(conf, vals) if abs(c - round(v, 2)) < 5e-3)
        out["candidates"][name] = {
            "pearson": _pearson(conf, vals),
            "spearman": _spearman(conf, vals),
            "share_equal_after_2dp_rounding": exact2 / len(rows),
            "max_abs_diff": max(abs(c - v) for c, v in zip(conf, vals)),
            "mean_abs_diff": statistics.fmean(abs(c - v) for c, v in zip(conf, vals)),
        }
    best = max(out["candidates"], key=lambda nm: out["candidates"][nm]["share_equal_after_2dp_rounding"])
    out["best_match"] = best

    # Sharp diagnostic: on mismatches, is confidence keyed to the chosen option or to the max?
    mm = [r for r in rows if r.mismatch]
    if mm:
        cm = sum(1 for r in mm if abs(r.confidence - round((r.p_choice - uni) / (1 - uni), 2)) < 5e-3)
        mx = sum(1 for r in mm if abs(r.confidence - round((r.p_max - uni) / (1 - uni), 2)) < 5e-3)
        out["on_mismatches"] = {"n": len(mm), "matches_formula_on_p_choice": cm, "matches_formula_on_p_max": mx}
    return out


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def _spearman(a: list[float], b: list[float]) -> float:
    return _pearson(_rank(a), _rank(b))


def determinism_block(rows: list[Row]) -> dict[str, Any]:
    groups: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        if r.experiment == "determinism":
            groups[r.date].append(r)
    groups = {d: sorted(rs, key=lambda r: r.repeat) for d, rs in groups.items() if len(rs) >= 2}
    if not groups:
        return {"n_dates": 0}
    choice_changed = sum(1 for rs in groups.values() if len({r.choice for r in rs}) > 1)
    probs_changed = sum(1 for rs in groups.values() if len({tuple(sorted(r.probabilities.items())) for r in rs}) > 1)
    conf_changed = sum(1 for rs in groups.values() if len({r.confidence for r in rs}) > 1)
    n = len(groups)
    # How large are the probability wobbles when they happen?
    deltas = []
    for rs in groups.values():
        for opt in config.OPTIONS:
            vs = [r.probabilities.get(opt, 0.0) for r in rs]
            deltas.append(max(vs) - min(vs))
    return {
        "n_dates": n,
        "repeats_per_date": statistics.fmean(len(rs) for rs in groups.values()),
        "choice_changed": proportion(choice_changed, n),
        "probabilities_changed": proportion(probs_changed, n),
        "confidence_changed": proportion(conf_changed, n),
        "max_per_option_swing": max(deltas) if deltas else 0.0,
        "mean_per_option_swing": statistics.fmean(deltas) if deltas else 0.0,
    }


def batching_block(rows: list[Row]) -> dict[str, Any]:
    batched = [r for r in rows if r.experiment == "batch"]
    if not batched:
        return {"ran": False}
    dates = {r.date for r in batched}
    single = [r for r in rows if r.experiment in ("main", "determinism") and r.date in dates]
    b = proportion(sum(1 for r in batched if r.mismatch), len(batched))
    s = proportion(sum(1 for r in single if r.mismatch), len(single))
    odds, p = fisher_2x2(b["k"], b["n"] - b["k"], s["k"], s["n"] - s["k"])
    return {"ran": True, "batched": b, "single_same_dates": s, "batch_sizes": dict(Counter(r.batch_size for r in batched)),
            "fisher": {"odds_ratio": odds, "p_value": p}}


def analyse(rows: list[Row]) -> dict[str, Any]:
    main = [r for r in rows if r.experiment == "main"]
    all_single = [r for r in rows if r.batch_size == 1]
    ties = sum(1 for r in main if r.is_tie)
    unknown = [r.p_unknown for r in main if r.p_unknown == r.p_unknown]
    sums = [r.prob_sum for r in rows]
    return {
        "meta": {
            "n_records_questions_total": len(rows),
            "n_main": len(main),
            "served_models": dict(Counter(r.served_model for r in rows)),
            "requested_model": config.MODEL,
            "seed": config.SEED,
            "options": list(config.OPTIONS),
            "state": config.STATE,
            "question_template": config.QUESTION_TEMPLATE,
            "date_range": [config.DATE_START.isoformat(), config.DATE_END.isoformat()],
            "tolerance": config.TOL,
            "choice_always_in_criteria": all(r.choice_in_criteria for r in rows),
            "prob_sum_min": min(sums), "prob_sum_max": max(sums),
            "quantisation": decimals_used(rows),
        },
        "mismatch_main": mismatch_block(main),
        "mismatch_all_single": mismatch_block(all_single),
        "gap_buckets_main": gap_buckets(main),
        "gap_buckets_all_single": gap_buckets(all_single),
        "ties_main": proportion(ties, len(main)),
        "accuracy_main": accuracy_block(main, config.SEED),
        "unknown_main": {
            "mean": statistics.fmean(unknown) if unknown else float("nan"),
            "min": min(unknown) if unknown else float("nan"),
            "max": max(unknown) if unknown else float("nan"),
            "median": statistics.median(unknown) if unknown else float("nan"),
            "share_argmax": sum(1 for r in main if "Unknown" in r.argmax_options) / len(main) if main else float("nan"),
            "share_chosen": sum(1 for r in main if r.choice == "Unknown") / len(main) if main else float("nan"),
        },
        "confidence_main": confidence_block(main),
        "determinism": determinism_block(rows),
        "batching": batching_block(rows),
    }


# --------------------------------------------------------------------------- outputs
CSV_FIELDS = ["record_id", "experiment", "repeat", "batch_size", "qname", "date", "truth", "served_model",
              "choice", "p_choice", "p_max", "deficit", "mismatch", "is_tie", "n_argmax_options",
              "top2_gap", "gap_bucket", "confidence", "p_unknown", "prob_sum", "choice_correct",
              *[f"p_{o}" for o in config.OPTIONS]]


def write_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            d = {
                "record_id": r.record_id, "experiment": r.experiment, "repeat": r.repeat,
                "batch_size": r.batch_size, "qname": r.qname, "date": r.date, "truth": r.truth,
                "served_model": r.served_model, "choice": r.choice, "p_choice": r.p_choice,
                "p_max": r.p_max, "deficit": round(r.deficit, 10), "mismatch": int(r.mismatch),
                "is_tie": int(r.is_tie), "n_argmax_options": len(r.argmax_options),
                "top2_gap": round(r.top2_gap, 10), "gap_bucket": r.gap_bucket,
                "confidence": r.confidence, "p_unknown": r.p_unknown, "prob_sum": round(r.prob_sum, 10),
                "choice_correct": int(r.choice_correct),
            }
            for o in config.OPTIONS:
                d[f"p_{o}"] = r.probabilities.get(o, "")
            w.writerow(d)


def find_repro(rows: list[Row]) -> Row | None:
    """The clearest mismatch: largest deficit, then the cleanest gap structure."""
    mm = [r for r in rows if r.mismatch]
    if not mm:
        return None
    return max(mm, key=lambda r: (round(r.deficit, 6), -len(r.argmax_options), r.record_id))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Analyse collected Jev Choice responses (no API calls)")
    ap.add_argument("--raw", default=str(config.RAW_JSONL))
    ap.add_argument("--csv", default=str(config.PER_REQUEST_CSV))
    ap.add_argument("--report", default=str(config.REPORT_MD))
    ap.add_argument("--charts-dir", default=str(config.CHARTS_DIR))
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args(argv)

    rows = load_rows(Path(args.raw))
    if not rows:
        raise SystemExit(f"no usable records in {args.raw}; run choice_audit.collect first")
    print(f"[analyse] {len(rows)} questions from {args.raw}", flush=True)

    results = analyse(rows)
    write_csv(rows, Path(args.csv))
    results_path = Path(args.csv).with_name("results.json")
    repro = find_repro([r for r in rows if r.experiment == "main"]) or find_repro(rows)
    results["repro_record_id"] = repro.record_id if repro else None
    results_path.write_text(json.dumps(results, indent=2, default=str))

    charts: dict[str, str] = {}
    if not args.no_charts:
        from .charts import build_all
        charts = build_all(rows, results, Path(args.charts_dir))
        print(f"[charts] {len(charts)} written to {args.charts_dir}", flush=True)

    from .report import write_report
    write_report(rows, results, charts, Path(args.report), Path(args.raw))
    print(f"[analyse] wrote {args.csv}, {results_path}, {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
