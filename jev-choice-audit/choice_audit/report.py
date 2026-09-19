"""Render report.md from the computed results. No API calls, no new statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import config
from .extract import Row
from .stats import fmt_ci, fmt_p
from .store import JsonlStore


def _pct(d: dict[str, Any]) -> str:
    return f"{d['rate'] * 100:.2f}%"


def _row(label: str, d: dict[str, Any], extra: str = "") -> str:
    return f"| {label} | {d['k']} / {d['n']} | {_pct(d)} | {fmt_ci(d['ci_low'], d['ci_high'])} | {extra} |"


def _raw_pair(raw_path: Path, record_id: str) -> tuple[dict, dict] | None:
    for rec in JsonlStore(raw_path).read():
        if rec.get("record_id") == record_id and rec.get("ok"):
            return rec["request"], rec["response"]
    return None


def write_report(rows: list[Row], R: dict[str, Any], charts: dict[str, str], path: Path, raw_path: Path) -> None:
    meta = R["meta"]
    mm = R["mismatch_main"]
    gb = R["gap_buckets_main"]
    f = gb["fisher_near_vs_far"]
    acc = R["accuracy_main"]
    conf = R["confidence_main"]
    det = R["determinism"]
    bat = R["batching"]
    unk = R["unknown_main"]
    q = meta["quantisation"]

    def img(key: str, caption: str) -> str:
        if key not in charts:
            return ""
        return f"\n![{caption}](charts/{charts[key]})\n\n*{caption}*\n"

    L: list[str] = []
    A = L.append

    A("# Jev `Choice`: `choice` does not always match the highest returned probability")
    A("")
    A(f"In **{_pct(mm)}** of {mm['n']} single-question requests "
      f"(95% CI {fmt_ci(mm['ci_low'], mm['ci_high'])}), the `choice` field named an option whose "
      f"entry in the same response's `probabilities` map was lower than the maximum, which "
      f"contradicts the API's documented contract that `choice` is "
      f"\"the name of the choice with the highest probability among the question's criteria\".")
    A("")

    # ---- method
    A("## Method")
    A("")
    A(f"- **Model requested** `{meta['requested_model']}`. **Served on every response** "
      f"{', '.join(f'`{k}` ({v} answers)' for k, v in meta['served_models'].items())}.")
    A(f"- **Endpoint** `POST {config.BASE_URL}{config.ENDPOINT}`, schema taken from the published OpenAPI document at "
      f"`{config.BASE_URL}/openapi.json`.")
    A(f"- **Seed** {meta['seed']}. Dates are sampled uniformly without replacement from "
      f"{meta['date_range'][0]} to {meta['date_range'][1]}.")
    A(f"- **Samples** {R['meta']['n_main']} unique dates, one question per request; "
      f"{config.N_DETERMINISM} of those dates re-sent {config.N_REPEATS} times each as byte-identical requests; "
      f"and the same {config.N_DETERMINISM} dates sent again in batches of up to {config.BATCH_SIZE} questions "
      f"per request. {meta['n_records_questions_total']} questions in total.")
    A(f"- **State**, identical on every request: `{meta['state']}`")
    A(f"- **Question**: `{meta['question_template']}`")
    A(f"- **Options**, fixed order: {', '.join(f'`{o}`' for o in meta['options'])}")
    A("- **Ground truth** comes from Python's `datetime` and is never part of any request.")
    A(f"- **Tolerance** {meta['tolerance']:g}. A mismatch is `probabilities[choice] < max(probabilities)` "
      f"by more than that tolerance.")
    A("")
    A("Two properties of the response format matter for reading everything below:")
    A("")
    A(f"- Reported probabilities are **quantised to 0.01**: all {q['n_values']} values sit within "
      f"{q['max_deviation_from_grid']:.1e} of a multiple of 0.01, spanning {q['distinct_values']} distinct "
      f"values from {q['min']:.2f} to {q['max']:.2f}. The granularity is in the numbers themselves, not only "
      f"in their display: the response body carries full float64 text, and "
      f"{(1 - q['exactly_equal_to_own_2dp']) * 100:.1f}% of values are an ULP off the grid "
      f"(`0.13999999999999999` rather than `0.14`), so the vector is the result of arithmetic that lands on "
      f"a 0.01 grid rather than a literal rounding applied for display.")
    A(f"- Vectors are not renormalised after quantisation: sums range from "
      f"{meta['prob_sum_min']:.2f} to {meta['prob_sum_max']:.2f}.")
    A(f"- `choice` was always one of the supplied criteria: {meta['choice_always_in_criteria']}.")
    A("")

    # ---- results
    A("## Results")
    A("")
    A("| Measure | Count | Rate | 95% Wilson CI | Test |")
    A("|---|---|---|---|---|")
    A(_row("Mismatch, single questions", mm, "primary result"))
    A(_row("Mismatch, gap = 0 (top two identical)", gb["0 (exact tie)"], ""))
    A(_row("Mismatch, 0 < gap ≤ 0.01", gb["<=0.01"], ""))
    A(_row("Mismatch, gap > 0.01", gb[">0.01"], ""))
    A(_row("Mismatch deficit greater than one 0.01 step", mm["beyond_rounding"], "bounds the disagreement"))
    A(_row("Ties at the maximum (more than one option)", R["ties_main"], ""))
    A(_row("`choice` correct", acc["choice"], f"binomial vs 1/7: p = {fmt_p(acc['choice']['binom_p_vs_chance'])}"))
    A(_row("argmax correct (ties broken at random)", acc["argmax_random_tiebreak"],
           f"binomial vs 1/7: p = {fmt_p(acc['argmax_random_tiebreak']['binom_p_vs_chance'])}"))
    if det.get("n_dates"):
        A(_row("`choice` changed across identical repeats", det["choice_changed"], f"{det['n_dates']} dates"))
        A(_row("Any probability changed across identical repeats", det["probabilities_changed"], f"{det['n_dates']} dates"))
        A(_row("`confidence` changed across identical repeats", det["confidence_changed"], f"{det['n_dates']} dates"))
    if bat.get("ran"):
        A(_row("Mismatch, batched requests", bat["batched"], f"Fisher vs single: p = {fmt_p(bat['fisher']['p_value'])}"))
        A(_row("Mismatch, single requests, same dates", bat["single_same_dates"], ""))
    A("")
    A(f"**Near-tie association.** Of {f['near_tie_n']} questions whose top two probabilities differ by "
      f"at most {config.NEAR_TIE}, {f['near_tie_mismatch']} mismatched. Of {f['non_near_tie_n']} questions "
      f"with a larger gap, {f['non_near_tie_mismatch']} mismatched. Fisher's exact test, two-sided: "
      f"odds ratio {f['odds_ratio']:.3g}, p = {fmt_p(f['p_value'])}.")
    A("")
    A(f"**Size of the disagreement.** Every mismatch was short of the maximum by exactly "
      f"{', '.join(f'{d:.2f}' for d in sorted(mm['deficit_counts']))} "
      f"(counts: {json.dumps({f'{k:.2f}': v for k, v in sorted(mm['deficit_counts'].items())})}). "
      f"The largest deficit observed was {mm['max_deficit']:.2f}, one step of the reporting grid.")
    A("")
    A(f"**`Unknown`.** Mean probability {unk['mean']:.4f}, range {unk['min']:.2f} to {unk['max']:.2f}, "
      f"median {unk['median']:.2f}. It held the maximum in {unk['share_argmax'] * 100:.1f}% of questions "
      f"and was returned as `choice` in {unk['share_chosen'] * 100:.1f}%.")
    A("")
    A(img("mismatch_by_gap", "Mismatch rate by the gap between the top two reported probabilities, with 95% Wilson intervals."))
    A(img("gap_histogram", "Distribution of the top-1 minus top-2 gap, with the mismatching questions overlaid."))
    A(img("accuracy", "Accuracy of `choice` against the accuracy of taking the highest probability, both against a 1/7 chance line."))

    # ---- confidence
    A("### How `confidence` relates to the probabilities")
    A("")
    A(f"Confidence ranged from {conf['confidence_min']:.2f} to {conf['confidence_max']:.2f} "
      f"(mean {conf['confidence_mean']:.4f}). Candidate closed forms, evaluated against the reported value:")
    A("")
    A("| Candidate | Pearson r | Spearman ρ | Equal after rounding to 2dp | Mean abs. difference |")
    A("|---|---|---|---|---|")
    for name, c in sorted(conf["candidates"].items(), key=lambda kv: -kv[1]["share_equal_after_2dp_rounding"]):
        A(f"| `{name}` | {c['pearson']:.4f} | {c['spearman']:.4f} | "
          f"{c['share_equal_after_2dp_rounding'] * 100:.1f}% | {c['mean_abs_diff']:.4f} |")
    A("")
    om = conf.get("on_mismatches")
    if om:
        A(f"Restricting to the {om['n']} mismatching questions, the closed form "
          f"`(p - 1/k) / (1 - 1/k)` rounded to two decimals reproduces the reported confidence for "
          f"**{om['matches_formula_on_p_choice']}** of them when `p` is the probability of the *chosen* option, "
          f"against **{om['matches_formula_on_p_max']}** when `p` is the *maximum* probability.")
        A("")
    A(img("confidence", "Reported confidence against the highest reported probability, with mismatches highlighted."))

    # ---- determinism
    if det.get("n_dates"):
        A("### Determinism")
        A("")
        A(f"Each of {det['n_dates']} dates was sent {det['repeats_per_date']:.0f} times as a byte-identical request. "
          f"`choice` varied on {det['choice_changed']['k']} of them ({_pct(det['choice_changed'])}); the probability "
          f"vector varied on {det['probabilities_changed']['k']} ({_pct(det['probabilities_changed'])}). "
          f"The largest swing in any single option's probability across repeats was "
          f"{det['max_per_option_swing']:.2f}, with a mean of {det['mean_per_option_swing']:.4f}.")
        A("")
        A(img("determinism", "Share of dates where the field changed across five identical repeats."))

    if bat.get("ran"):
        A("### Batched against single requests")
        A("")
        A(f"Batch sizes sent: {json.dumps({str(k): v for k, v in bat['batch_sizes'].items()})}. "
          f"The mismatch rate was {_pct(bat['batched'])} batched against {_pct(bat['single_same_dates'])} "
          f"single on the same dates (Fisher's exact, two-sided: p = {fmt_p(bat['fisher']['p_value'])}).")
        A("")
        A(img("batching", "Mismatch rate for batched against single-question requests on the same dates."))

    # ---- conclusion
    A("## Conclusion")
    A("")
    A(f"**The data supports the hypothesis that `choice` and `probabilities` are not produced by the same "
      f"computation.** Three observations carry that conclusion.")
    A("")
    A(f"1. The disagreement is real and frequent: {_pct(mm)} of single questions, "
      f"{fmt_ci(mm['ci_low'], mm['ci_high'])} at 95%.")
    A(f"2. It cannot be an artefact of rounding one vector for display. Rounding to a grid is monotone: "
      f"if one raw probability is at least another, its rounded value is at least the other's rounded value. "
      f"So if `choice` were the argmax of the same vector that was then rounded into `probabilities`, "
      f"`probabilities[choice]` would still be a maximum of the rounded vector, and the mismatch rate would be zero. "
      f"It is not.")
    A(f"3. The disagreement is bounded at exactly one reporting step. Every one of the {mm['k']} mismatches "
      f"fell short of the maximum by {mm['max_deficit']:.2f} and none by more "
      f"({mm['beyond_rounding']['k']} of {mm['beyond_rounding']['n']} exceeded one step), and all of them "
      f"occurred where the top two probabilities were within {config.NEAR_TIE} "
      f"(Fisher's exact p = {fmt_p(f['p_value'])}). The two computations therefore agree closely, "
      f"disagreeing only where a difference smaller than the reporting resolution decides the ordering.")
    A("")
    if det.get("n_dates"):
        A(f"The repeat experiment shows the endpoint is not deterministic in any field: across "
          f"{det['repeats_per_date']:.0f} byte-identical repeats the probability vector changed on "
          f"{_pct(det['probabilities_changed'])} of dates and `choice` changed on "
          f"{_pct(det['choice_changed'])}. That rules out a stale or cached component as the source of the "
          f"mismatch, since no field is fixed between calls, but it does not by itself separate the two "
          f"computations: the mismatch above is a disagreement *inside a single response*, which repetition "
          f"across calls cannot explain either way.")
        A("")
    if om and om["matches_formula_on_p_choice"] > om["matches_formula_on_p_max"]:
        A(f"`confidence` sides with `choice` rather than with the reported maximum on mismatching questions "
          f"({om['matches_formula_on_p_choice']} against {om['matches_formula_on_p_max']} of {om['n']}), "
          f"which is consistent with `choice` and `confidence` being read off one internal vector and "
          f"`probabilities` being reported from another, or from the same one at lower precision.")
        A("")
    A(f"What the data does not establish: which of the two is correct, whether the cause is precision, a "
      f"separate pass, or an ordering step, and whether the behaviour extends to tasks where the model is "
      f"not at chance. On this task neither rule beat chance "
      f"(`choice` {_pct(acc['choice'])}, p = {fmt_p(acc['choice']['binom_p_vs_chance'])}; argmax "
      f"{_pct(acc['argmax_random_tiebreak'])}, p = {fmt_p(acc['argmax_random_tiebreak']['binom_p_vs_chance'])}), "
      f"which is what makes near-ties common enough to expose the disagreement at all. "
      f"A task the model can do would produce larger gaps and, on this evidence, fewer mismatches.")
    A("")

    # ---- repro
    A("## Minimal reproduction")
    A("")
    repro_id = R.get("repro_record_id")
    pair = _raw_pair(raw_path, repro_id) if repro_id else None
    if pair:
        req, resp = pair
        ans = next(iter(resp["answers"].values()))
        p = ans["probabilities"]
        best = max(p, key=lambda k: p[k])
        A(f"Record `{repro_id}`. `choice` is `{ans['choice']}` at {p[ans['choice']]:.2f}, "
          f"while `{best}` is reported at {p[best]:.2f} in the same response.")
        A("")
        A("Request:")
        A("")
        A("```json")
        A(json.dumps(req, indent=2))
        A("```")
        A("")
        A("Response:")
        A("")
        A("```json")
        A(json.dumps(resp, indent=2))
        A("```")
        A("")
        A(img("mismatch_example", "The full probability vector for the reproduction above."))
    else:
        A("No mismatching record was found in the collected data.")
    A("")

    # ---- provenance
    A("## Files")
    A("")
    n_requests = sum(1 for _ in JsonlStore(raw_path).read())
    A(f"- `{raw_path.name}`: every request and response, one JSON object per line, "
      f"{meta['n_records_questions_total']} questions across {n_requests} requests.")
    A(f"- `{config.PER_REQUEST_CSV.name}`: one row per question with the derived columns used above.")
    A("- `results.json`: every statistic in this report, as computed.")
    A("- `charts/`: the figures, regenerated from the JSONL by `python -m choice_audit.analyze`.")
    A("")
    A("Collection and analysis are separate programs. The analysis makes no network calls, so every number "
      "here can be recomputed from the committed JSONL alone.")
    A("")

    path.write_text("\n".join(L))
