"""Collection: send the planned requests and append every raw exchange to JSONL.

    python -m choice_audit.collect                 # all experiments, resumable
    python -m choice_audit.collect --experiment main --limit 20
    python -m choice_audit.collect --dry-run       # print the plan, send nothing

Resumability: a record_id already present in the JSONL with ok=true is skipped,
so re-running after a crash or a rate-limit wall costs nothing.

This module performs no analysis. It writes what the API said, verbatim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config
from .client import Counter, JevClient
from .plan import Unit, build_units
from .store import JsonlStore


def run(units: list[Unit], store: JsonlStore, client: JevClient, workers: int) -> tuple[int, int]:
    todo = [u for u in units if not store.has(u.record_id)]
    skipped = len(units) - len(todo)
    print(f"[plan] {len(units)} requests, {skipped} already done, {len(todo)} to send", flush=True)
    if not todo:
        return 0, 0

    done = failed = 0
    t0 = time.time()

    def one(unit: Unit) -> dict:
        body = unit.body()
        result = client.post(body)
        record = {
            "record_id": unit.record_id,
            "experiment": unit.experiment,
            "repeat": unit.repeat,
            "requested_model": config.MODEL,
            "batch_size": len(unit.items),
            "items": [it.as_meta() for it in unit.items],
            "request": body,
            "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            **result,
        }
        if result.get("ok"):
            record["served_model"] = result["response"].get("model")
        return record

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(one, u): u for u in todo}
        for fut in as_completed(futures):
            unit = futures[fut]
            try:
                record = fut.result()
            except Exception as e:  # never lose the rest of the run to one bad unit
                record = {
                    "record_id": unit.record_id,
                    "experiment": unit.experiment,
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
            store.append(record)
            if record.get("ok"):
                done += 1
            else:
                failed += 1
            n = done + failed
            if n % 50 == 0 or n == len(todo):
                rate = n / max(1e-9, time.time() - t0)
                print(f"[send] {n}/{len(todo)} ok={done} failed={failed} {rate:.1f} req/s", flush=True)
    return done, failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Collect Jev Choice responses")
    ap.add_argument("--experiment", choices=["main", "determinism", "batch", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="send at most N requests (after resume filtering)")
    ap.add_argument("--out", default=str(config.RAW_JSONL))
    ap.add_argument("--workers", type=int, default=config.WORKERS)
    ap.add_argument("--rate", type=float, default=config.MAX_REQUESTS_PER_SECOND, help="max requests per second")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit without sending")
    args = ap.parse_args(argv)

    units = build_units()
    if args.experiment != "all":
        units = [u for u in units if u.experiment == args.experiment]

    if args.dry_run:
        for u in units[:5]:
            print(u.record_id, [i.date.isoformat() for i in u.items])
        print(f"... {len(units)} requests, {sum(len(u.items) for u in units)} questions")
        return 0

    store = JsonlStore(Path(args.out))
    if args.limit is not None:
        pending = [u for u in units if not store.has(u.record_id)][: args.limit]
        units = pending

    counter = Counter()
    client = JevClient(rate=args.rate, counter=counter)
    t0 = time.time()
    done, failed = run(units, store, client, args.workers)
    print(
        f"[done] sent={counter.requests} ok={done} failed={failed} retries={counter.retries} "
        f"tokens_in={counter.input_tokens} tokens_out={counter.output_tokens} "
        f"elapsed={time.time() - t0:.1f}s -> {args.out}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
