"""Put every package summary under all 40 lenses. One request per package."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .client import Jev, Store
from .lenses import LENSES, questions, state_for

DATA = Path(__file__).resolve().parents[1] / "data"
RAW, MATRIX = DATA / "readings.jsonl", DATA / "matrix.json"


def main() -> int:
    corpus = json.loads((DATA / "corpus.json").read_text())
    store, jev, qs = Store(RAW), Jev(), questions()
    todo = [p for p in corpus if p["name"] not in store.done]
    print(f"[measure] {len(corpus)} packages, {len(todo)} to read, {len(LENSES)} lenses each", flush=True)

    def one(pkg):
        res = jev.ask(state_for(pkg), qs)
        rec = {"record_id": pkg["name"], "summary": pkg["summary"], **res}
        if res.get("ok"):
            rec["readings"] = {k: v["noul"] for k, v in res["response"]["answers"].items()}
            rec["served_model"] = res["response"].get("model")
        store.append(rec)
        return rec

    if todo:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for i, _ in enumerate(ex.map(one, todo), 1):
                if i % 50 == 0 or i == len(todo):
                    print(f"[measure] {i}/{len(todo)}", flush=True)
        print(f"[usage] {jev.requests} requests, {jev.input_tokens} input tokens", flush=True)

    by = {p["name"]: p for p in corpus}
    rows = []
    for name, rec in store.done.items():
        if name in by and rec.get("readings"):
            rows.append({**by[name], "readings": rec["readings"], "served_model": rec.get("served_model")})
    rows.sort(key=lambda r: r["rank"])
    MATRIX.write_text(json.dumps({"lenses": {k: {"group": g, "text": t} for k, (g, t) in LENSES.items()},
                                  "packages": rows}, indent=1))
    print(f"[measure] {len(rows)} packages x {len(LENSES)} lenses -> {MATRIX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
