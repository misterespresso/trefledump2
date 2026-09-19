"""Bundle the measured matrix and its analysis into one JavaScript payload.

The published page is static: it ships every reading it needs and does all of its
arithmetic in the browser, so nothing here is precomputed beyond what analyse.py
already worked out.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, PAGE = ROOT / "data", ROOT / "page"


def main() -> int:
    matrix = json.loads((DATA / "matrix.json").read_text())
    analysis = json.loads((DATA / "analysis.json").read_text())

    keys = list(matrix["lenses"])
    spread = {s["lens"]: s for s in analysis["lens_spread"]}
    coords = {c["name"]: c for c in analysis["coords"]}

    lenses = [{
        "key": k,
        "group": matrix["lenses"][k]["group"],
        "text": matrix["lenses"][k]["text"],
        "mean": round(spread[k]["mean"], 4),
        "sd": round(spread[k]["sd"], 4),
    } for k in keys]

    packages = []
    for p in matrix["packages"]:
        c = coords[p["name"]]
        packages.append({
            "name": p["name"],
            "rank": p["rank"],
            "summary": p["summary"],
            "topics": p["topics"],
            "maturity": p["maturity"],
            "x": round(c["x"], 3),
            "y": round(c["y"], 3),
            "r": [round(p["readings"][k], 2) for k in keys],
        })

    payload = {
        "lenses": lenses,
        "packages": packages,
        "validation": analysis["validation"],
        "components": analysis["components"][:3],
        "pairs": analysis["strongest_pairs"][:6],
        "meanAurocLenses": analysis["mean_auroc_lenses"],
        "meanAurocTfidf": analysis["mean_auroc_tfidf"],
        "model": analysis["served_models"][0],
    }

    PAGE.mkdir(exist_ok=True)
    out = PAGE / "data.js"
    out.write_text("window.SCOPE = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(f"{out} — {len(packages)} packages x {len(lenses)} lenses, {out.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
