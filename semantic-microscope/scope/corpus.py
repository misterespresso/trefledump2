"""Build the corpus: PyPI package summaries for the most-downloaded packages.

The ranking is hugovk/top-pypi-packages, a published list derived from PyPI's own
download statistics, so the sample is reproducible and not a matter of my taste.
We take the top N by download count and keep those whose summary is a real sentence.

A package summary is a one-line pitch for a piece of software: the shortest honest
example of technical rhetoric that is also freely available.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parents[1] / "data"
CORPUS = DATA / "corpus.json"
UA = {"User-Agent": "semantic-microscope/1.0"}
DEP = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


TOP_URL = ("https://raw.githubusercontent.com/hugovk/top-pypi-packages/main/"
           "top-pypi-packages.min.json")


def top_names(n: int) -> list[str]:
    req = urllib.request.Request(TOP_URL, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        rows = json.load(r)["rows"]
    return [row["project"] for row in rows[:n]]


def fetch(name: str) -> dict[str, Any] | None:
    url = f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json"
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
    except Exception:
        return None


def deps_of(payload: dict[str, Any]) -> list[str]:
    out = []
    for spec in (payload.get("info", {}).get("requires_dist") or []):
        if "extra ==" in spec:          # optional extras pull in long tails; skip them
            continue
        m = DEP.match(spec)
        if m:
            out.append(m.group(1).lower())
    return out


def build(target: int = 500) -> list[dict[str, Any]]:
    names = top_names(target)
    with ThreadPoolExecutor(max_workers=16) as ex:
        payloads = list(ex.map(fetch, names))
    seen = {n: p for n, p in zip(names, payloads) if p}

    rows = []
    for rank, (name, p) in enumerate(seen.items(), start=1):
        info = p.get("info", {})
        summary = (info.get("summary") or "").strip()
        # A usable pitch: long enough to say something, short enough to be one line.
        if not (25 <= len(summary) <= 200):
            continue
        cls = info.get("classifiers") or []
        rows.append({
            "rank": rank,
            "name": info.get("name") or name,
            "summary": summary,
            "topics": sorted({c.split(" :: ", 1)[1].split(" :: ")[0]
                              for c in cls if c.startswith("Topic :: ")}),
            "audience": sorted({c.split(" :: ")[-1] for c in cls if c.startswith("Intended Audience")}),
            "maturity": next((c.split(" :: ")[-1] for c in cls if c.startswith("Development Status")), None),
            "requires_python": info.get("requires_python"),
            "n_releases": len(p.get("releases") or {}),
        })
    return rows


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    rows = build()
    CORPUS.write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} of {500} top packages have a usable summary -> {CORPUS}")
    for r in rows[:6]:
        print(f"  {r['name']:22s} | {r['summary'][:74]}")
