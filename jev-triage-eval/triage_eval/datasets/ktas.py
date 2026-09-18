"""Loader for the KTAS mistriage dataset (Moon et al., PLOS ONE 2019)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .base import MENTAL_LABELS, TriageRecord, _num, plausible

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "ktas" / "ktas_triage.csv"

ARRIVAL = {
    1: "walked in",
    2: "public ambulance (119)",
    3: "private car",
    4: "private ambulance",
    5: "public transport or police",
    6: "wheelchair",
    7: "other",
}


def read_ktas_csv(path: Path | str = DEFAULT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="latin-1", decimal=",")
    df.columns = [c.strip() for c in df.columns]
    return df


def _mental(code: Optional[float]) -> Optional[str]:
    if code is None:
        return None
    i = int(code) - 1
    return MENTAL_LABELS[i] if 0 <= i < len(MENTAL_LABELS) else None


def load_ktas(path: Path | str = DEFAULT_PATH, limit: Optional[int] = None) -> list[TriageRecord]:
    df = read_ktas_csv(path)
    records: list[TriageRecord] = []
    for i, row in df.iterrows():
        pain_flag = _num(row.get("Pain"))
        nrs = plausible(_num(row.get("NRS_pain")), 0, 10)
        rec = TriageRecord(
            record_id=f"ktas-{i:04d}",
            dataset="ktas",
            true_acuity=int(row["KTAS_expert"]),
            nurse_acuity=int(row["KTAS_RN"]) if _num(row.get("KTAS_RN")) else None,
            chief_complaint=str(row.get("Chief_complain", "")).strip(),
            age=_num(row.get("Age")),
            sex={1: "female", 2: "male"}.get(int(_num(row.get("Sex")) or 0)),
            arrival_mode=ARRIVAL.get(int(_num(row.get("Arrival mode")) or 0)),
            injury={1: False, 2: True}.get(int(_num(row.get("Injury")) or 0)),
            mental_status=_mental(_num(row.get("Mental"))),
            pain_present=None if pain_flag is None else pain_flag == 1,
            pain_score=nrs,
            sbp=plausible(_num(row.get("SBP")), 30, 300),
            dbp=plausible(_num(row.get("DBP")), 10, 200),
            hr=plausible(_num(row.get("HR")), 20, 300),
            rr=plausible(_num(row.get("RR")), 4, 80),
            temp_c=plausible(_num(row.get("BT")), 25, 45),
            spo2=plausible(_num(row.get("Saturation")), 40, 100),
            extra={
                "hospital": {1: "local ED", 2: "regional ED"}.get(int(_num(row.get("Group")) or 0)),
                "patients_per_hour": _num(row.get("Patients number per hour")),
            },
        )
        records.append(rec)
        if limit and len(records) >= limit:
            break
    return records
