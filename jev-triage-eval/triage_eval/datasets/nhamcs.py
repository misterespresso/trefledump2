"""Loader for NHAMCS Emergency Department public-use files (CDC/NCHS).

Download a year's Stata (or SAS) file yourself, for example
https://ftp.cdc.gov/pub/Health_Statistics/NCHS/dataset_documentation/nhamcs/stata/ed2022-stata.zip
and unzip it into data/nhamcs/. The loader reads the .dta/.sas7bdat with
pyreadstat, keeps visits with a recorded 5-level triage (IMMEDR 1-5) and turns
the reason-for-visit codes into text using the file's own value labels.

NHAMCS-ED variable notes (public-use codebook):
  IMMEDR   1 immediate, 2 emergent, 3 urgent, 4 semi-urgent, 5 non-urgent;
           -9 blank, -8 unknown, 0 no triage, 7 ED does not triage
  TEMPF    temperature in tenths of a degree F (986 = 98.6 F); -9 blank
  PULSE, RESPR, BPSYS, BPDIAS, POPCT, PAINSCALE  -9 blank, -8 unknown
  SEX      1 female, 2 male;  ARREMS 1 arrived by EMS, 2 no
  INJURY   1 yes (older files 2 = no, 3 = questionable; newer 0 = no)
  RFV1..3  reason-for-visit codes (labelled in the Stata/SAS file)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import TriageRecord, _num, plausible

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "nhamcs"


def _find_file(path: Optional[Path | str]) -> Path:
    if path is not None:
        p = Path(path)
        if p.is_file():
            return p
        cands = sorted(p.glob("*.dta")) + sorted(p.glob("*.sas7bdat"))
    else:
        cands = sorted(DEFAULT_DIR.glob("*.dta")) + sorted(DEFAULT_DIR.glob("*.sas7bdat"))
    if not cands:
        raise FileNotFoundError(
            "No NHAMCS .dta or .sas7bdat file found. Download ed<year>-stata.zip from "
            "ftp.cdc.gov/pub/Health_Statistics/NCHS/dataset_documentation/nhamcs/stata/ "
            f"and unzip it into {DEFAULT_DIR} or pass --nhamcs-path."
        )
    return cands[-1]


def read_nhamcs(path: Optional[Path | str] = None) -> tuple[pd.DataFrame, dict]:
    import pyreadstat  # heavy import, keep local

    f = _find_file(path)
    reader = pyreadstat.read_dta if f.suffix == ".dta" else pyreadstat.read_sas7bdat
    df, meta = reader(str(f), apply_value_formats=False)
    df.columns = [c.upper() for c in df.columns]
    labels = {k.upper(): v for k, v in (meta.variable_value_labels or {}).items()}
    return df, labels


def _label(labels: dict, var: str, code) -> Optional[str]:
    if code is None:
        return None
    table = labels.get(var)
    if not table:
        return None
    for k, v in table.items():
        try:
            if float(k) == float(code):
                return str(v)
        except (TypeError, ValueError):
            if str(k) == str(code):
                return str(v)
    return None


def _clean_rfv(text: Optional[str], code) -> Optional[str]:
    if text is None:
        return None
    t = text.strip()
    # Stata labels often look like "10500 Chest pain and related symptoms (1050.0)"
    if code is not None and t.startswith(str(int(float(code)))):
        t = t[len(str(int(float(code)))):].strip(" -:")
    return t or None


def load_nhamcs(
    path: Optional[Path | str] = None,
    limit: Optional[int] = None,
    sample: Optional[int] = None,
    seed: int = 0,
    adults_only: bool = False,
) -> list[TriageRecord]:
    df, labels = read_nhamcs(path)
    if "IMMEDR" not in df.columns:
        raise ValueError("IMMEDR column not found; is this an NHAMCS *ED* file?")
    df = df[df["IMMEDR"].isin([1, 2, 3, 4, 5])].copy()
    if adults_only and "AGE" in df.columns:
        df = df[df["AGE"] >= 18]
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed)
    if "RFV1" in df.columns and not labels.get("RFV1"):
        warnings.warn("RFV1 has no value labels in this file; chief complaints will be raw codes.")

    records: list[TriageRecord] = []
    for idx, row in df.iterrows():
        rfv = []
        for var in ("RFV1", "RFV2", "RFV3"):
            code = _num(row.get(var))
            if code is None or code <= 0:
                continue
            txt = _clean_rfv(_label(labels, var, code), code)
            rfv.append(txt or f"reason-for-visit code {int(code)}")
        complaint = "; ".join(rfv) if rfv else "not recorded"

        tempf = _num(row.get("TEMPF"))
        if tempf is not None and tempf > 0:
            if tempf > 200:  # tenths of a degree
                tempf = tempf / 10.0
            temp_c = round((tempf - 32) * 5 / 9, 1)
        else:
            temp_c = None

        pain = _num(row.get("PAINSCALE"))
        pain = plausible(pain, 0, 10)
        injury_code = _num(row.get("INJURY"))
        arrems = _num(row.get("ARREMS"))
        sex = _num(row.get("SEX"))
        rec = TriageRecord(
            record_id=f"nhamcs-{int(idx):06d}",
            dataset="nhamcs",
            true_acuity=int(row["IMMEDR"]),
            chief_complaint=complaint,
            age=plausible(_num(row.get("AGE")), 0, 120),
            sex={1: "female", 2: "male"}.get(int(sex) if sex else 0),
            arrival_mode={1: "ambulance (EMS)", 2: "not by ambulance"}.get(int(arrems) if arrems else 0),
            injury=None if injury_code is None or injury_code < 0 else injury_code == 1,
            mental_status=None,  # not recorded at triage in the public-use file
            pain_present=None if pain is None else pain > 0,
            pain_score=pain,
            sbp=plausible(_num(row.get("BPSYS")), 30, 300),
            dbp=plausible(_num(row.get("BPDIAS")), 10, 200),
            hr=plausible(_num(row.get("PULSE")), 20, 300),
            rr=plausible(_num(row.get("RESPR")), 4, 80),
            temp_c=plausible(temp_c, 25, 45),
            spo2=plausible(_num(row.get("POPCT")), 40, 100),
            extra={"year": _num(row.get("YEAR")), "weight": _num(row.get("PATWT"))},
        )
        records.append(rec)
        if limit and len(records) >= limit:
            break
    return records
