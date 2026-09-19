"""Every knob of the experiment in one place, so a run is reproducible from this file."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

# --- identity of the run -----------------------------------------------------
SEED = 20260919
MODEL = "jev-latest"  # the alias we request; the served version is logged per response

# --- the task ----------------------------------------------------------------
# The date lives in the question, never the state, so several dates can share one
# request in the batching experiment. Ground truth is never sent to the API.
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
OPTIONS = (*DAYS, "Unknown")  # fixed order, used verbatim as the criteria keys
STATE = "Hello."  # a neutral greeting, identical on every request, carrying no hints
QUESTION_TEMPLATE = "What day of the week is {date} in the Gregorian calendar?"

DATE_START = dt.date(2027, 1, 1)
DATE_END = dt.date(2099, 12, 31)

# --- sample sizes ------------------------------------------------------------
N_MAIN = 1000  # unique dates, one question per request
N_DETERMINISM = 50  # dates drawn from the main sample
N_REPEATS = 5  # times each determinism date is re-sent, unchanged
BATCH_SIZE = 4  # questions per request in the batching experiment

# --- numerics ----------------------------------------------------------------
TOL = 1e-9  # float tolerance for comparing probabilities
NEAR_TIE = 0.01  # gap at or below this counts as a near-tie
CHANCE = 1.0 / 7.0  # 7 days are reachable; "Unknown" is never the right answer

# --- transport ---------------------------------------------------------------
BASE_URL = "https://api.typesafe.ai"
ENDPOINT = "/v1/systemone"
API_KEY_ENV = "TYPESAFE_API_KEY"
TIMEOUT_S = 30.0
MAX_ATTEMPTS = 6
BACKOFF_INITIAL_S = 1.0
BACKOFF_MAX_S = 60.0
RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
MAX_REQUESTS_PER_SECOND = 10.0  # published limit is 1,200/min; stay well under
WORKERS = 6

# --- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHARTS_DIR = ROOT / "charts"
RAW_JSONL = DATA_DIR / "raw_responses.jsonl"
PER_REQUEST_CSV = DATA_DIR / "per_question_results.csv"
REPORT_MD = ROOT / "report.md"


def question_text(date: dt.date) -> str:
    return QUESTION_TEMPLATE.format(date=date.isoformat())


def truth(date: dt.date) -> str:
    """Ground truth from the standard library. Never sent to the API."""
    return DAYS[date.weekday()]
