from .base import ACUITY_LEVELS, TriageRecord, records_to_frame
from .ktas import load_ktas
from .nhamcs import load_nhamcs

LOADERS = {"ktas": load_ktas, "nhamcs": load_nhamcs}

__all__ = ["ACUITY_LEVELS", "TriageRecord", "records_to_frame", "load_ktas", "load_nhamcs", "LOADERS"]
