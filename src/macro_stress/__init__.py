"""Reconstructed CompSci 390B macro-stress / safe-haven analysis.

The original course notebooks were not recovered. Modules here implement
the methodology in the May 2026 final report. See docs/PROVENANCE.md.
"""

from .data import (
    CANONICAL_CSV_NAME,
    EXPECTED_N_ROWS,
    SAMPLE_END,
    SAMPLE_START,
    find_dataset,
    load_raw,
    validate_dataset,
)
from .features import FEATURE_COUNT, build_supervised_frame, log_returns
from .regimes import add_stress_flags

__all__ = [
    "CANONICAL_CSV_NAME",
    "EXPECTED_N_ROWS",
    "FEATURE_COUNT",
    "SAMPLE_END",
    "SAMPLE_START",
    "add_stress_flags",
    "build_supervised_frame",
    "find_dataset",
    "load_raw",
    "log_returns",
    "validate_dataset",
]
