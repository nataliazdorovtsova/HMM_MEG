"""Loaders for the MATLAB -> Python data contract.

Three tidy files are produced by ``HMM_ExportForPython.m``:

- ``subject_level.csv``: one row per subject (demographics, cognition,
  behaviour, and the subject-level HMM scalars: switching rate, entropy
  rate, max FO).
- ``state_level.csv``: long format, one row per subject x state (FO,
  mean lifetime, mean interval per state).
- ``transition_matrices.csv``: long format, one row per subject x
  from_state x to_state (transition probability).

Each loader validates the expected columns are present and raises a clear
error otherwise, rather than failing downstream inside a model fit.
"""

from __future__ import annotations

import pandas as pd

SUBJECT_LEVEL_COLUMNS = [
    "subject_id", "age", "sex", "WASI_T", "SDQ_total", "SDQ_hyperactivity",
    "SDQ_conduct", "SDQ_peerproblems", "SDQ_emotion", "SDQ_prosocial",
    "switching_rate", "entropy_rate", "max_FO",
]

STATE_LEVEL_COLUMNS = [
    "subject_id", "state", "FO", "lifetime_mean", "interval_mean",
]

TRANSITION_COLUMNS = ["subject_id", "from_state", "to_state", "probability"]


def _validate(df: pd.DataFrame, expected_columns: list[str], name: str) -> pd.DataFrame:
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing expected columns: {missing}")
    return df


def load_subject_level(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _validate(df, SUBJECT_LEVEL_COLUMNS, "subject_level.csv")


def load_state_level(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _validate(df, STATE_LEVEL_COLUMNS, "state_level.csv")


def load_transition_matrices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _validate(df, TRANSITION_COLUMNS, "transition_matrices.csv")


def merged_state_level(subject_level: pd.DataFrame, state_level: pd.DataFrame) -> pd.DataFrame:
    """Join state-level metrics with subject-level covariates for modelling.

    Every row in the result is one subject x state observation with the
    subject's age/sex/cognition/behaviour attached, ready for a mixed model
    with ``state`` as a within-subject factor.
    """
    return state_level.merge(subject_level, on="subject_id", how="left", validate="many_to_one")


def transition_matrix_for_subject(transitions: pd.DataFrame, subject_id) -> pd.DataFrame:
    """Return the from_state x to_state transition matrix for one subject as a wide DataFrame."""
    sub = transitions[transitions["subject_id"] == subject_id]
    wide = sub.pivot(index="from_state", columns="to_state", values="probability")
    return wide.sort_index().sort_index(axis=1)
