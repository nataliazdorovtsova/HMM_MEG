"""Entropy-rate calculations over HMM state transition matrices.

This is a direct port of the math in ``matlab/legacy/HMM_Entropy.m`` (stationary
distribution -> entropy rate), which is plain linear algebra over an
already-computed transition matrix and does not depend on any HMM-MAR
toolbox call. It exists here so the entropy rate reported in
``subject_level.csv`` (computed in MATLAB) can be independently
cross-checked from the exported ``transition_matrices.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.io import transition_matrix_for_subject


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Stationary distribution mu of a row-stochastic transition matrix P.

    Solves mu @ P = mu, i.e. mu is a left eigenvector of P with eigenvalue 1,
    equivalently a right null-space vector of (P.T - I). Mirrors MATLAB's
    ``null(P' - eye(n))`` followed by normalisation in matlab/legacy/HMM_Entropy.m.
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if P.shape != (n, n):
        raise ValueError(f"P must be square, got shape {P.shape}")

    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    mu = np.real(eigenvectors[:, idx])
    mu = mu / mu.sum()
    return mu


def entropy_rate(P: np.ndarray, mu: np.ndarray | None = None, fs: float | None = None) -> float:
    """Entropy rate of a Markov chain with transition matrix P, in nats per unit time.

    If ``fs`` (samples/sec) is given, the result is converted to nats/second
    by multiplying by ``fs`` (matching the ``Ent_rate * 4`` conversion in
    HMM_Entropy.m for data sampled at 4 samples/second post-HMM-inference).
    """
    P = np.asarray(P, dtype=float)
    if mu is None:
        mu = stationary_distribution(P)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_P = np.where(P > 0, np.log(P), 0.0)
    state_entropy = -np.sum(P * log_P, axis=1)
    rate = float(np.dot(mu, state_entropy))

    if fs is not None:
        rate *= fs
    return rate


def entropy_rate_by_subject(transitions: pd.DataFrame, fs: float | None = 4.0) -> pd.DataFrame:
    """Compute entropy rate (and stationary distribution) for every subject.

    ``fs`` defaults to 4 to match matlab/legacy/HMM_Entropy.m's nats/second conversion;
    pass ``None`` to get nats per unit time instead.
    """
    rows = []
    for subject_id in sorted(transitions["subject_id"].unique()):
        P = transition_matrix_for_subject(transitions, subject_id).to_numpy()
        mu = stationary_distribution(P)
        rate = entropy_rate(P, mu=mu, fs=fs)
        rows.append({"subject_id": subject_id, "entropy_rate": rate})
    return pd.DataFrame(rows)
