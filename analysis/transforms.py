"""Distribution-shape fixes for outcomes whose residuals violate the normality
diagnostic in `models.py` - e.g. fractional occupancy, a bounded proportion
with a strong floor effect that produces severely non-normal residuals in a
plain linear/mixed model (Shapiro-Wilk p ~ 1e-23 on the real HMM_MEG data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


def rank_based_inverse_normal_transform(x: pd.Series, c: float = 3 / 8) -> pd.Series:
    """Blom's rank-based inverse-normal transform.

    Ranks `x` (average rank for ties), then maps ranks onto normal quantiles
    via `Phi^-1((rank - c) / (n - 2c + 1))`. Outcome-agnostic: the same
    transform applies to FO, lifetime, interval, entropy rate, or switching
    rate without any outcome-specific handling, unlike a logit transform
    (which only makes sense for a bounded proportion like FO).
    """
    n = x.shape[0]
    ranks = rankdata(x, method="average")
    quantiles = (ranks - c) / (n - 2 * c + 1)
    return pd.Series(norm.ppf(quantiles), index=x.index, name=f"{x.name}_rankint")


def log_transform(x: pd.Series) -> pd.Series:
    """Natural-log transform for strictly-positive duration/waiting-time outcomes
    (state lifetimes, intervals). More principled than the generic rank-INT for
    this specific data type: durations are commonly ~log-normal, and log-scale
    coefficients are directly interpretable as approximate proportional change,
    unlike rank-transformed coefficients. Raises if any value is <= 0 rather than
    silently producing -inf/NaN - a duration of 0 or less should never occur
    (the exported lifetime/interval means are always >= 1 sample), so it signals
    a real data problem rather than something to paper over with an epsilon.
    """
    if (x <= 0).any():
        raise ValueError(f"log_transform requires strictly positive values; {x.name} has non-positive entries")
    return pd.Series(np.log(x), index=x.index, name=f"{x.name}_log")
