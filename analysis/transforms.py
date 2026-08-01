"""Distribution-shape fixes for outcomes whose residuals violate the normality
diagnostic in `models.py` - e.g. fractional occupancy, a bounded proportion
with a strong floor effect that produces severely non-normal residuals in a
plain linear/mixed model (Shapiro-Wilk p ~ 1e-23 on the real HMM_MEG data).
"""

from __future__ import annotations

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
