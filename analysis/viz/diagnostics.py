"""Model-fit diagnostic plots - distinct from `figures.py`, which is the
paper's actual figure set. A Shapiro-Wilk p-value alone can be misleading at
larger sample sizes (it flags small, possibly-inconsequential deviations),
so every non-normality call in `models.py` diagnostics should be checked
visually here before being treated as a real problem or a resolved one.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analysis.viz.style import STATE_PALETTE, TEXT_MUTED, TEXT_PRIMARY, apply_style


def residual_qq_plot(residuals: np.ndarray, title: str | None = None, ax=None):
    """QQ plot of model residuals against a normal distribution."""
    apply_style()
    ax = ax or plt.gca()

    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, s=16, alpha=0.6, color=STATE_PALETTE[0])
    ax.plot(osm, slope * osm + intercept, color=TEXT_PRIMARY, linewidth=2)

    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles (residuals)")
    if title:
        ax.set_title(title, color=TEXT_PRIMARY)
    ax.tick_params(colors=TEXT_MUTED)
    return ax
