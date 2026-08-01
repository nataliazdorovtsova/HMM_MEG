"""Shared plotting style: one place to fix font sizes, palette, and chart
chrome so every figure is consistent, instead of the ~10 near-identical
hand-tuned `FontSize`/`Position`/`ylim` blocks in the old MATLAB scripts.

These are static, print-destined publication figures (not an interactive
dashboard), so this module targets a single light/white surface rather than
light+dark theming.

Palette values are the first 7 categorical slots of the validated default
palette (see the `dataviz` skill's `references/palette.md`): documented as
passing the adjacent-pair CVD/contrast checks in both light and dark modes
(worst adjacent CVD deltaE 9.1 light / 8.4 dark, both above the deltaE >= 8
target), which is the relevant check for these plots (states sit side by
side in bar/box/violin/strip charts, never in an all-pairs scatter/small-
multiples layout). One slot per HMM state; do not reassign to another state
across figures - color follows the entity (state identity), never a
plot-specific ranking.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

# Categorical: one fixed hue per HMM state (slots 1-7 of the validated palette).
STATE_PALETTE = [
    "#2a78d6",  # state 1 - blue
    "#eb6834",  # state 2 - orange
    "#1baf7a",  # state 3 - aqua
    "#eda100",  # state 4 - yellow
    "#e87ba4",  # state 5 - magenta
    "#008300",  # state 6 - green
    "#4a3aa7",  # state 7 - violet
]

# Sequential: single hue, light -> dark, for magnitude (e.g. transition probability heatmaps).
SEQUENTIAL_CMAP = "Blues"

# Diverging: blue <-> red with a neutral gray midpoint, for polarity (e.g. correlation matrices).
DIVERGING_CMAP = "RdBu_r"

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


def state_palette(n_states: int) -> list[str]:
    """Fixed-order categorical colours for `n_states` HMM states.

    Raises if more states are requested than the validated palette covers,
    rather than silently cycling colours (a repeated hue would misrepresent
    two different states as the same identity).
    """
    if n_states > len(STATE_PALETTE):
        raise ValueError(
            f"Only {len(STATE_PALETTE)} validated categorical colours are available, "
            f"got {n_states} states. Fold extra states into 'Other' or facet instead "
            f"of adding an unvalidated colour."
        )
    return STATE_PALETTE[:n_states]


def apply_style() -> None:
    """Apply the shared figure style. Call once at the top of any figure script."""
    sns.set_theme(style="white", rc={
        "axes.edgecolor": TEXT_MUTED,
        "axes.labelcolor": TEXT_PRIMARY,
        "text.color": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "grid.color": GRIDLINE,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })
