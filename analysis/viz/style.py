"""Shared figure style, implementing the project style guide.

The style guide refers to a `geodesic_style` package which is not available
in this environment, so its specification is implemented here directly:
the Wong colourblind-safe palette, serif typography, the type scale, L-shaped
axes, the standard figure sizes, and `save_figure` writing both PNG and PDF.

Colours are assigned by semantic meaning rather than arbitrarily. Each HMM
state keeps the same colour in every figure, so a reader can carry state
identity across panels; a colour never signifies a plot-specific ranking.

Reference: Wong, B. (2011). Points of view: Color blindness. Nature Methods,
8(6), 441.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# --- Wong palette -----------------------------------------------------------

COLORS = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Black is reserved for axes and text; yellow has low contrast on white, so
# both are excluded from the automatic cycle but remain available by key.
COLOR_CYCLE = [
    COLORS["orange"], COLORS["blue"], COLORS["bluish_green"], COLORS["vermillion"],
    COLORS["reddish_purple"], COLORS["sky_blue"], COLORS["yellow"],
]

NEUTRAL_LIGHT_BLUE = "#B8D4E8"
NEUTRAL_GREY = "#AAAAAA"

# One fixed colour per HMM state. Seven states exhaust the palette once black
# is set aside, so yellow is included here; it carries a dark edge in every
# filled mark to compensate for its low contrast on white.
STATE_PALETTE = COLOR_CYCLE[:7]

SEQUENTIAL_CMAP = "Blues"
DIVERGING_CMAP = "RdBu_r"

TEXT_PRIMARY = COLORS["black"]
TEXT_SECONDARY = "#333333"
TEXT_MUTED = "#555555"

FIGURE_SIZES = {
    "single_column": (5.5, 4.0),
    "double_column": (10.0, 4.5),
    "full_page": (10.0, 10.0),
    "multi_panel_tall": (14.0, 15.0),
}

# Times New Roman is unavailable in this environment; the style guide lists
# DejaVu Serif as the alternative, with Liberation Serif and Nimbus Roman as
# metric-compatible fallbacks should the figures be rebuilt elsewhere.
SERIF_STACK = ["Times New Roman", "DejaVu Serif", "Liberation Serif", "Nimbus Roman"]


def state_palette(n_states: int) -> list[str]:
    """Fixed-order colours for `n_states` HMM states.

    Raises rather than cycling if more states are requested than the palette
    provides: a repeated colour would assert that two distinct states are the
    same entity.
    """
    if n_states > len(STATE_PALETTE):
        raise ValueError(
            f"The Wong palette provides {len(STATE_PALETTE)} colours once black is reserved "
            f"for axes and text; {n_states} states were requested. Facet the figure or group "
            f"states rather than introducing a colour outside the palette."
        )
    return STATE_PALETTE[:n_states]


def apply_style() -> None:
    """Apply the project figure style. Call once before building a figure."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": SERIF_STACK,
        "mathtext.fontset": "dejavuserif",
        "axes.prop_cycle": plt.cycler(color=COLOR_CYCLE),

        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "font.size": 11,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": TEXT_PRIMARY,
        "axes.labelcolor": TEXT_PRIMARY,
        "text.color": TEXT_PRIMARY,
        "xtick.color": TEXT_PRIMARY,
        "ytick.color": TEXT_PRIMARY,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "legend.frameon": False,
        "lines.linewidth": 1.5,

        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 150,
    })


def reference_line(ax, value: float, axis: str = "y") -> None:
    """Draw a reference line (e.g. zero, chance) in the house style: grey,
    dashed, thin and low-opacity, so it never competes with the data."""
    drawer = ax.axhline if axis == "y" else ax.axvline
    drawer(value, color=NEUTRAL_GREY, linestyle="--", linewidth=0.7, alpha=0.5, zorder=0)


def save_figure(fig, path_without_extension: str | Path) -> list[Path]:
    """Save a figure as both PNG (150 DPI) and PDF, per the style guide."""
    stem = Path(path_without_extension)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, kwargs in ((".png", {"dpi": 150}), (".pdf", {})):
        target = stem.with_suffix(suffix)
        fig.savefig(target, bbox_inches="tight", **kwargs)
        written.append(target)
    return written
