"""One function per paper figure, built on the shared style in `style.py`.

Each function returns the created `matplotlib.axes.Axes` so callers can add
figure-specific titles/annotations and save with `fig.savefig(...)`, rather
than each figure hard-coding its own font sizes and axis limits as the old
MATLAB scripts did.
"""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.viz.style import (
    DIVERGING_CMAP,
    SEQUENTIAL_CMAP,
    STATE_PALETTE,
    TEXT_PRIMARY,
    apply_style,
    state_palette,
)


def corr_heatmap(df: pd.DataFrame, columns: list[str], ax=None):
    """Correlation heatmap for `columns`, diverging colour (polarity: -1..1)."""
    apply_style()
    ax = ax or plt.gca()

    corr = df[columns].corr()
    sns.heatmap(
        corr, ax=ax, cmap=DIVERGING_CMAP, vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", square=True, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    # Wrapped, not shrunk: a long title at full font size can run past the
    # axes into the colorbar (seen on the real figure) - wrapping avoids that
    # without reintroducing the small, illegible fonts reviewers complained
    # about in the original figures.
    title = "\n".join(textwrap.wrap("Correlations between cognitive and behavioural scores", width=40))
    ax.set_title(title, color=TEXT_PRIMARY)
    return ax


def state_metric_distribution(
    df: pd.DataFrame,
    metric: str,
    state_col: str = "state",
    subject_col: str = "subject_id",
    ylabel: str | None = None,
    log_scale: bool = False,
    ax=None,
):
    """Distribution of `metric` per state, with individual subject points overlaid.

    Reviewer #3 (concern 2b) explicitly asked for per-state distributions
    with individual data points shown, for every metric of interest - not
    just lifetimes/intervals as in the original Figure 4.

    `log_scale=True` for metrics like interval time, where one state (e.g. a
    rarely-visited one) can span orders of magnitude more range than the
    others - on a linear axis that state's range dwarfs the rest, crushing
    every other state's variation into an unreadable sliver near zero.
    """
    apply_style()
    ax = ax or plt.gca()

    n_states = df[state_col].nunique()
    palette = state_palette(n_states)
    order = sorted(df[state_col].unique())

    sns.violinplot(
        data=df, x=state_col, y=metric, order=order, hue=state_col,
        palette=palette, inner=None, cut=0, legend=False, ax=ax,
        log_scale=log_scale,
    )
    for patch in ax.collections:
        patch.set_alpha(0.5)
    sns.stripplot(
        data=df, x=state_col, y=metric, order=order, hue=state_col,
        palette=palette, size=4, alpha=0.7, jitter=0.2, legend=False, ax=ax,
    )

    ax.set_xlabel("State")
    ax.set_ylabel(ylabel or metric)
    return ax


def cognition_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ax=None,
):
    """Scatter + regression line for a cognition/behaviour-vs-metric relationship.

    Replaces the ~10 near-identical `fitlm`/`plot` blocks in
    HMM_CognitiveBehaviouralAnalyses.m with one reusable function.
    """
    apply_style()
    ax = ax or plt.gca()

    sns.regplot(
        data=df, x=x, y=y, ax=ax,
        scatter_kws={"color": STATE_PALETTE[0], "alpha": 0.7, "s": 40},
        line_kws={"color": TEXT_PRIMARY, "linewidth": 2},
    )
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    return ax


def transition_heatmap(
    mean_transition_matrix: np.ndarray, state_labels: list[str] | None = None,
    mask_diagonal: bool = True, ax=None,
):
    """Heatmap of the (mean, across subjects) state transition matrix. Sequential colour: magnitude.

    `mask_diagonal=True` (default) blanks the self-transition cells before
    plotting, matching the original HMM_Entropy.m script's approach
    (`Ao(find(eye(nstate))) = 0`). At this HMM's sampling rate, self-
    transition probability is ~0.99+ for every state, so on a shared 0-1
    colour scale the diagonal saturates dark and every off-diagonal cell -
    the actually interesting "which state do you switch to" structure -
    is indistinguishable near-white. Masking the diagonal both hides the
    uninformative self-transitions and lets the colour scale stretch across
    the off-diagonal values that actually vary.
    """
    apply_style()
    ax = ax or plt.gca()

    n_states = mean_transition_matrix.shape[0]
    labels = state_labels or [f"State {i + 1}" for i in range(n_states)]

    matrix = mean_transition_matrix.astype(float).copy()
    mask = np.eye(n_states, dtype=bool) if mask_diagonal else None
    vmax = np.nanmax(matrix[~mask]) if mask_diagonal else 1.0

    sns.heatmap(
        matrix, ax=ax, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=vmax, mask=mask,
        square=True, linewidths=0.5, linecolor="white",
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"label": "Transition probability" + (" (self-transitions masked)" if mask_diagonal else "")},
    )
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    return ax
