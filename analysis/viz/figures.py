"""One function per paper figure, built on the shared style in `style.py`.

Each function returns the created `matplotlib.axes.Axes` so callers can add
figure-specific titles/annotations and save with `fig.savefig(...)`, rather
than each figure hard-coding its own font sizes and axis limits as the old
MATLAB scripts did.
"""

from __future__ import annotations

import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.viz.style import (
    NODE_YELLOW,
    PIYG_NEGATIVE,
    PIYG_POSITIVE,
    DIVERGING_CMAP,
    DIVERGING_CMAP_PIYG,
    SEQUENTIAL_CMAP,
    STATE_PALETTE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
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


def transition_network(
    mean_transition_matrix: np.ndarray,
    node_values: dict[int, float],
    node_sizes: dict[int, float],
    significant_states: set[int] | None = None,
    state_labels: dict[int, str] | None = None,
    vlim: float | None = None,
    layout: str = "circular",
    seed: int = 7,
    ax=None,
):
    """Network ("cruciform") view of the state transition structure, with each
    state coloured by how its *transitions-into* measure relates to cognition.

    Deliberately different from the original manuscript's Figure 8A, which
    drew one coloured edge per transition cell thresholded at p < 0.025
    uncorrected - effects the paper itself reported as not surviving
    correction. Drawing them as findings overstates them. Here:

    - **edges** carry no inference. They show the mean transition structure
      (width and grey level track mean probability, self-transitions
      excluded), so the layout still conveys which states are hubs.
    - **nodes** carry the inference. Fill is the diverging colour scale
      applied to `node_values` (the correlation between transitions into
      that state and cognitive ability); states in `significant_states`
      get a heavy ring, so significance is marked once, explicitly, rather
      than implied by which elements were drawn at all.

    `node_sizes` is normally each state's mean fractional occupancy.
    """
    import networkx as nx

    apply_style()
    ax = ax or plt.gca()

    n_states = mean_transition_matrix.shape[0]
    states = list(range(1, n_states + 1))
    significant_states = significant_states or set()
    labels = state_labels or {k: f"State {k}" for k in states}

    off_diagonal = mean_transition_matrix.astype(float).copy()
    np.fill_diagonal(off_diagonal, 0.0)

    graph = nx.DiGraph()
    graph.add_nodes_from(states)
    for i in states:
        for j in states:
            if i != j and off_diagonal[i - 1, j - 1] > 0:
                graph.add_edge(i, j, weight=float(off_diagonal[i - 1, j - 1]))

    if layout == "circular":
        # Deterministic and collision-free: every node gets equal room and
        # labels sit radially outside, so nothing overlaps regardless of the
        # data. A force layout on a near-complete graph (every state reaches
        # every other) produces an arbitrary, lopsided arrangement that
        # invites reading structure into what is really just the optimiser's
        # starting point.
        angles = {s: np.pi / 2 - 2 * np.pi * i / n_states for i, s in enumerate(states)}
        pos = {s: np.array([np.cos(a), np.sin(a)]) for s, a in angles.items()}
    else:
        pos = nx.spring_layout(graph, weight="weight", seed=seed, k=1.1, iterations=400)
        angles = {s: np.arctan2(*pos[s][::-1]) for s in states}

    weights = np.array([graph[u][v]["weight"] for u, v in graph.edges()])
    wmax = weights.max() if len(weights) else 1.0
    nx.draw_networkx_edges(
        graph, pos, ax=ax,
        width=[0.4 + 3.6 * (w / wmax) for w in weights],
        edge_color=[(0.55, 0.55, 0.55, 0.20 + 0.55 * (w / wmax)) for w in weights],
        arrowsize=9, connectionstyle="arc3,rad=0.10", node_size=2400,
    )

    limit = vlim or max(abs(v) for v in node_values.values())
    cmap = plt.get_cmap(DIVERGING_CMAP_PIYG)
    norm = mpl.colors.Normalize(vmin=-limit, vmax=limit)

    size_max = max(node_sizes.values())
    for state in states:
        size = 900 + 3400 * (node_sizes[state] / size_max)
        is_sig = state in significant_states
        ax.scatter(
            *pos[state], s=size, zorder=3,
            color=cmap(norm(node_values[state])),
            edgecolors=TEXT_PRIMARY if is_sig else TEXT_MUTED,
            linewidths=2.6 if is_sig else 0.8,
        )

    for state in states:
        x, y = pos[state]
        # push the label radially outward from the centre so it never sits on
        # its own node or a neighbour's
        dx, dy = np.cos(angles[state]), np.sin(angles[state])
        ax.annotate(
            labels[state] + (" *" if state in significant_states else ""),
            (x, y), xytext=(34 * dx, 34 * dy), textcoords="offset points",
            ha="center" if abs(dx) < 0.4 else ("left" if dx > 0 else "right"),
            va="center" if abs(dy) < 0.4 else ("bottom" if dy > 0 else "top"),
            fontsize=10,
            fontweight="bold" if state in significant_states else "normal",
            color=TEXT_PRIMARY if state in significant_states else TEXT_SECONDARY,
        )

    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.margins(0.34)
    colorbar = ax.figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
        orientation="horizontal", fraction=0.045, pad=0.04,
    )
    colorbar.set_label("Correlation with cognitive ability (transitions into state)")
    return ax


def transition_correlation_heatmap(
    correlation_matrix: np.ndarray,
    state_labels: list[str] | None = None,
    vlim: float | None = None,
    ax=None,
):
    """Cell-wise correlations between each transition probability and cognitive
    ability, on the diverging magenta-green scale.

    **Descriptive only.** These 49 cells are not the confirmatory analysis:
    testing them individually is the multiplicity problem the reanalysis
    removed, so no cell is marked significant here. The confirmatory test was
    over transitions *into* each state (column sums) - see `transition_network`.
    """
    apply_style()
    ax = ax or plt.gca()

    n_states = correlation_matrix.shape[0]
    labels = state_labels or [str(i + 1) for i in range(n_states)]
    limit = vlim or float(np.abs(correlation_matrix).max())

    sns.heatmap(
        correlation_matrix, ax=ax, cmap=DIVERGING_CMAP_PIYG,
        vmin=-limit, vmax=limit, center=0, square=True,
        linewidths=0.5, linecolor="white",
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"label": "Correlation with cognitive ability", "orientation": "horizontal",
                  "fraction": 0.05, "pad": 0.10},
    )
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.tick_params(rotation=0)
    return ax


def significant_transition_network(
    correlation_matrix: np.ndarray,
    pvalue_matrix: np.ndarray,
    node_sizes: dict[int, float],
    alpha: float = 0.05,
    state_labels: dict[int, str] | None = None,
    legend_labels: tuple[str, str] | None = None,
    layout: str = "circular",
    seed: int = 7,
    ax=None,
):
    """State network in the style of the original manuscript figure: yellow
    nodes, and one arrow per transition whose correlation with cognitive
    ability is significant at `alpha`.

    Arrows are a flat magenta (negative) or green (positive) at a single
    width - the sign is the only thing they encode, so a continuous colour
    ramp or varying thickness would imply a precision the display does not
    have. Self-transitions are drawn as loops when significant.

    **These are uncorrected cell-wise tests** (49 of them at K = 7). The
    figure is descriptive, and any caption must say so: the confirmatory
    analysis is over transitions *into* each state, not individual cells.
    `transition_network` shows that result.
    """
    import networkx as nx

    apply_style()
    ax = ax or plt.gca()

    n_states = correlation_matrix.shape[0]
    states = list(range(1, n_states + 1))
    labels = state_labels or {k: f"State {k}" for k in states}

    if layout == "circular":
        angles = {s: np.pi / 2 - 2 * np.pi * i / n_states for i, s in enumerate(states)}
        pos = {s: np.array([np.cos(a), np.sin(a)]) for s, a in angles.items()}
    else:
        scaffold = nx.complete_graph(states, create_using=nx.DiGraph)
        pos = nx.spring_layout(scaffold, seed=seed)
        angles = {s: np.arctan2(*pos[s][::-1]) for s in states}

    size_max = max(node_sizes.values())
    node_size = {s: 900 + 3400 * (node_sizes[s] / size_max) for s in states}

    graph = nx.DiGraph()
    graph.add_nodes_from(states)
    positive, negative, self_loops = [], [], []
    for i in states:
        for j in states:
            if pvalue_matrix[i - 1, j - 1] >= alpha:
                continue
            if i == j:
                self_loops.append((i, correlation_matrix[i - 1, j - 1] > 0))
            else:
                graph.add_edge(i, j)
                (positive if correlation_matrix[i - 1, j - 1] > 0 else negative).append((i, j))

    for edges, colour in ((negative, PIYG_NEGATIVE), (positive, PIYG_POSITIVE)):
        if edges:
            nx.draw_networkx_edges(
                graph, pos, edgelist=edges, ax=ax,
                width=2.0, edge_color=colour, alpha=0.9, arrowsize=14,
                connectionstyle="arc3,rad=0.12",
                node_size=[node_size[s] for s in states],
            )

    # Self-loops need explicit geometry: an arrow from a point to itself has
    # zero length and renders nothing.
    #
    # Their size is set in DATA units, not derived from the marker size in
    # points. The points-per-data ratio is not knowable here - `tight_layout`
    # resizes the axes after this function returns - so a radius computed from
    # the transform comes out too small and the arc gets drawn across its own
    # node. The layout is always a unit circle at a fixed aspect, so
    # data-space constants are stable across figure sizes.
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.8, 1.8)
    node_radius_data = {s: 0.155 + 0.105 * (node_sizes[s] / size_max) for s in states}
    for state, is_positive in self_loops:
        x, y = pos[state]
        dx, dy = np.cos(angles[state]), np.sin(angles[state])
        loop_radius = 0.125
        centre = (x + (node_radius_data[state] + loop_radius * 0.95) * dx,
                  y + (node_radius_data[state] + loop_radius * 0.95) * dy)
        base_angle = np.degrees(np.arctan2(dy, dx))
        colour = PIYG_POSITIVE if is_positive else PIYG_NEGATIVE
        # drawn above the nodes: a loop tucked behind its own node reads as a
        # stray stub rather than a self-transition
        ax.add_patch(mpl.patches.Arc(
            centre, 2 * loop_radius, 2 * loop_radius,
            theta1=base_angle + 115, theta2=base_angle - 115,
            lw=2.0, color=colour, zorder=4,
        ))
        head_angle = np.radians(base_angle + 115)
        tip = (centre[0] + loop_radius * np.cos(head_angle),
               centre[1] + loop_radius * np.sin(head_angle))
        tangent = head_angle - np.pi / 2
        ax.annotate(
            "", xy=tip,
            xytext=(tip[0] - 0.022 * np.cos(tangent), tip[1] - 0.022 * np.sin(tangent)),
            arrowprops=dict(arrowstyle="-|>", mutation_scale=13, lw=2.0, color=colour),
            zorder=4,
        )

    for state in states:
        ax.scatter(
            *pos[state], s=node_size[state], zorder=3,
            color=NODE_YELLOW, edgecolors=TEXT_SECONDARY, linewidths=1.0,
        )

    # Self-loops sit radially outward from their node, exactly where the label
    # would otherwise go, so looped states need their labels pushed clear.
    looped = {state for state, _ in self_loops}
    for state in states:
        x, y = pos[state]
        dx, dy = np.cos(angles[state]), np.sin(angles[state])
        offset = 104 if state in looped else 38
        ax.annotate(
            labels[state], (x, y), xytext=(offset * dx, offset * dy), textcoords="offset points",
            ha="center" if abs(dx) < 0.4 else ("left" if dx > 0 else "right"),
            va="center" if abs(dy) < 0.4 else ("bottom" if dy > 0 else "top"),
            fontsize=10, color=TEXT_PRIMARY,
        )

    positive_label, negative_label = legend_labels or (
        f"Positive correlation (p < {alpha:g}, uncorrected)",
        f"Negative correlation (p < {alpha:g}, uncorrected)",
    )
    ax.legend(
        handles=[
            mpl.lines.Line2D([], [], color=PIYG_POSITIVE, lw=2.4, label=positive_label),
            mpl.lines.Line2D([], [], color=PIYG_NEGATIVE, lw=2.4, label=negative_label),
        ],
        loc="upper left", bbox_to_anchor=(-0.04, 1.04), frameon=False, fontsize=10,
    )
    ax.set_axis_off()
    ax.set_aspect("equal")
    return ax
