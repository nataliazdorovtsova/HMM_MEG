"""Generate the numbered manuscript figures.

    python -m analysis.make_paper_figures \
        --export-dir /path/to/HMM_MEG_exports \
        --output-dir figures

Numbering follows the revised manuscript, in which the original Figure 6
(switching rate and entropy against cognitive ability) is dropped because
neither relationship survives multiplicity correction, and the subsequent
figures move up by one:

    Figure 1  cognitive and behavioural correlations
    Figure 2  preprocessing schematic        (not generated here)
    Figure 3  state maps, HCP Workbench      (not generated here)
    Figure 4  state intervals and lifetimes
    Figure 5  Viterbi path, transition matrices and transition digraph
              (components 5A-5D, assembled by hand)
    Figure 6  fractional occupancy against cognitive ability, states 1, 4 and 6
    Figure 7  transitions into each state against cognitive ability
    Figure 8  transitions into states 3, 4 and 6 against cognitive ability

Supplementary figures S1 and S2 show all seven states for occupancy and for
transitions respectively, so that a reader can see the states which do not
survive correction alongside those which do.

Figures 2 and 3 are not produced here: Figure 2 is a schematic containing no
data, and Figure 3 renders cortical surfaces through HCP Workbench.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from analysis.io import (
    load_state_level,
    load_subject_level,
    load_transition_matrices,
    merged_state_level,
)
from analysis.models import transition_column_sums
from analysis.viz.style import (
    DIVERGING_CMAP,
    FIGURE_SIZES,
    NEUTRAL_GREY,
    SEQUENTIAL_CMAP,
    TEXT_PRIMARY,
    apply_style,
    save_figure,
    state_palette,
)

FS = 250  # sampling rate of the state time courses, in Hz

STATE_LABELS = {
    1: "State 1\n(DMN+)", 2: "State 2\n(VT+)", 3: "State 3\n(DMN−)",
    4: "State 4\n(SM+, FT−)", 5: "State 5\n(V+, FT−)",
    6: "State 6\n(FP+, V−)", 7: "State 7\n(LP+, DMN−)",
}
SHORT_LABELS = {1: "DMN+", 2: "VT+", 3: "DMN−", 4: "SM+, FT−",
                5: "V+, FT−", 6: "FP+, V−", 7: "LP+, DMN−"}

BEHAVIOURAL_COLUMNS = [
    "WASI_T", "SDQ_total", "SDQ_hyperactivity", "SDQ_conduct",
    "SDQ_peerproblems", "SDQ_emotion", "SDQ_prosocial",
]
BEHAVIOURAL_LABELS = [
    "WASI-II MR", "SDQ total", "SDQ hyper-\nactivity", "SDQ conduct",
    "SDQ peer\nproblems", "SDQ emotion", "SDQ prosocial",
]

# States whose own slope survives correction across all 31 reported tests
OCCUPANCY_STATES = [1, 4, 6]
TRANSITION_STATES = [3, 4, 6]
ALL_STATES = list(range(1, 8))


def _load(export_dir: Path):
    subject_level = load_subject_level(str(export_dir / "subject_level.csv"))
    state_level = load_state_level(str(export_dir / "state_level.csv"))
    transitions = load_transition_matrices(str(export_dir / "transition_matrices.csv"))
    merged = merged_state_level(subject_level, state_level)
    merged["lifetime_ms"] = merged["lifetime_mean"] / FS * 1000
    merged["interval_ms"] = merged["interval_mean"] / FS * 1000
    return subject_level, merged, transitions


def _scatter(ax, x, y, xlabel, ylabel, colour):
    """Scatter with a least-squares fit and its 95% confidence band."""
    ax.scatter(x, y, s=34, color=colour, edgecolor=TEXT_PRIMARY, linewidth=0.4,
               alpha=0.85, zorder=3)
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(grid, slope * grid + intercept, color=TEXT_PRIMARY, linewidth=1.5, zorder=4)

    # 95% band from the standard error of prediction
    n = len(x)
    resid = y - (slope * x + intercept)
    se = np.sqrt(np.sum(resid**2) / (n - 2))
    band = se * np.sqrt(1 / n + (grid - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    ax.fill_between(grid, slope * grid + intercept - 1.96 * band,
                    slope * grid + intercept + 1.96 * band,
                    color=NEUTRAL_GREY, alpha=0.25, linewidth=0, zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def figure1_correlations(subject_level, output_dir: Path):
    apply_style()
    corr = subject_level[BEHAVIOURAL_COLUMNS].corr()
    n = len(BEHAVIOURAL_COLUMNS)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["full_page"])
    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING_CMAP, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            r = corr.iloc[i, j]
            ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if abs(r) > 0.6 else TEXT_PRIMARY)
    ax.set_xticks(range(n), BEHAVIOURAL_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(n), BEHAVIOURAL_LABELS)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation", fontsize=14, fontweight="bold")
    cbar.outline.set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_dir / "figure_1_correlations")
    plt.close(fig)


def figure4_intervals_lifetimes(merged, output_dir: Path):
    apply_style()
    palette = state_palette(7)
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["double_column"])

    for ax, column, label, letter, log in (
        (axes[0], "interval_ms", "Interval (ms)", "A", True),
        (axes[1], "lifetime_ms", "Lifetime (ms)", "B", False),
    ):
        data = [merged.loc[merged["state"] == s, column].to_numpy() for s in ALL_STATES]
        parts = ax.violinplot(data, positions=ALL_STATES, widths=0.75,
                              showextrema=False, showmedians=False)
        for body, colour in zip(parts["bodies"], palette):
            body.set_facecolor(colour)
            body.set_edgecolor(TEXT_PRIMARY)
            body.set_linewidth(0.6)
            body.set_alpha(0.55)
        for s, colour in zip(ALL_STATES, palette):
            values = merged.loc[merged["state"] == s, column].to_numpy()
            jitter = np.random.default_rng(s).normal(0, 0.055, len(values))
            ax.scatter(s + jitter, values, s=12, color=colour,
                       edgecolor=TEXT_PRIMARY, linewidth=0.3, alpha=0.85, zorder=3)
        if log:
            ax.set_yscale("log")
        ax.set_xticks(ALL_STATES, [str(s) for s in ALL_STATES])
        ax.set_xlabel("State")
        ax.set_ylabel(label)
        ax.set_title(letter, loc="left", fontsize=16, fontweight="bold")

    fig.tight_layout()
    save_figure(fig, output_dir / "figure_4_intervals_lifetimes")
    plt.close(fig)


def _transition_digraph(mean_matrix, mean_occupancy, palette):
    """Directed graph of the group mean transition matrix.

    Purely descriptive, and deliberately so: every off-diagonal transition is
    drawn, none is marked significant, and no arrow here corresponds to a test.
    The confirmatory transition result is over transitions *into* each state
    (the column sums), which Figures 7 and 8 report.

    Self-transitions are excluded rather than drawn as self-loops. At 250 Hz
    every state persists with probability ~0.91, so seven near-identical loops
    would dominate the figure while carrying no information about which state
    follows which; Figure 4 already reports state persistence as lifetimes.

    Nodes sit on a fixed circle rather than a force-directed layout, so the
    figure is reproducible and node position carries no meaning. Arrows are
    drawn in data coordinates on an equal-aspect axis with fixed limits, so
    later layout adjustments cannot detach them from their nodes.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import FancyArrowPatch

    n = len(ALL_STATES)
    angles = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    positions = np.column_stack([np.cos(angles), np.sin(angles)])

    off_diagonal = ~np.eye(n, dtype=bool)
    weights = mean_matrix[off_diagonal]
    norm = Normalize(vmin=0.0, vmax=weights.max())
    cmap = plt.get_cmap(SEQUENTIAL_CMAP)

    # Radius in data units, from mean occupancy. The square root keeps state 1
    # (mean FO ~0.015, an order of magnitude below the rest) visible while
    # still ranking the nodes by how much time each state actually holds.
    radii = 0.055 + 0.16 * np.sqrt(mean_occupancy / mean_occupancy.max())

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.set_aspect("equal")
    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.32, 1.32)
    ax.axis("off")

    # Weakest arrows first, so the strong transitions are never buried.
    edges = sorted(
        ((i, j) for i in range(n) for j in range(n) if i != j),
        key=lambda ij: mean_matrix[ij],
    )
    for i, j in edges:
        weight = mean_matrix[i, j]
        # Shorten each arrow to the node boundaries so the head is visible
        # against the circle rather than hidden underneath it.
        delta = positions[j] - positions[i]
        unit = delta / np.linalg.norm(delta)
        start = positions[i] + unit * radii[i]
        end = positions[j] - unit * (radii[j] + 0.02)
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=11,
            connectionstyle="arc3,rad=0.14",  # i->j and j->i curve opposite ways
            linewidth=0.6 + 3.4 * norm(weight),
            color=cmap(0.25 + 0.75 * norm(weight)),
            alpha=0.45 + 0.55 * norm(weight), zorder=1,
        ))

    for index, state in enumerate(ALL_STATES):
        ax.add_patch(plt.Circle(
            positions[index], radii[index], facecolor=palette[index],
            edgecolor=TEXT_PRIMARY, linewidth=0.8, zorder=2,
        ))
        # Push the label radially outwards, clear of the node and its arrows,
        # then anchor it on the side facing away from the circle. Without the
        # side-dependent anchoring, a centred label on a node at the 3 o'clock
        # or 9 o'clock position runs straight back over its own node.
        x, y = positions[index] * (1 + radii[index] + 0.10)
        horizontal = np.cos(angles[index])
        alignment = "center" if abs(horizontal) < 0.3 else ("left" if horizontal > 0 else "right")
        ax.text(
            x, y, f"{state}: {SHORT_LABELS[state]}",
            ha=alignment, va="center", fontsize=10, fontweight="bold", zorder=3,
        )

    colourbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.035, pad=0.02, shrink=0.82,
    )
    colourbar.set_label("Mean transition probability", fontsize=13, fontweight="bold")
    colourbar.outline.set_visible(False)
    return fig


def figure5_components(merged, transitions, export_dir: Path, output_dir: Path):
    """Components of Figure 5, saved separately for manual assembly."""
    apply_style()
    palette = state_palette(7)

    # 5A: Viterbi path, first 1000 timesteps (4 s at 250 Hz)
    vpath_file = export_dir / "vpath_subject1_first1000.csv"
    if vpath_file.exists():
        vpath = np.loadtxt(vpath_file, delimiter=",").astype(int).ravel()
        fig, ax = plt.subplots(figsize=(10, 2.6))
        for t, state in enumerate(vpath):
            ax.axvspan(t, t + 1, color=palette[state - 1], linewidth=0)
        ax.set_xlim(0, len(vpath))
        ax.set_yticks([])
        ax.set_xlabel("Timestep")
        ax.set_ylabel("State")
        ax.spines["left"].set_visible(False)
        handles = [plt.Rectangle((0, 0), 1, 1, color=palette[s - 1]) for s in ALL_STATES]
        ax.legend(handles, [f"{s}: {SHORT_LABELS[s]}" for s in ALL_STATES],
                  loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=7, fontsize=11)
        fig.tight_layout()
        save_figure(fig, output_dir / "figure_5A_viterbi_path")
        plt.close(fig)

    # 5B: two example single-subject transition matrices
    subjects = sorted(transitions["subject_id"].unique())
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["double_column"])
    for ax, subject in zip(axes, (subjects[0], subjects[-1])):
        wide = (transitions[transitions["subject_id"] == subject]
                .pivot(index="from_state", columns="to_state", values="probability")
                .sort_index().sort_index(axis=1).to_numpy())
        im = ax.imshow(wide, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1)
        ax.set_xticks(range(7), [str(s) for s in ALL_STATES])
        ax.set_yticks(range(7), [str(s) for s in ALL_STATES])
        ax.set_xlabel("To state")
        ax.set_ylabel("From state")
        ax.set_title(f"Participant {subject}", fontsize=14)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03)
    cbar.set_label("Transition probability", fontsize=14, fontweight="bold")
    cbar.outline.set_visible(False)
    save_figure(fig, output_dir / "figure_5B_single_subject_matrices")
    plt.close(fig)

    mean_matrix = (transitions.groupby(["from_state", "to_state"])["probability"]
                   .mean().unstack().sort_index().sort_index(axis=1).to_numpy())

    # 5C: directed graph of the same mean transition matrix
    mean_occupancy = merged.groupby("state")["FO"].mean().reindex(ALL_STATES).to_numpy()
    fig = _transition_digraph(mean_matrix, mean_occupancy, palette)
    save_figure(fig, output_dir / "figure_5C_transition_digraph")
    plt.close(fig)

    # 5D: group mean transition matrix, self-transitions masked
    masked = np.ma.masked_array(mean_matrix, mask=np.eye(7, dtype=bool))
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_column"])
    im = ax.imshow(masked, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=masked.max())
    for i in range(7):
        for j in range(7):
            if i != j:
                ax.text(j, i, f"{mean_matrix[i, j]:.3f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if mean_matrix[i, j] > 0.6 * masked.max() else TEXT_PRIMARY)
    ax.set_xticks(range(7), [str(s) for s in ALL_STATES])
    ax.set_yticks(range(7), [str(s) for s in ALL_STATES])
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Transition probability", fontsize=14, fontweight="bold")
    cbar.outline.set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_dir / "figure_5D_mean_transition_matrix")
    plt.close(fig)


def _state_scatter_grid(df, states, value_column, ylabel_template, output_stem,
                        output_dir: Path, ncols=3):
    apply_style()
    palette = state_palette(7)
    nrows = int(np.ceil(len(states) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                            figsize=(5.0 * ncols, 4.0 * nrows), squeeze=False)
    for index, state in enumerate(states):
        ax = axes[index // ncols][index % ncols]
        subset = df[df["state"] == state]
        _scatter(ax, subset["WASI_T"].to_numpy(), subset[value_column].to_numpy(),
                 "WASI-II matrix reasoning T-score",
                 ylabel_template.format(state=state), palette[state - 1])
        ax.set_title(f"State {state} ({SHORT_LABELS[state]})", fontsize=14)
    for index in range(len(states), nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")
    fig.tight_layout()
    save_figure(fig, output_dir / output_stem)
    plt.close(fig)


def figure6_occupancy(merged, output_dir: Path):
    _state_scatter_grid(merged, OCCUPANCY_STATES, "FO",
                        "Fractional occupancy", "figure_6_occupancy_cognition",
                        output_dir, ncols=3)


def figure8_transitions(transitions, subject_level, output_dir: Path):
    df = transition_column_sums(transitions).merge(subject_level, on="subject_id")
    _state_scatter_grid(df, TRANSITION_STATES, "transition_into_sum",
                        "Probability of entering state",
                        "figure_8_transitions_cognition", output_dir, ncols=3)


def figure7_transition_overview(transitions, subject_level, merged, output_dir: Path):
    """Correlation of entry probability with cognitive ability, per state,
    alongside the cell-wise correlation matrix."""
    apply_style()
    palette = state_palette(7)
    column_sums = transition_column_sums(transitions).merge(subject_level, on="subject_id")

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["double_column"],
                             gridspec_kw={"width_ratios": [1, 1.15]})

    ax = axes[0]
    rs = [pearsonr(column_sums.loc[column_sums.state == s, "transition_into_sum"],
                   column_sums.loc[column_sums.state == s, "WASI_T"])[0] for s in ALL_STATES]
    bars = ax.bar(ALL_STATES, rs, color=[palette[s - 1] for s in ALL_STATES],
                  edgecolor=TEXT_PRIMARY, linewidth=0.6, width=0.7)
    # Headroom first, so the significance markers sit clear of the bar ends and
    # of the axis; without it the marker on a negative bar lands on the tick label.
    span = max(abs(min(rs)), abs(max(rs)))
    ax.set_ylim(-span * 1.35, span * 1.35)
    pad = span * 0.06
    for state, bar, r in zip(ALL_STATES, bars, rs):
        if state in TRANSITION_STATES:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    r + pad if r > 0 else r - pad, "*",
                    ha="center", va="bottom" if r > 0 else "top",
                    fontsize=16, fontweight="bold")
    ax.axhline(0, color=TEXT_PRIMARY, linewidth=0.8, zorder=1)
    ax.set_xticks(ALL_STATES, [str(s) for s in ALL_STATES])
    ax.set_xlabel("State")
    ax.set_ylabel("Correlation with cognitive ability")
    ax.set_title("A", loc="left", fontsize=16, fontweight="bold")

    ax = axes[1]
    wide = transitions.pivot_table(index="subject_id",
                                   columns=["from_state", "to_state"], values="probability")
    cognition = subject_level.set_index("subject_id").loc[wide.index, "WASI_T"]
    cellwise = np.array([[pearsonr(wide[(i, j)], cognition)[0] for j in ALL_STATES]
                         for i in ALL_STATES])
    limit = float(np.abs(cellwise).max())
    im = ax.imshow(cellwise, cmap=DIVERGING_CMAP, vmin=-limit, vmax=limit)
    ax.set_xticks(range(7), [str(s) for s in ALL_STATES])
    ax.set_yticks(range(7), [str(s) for s in ALL_STATES])
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title("B", loc="left", fontsize=16, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation with\ncognitive ability", fontsize=12, fontweight="bold")
    cbar.outline.set_visible(False)

    fig.tight_layout()
    save_figure(fig, output_dir / "figure_7_transition_overview")
    plt.close(fig)


def supplementary_all_states(merged, transitions, subject_level, output_dir: Path):
    _state_scatter_grid(merged, ALL_STATES, "FO", "Fractional occupancy",
                        "figure_S1_occupancy_all_states", output_dir, ncols=4)
    df = transition_column_sums(transitions).merge(subject_level, on="subject_id")
    _state_scatter_grid(df, ALL_STATES, "transition_into_sum",
                        "Probability of entering state",
                        "figure_S2_transitions_all_states", output_dir, ncols=4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    output_dir = Path(args.output_dir)
    subject_level, merged, transitions = _load(export_dir)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        figure1_correlations(subject_level, output_dir)
        figure4_intervals_lifetimes(merged, output_dir)
        figure5_components(merged, transitions, export_dir, output_dir)
        figure6_occupancy(merged, output_dir)
        figure7_transition_overview(transitions, subject_level, merged, output_dir)
        figure8_transitions(transitions, subject_level, output_dir)
        supplementary_all_states(merged, transitions, subject_level, output_dir)

    for path in sorted(output_dir.glob("*.png")):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
