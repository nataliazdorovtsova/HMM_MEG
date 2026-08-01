"""Regenerate every supplement figure from the exported data.

Run as a script so the figures in `supplement/figures/` are reproducible
artifacts rather than one-off outputs pasted in by hand:

    python -m analysis.make_figures \
        --export-dir /path/to/HMM_MEG_exports \
        --output-dir supplement/figures

The export directory holds per-participant data and is deliberately outside
this repo (see `.gitignore` and the note in `HMM_ExportForPython.m`); only
the rendered figures are versioned.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.io import (
    load_state_level,
    load_subject_level,
    load_transition_matrices,
    merged_state_level,
)
from analysis.models import fit_state_metric_model, fit_subject_scalar_model, transition_column_sums
from analysis.transforms import rank_based_inverse_normal_transform
from analysis.viz.diagnostics import residual_qq_plot
from analysis.viz.figures import (
    cognition_scatter,
    corr_heatmap,
    state_metric_distribution,
    transition_heatmap,
)

FS = 250  # HMM_FinishPreprocessing.m: options.downsample = 250

BEHAVIOURAL_COLUMNS = [
    "age", "WASI_T", "SDQ_total", "SDQ_hyperactivity",
    "SDQ_conduct", "SDQ_peerproblems", "SDQ_emotion", "SDQ_prosocial",
]

# States surviving the final multiplicity correction across FO, interval and
# transition-into-state (see supplement section 7).
HIGHLIGHT_STATES = [3, 4, 6, 7]


def _load(export_dir: Path):
    subject_level = load_subject_level(str(export_dir / "subject_level.csv"))
    state_level = load_state_level(str(export_dir / "state_level.csv"))
    transitions = load_transition_matrices(str(export_dir / "transition_matrices.csv"))

    merged = merged_state_level(subject_level, state_level)
    merged["lifetime_sec"] = merged["lifetime_mean"] / FS
    merged["interval_sec"] = merged["interval_mean"] / FS
    return subject_level, merged, transitions


def figure_correlations(subject_level, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 7))
    corr_heatmap(subject_level, columns=BEHAVIOURAL_COLUMNS, ax=ax)
    fig.tight_layout()
    path = output_dir / "cognitive_behavioural_correlations.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def figure_state_distributions(merged, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    state_metric_distribution(merged, metric="FO", ylabel="Fractional occupancy", ax=axes[0])
    state_metric_distribution(merged, metric="lifetime_sec", ylabel="Mean lifetime (s)", ax=axes[1])
    # log scale: state 1's interval range (up to ~22s) otherwise crushes every
    # other state into an unreadable sliver near zero
    state_metric_distribution(
        merged, metric="interval_sec", ylabel="Mean interval (s)", log_scale=True, ax=axes[2],
    )
    fig.tight_layout()
    path = output_dir / "state_metric_distributions.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def figure_cognition_scatters(merged, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, state in zip(axes.ravel(), HIGHLIGHT_STATES):
        cognition_scatter(
            merged[merged["state"] == state], x="WASI_T", y="FO",
            xlabel="WASI-II matrix reasoning T-score",
            ylabel=f"Fractional occupancy (state {state})",
            ax=ax,
        )
    fig.tight_layout()
    path = output_dir / "cognition_fo_scatters.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def figure_transition_heatmap(transitions, output_dir: Path) -> Path:
    mean_matrix = (
        transitions.groupby(["from_state", "to_state"])["probability"]
        .mean().unstack().sort_index().sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(7.5, 6))
    transition_heatmap(mean_matrix.to_numpy(), ax=ax)
    fig.tight_layout()
    path = output_dir / "transition_matrix_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def figure_residual_qq(subject_level, merged, output_dir: Path) -> Path:
    """Raw vs rank-INT residual QQ plots for FO and entropy rate.

    Uses the MixedLM/OLS fits exactly as reported in the supplement's
    diagnostics section, so the figure matches the Shapiro-Wilk numbers
    quoted alongside it.
    """
    merged = merged.copy()
    subject_level = subject_level.copy()
    merged["FO_rankint"] = rank_based_inverse_normal_transform(merged["FO"])
    subject_level["entropy_rate_rankint"] = rank_based_inverse_normal_transform(
        subject_level["entropy_rate"]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fo_raw, _ = fit_state_metric_model(merged, metric="FO")
        fo_rank, _ = fit_state_metric_model(merged, metric="FO_rankint")
        ent_raw, _ = fit_subject_scalar_model(subject_level, outcome="entropy_rate")
        ent_rank, _ = fit_subject_scalar_model(subject_level, outcome="entropy_rate_rankint")

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    residual_qq_plot(fo_raw.resid.values, title="FO (raw) residuals", ax=axes[0, 0])
    residual_qq_plot(fo_rank.resid.values, title="FO (rank-INT) residuals", ax=axes[0, 1])
    residual_qq_plot(ent_raw.resid.values, title="Entropy rate (raw) residuals", ax=axes[1, 0])
    residual_qq_plot(ent_rank.resid.values, title="Entropy rate (rank-INT) residuals", ax=axes[1, 1])
    fig.tight_layout()
    path = output_dir / "residual_qq_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-dir", required=True,
        help="Directory holding subject_level.csv, state_level.csv, transition_matrices.csv",
    )
    parser.add_argument(
        "--output-dir", default="supplement/figures",
        help="Where to write the rendered figures (default: supplement/figures)",
    )
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_level, merged, transitions = _load(export_dir)

    for path in [
        figure_correlations(subject_level, output_dir),
        figure_state_distributions(merged, output_dir),
        figure_cognition_scatters(merged, output_dir),
        figure_transition_heatmap(transitions, output_dir),
        figure_residual_qq(subject_level, merged, output_dir),
    ]:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
