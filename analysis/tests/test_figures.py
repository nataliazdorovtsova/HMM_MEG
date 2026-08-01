import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from analysis.viz.figures import (
    cognition_scatter,
    corr_heatmap,
    state_metric_distribution,
    transition_correlation_heatmap,
    transition_heatmap,
)


@pytest.fixture(autouse=True)
def _close_figures_between_tests():
    # Several tests below call the figure functions without an explicit `ax`,
    # which falls back to matplotlib's global `plt.gca()` inside them - without
    # this, a log-scale axis set by one test can leak into the next test's
    # "fresh" gca() and produce confusing failures unrelated to what's being
    # tested.
    yield
    plt.close("all")


@pytest.fixture
def synthetic_subject_level():
    rng = np.random.default_rng(0)
    n = 20
    return pd.DataFrame({
        "age": rng.uniform(8, 13, n),
        "WASI_T": rng.normal(50, 10, n),
        "SDQ_total": rng.normal(10, 5, n),
    })


@pytest.fixture
def synthetic_state_df():
    rng = np.random.default_rng(1)
    rows = []
    for subject_id in range(1, 11):
        for state in range(1, 8):
            rows.append({"subject_id": subject_id, "state": state, "metric": rng.uniform(0.01, 1.0)})
    return pd.DataFrame(rows)


def test_corr_heatmap_runs_and_wraps_title(synthetic_subject_level):
    ax = corr_heatmap(synthetic_subject_level, columns=["age", "WASI_T", "SDQ_total"])
    # title should be wrapped (multi-line), not a single unbroken long string
    assert "\n" in ax.get_title()


def test_state_metric_distribution_runs_with_and_without_log_scale(synthetic_state_df):
    ax_linear = state_metric_distribution(synthetic_state_df, metric="metric")
    assert ax_linear.get_yscale() == "linear"

    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    ax_log = state_metric_distribution(synthetic_state_df, metric="metric", log_scale=True, ax=ax)
    assert ax_log.get_yscale() == "log"


def test_state_metric_distribution_rejects_more_states_than_palette(synthetic_state_df):
    extra_state = synthetic_state_df.copy()
    extra_state["state"] = 8  # an 8th state; only 7 validated colours exist
    combined = pd.concat([synthetic_state_df, extra_state], ignore_index=True)
    with pytest.raises(ValueError):
        state_metric_distribution(combined, metric="metric")


def test_cognition_scatter_runs(synthetic_subject_level):
    ax = cognition_scatter(synthetic_subject_level, x="WASI_T", y="SDQ_total")
    assert ax.get_xlabel() == "WASI_T"
    assert ax.get_ylabel() == "SDQ_total"


def test_transition_heatmap_masks_diagonal_by_default():
    rng = np.random.default_rng(0)
    matrix = rng.dirichlet(np.ones(7), size=7)  # rows sum to 1, like a real transition matrix
    np.fill_diagonal(matrix, 0.99)  # dominant self-transitions, like the real HMM data

    ax = transition_heatmap(matrix)
    # the colorbar is appended to the same figure as its own Axes; its label
    # should say the diagonal was masked
    colorbar_ax = ax.figure.axes[-1]
    assert "masked" in colorbar_ax.get_ylabel()


def test_transition_heatmap_can_show_diagonal():
    rng = np.random.default_rng(0)
    matrix = rng.dirichlet(np.ones(7), size=7)

    ax = transition_heatmap(matrix, mask_diagonal=False)
    colorbar_ax = ax.figure.axes[-1]
    assert "masked" not in colorbar_ax.get_ylabel()





def test_transition_correlation_heatmap_is_symmetric_about_zero():
    rng = np.random.default_rng(0)
    correlations = rng.uniform(-0.4, 0.5, size=(7, 7))

    ax = transition_correlation_heatmap(correlations)

    # a diverging scale must be centred on zero or the colours mislead
    mesh = ax.collections[0]
    assert mesh.norm.vmin == pytest.approx(-mesh.norm.vmax)



