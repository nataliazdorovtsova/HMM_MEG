import matplotlib

matplotlib.use("Agg")

import numpy as np

from analysis.viz.diagnostics import residual_qq_plot


def test_residual_qq_plot_runs_and_labels_axes():
    rng = np.random.default_rng(0)
    residuals = rng.normal(size=100)

    ax = residual_qq_plot(residuals, title="test")

    assert ax.get_xlabel() == "Theoretical quantiles"
    assert ax.get_ylabel() == "Sample quantiles (residuals)"
    assert ax.get_title() == "test"
