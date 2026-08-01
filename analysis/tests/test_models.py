import math

import numpy as np
import pandas as pd
import pytest

from analysis.correction import correct_pvalues
from analysis.models import fit_state_metric_model, fit_subject_scalar_model

N_SUBJECTS = 30
N_STATES = 7


@pytest.fixture
def synthetic_subject_level():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "subject_id": np.arange(1, N_SUBJECTS + 1),
        "age": rng.uniform(8, 13, N_SUBJECTS),
        "sex": rng.integers(1, 3, N_SUBJECTS),
        "WASI_T": rng.normal(50, 10, N_SUBJECTS),
        "SDQ_total": rng.normal(10, 5, N_SUBJECTS),
        "switching_rate": rng.uniform(0, 0.1, N_SUBJECTS),
        "entropy_rate": rng.uniform(0, 2, N_SUBJECTS),
        "max_FO": rng.uniform(0.2, 0.6, N_SUBJECTS),
    })


@pytest.fixture
def synthetic_state_level(synthetic_subject_level):
    rng = np.random.default_rng(1)
    rows = []
    for subject_id in synthetic_subject_level["subject_id"]:
        for state in range(1, N_STATES + 1):
            rows.append({
                "subject_id": subject_id,
                "state": state,
                "FO": rng.dirichlet(np.ones(N_STATES))[0],
                "lifetime_mean": rng.uniform(50, 500),
                "interval_mean": rng.uniform(500, 5000),
            })
    state_df = pd.DataFrame(rows)
    return state_df.merge(synthetic_subject_level, on="subject_id", how="left")


def test_fit_state_metric_model_runs_and_returns_diagnostics(synthetic_state_level):
    result, diagnostics = fit_state_metric_model(synthetic_state_level, metric="FO")

    assert result.params is not None
    assert "normality" in diagnostics
    assert "shapiro_p" in diagnostics["normality"]
    assert "vif" in diagnostics
    assert set(diagnostics["vif"].columns) == {"term", "VIF"}


def test_fit_subject_scalar_model_runs_and_returns_diagnostics(synthetic_subject_level):
    result, diagnostics = fit_subject_scalar_model(synthetic_subject_level, outcome="entropy_rate")

    assert result.params is not None
    assert "normality" in diagnostics
    assert "vif" in diagnostics


def test_fit_subject_scalar_model_extra_terms(synthetic_subject_level):
    result, _ = fit_subject_scalar_model(
        synthetic_subject_level, outcome="entropy_rate", extra_terms=" + SDQ_total",
    )
    assert "SDQ_total" in result.params.index


def test_correct_pvalues_applies_one_correction_across_whole_family():
    pvalues = {"FO_state_main_effect": 0.001, "FO_state_x_cognition": 0.04,
               "entropy_cognition": 0.03, "switching_rate_cognition": 0.20}
    corrected = correct_pvalues(pvalues)

    assert set(corrected["term"]) == set(pvalues.keys())
    assert (corrected["p_adjusted"] >= corrected["p_raw"]).all()


def test_correct_pvalues_rejects_empty_input():
    with pytest.raises(ValueError):
        correct_pvalues({})


def test_correct_pvalues_excludes_nan_terms_instead_of_poisoning_whole_result():
    # A mixed model's "Group Var" random-effect row has no real p-value (NaN).
    # Feeding that straight into statsmodels' multipletests makes every
    # p_adjusted come back NaN - regression test for that failure mode.
    pvalues = {"FO_state_main_effect": 0.001, "FO_state_x_cognition": 0.04, "FO_Group_Var": float("nan")}

    with pytest.warns(UserWarning, match="FO_Group_Var"):
        corrected = correct_pvalues(pvalues)

    assert set(corrected["term"]) == set(pvalues.keys())
    real_terms = corrected[corrected["term"] != "FO_Group_Var"]
    assert real_terms["p_adjusted"].notna().all()
    dropped_row = corrected[corrected["term"] == "FO_Group_Var"].iloc[0]
    assert math.isnan(dropped_row["p_adjusted"])
    assert dropped_row["significant"] == False  # noqa: E712 (numpy/pandas bool, not `is False`)


def test_correct_pvalues_all_nan_raises():
    with pytest.raises(ValueError):
        correct_pvalues({"a": float("nan"), "b": float("nan")})
