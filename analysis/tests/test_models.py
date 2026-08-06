import math

import numpy as np
import pandas as pd
import pytest

from analysis.correction import correct_pvalues
from analysis.models import (
    fit_state_metric_model,
    fit_state_metric_model_cluster_ols,
    fit_state_metric_model_simple_slopes,
    fit_subject_scalar_model,
    state_slope_terms,
    transition_column_sums,
)

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


@pytest.fixture
def synthetic_transitions(synthetic_subject_level):
    rng = np.random.default_rng(2)
    rows = []
    for subject_id in synthetic_subject_level["subject_id"]:
        # each subject's matrix rows should sum to 1, like a real transition matrix
        matrix = rng.dirichlet(np.ones(N_STATES), size=N_STATES)
        for from_state in range(1, N_STATES + 1):
            for to_state in range(1, N_STATES + 1):
                rows.append({
                    "subject_id": subject_id,
                    "from_state": from_state,
                    "to_state": to_state,
                    "probability": matrix[from_state - 1, to_state - 1],
                })
    return pd.DataFrame(rows)



def test_transition_column_sums_matches_hand_computed_expectation(synthetic_transitions):
    expected = synthetic_transitions[
        (synthetic_transitions["subject_id"] == 1) & (synthetic_transitions["to_state"] == 3)
    ]["probability"].sum()

    result = transition_column_sums(synthetic_transitions)

    actual = result[(result["subject_id"] == 1) & (result["state"] == 3)]["transition_into_sum"].iloc[0]
    assert actual == pytest.approx(expected)


def test_fit_state_metric_model_cluster_ols_runs_and_returns_diagnostics(synthetic_state_level):
    result, diagnostics = fit_state_metric_model_cluster_ols(synthetic_state_level, metric="FO")

    assert result.params is not None
    assert result.cov_type == "cluster"
    assert "normality" in diagnostics
    assert "vif" in diagnostics


def test_simple_slopes_gives_one_coefficient_per_state(synthetic_state_level):
    result, _ = fit_state_metric_model_simple_slopes(synthetic_state_level, metric="FO")
    terms = state_slope_terms(N_STATES)

    assert len(terms) == N_STATES
    for term in terms:
        assert term in result.params.index, f"missing per-state slope: {term}"


def test_simple_slopes_reparameterisation_preserves_the_fit(synthetic_state_level):
    # Reparameterising must not change the model, only the basis: identical
    # fitted values and residuals. If this ever fails, the two functions are
    # fitting genuinely different models and their results aren't comparable.
    treatment, _ = fit_state_metric_model_cluster_ols(synthetic_state_level, metric="FO")
    simple, _ = fit_state_metric_model_simple_slopes(synthetic_state_level, metric="FO")

    assert np.allclose(treatment.fittedvalues, simple.fittedvalues, atol=1e-8)
    assert treatment.df_model == simple.df_model


def test_simple_slope_equals_reference_slope_plus_its_contrast(synthetic_state_level):
    # State k's simple slope should equal (reference slope + difference term)
    # from the treatment-coded fit. This is the identity that makes the
    # earlier misreading concrete: the treatment-coded interaction alone is
    # NOT state k's slope.
    treatment, _ = fit_state_metric_model_cluster_ols(synthetic_state_level, metric="FO")
    simple, _ = fit_state_metric_model_simple_slopes(synthetic_state_level, metric="FO")

    reference_slope = treatment.params["center(WASI_T)"]
    for k in range(2, N_STATES + 1):
        expected = reference_slope + treatment.params[f"C(state)[T.{k}]:center(WASI_T)"]
        actual = simple.params[f"C(state)[{k}]:center(WASI_T)"]
        assert actual == pytest.approx(expected, abs=1e-8)


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
    assert not dropped_row["significant"]


def test_correct_pvalues_all_nan_raises():
    with pytest.raises(ValueError):
        correct_pvalues({"a": float("nan"), "b": float("nan")})


def test_sensitivity_report_shows_both_correction_families():
    # The per-metric family is deliberately reported alongside the primary
    # one; if this ever silently drops to a single family, the paper would be
    # claiming a robustness check it no longer performs.
    from analysis.run_final_correction import sensitivity_report

    results = {
        "FO": {f"C(state)[{k}]:center(WASI_T)": p
               for k, p in zip(range(1, 8), [0.004, 0.33, 0.017, 0.006, 0.16, 0.004, 0.012])},
        "lifetime": {f"C(state)[{k}]:center(WASI_T)": 0.5 for k in range(1, 8)},
        "entropy_rate": {"center(WASI_T)": 0.02, "center(age)": 0.17},
    }
    report = sensitivity_report(results)

    assert "across all" in report and "within metric" in report
    # a narrower family can only ever promote terms, never demote them
    lines = {l.split()[0]: l for l in report.splitlines()[1:]}
    assert "FO" in lines
    assert "lifetime" in lines and lines["lifetime"].count("none") == 2
