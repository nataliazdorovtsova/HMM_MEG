import numpy as np
import pandas as pd
import pytest

from analysis.permutation import permutation_test_state_metric_model, permutation_test_subject_scalar_model

N_SUBJECTS = 20
N_STATES = 7


@pytest.fixture
def synthetic_subject_level():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "subject_id": np.arange(1, N_SUBJECTS + 1),
        "age": rng.uniform(8, 13, N_SUBJECTS),
        "sex": rng.integers(1, 3, N_SUBJECTS),
        "WASI_T": rng.normal(50, 10, N_SUBJECTS),
        "entropy_rate": rng.uniform(0, 2, N_SUBJECTS),
    })


@pytest.fixture
def synthetic_state_level(synthetic_subject_level):
    rng = np.random.default_rng(1)
    rows = []
    for subject_id in synthetic_subject_level["subject_id"]:
        for state in range(1, N_STATES + 1):
            rows.append({"subject_id": subject_id, "state": state, "FO": rng.dirichlet(np.ones(N_STATES))[0]})
    return pd.DataFrame(rows).merge(synthetic_subject_level, on="subject_id", how="left")


def test_permutation_test_state_metric_model_shape_and_range(synthetic_state_level):
    terms = ["center(WASI_T)", "C(state)[T.3]:center(WASI_T)"]
    result = permutation_test_state_metric_model(
        synthetic_state_level, metric="FO", terms=terms, n_permutations=5, seed=0,
    )

    assert set(result["term"]) == set(terms)
    assert (result["n_valid_permutations"] <= 5).all()
    assert (result["n_valid_permutations"] > 0).all()
    assert result["perm_p"].between(0, 1).all()


def test_permutation_cluster_ols_path_gives_same_p_values_as_mixedlm(synthetic_state_level):
    # The cluster-OLS fast path is only legitimate because the two estimators
    # produce the same coefficients when the random-effect variance is ~0
    # (verified to ~1e-17 on the real data). Same seed => same permutations,
    # so the resulting p-values should match too. If a future change makes
    # the two paths diverge, this catches it.
    terms = ["center(WASI_T)", "C(state)[T.3]:center(WASI_T)"]
    mixed = permutation_test_state_metric_model(
        synthetic_state_level, metric="FO", terms=terms, n_permutations=10, seed=0,
    )
    ols = permutation_test_state_metric_model(
        synthetic_state_level, metric="FO", terms=terms, n_permutations=10, seed=0, use_cluster_ols=True,
    )

    merged = mixed.merge(ols, on="term", suffixes=("_mixed", "_ols"))
    assert np.allclose(merged["observed_coef_mixed"], merged["observed_coef_ols"], atol=1e-8)
    assert np.allclose(merged["perm_p_mixed"], merged["perm_p_ols"], atol=1e-12)


def test_permutation_cluster_ols_path_is_faster(synthetic_state_level):
    # Guards the actual reason the fast path exists - if MixedLM ever became
    # as fast, or the OLS path accidentally started fitting MixedLM, the
    # 5000-permutation runs would silently become 200x slower.
    import time

    terms = ["center(WASI_T)"]
    t0 = time.perf_counter()
    permutation_test_state_metric_model(
        synthetic_state_level, metric="FO", terms=terms, n_permutations=5, seed=0,
    )
    mixed_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    permutation_test_state_metric_model(
        synthetic_state_level, metric="FO", terms=terms, n_permutations=5, seed=0, use_cluster_ols=True,
    )
    ols_elapsed = time.perf_counter() - t0

    assert ols_elapsed < mixed_elapsed


def test_permutation_test_subject_scalar_model_shape_and_range(synthetic_subject_level):
    terms = ["center(WASI_T)", "center(age)"]
    result = permutation_test_subject_scalar_model(
        synthetic_subject_level, outcome="entropy_rate", terms=terms, n_permutations=20, seed=0,
    )

    assert set(result["term"]) == set(terms)
    assert (result["n_valid_permutations"] == 20).all()
    assert result["perm_p"].between(0, 1).all()


def test_permutation_p_value_never_zero():
    # The +1/+1 correction means an observed effect that's the most extreme
    # of all permutations still gets p = 1/(n+1), never exactly 0 - avoids
    # ever reporting an impossible p=0.
    from analysis.permutation import _two_sided_perm_p

    p = _two_sided_perm_p(observed=100.0, null_values=np.zeros(9))
    assert p == pytest.approx(1 / 10)
