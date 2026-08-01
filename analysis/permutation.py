"""Permutation-based inference: sidesteps the residual-normality assumption
entirely (per Reviewer #1's concern about parametric tests being unsuitable
here) rather than transforming the outcome to satisfy it, as `transforms.py`
does. Used as an independent, assumption-free check against the raw and
rank-INT parametric results in `models.py`.

Method: permute the subject-level covariate of interest (typically
cognition) across subjects, holding state/age/sex/design fixed, and refit
the same model each time to build a null distribution for each fixed-effect
coefficient. For the per-state mixed model, permuting at the subject level
(not row level) preserves each subject's own 7-state block intact, which is
required for validity under a within-subject repeated-measures design.

Limitation, stated plainly rather than glossed over: this is a simple
covariate-permutation test, not a full Freedman-Lane permutation of
nuisance-adjusted residuals. It assumes subjects are exchangeable with
respect to the *other* covariates in the model (age, sex) - if cognition is
itself correlated with age in this sample, that correlation is preserved
under the permutation for age's own coefficient, but the cognition
permutation still slightly disturbs that joint structure. This is the
standard, commonly-used approach in this literature; a full Freedman-Lane
implementation would be a further refinement if reviewers push back on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.models import (
    fit_state_metric_model,
    fit_state_metric_model_cluster_ols,
    fit_state_metric_model_simple_slopes,
    fit_subject_scalar_model,
)


def _permute_subject_covariate(
    df: pd.DataFrame, covariate_col: str, subject_col: str, rng: np.random.Generator
) -> pd.DataFrame:
    """Return a copy of `df` with `covariate_col` shuffled across subjects.

    Each subject's own rows all get the same (shuffled) value, since the
    covariate is a subject-level attribute repeated across that subject's
    state rows.
    """
    subject_values = df[[subject_col, covariate_col]].drop_duplicates(subject_col).set_index(subject_col)
    shuffled_index = rng.permutation(subject_values.index.to_numpy())
    shuffled_values = pd.Series(subject_values[covariate_col].to_numpy(), index=shuffled_index)

    permuted = df.copy()
    permuted[covariate_col] = permuted[subject_col].map(shuffled_values)
    return permuted


def _two_sided_perm_p(observed: float, null_values: np.ndarray) -> float:
    n = len(null_values)
    exceed = np.sum(np.abs(null_values) >= np.abs(observed))
    return (exceed + 1) / (n + 1)


def permutation_test_state_metric_model(
    df: pd.DataFrame,
    metric: str,
    terms: list[str],
    cognition_col: str = "WASI_T",
    state_col: str = "state",
    subject_col: str = "subject_id",
    n_permutations: int = 1000,
    seed: int = 0,
    use_cluster_ols: bool = False,
    simple_slopes: bool = False,
) -> pd.DataFrame:
    """Permutation p-values for `terms` (coefficient names as they appear in
    the fitted model's `.params`, e.g. `C(state)[T.3]:center(WASI_T)`) from
    `fit_state_metric_model`, permuting `cognition_col` across subjects.

    `use_cluster_ols=True` swaps the per-permutation refit from MixedLM to
    `fit_state_metric_model_cluster_ols` - ~200x faster (0.03s vs 6.8s per
    fit on the real data), which is what makes 5000+ permutations practical
    at all. Two reasons this is a fair swap rather than a shortcut:

    1. The null distribution here is built from *coefficients*, not standard
       errors or p-values, so the robust-SE machinery that distinguishes the
       two functions has no effect on the permutation p-value whatsoever.
       Only the coefficient estimator matters.
    2. MixedLM's coefficients differ from OLS's only through the GLS
       weighting implied by a non-zero random-effect variance - and that
       variance collapses to ~0 in every model in this project, so the two
       estimators coincide in practice. Verified empirically before use
       (see the supplement's high-resolution permutation section).

    Do not enable this blindly on a dataset where the random-effect variance
    is genuinely non-zero: there the two estimators would legitimately
    differ, and the MixedLM path is the right one.

    `simple_slopes=True` uses `fit_state_metric_model_simple_slopes`, whose
    coefficients are each state's own cognition slope rather than its
    difference from the reference state. Pass the term names from
    `analysis.models.state_slope_terms` with it. This is what the paper's
    per-state claims need; see that function's docstring.
    """
    rng = np.random.default_rng(seed)
    if simple_slopes:
        fit_fn = fit_state_metric_model_simple_slopes
    elif use_cluster_ols:
        fit_fn = fit_state_metric_model_cluster_ols
    else:
        fit_fn = fit_state_metric_model

    observed_result, _ = fit_fn(df, metric=metric, cognition_col=cognition_col, state_col=state_col, subject_col=subject_col)
    observed = {term: observed_result.params[term] for term in terms}

    null_values = {term: np.empty(n_permutations) for term in terms}
    for i in range(n_permutations):
        permuted_df = _permute_subject_covariate(df, cognition_col, subject_col, rng)
        result, _ = fit_fn(permuted_df, metric=metric, cognition_col=cognition_col, state_col=state_col, subject_col=subject_col)
        for term in terms:
            null_values[term][i] = result.params.get(term, np.nan)

    rows = []
    for term in terms:
        valid_null = null_values[term][~np.isnan(null_values[term])]
        rows.append({
            "term": term,
            "observed_coef": observed[term],
            "n_valid_permutations": len(valid_null),
            "perm_p": _two_sided_perm_p(observed[term], valid_null),
        })
    return pd.DataFrame(rows)


def permutation_test_subject_scalar_model(
    df: pd.DataFrame,
    outcome: str,
    terms: list[str],
    cognition_col: str = "WASI_T",
    subject_col: str = "subject_id",
    n_permutations: int = 2000,
    seed: int = 0,
    extra_terms: str = "",
) -> pd.DataFrame:
    """Permutation p-values for `terms` from `fit_subject_scalar_model`,
    permuting `cognition_col` across subjects (one row per subject, so this
    is a standard full permutation - no repeated-measures subtlety).
    """
    rng = np.random.default_rng(seed)

    observed_result, _ = fit_subject_scalar_model(df, outcome=outcome, cognition_col=cognition_col, extra_terms=extra_terms)
    observed = {term: observed_result.params[term] for term in terms}

    null_values = {term: np.empty(n_permutations) for term in terms}
    for i in range(n_permutations):
        permuted_df = _permute_subject_covariate(df, cognition_col, subject_col, rng)
        result, _ = fit_subject_scalar_model(permuted_df, outcome=outcome, cognition_col=cognition_col, extra_terms=extra_terms)
        for term in terms:
            null_values[term][i] = result.params.get(term, np.nan)

    rows = []
    for term in terms:
        valid_null = null_values[term][~np.isnan(null_values[term])]
        rows.append({
            "term": term,
            "observed_coef": observed[term],
            "n_valid_permutations": len(valid_null),
            "perm_p": _two_sided_perm_p(observed[term], valid_null),
        })
    return pd.DataFrame(rows)
