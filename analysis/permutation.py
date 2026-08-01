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

from analysis.models import fit_state_metric_model, fit_subject_scalar_model


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
) -> pd.DataFrame:
    """Permutation p-values for `terms` (coefficient names as they appear in
    the fitted model's `.params`, e.g. `C(state)[T.3]:center(WASI_T)`) from
    `fit_state_metric_model`, permuting `cognition_col` across subjects.
    """
    rng = np.random.default_rng(seed)

    observed_result, _ = fit_state_metric_model(df, metric=metric, cognition_col=cognition_col, state_col=state_col, subject_col=subject_col)
    observed = {term: observed_result.params[term] for term in terms}

    null_values = {term: np.empty(n_permutations) for term in terms}
    for i in range(n_permutations):
        permuted_df = _permute_subject_covariate(df, cognition_col, subject_col, rng)
        result, _ = fit_state_metric_model(permuted_df, metric=metric, cognition_col=cognition_col, state_col=state_col, subject_col=subject_col)
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
