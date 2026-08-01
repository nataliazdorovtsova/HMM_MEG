"""Mixed-effects model specifications replacing the per-state/per-metric GLMs.

Design, per the agreed redesign:

- Per-state metrics (FO, lifetime, interval) are modelled with ONE mixed
  model each, `state` as a within-subject categorical fixed effect and
  `subject_id` as a random intercept, instead of one separate GLM per
  state. `age` and `cognition` are both included as an interaction term
  (effects of interest, not covariates-only), alongside `sex`.
- Subject-level scalars (switching rate, entropy rate) are modelled with
  ordinary least squares, since there is no repeated-measures structure
  to account for. Given switching rate and entropy rate correlate at
  r ~ 0.998 (see HMM_Entropy.m), only one of them should be treated as
  the primary outcome per family in `correction.py` — see the plan notes
  in the repo root about which is primary vs. robustness-check.

Every fit function returns the fitted statsmodels result *and* a
diagnostics dict (residual normality, VIF), so diagnostics are never an
afterthought bolted on after the fact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor


def _vif(df: pd.DataFrame, formula_rhs: str) -> pd.DataFrame:
    """Variance inflation factors for the fixed-effect design matrix of `formula_rhs`."""
    design = dmatrix(formula_rhs, df, return_type="dataframe")
    # drop the intercept column for VIF, which is not meaningful for it
    design = design.drop(columns=[c for c in design.columns if c == "Intercept"])
    vifs = {
        col: variance_inflation_factor(design.values, i)
        for i, col in enumerate(design.columns)
    }
    return pd.DataFrame({"term": list(vifs.keys()), "VIF": list(vifs.values())})


def _residual_normality(residuals: np.ndarray) -> dict:
    stat, p = shapiro(residuals)
    return {"shapiro_W": float(stat), "shapiro_p": float(p), "normal_at_0.05": bool(p > 0.05)}


def fit_state_metric_model(
    df: pd.DataFrame,
    metric: str,
    cognition_col: str = "WASI_T",
    state_col: str = "state",
    subject_col: str = "subject_id",
):
    """Fit `metric ~ C(state) * cognition + age * cognition + C(sex)` with subject random intercept.

    `df` must be the long-format subject x state table produced by
    `analysis.io.merged_state_level`. `age` and `cognition` are mean-centered
    (via patsy's `center()`) before entering their interaction terms - on
    real data, leaving them uncentered produced VIFs in the hundreds purely
    from the structural correlation between a raw variable and its own
    interaction term, not from any real collinearity between age and
    cognition themselves.
    """
    fixed_rhs = (
        f"C({state_col}) * center({cognition_col}) + center(age) * center({cognition_col}) + C(sex)"
    )
    formula = f"{metric} ~ {fixed_rhs}"
    model = smf.mixedlm(formula, df, groups=df[subject_col])
    result = model.fit(reml=True)

    diagnostics = {
        "normality": _residual_normality(result.resid.values),
        "vif": _vif(df, fixed_rhs),
    }
    return result, diagnostics


def fit_state_metric_model_cluster_ols(
    df: pd.DataFrame,
    metric: str,
    cognition_col: str = "WASI_T",
    state_col: str = "state",
    subject_col: str = "subject_id",
):
    """Same formula as `fit_state_metric_model`, but OLS with cluster-robust
    (subject-clustered) standard errors instead of a mixed model's random
    intercept.

    Motivation: across every mixed model fit in this project (FO, lifetime,
    interval, transition-into-state), the random-effect variance ("Group
    Var") has repeatedly come out at or near 0, and the transition-into-state
    model failed to converge on the real data. A near-zero variance component
    is a sign the mixed model isn't buying anything a subject-clustered OLS
    wouldn't already give: cluster-robust SEs still account for each
    subject's 7 state-level rows being correlated, without trying to estimate
    a variance component the data doesn't support - and OLS is numerically
    far more stable than MixedLM's iterative optimizer. Use this as a
    robustness check against the mixed-model result, not a blind replacement
    for it.
    """
    fixed_rhs = (
        f"C({state_col}) * center({cognition_col}) + center(age) * center({cognition_col}) + C(sex)"
    )
    formula = f"{metric} ~ {fixed_rhs}"
    result = smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": df[subject_col]})

    diagnostics = {
        "normality": _residual_normality(result.resid.values),
        "vif": _vif(df, fixed_rhs),
    }
    return result, diagnostics


def fit_state_metric_model_simple_slopes(
    df: pd.DataFrame,
    metric: str,
    cognition_col: str = "WASI_T",
    state_col: str = "state",
    subject_col: str = "subject_id",
):
    """Per-state cognition slopes, each estimated in its own right.

    `fit_state_metric_model` uses treatment coding, so its
    `C(state)[T.k]:cognition` terms are the DIFFERENCE between state k's
    cognition slope and the reference state's, and the bare `cognition`
    term is the reference state's slope. That is the right model, but the
    wrong parameterisation for the question this paper asks, which is "does
    occupancy of state k relate to cognitive ability?" - not "does state k
    differ from state 1?".

    On the real data the distinction changed conclusions rather than just
    wording: mean interval showed a significant difference-from-state-1 for
    every state, but only state 1 itself had a non-zero slope; and states 2
    and 5 looked significant for FO as differences while having no effect
    of their own.

    Dropping the standalone cognition main effect and keeping only
    `C(state):cognition` makes patsy emit one coefficient per state, each
    being that state's own slope. The model fit is identical (same design
    space, same residuals) - only the basis changes, so this is a
    reparameterisation rather than a different model. Age is kept as
    `center(age) + center(age):cognition` written out explicitly, because
    the `*` shorthand would reintroduce the cognition main effect and
    collapse the per-state terms back into contrasts.
    """
    fixed_rhs = (
        f"C({state_col}) + C({state_col}):center({cognition_col})"
        f" + center(age) + center(age):center({cognition_col}) + C(sex)"
    )
    formula = f"{metric} ~ {fixed_rhs}"
    result = smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": df[subject_col]})

    diagnostics = {"normality": _residual_normality(result.resid.values)}
    return result, diagnostics


def state_slope_terms(n_states: int = 7, cognition_col: str = "WASI_T", state_col: str = "state") -> list[str]:
    """Coefficient names for the per-state slopes from
    `fit_state_metric_model_simple_slopes`, in state order."""
    return [f"C({state_col})[{k}]:center({cognition_col})" for k in range(1, n_states + 1)]


def fit_subject_scalar_model(
    df: pd.DataFrame,
    outcome: str,
    cognition_col: str = "WASI_T",
    extra_terms: str = "",
):
    """Fit `outcome ~ cognition * age + C(sex) [+ extra_terms]` via OLS.

    `extra_terms` lets behavioural covariates (e.g. `+ SDQ_total`) be added
    without duplicating the base formula for every behavioural outcome.
    `cognition` and `age` are mean-centered before their interaction term -
    see `fit_state_metric_model` for why.
    """
    fixed_rhs = f"center({cognition_col}) * center(age) + C(sex){extra_terms}"
    formula = f"{outcome} ~ {fixed_rhs}"
    result = smf.ols(formula, df).fit()

    diagnostics = {
        "normality": _residual_normality(result.resid.values),
        "vif": _vif(df, fixed_rhs),
    }
    return result, diagnostics


def fit_transition_matrix_model(transitions_long: pd.DataFrame, subject_level: pd.DataFrame,
                                  cognition_col: str = "WASI_T"):
    """Single, planned model for whether cognition relates to the probability of
    transitioning INTO each state, replacing the 49-cell-then-7-column-sum
    sequence in HMM_Entropy.m (49 cell-wise correlations found nothing, so the
    7 column-sum correlations were tried as an unplanned second attempt - the
    exact "moved the goalposts after the first test failed" problem Reviewer #3
    flagged, concern 5).

    An earlier version of this function fit the full K x K x cognition
    three-way interaction (49 cells) as fixed effects. On the real data (46
    subjects) that raised `LinAlgError: Singular matrix` - over-parameterized,
    not just slow to converge. Rather than patch around that, this collapses
    each subject's transition matrix to its column sums (total probability of
    transitioning into each state, summed over the state transitioned from)
    BEFORE fitting - which is what the column-sum analysis was actually asking
    in the first place. That makes it structurally identical to
    `fit_state_metric_model` (7 states x 46 subjects).

    Uses `fit_state_metric_model_cluster_ols` rather than the MixedLM version:
    on the real data the mixed model ran but failed to converge, and its
    random-effect variance collapsed to ~0 (the same pattern seen in the
    FO/lifetime/interval mixed models) - cluster-robust OLS gives the same
    subject-level correlation adjustment without the convergence problem.

    `transitions_long` must have columns subject_id, from_state, to_state,
    probability (as produced by analysis.io.load_transition_matrices).
    """
    column_sums = transition_column_sums(transitions_long)
    df = column_sums.merge(subject_level, on="subject_id", how="left")
    return fit_state_metric_model_cluster_ols(df, metric="transition_into_sum", cognition_col=cognition_col)


def transition_column_sums(transitions_long: pd.DataFrame) -> pd.DataFrame:
    """Per subject, total probability of transitioning INTO each state
    (summed over the state transitioned from). Returned in the same
    subject_id/state long-format shape `fit_state_metric_model` expects.
    """
    return (
        transitions_long.groupby(["subject_id", "to_state"])["probability"]
        .sum()
        .reset_index()
        .rename(columns={"to_state": "state", "probability": "transition_into_sum"})
    )
