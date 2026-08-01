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
    """Single omnibus model over the full transition matrix, replacing the
    49-cell-then-7-column-sum sequence in HMM_Entropy.m.

    `transitions_long` must have columns subject_id, from_state, to_state,
    probability (as produced by analysis.io.load_transition_matrices).
    from_state and to_state are both entered as categorical fixed effects
    together with cognition, so the column-sum view becomes a planned
    follow-up decomposition of this one test rather than a second,
    independently-corrected round of testing.

    Note: with K states this fits a K x K x cognition three-way interaction
    (49 cells for K=7), which is a lot of fixed-effect parameters relative
    to ~46 subjects. If this fails to converge on the real data, the first
    thing to drop is the three-way interaction term (keep
    `C(from_state)*C(to_state) + cognition` and test cognition's main
    effect plus its two-way interaction with `to_state` only, which is
    what the column-sum follow-up actually asks).
    """
    df = transitions_long.merge(subject_level, on="subject_id", how="left")
    fixed_rhs = f"C(from_state) * C(to_state) * center({cognition_col}) + center(age) + C(sex)"
    formula = f"probability ~ {fixed_rhs}"
    model = smf.mixedlm(formula, df, groups=df["subject_id"])
    result = model.fit(reml=True)

    diagnostics = {
        "normality": _residual_normality(result.resid.values),
    }
    return result, diagnostics
