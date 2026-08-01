"""Multiplicity correction applied once, across the full family of tests
actually reported in the paper - replacing the per-script `multicmp` calls
that only corrected within their own sub-family (e.g. 7 FO tests corrected
against each other, then switching rate corrected on its own, etc.).

Usage: collect every fixed-effect term of interest from every fitted model
into one dict of {label: p_value}, then call `correct_pvalues` once on the
whole set before reporting any of them as significant.
"""

from __future__ import annotations

import math
import warnings

import pandas as pd
from statsmodels.stats.multitest import multipletests


def correct_pvalues(pvalues: dict[str, float], method: str = "fdr_bh", alpha: float = 0.05) -> pd.DataFrame:
    """Apply one multiplicity correction across every test label/p-value pair given.

    `method` defaults to Benjamini-Hochberg FDR ('fdr_bh'); pass 'holm' for
    the more conservative family-wise error rate control if preferred.

    Terms with a NaN p-value (e.g. a mixed model's random-effect "Group Var"
    row, which is a variance component, not a hypothesis test with a normal
    Wald p-value) are excluded from the correction rather than silently
    poisoning it - `statsmodels.stats.multitest.multipletests` returns all-NaN
    output if even one input p-value is NaN. Dropped terms are still returned
    in the result (with `p_adjusted` NaN) so they don't just disappear from
    the report, and a warning names them.
    """
    if not pvalues:
        raise ValueError("pvalues is empty - nothing to correct")

    valid = {label: p for label, p in pvalues.items() if p is not None and not math.isnan(p)}
    dropped = [label for label in pvalues if label not in valid]

    if dropped:
        warnings.warn(
            f"Excluding {len(dropped)} term(s) with a NaN p-value from multiplicity "
            f"correction (not real hypothesis tests, e.g. a variance component): {dropped}",
            stacklevel=2,
        )
    if not valid:
        raise ValueError("All p-values were NaN - nothing to correct")

    labels = list(valid.keys())
    raw_p = [valid[label] for label in labels]
    reject, p_adj, _, _ = multipletests(raw_p, alpha=alpha, method=method)

    result = pd.DataFrame({
        "term": labels,
        "p_raw": raw_p,
        "p_adjusted": p_adj,
        "significant": reject,
    })

    if dropped:
        result = pd.concat([
            result,
            pd.DataFrame({"term": dropped, "p_raw": float("nan"), "p_adjusted": float("nan"), "significant": False}),
        ], ignore_index=True)

    return result.sort_values("p_adjusted").reset_index(drop=True)
