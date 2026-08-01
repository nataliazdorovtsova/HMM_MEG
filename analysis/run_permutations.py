"""Run permutation tests for every headline outcome and save the p-values.

    python -m analysis.run_permutations \
        --export-dir /path/to/HMM_MEG_exports \
        --output supplement/permutation_pvalues_10000.json \
        --n-permutations 10000

Permutation inference is used for the headline tests because residuals are
badly non-normal for every HMM metric here (see the supplement's diagnostics
section) and permutation makes no normality assumption at all.

Runtime: roughly 4 minutes per state-level metric at 10,000 permutations,
using the cluster-OLS fast path (`use_cluster_ols=True`). The MixedLM path
would take ~19 hours for the same work; the two are verified to agree to
~1e-17 on this data because the random-effect variance is zero. Do not
enable the fast path on data where that variance is genuinely non-zero.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

from analysis.io import (
    load_state_level,
    load_subject_level,
    load_transition_matrices,
    merged_state_level,
)
from analysis.models import transition_column_sums
from analysis.permutation import (
    permutation_test_state_metric_model,
    permutation_test_subject_scalar_model,
)

FS = 250  # HMM_FinishPreprocessing.m: options.downsample = 250

STATE_TERMS = [
    "center(WASI_T)",
    *[f"C(state)[T.{k}]:center(WASI_T)" for k in range(2, 8)],
]
ENTROPY_TERMS = ["center(WASI_T)", "center(age)", "center(WASI_T):center(age)"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output", default="supplement/permutation_pvalues_10000.json")
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    subject_level = load_subject_level(str(export_dir / "subject_level.csv"))
    state_level = load_state_level(str(export_dir / "state_level.csv"))
    transitions = load_transition_matrices(str(export_dir / "transition_matrices.csv"))

    merged = merged_state_level(subject_level, state_level)
    merged["lifetime_sec"] = merged["lifetime_mean"] / FS
    merged["interval_sec"] = merged["interval_mean"] / FS
    transition_df = transition_column_sums(transitions).merge(subject_level, on="subject_id", how="left")

    results: dict[str, dict[str, float]] = {}

    state_jobs = [
        ("FO", merged, "FO"),
        ("lifetime", merged, "lifetime_sec"),
        ("interval", merged, "interval_sec"),
        ("transition_into", transition_df, "transition_into_sum"),
    ]
    for label, df, metric in state_jobs:
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = permutation_test_state_metric_model(
                df, metric=metric, terms=STATE_TERMS,
                n_permutations=args.n_permutations, seed=args.seed, use_cluster_ols=True,
            )
        print(f"=== {label} ({args.n_permutations} permutations, {time.time() - t0:.0f}s) ===", flush=True)
        print(res.to_string(index=False), flush=True)
        results[label] = dict(zip(res["term"], res["perm_p"]))

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ent = permutation_test_subject_scalar_model(
            subject_level, outcome="entropy_rate", terms=ENTROPY_TERMS,
            n_permutations=args.n_permutations, seed=args.seed,
        )
    print(f"=== entropy_rate ({args.n_permutations} permutations, {time.time() - t0:.0f}s) ===", flush=True)
    print(ent.to_string(index=False), flush=True)
    results["entropy_rate"] = dict(zip(ent["term"], ent["perm_p"]))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
