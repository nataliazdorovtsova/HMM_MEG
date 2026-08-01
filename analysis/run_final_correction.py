"""Apply ONE multiplicity correction across the entire family of reported tests.

    python -m analysis.run_final_correction \
        --pvalues supplement/permutation_pvalues_10000.json

This is the step that answers the reviewers' central methodological
complaint: in the original submission each metric was corrected only within
its own small sub-family (7 FO tests against each other, switching rate on
its own, and so on), so nothing was ever corrected across the full set of
tests the paper actually reported.

Switching rate and max FO are deliberately absent from the input file:
switching rate correlates with entropy rate at r = 0.998 and is reported as
a robustness check rather than an independent hypothesis, and max FO is a
supplementary summary. Including either would spend correction budget on a
test that is not an independent claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.correction import correct_pvalues

METRIC_ORDER = ["FO", "interval", "transition_into", "lifetime"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pvalues", default="supplement/permutation_pvalues_10000.json")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--method", default="fdr_bh")
    args = parser.parse_args()

    results = json.loads(Path(args.pvalues).read_text())
    all_pvalues = {
        f"{metric}__{term}": p
        for metric, terms in results.items()
        for term, p in terms.items()
    }

    print(f"Tests in the family: {len(all_pvalues)}  (method={args.method}, alpha={args.alpha})")
    corrected = correct_pvalues(all_pvalues, method=args.method, alpha=args.alpha)
    print()
    print(corrected.to_string(index=False))

    print()
    print("=" * 74)
    print("BY STATE - state x cognition interaction terms (p_adjusted)")
    print("=" * 74)
    print(f"{'State':<7}" + "".join(f"{m:<18}" for m in METRIC_ORDER))
    for state in range(2, 8):
        row = f"{state:<7}"
        for metric in METRIC_ORDER:
            match = corrected[corrected["term"] == f"{metric}__C(state)[T.{state}]:center(WASI_T)"]
            if match.empty:
                row += f"{'-':<18}"
            else:
                marker = "*" if match["significant"].iloc[0] else " "
                row += f"{match['p_adjusted'].iloc[0]:.4f}{marker}".ljust(18)
        print(row)
    print("\n* = significant after correction")

    print()
    print("Main / subject-level effects:")
    for _, r in corrected[~corrected["term"].str.contains(r"C\(state\)")].iterrows():
        marker = "*" if r["significant"] else " "
        print(f"  {marker} {r['term']:<45} p_raw={r['p_raw']:.5f}  p_adj={r['p_adjusted']:.5f}")


if __name__ == "__main__":
    main()
