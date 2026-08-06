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
    print("BY STATE - each state's own cognition slope (p_adjusted)")
    print("=" * 74)
    print(f"{'State':<7}" + "".join(f"{m:<18}" for m in METRIC_ORDER))
    for state in range(1, 8):
        row = f"{state:<7}"
        for metric in METRIC_ORDER:
            term = f"{metric}__C(state)[{state}]:center(WASI_T)"
            match = corrected[corrected["term"] == term]
            if match.empty:
                row += f"{'-':<18}"
            else:
                marker = "*" if match["significant"].iloc[0] else " "
                row += f"{match['p_adjusted'].iloc[0]:.4f}{marker}".ljust(18)
        print(row)
    print("\n* = significant after correction")

    print()
    print("Subject-level effects:")
    for _, r in corrected[~corrected["term"].str.contains(r"C\(state\)")].iterrows():
        marker = "*" if r["significant"] else " "
        print(f"  {marker} {r['term']:<45} p_raw={r['p_raw']:.5f}  p_adj={r['p_adjusted']:.5f}")

    print()
    print("=" * 74)
    print("SENSITIVITY - correcting within each metric instead of across all")
    print("=" * 74)
    print(sensitivity_report(results, method=args.method, alpha=args.alpha))


def sensitivity_report(results: dict, method: str = "fdr_bh", alpha: float = 0.05) -> str:
    """Compare the primary correction family against a per-metric one.

    The primary analysis corrects across every reported term at once. A
    reasonable alternative is to correct within each metric separately, on the
    grounds that each answers a distinct question. This reports both, because
    switching to the narrower family *after* seeing that it promotes two
    borderline states is exactly the move the original submission was
    criticised for - showing both makes the choice visible rather than
    load-bearing.
    """
    everything = {f"{m}__{t}": p for m, terms in results.items() for t, p in terms.items()}
    primary = correct_pvalues(everything, method=method, alpha=alpha)
    primary_hits = set(primary.loc[primary["significant"], "term"])

    lines = [f"{'Metric':<18}{'across all ' + str(len(everything)):<22}{'within metric'}"]
    for metric in METRIC_ORDER + ["entropy_rate"]:
        if metric not in results:
            continue
        within = correct_pvalues(results[metric], method=method, alpha=alpha)
        within_hits = set(within.loc[within["significant"], "term"])

        def states(terms, prefix=""):
            out = sorted(
                int(t.split("[")[1].split("]")[0])
                for t in terms if t.startswith(prefix) and "C(state)[" in t
            )
            return str(out) if out else "none"

        if metric == "entropy_rate":
            a = "sig" if f"{metric}__center(WASI_T)" in primary_hits else "ns"
            b = "sig" if "center(WASI_T)" in within_hits else "ns"
            lines.append(f"{metric:<18}{'cognition: ' + a:<22}cognition: {b}")
        else:
            lines.append(
                f"{metric:<18}{states(primary_hits, metric + '__'):<22}{states(within_hits)}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
