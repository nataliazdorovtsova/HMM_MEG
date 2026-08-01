# Supplementary Methods: Statistical Redesign

Living document tracking the statistical redesign done for resubmission, and the evidence
behind each decision - written so the choices are justifiable to reviewers, not just to us.
Numbers here come from the real exported dataset (46 subjects, K=7 HMM states), generated via
`HMM_ExportForPython.m` and analysed with the `analysis/` Python package in this repo.

**Status:** complete through the final comprehensive correction pass (§7). All decisions
resolved (§8). Remaining work is manuscript-side: writing up the recommended framing (§7) with
final author sign-off, plus the non-code items in the original plan (reference list gaps,
literature review additions - see repo root plan notes, not reproduced here).

## 1. Motivation

Both reviewers of the original submission converged on the same structural problem: HMM
summary metrics (fractional occupancy, switching rate, entropy rate, lifetimes, intervals,
transition probabilities) were each tested in separate per-state/per-metric GLMs, corrected
for multiplicity only within their own small sub-family, with age/sex as covariates-only
despite the developmental sample, and no diagnostics for normality or multicollinearity.
Specific comments motivating each change below are catalogued in the original reviewer
document (see repo root).

## 2. Model specification

Per-state metrics (FO, lifetime, interval) are modelled with **one mixed model per metric**,
`state` as a categorical fixed effect and `subject_id` as a random intercept, replacing the
old `for i = 1:7` loop of separate GLMs:

```
metric ~ C(state) * center(cognition) + center(age) * center(cognition) + C(sex)
```

Subject-level scalars (entropy rate, switching rate) are modelled by OLS:

```
outcome ~ center(cognition) * center(age) + C(sex)
```

`age` and `cognition` are **mean-centered** before entering interaction terms (via patsy's
`center()`). This is not cosmetic: on the real data, leaving them uncentered produced
Variance Inflation Factors of 66-347 (see §3) purely from the structural correlation between
a raw variable and its own interaction term - centering brought every VIF under 10.

Code: `analysis/models.py::fit_state_metric_model`, `fit_subject_scalar_model`.

## 3. Diagnostics

### 3.1 Multicollinearity (VIF)

| Term | VIF, uncentered | VIF, centered |
|---|---|---|
| `age` | 218.4 | 1.22 |
| `age:WASI_T` | 347.7 | 1.15 |
| `WASI_T` | 98.2 | 7.21 |
| `C(state)[T.k]` (each) | 66.1 | 1.15 |
| `C(state)[T.k]:WASI_T` (each) | 66.2 | 2.00 |
| `C(sex)[T.2]` | 2.18 | 1.88 |

All centered VIFs are comfortably under the conventional concern threshold of ~10 (max is
7.21, the `WASI_T` main effect).

### 3.2 Residual normality (Shapiro-Wilk)

| Model | Raw outcome | Rank-INT outcome (Blom's transform) |
|---|---|---|
| FO (mixed model) | W = 0.720, p = 7.2e-23 | W = 0.989, p = 0.013 |
| Entropy rate (OLS) | W = 0.817, p = 4.7e-6 | W = 0.985, p = 0.817 |

Entropy rate: rank-INT fully resolves the violation. FO: rank-INT produces a large practical
improvement (W: 0.72 -> 0.989) but remains formally below p = 0.05 at n = 322 observations,
where Shapiro-Wilk is known to be highly sensitive to small deviations. See the QQ plot
(`figures/residual_qq_comparison.png`): the residual FO (rank-INT) plot tracks the reference
line closely across nearly the entire range, with the departure concentrated in a handful of
extreme low-FO points rather than a global distributional problem. Those specific
low-FO observations are worth a manual check (possible floor effect: a state some subjects
almost never visit) before treating this as fully resolved.

![Residual QQ plots, raw vs rank-INT, FO and entropy rate](figures/residual_qq_comparison.png)

Code: `analysis/transforms.py::rank_based_inverse_normal_transform`,
`analysis/viz/diagnostics.py::residual_qq_plot`.

## 4. Three-way comparison: does the state x cognition finding replicate across methods?

The critical question, given the diagnostics above: which HMM states show a cognition
relationship, and does that answer depend on which of the three defensible analysis choices
we make?

| State x cognition term | Raw FO (parametric, FDR-corrected) | Rank-INT FO (parametric, FDR-corrected) | Permutation (raw FO, subject-level covariate permutation, 500 perms, FDR-corrected) |
|---|---|---|---|
| State 2 | not significant | not significant | not significant, p_adj = 0.062 |
| State 3 | **significant**, p_adj = 0.0017 | not significant, p_adj = 0.12 | **significant**, p_adj = 0.037 |
| State 4 | not significant | **significant**, p_adj = 0.0016 | **significant**, p_adj = 0.024 |
| State 5 | not significant | not significant | not significant, p_adj = 0.069 |
| State 6 | not significant | **significant**, p_adj = 0.0010 | **significant**, p_adj = 0.020 |
| State 7 | not significant | **significant**, p_adj = 0.0065 | **significant**, p_adj = 0.024 |

**This is not a minor detail.** The choice between raw and rank-transformed FO changes which
states the paper's central claim is about - state 3 alone vs. states 4/6/7. Notably, states 4
and 6 are the fronto-parietal states the *original* submission's discussion centered on
(per Reviewer #1's quote of our own text: "States 4 and 6... fronto-parietal activation and
suppression... predicts increases in cognitive ability"), so the rank-INT result is more
consistent with our original narrative than the raw-FO result is. That's circumstantial
supporting evidence for rank-INT, not proof - a transform can shift which effect looks
significant simply by reweighting the tails differently, independent of which is "more true."

The permutation test (`analysis/permutation.py`) is the tie-breaker: it tests the raw,
untransformed FO model but replaces the parametric Wald p-value with an empirical null built
by permuting each subject's cognition score across subjects 500 times and refitting, which
sidesteps the normality question entirely rather than resolving it via transformation.

**Result: the permutation test recovers state 3 (found only by the raw model) AND states
4/6/7 (found only by rank-INT), all four surviving the same FDR correction** (p_adj 0.020-
0.037). States 2 and 5 come close but don't clear it (p_adj 0.062, 0.069). Rather than one
parametric approach being "right," each looks to have been underpowered for part of the true
signal in a different way - state 3's effect is apparently sensitive to the severe FO
skew that only rank-INT fixes, while states 4/6/7's effects were presumably attenuated by
that same skew in the raw model. The permutation result, being assumption-free, is the most
complete picture we have and is the recommended basis for the manuscript's reported states -
**pending final author sign-off**, since this determines the paper's central claim.

Permutation caveat (stated plainly): this is a simple covariate-permutation test, not a full
Freedman-Lane permutation of nuisance-adjusted residuals - it assumes subjects are
exchangeable with respect to age/sex as well as cognition. Treat as the standard, commonly
used approach in this literature; a full Freedman-Lane implementation would be a further
refinement if reviewers push back on this point specifically.

## 5. Lifetime/interval metrics: original method unrecoverable, replacement decided

**Decided**: raw Viterbi-path run-length encoding, no minimum-duration threshold. Implemented
in `HMM_ExportForPython.m`, already reflected in the real `state_level.csv` export.

Investigation: `Intervals_k7_3.mat`/`LifeTimes_k7_3.mat` (and their flattened
`IntervalsFull.csv`/`LifetimesFull.csv` exports) store only per-state vectors concatenated
across all subjects with no subject index - the per-subject values needed for the mixed-model
redesign are not recoverable from them, and the script that originally produced them is not
present in this repo. Two independent attempts to reproduce their values instead - (1) raw
Viterbi-path run-length encoding, (2) the real `getStateLifeTimes`/`getStateIntervalTimes`
HMM-MAR functions on the true `Gamma`, both with and without a 12-sample (~50ms) threshold -
all produced 10-100x more, much shorter events than the old export (e.g. state 4: our
count 90k-94k events, old count 732; our mean 11-20 samples, old mean 59.75). The gap is far
too large to be the stated 50ms exclusion alone, and chasing a larger threshold purely to
match an undocumented old number would mean adopting the exact kind of unmotivated parameter
Reviewer #3 already flagged (concern 2c).

Decision: rather than reverse-engineer an unknown original method, adopt the simplest fully
transparent definition - raw Viterbi-path run-length encoding, no threshold - and disclose
this reasoning directly if asked. A genuine missing value (`NaN`) is recorded when a subject
never visits a state at all, rather than a fabricated zero, which also directly answers
Reviewer #3's question about whether some subjects never visit certain states.

## 6. Extended results: lifetime, interval, transition-into-state, switching rate, max FO

### 6.1 Lifetime and interval

Same diagnostics-first approach as FO: raw residuals badly non-normal (lifetime W=0.536,
p=2.8e-28; interval W=0.426, p=9.6e-31 - worse than FO, expected for duration/waiting-time
data). Rank-INT outperformed a log-seconds transform on both (lifetime W=0.93 vs 0.79; interval
W=0.98 vs 0.91), so permutation testing (500 perms, raw seconds, same tie-breaker logic as FO)
was run directly rather than trusting either transform's parametric p-values:

| State x cognition term | Lifetime (perm_p, uncorrected) | Interval (perm_p, uncorrected) |
|---|---|---|
| State 2 | 0.076 | 0.014 |
| State 3 | 0.679 (not significant) | 0.014 |
| State 4 | **0.050** | 0.014 |
| State 5 | 0.068 | 0.016 |
| State 6 | 0.056 | 0.014 |
| State 7 | 0.056 | 0.016 |
| Main cognition effect | 0.044 | 0.014 |

Lifetime: only state 4 clearly stands out (consistent with the FO/rank-INT and FO/permutation
results), others marginal. **Interval: every state shows an almost identical effect size
(-0.146 to -0.152) and p-value** - this doesn't differentiate between states at all, which
fits the interpretation that interval time is largely just the inverse of how often a state is
visited overall (i.e., redundant with FO/occupancy) rather than adding state-specific
information - a concrete instance of the exact metric-redundancy concern Reviewer #1 raised
(general comments, and comment 6 in Results). These raw p-values still need the one
comprehensive correction pass (§7) before being called significant or not.

### 6.2 Transition-into-state (replaces the 49-cell/7-column-sum sequence)

`fit_transition_matrix_model` (see `analysis/models.py`) was rebuilt after the original full
K x K x cognition three-way interaction crashed on the real data (`LinAlgError: Singular
matrix` - over-parameterized for 46 subjects, 49 cells). It now fits column sums (total
probability of transitioning INTO each state) as the single planned test, structurally
identical to the FO/lifetime/interval models.

The mixed-model version of this ran without crashing but failed to converge. Rather than
report an unconverged result, it was refit as OLS with subject-clustered robust standard
errors (`fit_state_metric_model_cluster_ols`) - motivated by a **project-level observation**:
the random-effect ("Group Var") variance collapsed to ~0 in essentially every mixed model
fitted in this project (FO, lifetime, interval, transition-into-state), suggesting the
per-subject random intercept wasn't doing meaningful work once state was accounted for.
Cluster-robust OLS still accounts for each subject's 7 correlated state-level rows, without
estimating a variance component the data doesn't support, and is numerically far more stable.
This ran cleanly with no convergence issues.

**One structural caveat, not a bug**: because each subject's column sums across all 7 states
are constrained to total exactly 7 (transition matrix rows sum to 1), any subject-level
covariate that doesn't interact with state - here, the `age` and `sex` main effects - is
mathematically unidentifiable in this specific model (their coefficients and SEs come out as
~0 with near-zero uncertainty, e.g. 1e-16). **Do not interpret those two terms as "no effect" -
they are structurally inestimable here, not tested.** Only the state main effects and
state x cognition interactions are meaningful in this model.

### 6.3 Switching rate and max FO

Confirmed on real data: switching rate and entropy rate correlate at r=0.998 (matches the
original HMM_Entropy.m note). Both switching_rate and max_FO show the same non-normal-residual
pattern as every other outcome in this project (Shapiro p=1.7e-5 and 5.0e-6). **Decided**:
switching rate is reported as a robustness/consistency check alongside entropy rate (not an
independent claim, given the near-perfect correlation), and max_FO is reported as a
supplementary summary. Neither is entered into the main multiplicity-correction family in §7 -
including a variable that isn't an independent hypothesis would just dilute the correction
budget for the outcomes that matter.

### 6.4 Figures

`analysis/viz/figures.py` was run against the real data for the first time and visually
inspected (not just executed) - this caught two real problems no test would have:

1. **Correlation heatmap**: title text overlapped the colorbar at full length. Fixed by
   wrapping the title (not shrinking the font - reviewers already complained the original
   figures' text was too small).
2. **Transition heatmap**: at this HMM's ~250Hz effective sampling rate, self-transition
   probability is ~0.99+ for every state, so the raw matrix on a 0-1 colour scale showed
   nothing but a dark diagonal - the actually informative "which state do you switch to"
   structure was invisible. Fixed by masking the diagonal before plotting and rescaling colour
   to the off-diagonal range, matching the original `HMM_Entropy.m` script's own approach
   (`Ao(find(eye(nstate))) = 0`) rather than inventing a new convention.

Also added: a log-scale option for `state_metric_distribution`, needed because interval time's
state 1 range (up to 22s) otherwise crushes every other state's variation into an unreadable
sliver near zero on a linear axis.

All figures are regenerated from the exported data by `analysis/make_figures.py`, so
`supplement/figures/` holds reproducible artifacts rather than hand-pasted one-offs:

```
python -m analysis.make_figures --export-dir <export dir> --output-dir supplement/figures
```

**Per-state distributions** (Reviewer #3 concern 2b: all three metrics shown, with individual
subject points overlaid - the original Figure 4 showed only lifetimes/intervals, and never
individual data points):

![Per-state distributions of FO, lifetime and interval](figures/state_metric_distributions.png)

**Cognition vs. fractional occupancy** for the four states surviving correction (§7). Note the
direction: state 3 is **negative** (higher cognitive ability -> less time in that state) while
states 4, 6 and 7 are all **positive**:

![Cognition vs FO for states 3, 4, 6, 7](figures/cognition_fo_scatters.png)

**Mean transition matrix**, self-transitions masked:

![Mean state transition matrix](figures/transition_matrix_heatmap.png)

**Cognitive/behavioural correlations**:

![Correlations between cognitive and behavioural scores](figures/cognitive_behavioural_correlations.png)

## 7. Final result: one correction pass across the whole family

All blockers in §8 are resolved, so the comprehensive correction has been run for real
(`analysis/correction.py`, BH-FDR, alpha=0.05) across every state x cognition term from FO,
lifetime, interval, transition-into-state, and entropy rate - 31 tests total. Switching rate and
max_FO are deliberately excluded (§6.3: robustness/supplementary, not independent claims -
including them would only dilute the correction budget for outcomes that matter).

All p-values below are from **10,000-permutation** tests (see §7.1 on why the earlier
500-permutation figures were refined). Adjusted values below ~0.031 are tied at the BH-FDR
step-up threshold, which is expected when many raw p-values cluster tightly.

**Significant after correction** (16 of 31 terms):

| State | FO | Interval | Transition-into-state | Lifetime |
|---|---|---|---|---|
| 2 | ns (0.067) | **0.030** | ns (0.161) | ns (0.106) |
| 3 | **0.037** | **0.030** | **0.030** | ns (0.700) |
| 4 | **0.030** | **0.030** | *marginal (0.051)* | ns (0.086) |
| 5 | ns (0.080) | **0.030** | ns (0.146) | ns (0.106) |
| 6 | **0.030** | **0.030** | **0.040** | ns (0.090) |
| 7 | **0.030** | **0.030** | *marginal (0.075)* | ns (0.090) |

Subject-level / main effects: entropy rate with **age** p_adj = **0.030**; entropy rate with
**cognition** p_adj = **0.040**; FO main effect of cognition **0.030**; interval main effect
**0.030**. The entropy rate **age x cognition interaction is not significant** (p_adj = 0.70) -
i.e. the cognition relationship does not itself change across this age range, which is worth
reporting rather than omitting.

### Reading the pattern: *how often*, not *how long*

The four metrics are not interchangeable, and the pattern across them is the actual finding:

- **Fractional occupancy** gives **state-specific** effects: states **3, 4, 6, 7** significant;
  states 2 and 5 not. Direction matters - state 3 is **negative** (higher cognitive ability ->
  *less* occupancy) while 4, 6, 7 are **positive** (see `figures/cognition_fo_scatters.png`).
- **Transition-into-state** is the most conservative and also state-specific: only **3 and 6**
  clear correction, with **4 (0.051) and 7 (0.075) marginal**. Same direction pattern as FO.
- **Interval** is significant for **every state**, with near-identical coefficients
  (-0.146 to -0.152). That uniformity is the tell: mean interval between visits to *any* state
  shrinks whenever a child switches states more overall, so this is very likely the global
  switching-rate/entropy effect re-expressed once per state, **not six independent
  state-specific findings**. It should be reported as such, not as the paper's broadest result.
- **Lifetime survives for no state at all** (best p_adj = 0.086).

Taken together: individual differences here are in **how often children enter these states, not
how long they stay once there**. That is a direct, evidence-based answer to Reviewer #3's
concern 5 ("higher fractional occupancies can be a consequence of more transitions into a
state… or by longer life times, which is not tested") - it is now tested, and the answer is
transitions rather than durations.

**States 3 and 6 are significant on both state-specific metrics**; states 4 and 7 are
significant on occupancy and marginal on transitions. This convergence across
independently-computed metrics under one correction spanning the entire paper is substantially
more defensible than any single analysis in isolation.

Recommended framing for the manuscript: **states 3, 4, 6 and 7** as the cognition-related
states (noting 4 and 7 rest on occupancy with marginal transition support), entropy rate's
**age** relationship as a second headline result, the *how often not how long* contrast as the
mechanistic story, and switching rate / max FO / lifetime as supporting detail. This is a
recommendation pending author sign-off.

### 7.1 Why the p-values changed from the 500-permutation run

The earlier pass used 500 permutations, which bounds resolution at 1/501 ≈ 0.002 and leaves
real Monte-Carlo noise on every estimate. All headline tests were re-run at **10,000
permutations** (feasible because of the verified cluster-OLS fast path - see
`analysis/permutation.py`). Conclusions are stable for FO, interval and lifetime; the
meaningful change is in **transition-into-state**, where states 4 and 7 moved from significant
to marginal (0.051, 0.075). The earlier figures for that metric came from a *parametric*
cluster-OLS fit rather than a permutation test, so this is a like-for-like improvement, not an
instability in the result. Reporting the 10,000-permutation values throughout.

## 8. Resolved decisions

1. **Switching rate vs. entropy rate framing**: switching rate stays a robustness/consistency
   check alongside entropy rate, not an independent claim (§6.3). Author-confirmed.
2. **Transition-into-state's non-convergence**: resolved by refitting as cluster-robust OLS
   instead of a mixed model (§6.2) - runs cleanly, no convergence issues.
3. **Low-FO outlier points** (§3.2, §6.4 investigation): identified as two distinct patterns -
   subjects 1/41/23/14 (mildly low state-3 occupancy, unremarkable otherwise) and subjects
   35/10/39/29 (single-state-dominance profile, high `max_FO`, the same signature the original
   pipeline used to exclude a different participant for corrupted data). **Author decision:
   leave as-is** - extensive manual preprocessing/QC was already done on this dataset (including
   excluding a participant for corrupted data), and no further action was requested.

## 9. Reproducibility

- Code: `analysis/` package, this repo (`HMM_MEG`)
- Data: exported via `HMM_ExportForPython.m` from `hmm_k7_clean3.mat` / `Gamma_k7_clean3.mat` /
  `Xi_k7_clean3.mat` (K=7 HMM fit, 46 subjects)
- Python environment: see `requirements.txt` / `pyproject.toml`; analyses run in a conda env
  (`hmm_analysis`, Python 3.11) on the CBU cluster
- Entropy-rate computation independently cross-validated against the original MATLAB
  `HMM_Entropy.m` output: correlation 1.000000, max abs. difference 0.000049 (consistent with
  CSV rounding, not a computational discrepancy)
