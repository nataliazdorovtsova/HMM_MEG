# HMM_MEG

Analysis code for *"The entropy of resting-state neural dynamics is a marker of general
cognitive ability in childhood"* (Zdorovtsova et al.) — Hidden Markov Modelling of
resting-state MEG in 46 children aged 8–13, related to cognitive ability and behaviour.

The repository is split by language, because the two halves have different dependencies and
run at different stages:

| Directory | What it holds | Needs |
|---|---|---|
| **`analysis/`** | **The statistical analysis and figures.** Python. This is where the reported results come from. | Python 3.10+ (`requirements.txt`) |
| `matlab/` | MEG preprocessing, HMM inference, and the export that feeds `analysis/` | MATLAB, SPM12, OSL, HMM-MAR |
| `supplement/` | Methods supplement, figures, and the raw permutation p-values | — |
| `manuscript/` | Manuscript working copy and revision notes | — |

---

## analysis/ — statistics and figures

Everything downstream of the HMM. Install and run the tests with:

```bash
pip install -r requirements.txt
pytest
```

The analysis reads three tidy CSVs (see [the data contract](#the-data-contract)) and runs in
three steps:

```bash
# 1. permutation tests for every reported term (10,000 permutations)
python -m analysis.run_permutations --export-dir <export dir> \
    --output supplement/permutation_pvalues_10000.json

# 2. one multiplicity correction across the whole family of reported tests
python -m analysis.run_final_correction

# 3. regenerate every figure in supplement/figures/
python -m analysis.make_figures --export-dir <export dir> --output-dir supplement/figures
```

### Modules

| Module | Purpose |
|---|---|
| `io.py` | Load and validate the three exported CSVs |
| `models.py` | Model specifications. **`fit_state_metric_model_simple_slopes` is the one the paper reports** — it gives each state its own cognition slope rather than a contrast against a reference state |
| `permutation.py` | Permutation inference. Each term is tested against a null built by permuting a variable that term involves |
| `correction.py` | One BH-FDR pass across every reported term |
| `transforms.py` | Rank-based inverse-normal and log transforms (trialled; not used for the reported results) |
| `entropy.py` | Entropy rate from a transition matrix — a port of the MATLAB original, used to cross-check it |
| `viz/` | Shared plotting style, paper figures, and model diagnostics |
| `make_figures.py`, `make_tracker_pdf.py` | Reproducible figure and document generation |

Statistical choices, diagnostics, the results table, and an audit of the analysis are
documented in
[`supplement/statistical_methods_supplement.md`](supplement/statistical_methods_supplement.md).

---

## matlab/ — preprocessing, HMM, export

### `matlab/preprocessing/` — MEG data preparation

Run in this order. Steps 2–4 are done in the SPM12 GUI rather than by a script here.

1. **`HMM_MaxfilterAndICA.py`** — Maxfiltering and ICA artefact removal (Python: MNE + RED Tools)
2. Conversion to `.mat`/`.dat` (SPM12)
3. Bandpass filter 1–30 Hz (SPM12)
4. Coregistration to T1 scans (SPM12)
5. **`HMM_Beamforming.m`** — LCMV beamforming to source space
6. **`HMM_ApplyParcellation_38.m`** — 38-parcel parcellation
7. **`HMM_FinishPreprocessing.m`** — detrending, standardisation, leakage correction, Hilbert
   envelope, downsampling to 250 Hz

**`HMM_BadSegments.m`** inspects timeseries for bad segments and supports participant exclusions.

### `matlab/hmm/` — model inference

- **`HMM_Compute.m`** — HMM inference (HMM-MAR). Models were fitted for K = 4 to 14; K = 7 is
  reported, chosen for anatomically interpretable states
- **`HMM_Spectra.m`** — spectral estimation and decomposition across states
- **`HMM_Visualise.m`** — state maps in HCP Workbench

### `matlab/HMM_ExportForPython.m` — the handoff

Run after HMM inference. Produces the three CSVs the Python analysis consumes, and derives
per-subject lifetimes and intervals from the Viterbi path.

### `matlab/legacy/`

The 2023 analysis scripts, superseded by `analysis/` and kept only for provenance. They are
**not** part of the current results; their multiple-comparison handling is the main thing the
reanalysis set out to fix.

---

## The data contract

`matlab/HMM_ExportForPython.m` writes three tidy files, the only interface between the MATLAB
and Python halves:

| File | Shape |
|---|---|
| `subject_level.csv` | One row per subject: demographics, cognition, behaviour, switching rate, entropy rate, max FO |
| `state_level.csv` | One row per subject × state: fractional occupancy, mean lifetime, mean interval |
| `transition_matrices.csv` | One row per subject × from_state × to_state: transition probability |

These hold per-participant data and are **not** in this repository. Point the analysis at them
with `--export-dir`.

---

## Dependencies

**Python** — see `requirements.txt` (pandas, numpy, scipy, statsmodels, matplotlib, seaborn).

**MATLAB** —
[SPM12](https://www.fil.ion.ucl.ac.uk/spm/),
[OSL](https://github.com/OHBA-analysis/osl/) and its
[HMM-MAR toolbox](https://github.com/OHBA-analysis/HMM-MAR),
and [HCP Workbench](https://humanconnectome.org/software/connectome-workbench) for visualisation.

---

## Contact

Natalia Zdorovtsova — natalia.zdorovtsova@mrc-cbu.cam.ac.uk

A previous paper using related methods: Zdorovtsova et al. (2023),
[Cortex](https://doi.org/10.1016/j.cortex.2023.04.001).
