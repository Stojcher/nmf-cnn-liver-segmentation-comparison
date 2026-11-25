# Liver Segmentation on CHAOS T2-SPIR MRI — NMF vs CNN vs NMF+CNN

A controlled comparison of **four approaches** to liver segmentation, all sharing the
same preprocessing, the same 5-fold patient-level cross-validation, and the same
volumetric Dice metric:

1. **NMF-only** — NMF features + a light gradient-boosting pixel classifier + a
   liver-presence gate + intensity-gating boundary refinement.
2. **CNN-only** — a 2D U-Net trained on the ground-truth masks (no NMF).
3. **NMF+CNN (ROI cascade)** — NMF localizes a coarse ROI; the same U-Net segments inside it.
4. **NMF+CNN (channel fusion)** — the same U-Net, with the NMF component maps added
   as extra input channels.

Because the three CNN arms share one U-Net and training procedure, each fusion variant
differs from CNN-only by exactly **one controlled factor**, so any difference is
attributable to that factor.

## Results (5-fold CV, all 20 labeled patients, 40-epoch models)

| Approach | Full-volume Dice | Liver-only Dice | median | min | # ≥ 0.70 |
|----------|------------------|-----------------|--------|-----|----------|
| NMF-only | 0.743 ± 0.128 | 0.777 ± 0.123 | 0.780 | 0.282 | 17/20 |
| CNN-only | 0.855 ± 0.179 | 0.866 ± 0.181 | 0.894 | 0.084 | 19/20 |
| NMF+CNN (ROI) | 0.862 ± 0.170 | 0.874 ± 0.169 | 0.908 | 0.153 | 19/20 |
| NMF+CNN (channels) | **0.868 ± 0.139** | 0.874 ± 0.139 | 0.908 | **0.278** | 19/20 |

All CNN arms beat NMF-only (paired Wilcoxon p ≤ 0.0006). Among the CNN arms no NMF
variant gives a significant **mean** gain (channels vs CNN-only p = 0.55); NMF's durable
contribution is **robustness** — lowest variance and a far better worst case on the
domain-shift outlier (patient 15: 0.278 vs 0.084). Dice is the volumetric (3D)
per-patient Dice; ± is the population standard deviation over the 20 patients.

## Revision analyses (peer review, round 1)

A second set of analyses was added during revision — to strengthen the robustness
claim beyond the single outlier and to broaden the evaluation beyond Dice. Each is a
standalone, deterministic, CPU-only script that **reuses the cached fold models /
feature cache (no retraining)** and writes a Markdown table under `comparison/results/`.

| Script | Adds | Output | Reviewer point |
|---|---|---|---|
| `comparison/extra_metrics.py` | IoU, sensitivity, precision, HD95 & ASSD (3D, mm) for all 4 arms; re-verifies the Table 1 Dice | `results/extra_metrics.{md,json}` | more metrics |
| `comparison/effect_sizes.py` | Cohen's *d<sub>z</sub>* + 95% BCa bootstrap CIs for the paired arm differences (n=20, and n=19 without the outlier) | `results/effect_size_table.md` | small-N uncertainty |
| `comparison/robustness_stress.py` | synthetic stress test: Rician noise / bias field / k-space motion injected on the **raw** image, full pipeline re-run, over all 20 patients | `results/robustness_stress.{csv,md}`, `figs/robustness_curves.png` | N=1 robustness |
| `comparison/make_robustness_table.py` | motion table (paper Table 4) from the stress-test CSV | `results/table4_motion.md` | — |
| `comparison/k_ablation.py` | NMF component-count ablation (NMF-only CV for k ∈ {3,4,5,6}) | `results/k_ablation.{md,json}`, `figs/k_ablation.png` | k=5 justification |
| `comparison/channel_stability.py` | channel-ordering stability: which intensity-sorted channel the liver occupies | `results/channel_stability.md`, `figs/channel_stability.png` | sorting is ad-hoc |
| `comparison/cost_memory.py` | peak RAM (an isolated subprocess per arm) + preprocessing-vs-NMF timing split (`--breakdown`) | `results/cost_memory.md`, `results/cost_timing_breakdown.md` | cost evidence |
| `comparison/make_cost_table.py` | consolidated cost + memory table from the three cost outputs | `results/cost_table_final.md` | — |

```bash
python comparison/extra_metrics.py            # extra metrics (also re-verifies Dice reproduces)
python comparison/effect_sizes.py             # effect sizes + bootstrap CIs
python comparison/robustness_stress.py        # stress test (reuses cached models)
python comparison/make_robustness_table.py    # -> Table 4 (motion); --perturbation noise|bias
python comparison/k_ablation.py --ks 3,4,5,6  # k ablation (rebuilds the feat cache for each NEW k)
python comparison/channel_stability.py        # channel-ordering stability
python comparison/cost_memory.py              # peak RAM + timing (spawns isolated workers)
python comparison/cost_memory.py --breakdown  # preprocessing vs NMF split
python comparison/make_cost_table.py          # -> consolidated cost table
```

### Additional metrics (5-fold CV, 20 patients; mean — see `results/extra_metrics.md` for ±std)

| Approach | Dice | IoU | Sensitivity | Precision | HD95 (mm) | ASSD (mm) |
|---|---|---|---|---|---|---|
| NMF-only | 0.743 | 0.604 | 0.831 | 0.692 | 51.77 | 13.52 |
| CNN-only | 0.855 | 0.773 | 0.851 | 0.888 | 47.35 | 8.41 |
| NMF+CNN (ROI) | 0.862 | 0.784 | 0.846 | 0.919 | 31.72 | 6.82 |
| NMF+CNN (channels) | 0.868 | 0.785 | 0.857 | 0.906 | 23.61 | 5.36 |

The boundary metrics (HD95, ASSD) separate the CNN arms more clearly than Dice and
favor channel fusion; the low NMF-only precision (0.692) reflects its tendency to
over-segment.

### Robustness — synthetic motion stress (mean (worst-case) Dice over 20 patients)

| Approach | Clean | 10% | 20% | 30% | 40% |
|---|---|---|---|---|---|
| CNN-only | 0.855 (0.084) | 0.758 (0.030) | 0.647 (0.051) | 0.542 (0.046) | 0.423 (0.021) |
| NMF+CNN (ROI) | 0.862 (0.153) | 0.745 (0.019) | 0.578 (0.000) | 0.360 (0.000) | 0.187 (0.000) |
| NMF+CNN (channels) | 0.868 (**0.278**) | 0.719 (**0.257**) | 0.630 (**0.346**) | 0.540 (**0.162**) | 0.455 (**0.134**) |

Shared preprocessing (N4 + non-local-means) absorbs the noise and bias-field
corruption for every arm; **motion** — which survives preprocessing — is where the
arms separate. Channel fusion holds a far higher **worst-case** Dice than CNN-only at
every severity, while the ROI cascade degrades fastest. This reproduces, now over all
20 patients, the worst-case robustness first seen on the single outlier (patient 15).

**Other checks.** *Effect sizes:* the differences among the CNN arms are small
(|*d<sub>z</sub>*| ≤ 0.27, 95% CIs spanning 0; ≤ 0.15 with the outlier removed), while
NMF-only vs the CNN arms is a large effect (*d<sub>z</sub>* 1.15–1.81) — so the null
among the CNN arms is a genuinely small effect, not merely an undetected one.
*k-ablation:* full-volume Dice varies only 0.037 across k ∈ {3,4,5,6} (highest at
k=4), so the a-priori k=5 is a safe choice. *Channel stability:* no patient has the
liver pinned to one sorted channel in >80% of its slices — the intensity sort gives a
consistent channel *structure*, not a fixed anatomical assignment. *Cost:* the extra
NMF costs ~9 s/volume (one-time, cached) against 70–80 min of U-Net training, and peak
RAM stays ≤ 2.1 GB — negligible on the CPU-only setup (see `results/cost_table_final.md`).

## Requirements

- **Python 3.11** (CPU-only; no GPU required).
- Packages:

```bash
pip install numpy scipy scikit-learn scikit-image SimpleITK pydicom Pillow matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

| Package | Used for |
|---|---|
| numpy, scipy | arrays, morphology, Wilcoxon test |
| scikit-learn | NMF, HistGradientBoosting, RandomForest |
| scikit-image | filters, morphology, denoising, resize |
| SimpleITK | N4 bias-field correction |
| pydicom, Pillow | reading DICOM images and PNG ground-truth masks |
| torch | the 2D U-Net (CNN arms) |
| matplotlib | figures |

## Dataset

Uses the **CHAOS** challenge dataset (T2-SPIR MRI). The data is **not included in this
repository** (CHAOS has its own access terms) — download it from the
[CHAOS challenge](https://chaos.grand-challenge.org/) (data DOI
[10.5281/zenodo.3431873](https://doi.org/10.5281/zenodo.3431873)) and arrange the MR
volumes as:

```
Train_Sets/MR/<patient>/T2SPIR/DICOM_anon/*.dcm    # images
Train_Sets/MR/<patient>/T2SPIR/Ground/*.png        # liver masks (label 63)
```

Ground truth is provided only for `Train_Sets` (CHAOS withholds `Test_Sets` labels), so
all quantitative evaluation is 5-fold cross-validation on the 20 labeled `Train_Sets`
patients. Set the dataset path in `liver_nmf_segmentation.py` (`DATASET_PATH`).

## Quick start

```bash
# --- the four approaches (each writes comparison/results/<arm>.json) ---
python comparison/arm_nmf.py         # 1. NMF-only        (recomputes its CV, ~minutes)
python comparison/arm_cnn.py         # 2. CNN-only        )
python comparison/arm_nmf_cnn.py     # 3. NMF+CNN (ROI)   ) reload cached models -> re-score ~1-2 min
python comparison/arm_nmf_ch.py      # 4. NMF+CNN (chan.) )

# --- results & figures (read the already-saved JSONs) ---
python comparison/evaluate.py                  # table.md + paired Wilcoxon + summary charts
python comparison/save_final_figures.py        # all publication figures -> "comparison/Results final/"
python comparison/visualize_arms.py 15 10 22   # per-patient qualitative overlays (all approaches)
```

You usually **don't** need to re-run the arm scripts — their results are cached. For the
numbers/figures as you write, `evaluate.py` and `save_final_figures.py` are enough.

**Viewing more slices.** By default the overlays show a few *area-weighted representative
liver slices* per patient. To inspect **every liver slice** of a patient, pass a large
`--slices`:

```bash
python comparison/visualize_arms.py 15 --slices 100 --no-show
```

If the number exceeds the patient's liver-slice count, all of its liver slices are shown
(all four approaches side by side). Note that only **liver-containing** slices are ever
plotted — liver-free slices are still segmented and scored (they count toward the
full-volume Dice) but are not displayed.

## Script reference

| Script | Purpose | Arguments |
|---|---|---|
| `comparison/arm_nmf.py` | NMF-only CV → `results/nmf.json` | none |
| `comparison/arm_cnn.py` | CNN-only CV → `results/cnn.json` | `--fresh` `--epochs N` `--folds 0,1` `--smoke` `--input-size N` |
| `comparison/arm_nmf_cnn.py` | ROI cascade → `results/nmf_cnn.json` | same as arm_cnn |
| `comparison/arm_nmf_ch.py` | Channel fusion → `results/nmf_ch.json` | same as arm_cnn |
| `comparison/evaluate.py` | Aggregate 4 JSONs → `results/table.md`, `per_patient.csv`, `figs/` | none |
| `comparison/save_final_figures.py` | High-DPI figures → `Results final/` | optional demo patient ids (default `22 10 15`) |
| `comparison/visualize_arms.py` | Per-patient overlays (all arms) → `results/figs/` | patient ids; `--slices N`; `--no-show` |
| `liver_nmf_segmentation.py` | Standalone NMF pipeline | `cv` (run 5-fold CV) or none (segment+plot held-out) |

**CNN-arm flags:** the three CNN arms **cache their trained per-fold models** in
`comparison/models/`. Re-running without `--fresh` reloads them and only re-scores
(~1–2 min). Use `--fresh` to retrain from scratch, `--epochs N` to set the budget,
`--folds` to run a subset, `--smoke` for a quick 1-fold sanity run (writes to
`*_smoke` names so it can't overwrite the real results).

> To reproduce the headline 40-epoch results with a retrain you **must** pass
> `--fresh --epochs 40` (the default epoch count is 25). A full retrain of one CNN
> arm is ~1–2 h on CPU.

## Results file format

Each approach writes one JSON to `comparison/results/<arm>.json` — these four files are
the **canonical results**; the tables and figures are all derived from them. Each has
three keys:

| Key | Content |
|---|---|
| `arm` | approach name (`nmf`, `cnn`, `nmf_cnn`, `nmf_ch`) |
| `cfg` | the configuration used — for the CNN arms this includes `epochs: 40`, `threshold: 0.5`, `features`, `lr`, `batch_size`, … (how you confirm it's the 40-epoch setup) |
| `per_patient` | for each of the 20 patients, its held-out **volumetric Dice** |

Each `per_patient` entry holds two numbers:

- **`full_vol`** — Dice over the whole volume (all slices; penalizes false positives on
  non-liver slices) → this is the headline "full-volume Dice."
- **`liver_vol`** — Dice over liver-containing slices only.

```json
{ "arm": "nmf",
  "cfg": { "method": "NMF+HGB+gate+refine" },
  "per_patient": {
    "1":  { "full_vol": 0.8011, "liver_vol": 0.8198 },
    "15": { "full_vol": 0.2822, "liver_vol": 0.2901 }
  } }
```

`evaluate.py` reads all four and produces `table.md` (mean ± std, median, min, #≥0.70),
the paired Wilcoxon comparisons, `per_patient.csv`, and the figures — so you can
regenerate every reported number from these JSONs without retraining. (Patients appear in
cross-validation fold order, not numeric order; `evaluate.py` re-sorts them.)

## Caches (do not delete)

All three are deterministic functions of the raw data and are shared across arms:

| Cache | Built by | Content |
|---|---|---|
| `.feat_cache/` | `liver_nmf_segmentation` | per-patient preprocessed features (preprocess + NMF), ~28 min one-time |
| `comparison/.roi_cache/` | `nmf_roi.py` | per-patient ROI bounding boxes (ROI cascade) |
| `comparison/.nmf_maps_cache/` | `nmf_channels.py` | per-patient NMF component maps (channel fusion) |

Because every arm reads `.feat_cache/`, they all get **identical preprocessing**.

## Reproducibility

- **Inference is deterministic**: re-scoring the cached CNN models reproduces the stored
  `results/*.json` exactly, and the NMF-only pipeline (fixed `random_state`) reproduces
  its numbers exactly.
- **Training is not bit-identical**: the CNN data augmentation and CPU float ops are not
  fully deterministic, so a fresh retrain lands close but not identical — the
  cross-validation ± std captures this spread. The cached models are one valid draw.
- Folds are fixed (`make_folds`, seed 0) and identical across all four arms.

## Running from VS Code

The ▶ "Run Python File" button / F5 passes **no arguments**. That's fine for
`arm_nmf.py`, `evaluate.py`, `save_final_figures.py`, and `visualize_arms.py` (defaults).
To pass flags (`--fresh`, `--epochs 40`, custom patients), use the **integrated terminal**
instead. Select the Python interpreter that has the packages above.

## Not included in this repository

To keep the repo small, `.gitignore` excludes large or regenerable artifacts. After
cloning you will obtain or regenerate them:

| Excluded | How to get it |
|---|---|
| `Train_Sets/`, `Test_Sets/` | Download from the CHAOS challenge (see **Dataset**) |
| `.feat_cache/`, `comparison/.roi_cache/`, `comparison/.nmf_maps_cache/` | Built automatically on first run (deterministic; the feature cache is a ~28-min one-time step) |
| `comparison/models/` | Trained U-Net fold models — regenerate with `--fresh --epochs 40` (~1–2 h/arm on CPU) |
| `Archive/` | Development history (step1..step23), old experiments, the 25-epoch snapshot — not needed to run or reproduce the results |

The committed `comparison/results/*.json` already hold the reported 40-epoch numbers, so
`evaluate.py` reproduces the results table without any retraining.

## Project layout

```
liver_nmf_segmentation.py   ← the standalone NMF pipeline (approach 1's implementation)
comparison/                 ← the 4-approach framework (RUN THINGS FROM HERE)
  arm_nmf.py  arm_cnn.py  arm_nmf_cnn.py  arm_nmf_ch.py   one runner per approach
  common_data.py            shared data layer (cache -> slices, folds, volumetric Dice)
  cnn_core.py               shared U-Net + training/inference (used by arms 2-4)
  nmf_roi.py                unsupervised NMF ROI localizer (arm 3)
  nmf_channels.py           NMF component maps as CNN channels (arm 4)
  evaluate.py               aggregate arms -> table.md, stats, figures
  save_final_figures.py     publication figures -> "Results final/"
  visualize_arms.py         per-patient qualitative overlays
  # revision analyses (peer-review round 1) — reuse caches, no retraining:
  extra_metrics.py          IoU / sensitivity / precision / HD95 / ASSD (all arms)
  effect_sizes.py           Cohen's d_z + BCa bootstrap CIs
  robustness_stress.py      synthetic noise / bias / motion stress test
  make_robustness_table.py  stress CSV -> Table 4 (motion)
  k_ablation.py             NMF component-count ablation (k in {3,4,5,6})
  channel_stability.py      NMF channel-ordering stability
  cost_memory.py            peak RAM + preprocessing/NMF timing split
  make_cost_table.py        -> consolidated cost + memory table
  results/                  per-arm JSON + table.md + per_patient.csv + figs
                            + revision tables (extra_metrics, effect_size_table,
                            robustness_stress, table4_motion, k_ablation,
                            channel_stability, cost_*)
  models/                   cached fold models              [gitignored — regenerate]
  Results final/            final high-DPI PNGs
  .roi_cache/ .nmf_maps_cache/   deterministic input caches [gitignored — auto-built]
.feat_cache/                ← precomputed per-patient features [gitignored — auto-built]
Train_Sets/ , Test_Sets/    ← CHAOS dataset                  [gitignored — see Dataset]
Archive/                    ← full development history (step1..step23), old experiments,
                              the 25-epoch snapshot           [gitignored — not needed]
```

Items marked `[gitignored]` are **not** in the repository (see **Not included in this
repository** above).

## How it works (one paragraph)

Each T2-SPIR slice is preprocessed (body mask → N4 bias correction → non-local-means
denoising → intensity normalization) once and cached. The **NMF-only** arm builds
per-pixel features (intensity, multi-scale Gaussian, texture, position) plus NMF
cluster-context features, classifies pixels with gradient boosting, suppresses
false-positive slices with a random-forest presence gate, and trims leakage with an
intensity-gating refinement. The **CNN-only** arm trains a compact 2D U-Net on the same
preprocessed slices against the ground-truth masks. The two **NMF+CNN** arms reuse that
exact U-Net, differing only in how NMF is injected — as a coarse ROI the network refines
(cascade), or as extra NMF component-map input channels (channel fusion). All four are
evaluated identically with 5-fold patient-level cross-validation and volumetric Dice.

## Decision thresholds and tuning

The NMF-only pipeline and the CNN arms threshold the outputs of **different** models, so
their two probability thresholds are set independently and are not comparable:

- **NMF-only → 0.7** on the gradient-boosting pixel classifier's probability, forming the
  candidate mask (`PROB_THR` in `liver_nmf_segmentation.py`). This is a **fixed** operating
  point chosen once during development as a robust value — deliberately more conservative
  than 0.5, because the classifier is trained with class balancing that pushes probabilities
  upward. It is **not** re-tuned per fold.
- **CNN arms → 0.5** on the U-Net's sigmoid output (`cfg["threshold"]` in `cnn_core.py`),
  the conventional operating point for a sigmoid trained with a balanced Dice+BCE loss.

Only two hyperparameters are **grid-searched per fold**, on training data only: the
NMF-only presence-gate threshold (maximizing per-slice mean Dice) and the refinement
strength *k* (maximizing volumetric Dice). Everything else — `PROB_THR = 0.7`, the U-Net
0.5 threshold, `k = 5` NMF components, and the training configuration — is fixed a priori
and identical across folds.
