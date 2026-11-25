# -*- coding: utf-8 -*-
"""
Synthetic robustness stress test for the three CNN-based arms (CPU-only).

WHY: the paper's robustness claim currently rests on one natural outlier
(patient 15). This turns that N=1 observation into a CONTROLLED experiment:
corrupt every held-out volume with graded synthetic artifacts and measure how
each approach degrades, over all 20 patients, with a paired significance test.

DESIGN (the important part): corruption is injected at the ACQUISITION level --
on the raw DICOM pixels -- and then the FULL pipeline is re-run on the corrupted
image:

    raw DICOM --[perturb]--> body mask -> N4 -> NL-means -> normalize
              -> (recompute) NMF ROI / NMF channel maps -> U-Net inference -> Dice

This is the realistic test. N4 and NL-means exist precisely to remove bias fields
and noise, so injecting those AFTER preprocessing (as a naive version would) tests
a scenario that could never occur in deployment. Here the unsupervised NMF stage
and the preprocessing both face the same corruption as the network.

The trained fold models are REUSED (no retraining); only inference is run.

Perturbations (each a single severity, 0 = clean):
  * noise  : Rician noise (MRI magnitude noise), std = severity * signal level.
  * bias   : smooth multiplicative bias field, amplitude = severity.
  * motion : k-space phase-encode line corruption, fraction of lines = severity.

Metrics per approach, as a function of severity:
  * mean volumetric Dice over the 20 patients,
  * worst-case (minimum) Dice  <-- the robustness quantity,
  * paired Wilcoxon of each hybrid vs CNN-only (does NMF help, over N=20?).

Outputs (comparison/results/):
  robustness_stress.csv   long-form per-patient scores (written incrementally)
  robustness_stress.md    mean(worst-case) tables + paired Wilcoxon tables
  figs/robustness_curves.png   degradation curves (mean + worst-case)

HONESTY NOTES
  * A plausible outcome is that the shared preprocessing already absorbs the noise
    and bias (all arms robust, no arm difference) and only motion -- which survives
    N4/NL-means -- separates the arms, or that nothing separates them. Either result
    is reported faithfully; it is still far stronger than the current N=1 claim.
  * NMF-only is excluded on purpose: the claim under review is scoped to the
    CNN-based approaches, and perturbing the classical pipeline would require
    re-fitting its classifier/gate per fold.
  * Inference is deterministic; the only randomness is the perturbation draw, seeded
    per (patient, perturbation, severity), so the whole run is reproducible.
  * NMF recompute uses use_cache=False, so the .roi_cache / .nmf_maps_cache on disk
    are never read or written here -- the clean caches are untouched.

    python comparison/robustness_stress.py                      # full run
    python comparison/robustness_stress.py --quick              # fast subset
    python comparison/robustness_stress.py --perturbations motion --patients 1 15
    python comparison/robustness_stress.py --from-csv comparison/results/robustness_stress.csv
"""
import os
import sys
import csv
import time
import argparse

# torch first: avoids an intermittent c10.dll init failure (WinError 1114) when the
# scientific stack is imported before torch under memory pressure on Windows.
import torch  # noqa: F401,E402

import warnings

import numpy as np
from scipy.stats import wilcoxon
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)  # light NMF (max_iter=200)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_data as cd            # noqa: E402
import cnn_core                     # noqa: E402
import nmf_roi                      # noqa: E402
import nmf_channels                 # noqa: E402
import liver_nmf_segmentation as L  # noqa: E402

CFG = {**cnn_core.DEFAULT_CFG}                       # input_size=128, threshold=0.5
K = L.K

# approach -> (model-file prefix, U-Net input channels)
ARMS = {
    "CNN-only":           ("cnn",     1),
    "NMF+CNN (ROI)":      ("nmf_cnn", 1),
    "NMF+CNN (channels)": ("nmf_ch",  1 + K),
}
ARM_ORDER = ["CNN-only", "NMF+CNN (ROI)", "NMF+CNN (channels)"]
HYBRIDS = ["NMF+CNN (ROI)", "NMF+CNN (channels)"]
SHORT = {"CNN-only": "CNN", "NMF+CNN (ROI)": "ROI", "NMF+CNN (channels)": "ch"}

# published clean full-volume Dice (Table 1) -- used only as a harness sanity check
PUBLISHED_CLEAN = {"CNN-only": 0.855, "NMF+CNN (ROI)": 0.862, "NMF+CNN (channels)": 0.868}

DEFAULT_SEVERITIES = {
    "noise":  [0.02, 0.05, 0.10, 0.15],
    "bias":   [0.10, 0.20, 0.30, 0.40],
    "motion": [0.10, 0.20, 0.30, 0.40],
}
PT_ID = {"noise": 1, "bias": 2, "motion": 3}


# --------------------------------------------------------------------------
#  Acquisition-level perturbations  (operate on the RAW slice, non-negative)
# --------------------------------------------------------------------------
def _rician_noise(raw, sev, rng):
    """Rician magnitude noise: |(S + n1) + i n2|, n ~ N(0, std). std = sev*signal."""
    body = raw > 0
    ref = float(np.percentile(raw[body], 95)) if body.any() else (float(raw.max()) or 1.0)
    std = sev * ref
    n1 = rng.normal(0.0, std, raw.shape)
    n2 = rng.normal(0.0, std, raw.shape)
    return np.sqrt((raw + n1) ** 2 + n2 ** 2).astype(np.float32)


def _bias_field(raw, sev, rng):
    """Smooth low-frequency multiplicative field, mean ~1, amplitude = sev."""
    H, W = raw.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32) / float(max(H, W))
    fx, fy = rng.uniform(0.3, 0.9, 2)
    px, py = rng.uniform(0.0, 1.0, 2)
    field = np.sin(2 * np.pi * (xx * fx + px)) * np.cos(2 * np.pi * (yy * fy + py))
    return (raw * (1.0 + sev * field)).astype(np.float32)


def _motion(raw, sev, rng):
    """k-space phase-encode line corruption -> ghosting. sev = fraction of rows."""
    H, W = raw.shape
    F = np.fft.fftshift(np.fft.fft2(raw))
    n = int(round(sev * H))
    if n > 0:
        rows = rng.choice(H, size=n, replace=False)
        F[rows, :] *= np.exp(1j * rng.uniform(-np.pi, np.pi, size=n))[:, None]
    return np.abs(np.fft.ifft2(np.fft.ifftshift(F))).astype(np.float32)


PERTURB = {"noise": _rician_noise, "bias": _bias_field, "motion": _motion}


def rng_for(seed, patient, ptype, sev):
    """Deterministic, reproducible RNG per (patient, perturbation, severity)."""
    return np.random.default_rng([int(seed), int(patient), PT_ID[ptype],
                                  int(round(sev * 1000))])


# --------------------------------------------------------------------------
#  Build corrupted-and-preprocessed slices for one patient at one severity
# --------------------------------------------------------------------------
def corrupt_patient(clean_slices, vol, liver, ptype, sev, rng):
    """Perturb each slice's RAW image, re-run full preprocessing, keep clean GT.

    clean_slices : the patient's cached slices (gives the slice indices + GT set)
    vol, liver   : raw volume (N,H,W) and liver GT (N,H,W) from load_t2_volume
    """
    fn = PERTURB[ptype]
    out = []
    for s in clean_slices:
        si = s["slice"]
        raw_pert = np.clip(fn(vol[si], sev, rng), 0.0, None)
        norm, body = L.preprocess_incl(raw_pert)          # body mask re-derived here
        out.append(dict(slice=si, img=norm, gt=liver[si].astype(bool),
                        body=body, shape=tuple(norm.shape)))
    return out


# --------------------------------------------------------------------------
#  Scoring one patient volume with the three CNN arms
# --------------------------------------------------------------------------
def score_patient(models, slices, patient):
    """Return {arm: full_vol_dice}. NMF ROI / channels recomputed from `slices`."""
    res = {}
    preds, gts = cnn_core.predict_patient(models["CNN-only"], slices, None, CFG)
    res["CNN-only"] = cd.volumetric_dice(preds, gts)[0]

    bbox = nmf_roi.roi_for_patient(slices, patient=patient, use_cache=False)
    preds, gts = cnn_core.predict_patient(models["NMF+CNN (ROI)"], slices, bbox, CFG)
    res["NMF+CNN (ROI)"] = cd.volumetric_dice(preds, gts)[0]

    chans = nmf_channels.channels_for_patient(slices, patient=patient, k=K, use_cache=False)
    preds, gts = cnn_core.predict_patient(models["NMF+CNN (channels)"], slices, None,
                                          CFG, extras=chans)
    res["NMF+CNN (channels)"] = cd.volumetric_dice(preds, gts)[0]
    return res


def load_fold_models(fold):
    models = {}
    for arm, (prefix, in_ch) in ARMS.items():
        m = cnn_core.UNet(features=tuple(CFG["features"]), in_ch=in_ch)
        path = os.path.join(cd.MODELS_DIR, f"{prefix}_fold{fold}.pt")
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        models[arm] = m
    return models


# --------------------------------------------------------------------------
#  Run
# --------------------------------------------------------------------------
def run(ptypes, severities, patients, seed, csv_path):
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print("Loading cached clean slices ...", flush=True)
    clean = cd.load_slices(cd.ALL20)
    folds = cd.make_folds()
    fold_of = {p: i for i, f in enumerate(folds) for p in f}

    by_fold = {}
    for p in patients:
        by_fold.setdefault(fold_of[p], []).append(p)

    rows = []
    fh = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    writer.writerow(["perturbation", "severity", "approach", "patient", "dice_full"])

    def emit(pt, sev, res, p):
        for arm in ARM_ORDER:
            rows.append((pt, float(sev), arm, str(p), res[arm]))
            writer.writerow([pt, sev, arm, p, f"{res[arm]:.4f}"])
        fh.flush()

    t_start = time.time()
    for fold, fps in sorted(by_fold.items()):
        print(f"\n=== fold {fold}: patients {fps} ===", flush=True)
        models = load_fold_models(fold)
        for p in fps:
            # clean baseline from the cached slices (exact published preprocessing)
            cres = score_patient(models, clean[p], p)
            for pt in ptypes:                      # anchor every curve at severity 0
                emit(pt, 0.0, cres, p)
            print(f"  [clean]       patient {p:>2}: " +
                  "  ".join(f"{SHORT[a]}={cres[a]:.3f}" for a in ARM_ORDER), flush=True)

            raw = L.load_t2_volume(L.DATASET_PATH, p)
            vol, liver = raw["volume"], raw["liver_mask"]
            for pt in ptypes:
                for sev in severities[pt]:
                    t = time.time()
                    pert = corrupt_patient(clean[p], vol, liver, pt, sev,
                                           rng_for(seed, p, pt, sev))
                    res = score_patient(models, pert, p)
                    emit(pt, sev, res, p)
                    print(f"  [{pt}@{sev:<4}] patient {p:>2}: " +
                          "  ".join(f"{SHORT[a]}={res[a]:.3f}" for a in ARM_ORDER) +
                          f"   ({time.time()-t:.0f}s)", flush=True)
    fh.close()
    print(f"\nTotal compute: {(time.time()-t_start)/60:.1f} min")
    print(f"wrote {csv_path}")
    return rows


# --------------------------------------------------------------------------
#  Aggregation / outputs
# --------------------------------------------------------------------------
def _series(rows, pt, arm):
    bysev = {}
    for p_, s_, a_, _pat, d_ in rows:
        if p_ == pt and a_ == arm:
            bysev.setdefault(s_, []).append(d_)
    return bysev


def _paired(rows, pt, sev, arm_a, arm_b):
    """Per-patient paired (a, b) Dice at one severity -> (median diff a-b, p)."""
    da = {pat: d for p_, s_, a_, pat, d in rows if p_ == pt and s_ == sev and a_ == arm_a}
    db = {pat: d for p_, s_, a_, pat, d in rows if p_ == pt and s_ == sev and a_ == arm_b}
    pats = sorted(set(da) & set(db))
    a = np.array([da[x] for x in pats]); b = np.array([db[x] for x in pats])
    med = float(np.median(a - b))
    if len(a) < 1 or np.allclose(a, b):
        return med, float("nan")
    try:
        return med, float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        return med, float("nan")


def summarize(rows, out_dir, make_fig=True):
    ptypes = sorted({r[0] for r in rows}, key=lambda x: PT_ID.get(x, 99))
    severities = {pt: sorted({r[1] for r in rows if r[0] == pt}) for pt in ptypes}

    lines = ["# Synthetic robustness stress test\n",
             "Three CNN-based approaches; corruption injected on the raw image and "
             "the full pipeline (preprocessing + NMF + U-Net) re-run; cached fold "
             "models, no retraining. Cells are **mean Dice (worst-case Dice)** over "
             "the evaluated patients.\n"]

    # harness check: clean cohort mean vs the published Table 1 means (they match
    # only when all 20 patients are run; a subset legitimately differs).
    lines += ["\n## Harness check (clean baseline)\n",
              "| Approach | clean mean, this run | published (Table 1) |",
              "|---|---|---|"]
    for arm in ARM_ORDER:
        perpat = {pat: d for p_, s_, a_, pat, d in rows if a_ == arm and s_ == 0.0}
        cm = float(np.mean(list(perpat.values()))) if perpat else float("nan")
        lines.append(f"| {arm} | {cm:.3f} (n={len(perpat)}) | {PUBLISHED_CLEAN[arm]:.3f} |")

    for pt in ptypes:
        sevs = severities[pt]
        lines += [f"\n## {pt}\n",
                  "| Approach | " + " | ".join(f"sev {s:g}" for s in sevs) + " |",
                  "|" + "---|" * (len(sevs) + 1)]
        for arm in ARM_ORDER:
            bysev = _series(rows, pt, arm)
            cells = [f"{np.mean(bysev[s]):.3f} ({np.min(bysev[s]):.3f})"
                     if s in bysev else "-" for s in sevs]
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        # paired significance vs CNN-only (robustness test over N patients)
        lines += [f"\n_Paired Wilcoxon vs CNN-only (median Dice diff, p) at each "
                  f"severity > 0 — {pt}:_\n",
                  "| Comparison | " + " | ".join(f"sev {s:g}" for s in sevs if s > 0) + " |",
                  "|" + "---|" * (sum(1 for s in sevs if s > 0) + 1)]
        for hy in HYBRIDS:
            cells = []
            for s in sevs:
                if s <= 0:
                    continue
                med, pv = _paired(rows, pt, s, hy, "CNN-only")
                cells.append(f"{med:+.3f}, p={pv:.3f}" if pv == pv else f"{med:+.3f}, p=n/a")
            lines.append(f"| {hy} vs CNN-only | " + " | ".join(cells) + " |")

    md_path = os.path.join(out_dir, "robustness_stress.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {md_path}")

    if not make_fig:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"CNN-only": "#d1495b", "NMF+CNN (ROI)": "#2e8540",
              "NMF+CNN (channels)": "#1b6ca8"}
    # per-perturbation x-axis label (each severity axis means something different)
    xlabel = {"noise": "Rician noise σ (fraction of signal)",
              "bias": "bias-field amplitude",
              "motion": "fraction of k-space lines corrupted"}
    figdir = os.path.join(out_dir, "figs")
    os.makedirs(figdir, exist_ok=True)
    n = len(ptypes)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 7.4), squeeze=False)
    for col, pt in enumerate(ptypes):
        sevs = severities[pt]
        for arm in ARM_ORDER:
            bysev = _series(rows, pt, arm)
            xs = [s for s in sevs if s in bysev]
            mean = [float(np.mean(bysev[s])) for s in xs]
            worst = [float(np.min(bysev[s])) for s in xs]
            axes[0][col].plot(xs, mean, "-o", color=colors[arm], label=arm, lw=2, ms=4)
            axes[1][col].plot(xs, worst, "-o", color=colors[arm], label=arm, lw=2, ms=4)
        axes[0][col].set_title(f"{pt} — mean Dice")
        axes[1][col].set_title(f"{pt} — worst-case Dice")
        for r in (0, 1):
            axes[r][col].set_xlabel(xlabel.get(pt, "severity")); axes[r][col].set_ylim(0, 1)
            axes[r][col].grid(alpha=0.3)
    axes[0][0].set_ylabel("mean volumetric Dice")
    axes[1][0].set_ylabel("worst-case volumetric Dice")
    axes[0][0].legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig_path = os.path.join(figdir, "robustness_curves.png")
    fig.savefig(fig_path, dpi=300)
    print(f"wrote {fig_path}")


def _load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["perturbation"], float(r["severity"]), r["approach"],
                         r["patient"], float(r["dice_full"])))
    return rows


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perturbations", default="noise,bias,motion")
    ap.add_argument("--patients", nargs="*", default=None,
                    help="subset of patient ids (default: all 20)")
    ap.add_argument("--quick", action="store_true",
                    help="fast subset: 4 patients, 2 severities per perturbation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--from-csv", default=None,
                    help="skip compute; rebuild .md/.png from an existing results CSV")
    a = ap.parse_args()

    if a.from_csv:
        rows = _load_csv(a.from_csv)
        summarize(rows, cd.RESULTS_DIR, make_fig=not a.no_fig)
        return

    ptypes = [p.strip() for p in a.perturbations.split(",") if p.strip()]
    severities = {pt: list(DEFAULT_SEVERITIES[pt]) for pt in ptypes}
    patients = [str(p) for p in a.patients] if a.patients else list(cd.ALL20)
    if a.quick:
        severities = {pt: severities[pt][1::2] for pt in ptypes}
        if not a.patients:
            patients = list(dict.fromkeys(list(cd.make_folds()[0])[:3] + ["15"]))
    patients = [p for p in patients if p in cd.ALL20]

    rows = run(ptypes, severities, patients, a.seed,
               os.path.join(cd.RESULTS_DIR, "robustness_stress.csv"))
    summarize(rows, cd.RESULTS_DIR, make_fig=not a.no_fig)


if __name__ == "__main__":
    main()
