"""
Generate the FINAL, presentation-quality result figures for all four approaches
and save them to a single folder:  comparison/Results final/

Produces:
  01_results_table.png          metrics table (4 approaches x Dice stats)
  02_dice_boxplot.png           per-patient full-volume Dice by approach
  03_per_patient_dice.png       per-patient grouped bars
  qualitative_<approach>.png    (x4) liver mask (cyan) vs GT (red) on demo patients

All PNGs are high-DPI (200) with large, un-clipped titles.

    python comparison/save_final_figures.py
    python comparison/save_final_figures.py 22 33 15      # custom demo patients

Qualitative panels reuse the fold-correct predictors from visualize_arms (each
patient predicted with the model it was a TEST case in; the NMF pipeline is
retrained per needed fold -> a few minutes the first time).
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_data as cd            # noqa: E402

import matplotlib                   # noqa: E402
matplotlib.use("Agg")              # save-only, high quality
import matplotlib.pyplot as plt     # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from visualize_arms import Predictor, _pick_slices, _dice  # noqa: E402

# --- output folder (created next to this script) ------------------------------
OUT_DIR = os.path.join(cd.COMPARISON_DIR, "Results final")
os.makedirs(OUT_DIR, exist_ok=True)

_PRED_CMAP = ListedColormap(["cyan"])
DPI = 200

# canonical approach order + the exact labels visualize_arms.Predictor emits
ARMS = [("nmf", "NMF-only"), ("cnn", "CNN-only"),
        ("nmf_cnn", "NMF+CNN (ROI)"), ("nmf_ch", "NMF+CNN (channels)")]
PRED_LABELS = {"nmf": "NMF-only", "cnn": "CNN-only",
               "nmf_cnn": "NMF+CNN (ROI)", "nmf_ch": "NMF+CNN (ch)"}
DEMO_PATIENTS = ["22", "10", "15"]     # strong / median / failure (domain-shift)

# larger, readable defaults for slides/print
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
    "figure.titlesize": 16, "figure.titleweight": "bold",
    "savefig.dpi": DPI, "savefig.bbox": "tight",
})


def _safe(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")


# ==========================================================================
#  QUANTITATIVE FIGURES  (read the result JSONs; no model needed)
# ==========================================================================
def _load_results():
    data = {}
    for k, _ in ARMS:
        path = os.path.join(cd.RESULTS_DIR, f"{k}.json")
        if os.path.exists(path):
            data[k] = json.load(open(path))["per_patient"]
    pats = sorted(next(iter(data.values())).keys(), key=int)
    return data, pats


def save_metrics_table(data, pats):
    rows = []
    for k, lab in ARMS:
        if k not in data:
            continue
        f = np.array([data[k][p]["full_vol"] for p in pats])
        l = np.array([data[k][p]["liver_vol"] for p in pats])
        rows.append([lab, f"{f.mean():.3f} ± {f.std():.3f}",
                     f"{np.nanmean(l):.3f} ± {np.nanstd(l):.3f}",
                     f"{np.median(f):.3f}", f"{f.min():.3f}",
                     f"{int((f >= 0.70).sum())}/{len(f)}"])
    cols = ["Approach", "Full-volume Dice", "Liver-only Dice",
            "Median", "Min", "# ≥ 0.70"]
    col_widths = [0.24, 0.19, 0.19, 0.13, 0.13, 0.12]

    fig, ax = plt.subplots(figsize=(14, 0.9 + 0.6 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, colWidths=col_widths,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1, 2.0)
    for c in range(len(cols)):                       # header styling
        cell = tbl[0, c]
        cell.set_facecolor("#31456b")
        cell.set_text_props(color="white", fontweight="bold")
    fig.suptitle("Four-approach comparison — 5-fold patient-level CV "
                 "(CHAOS T2-SPIR, 40-epoch models)", y=1.02)
    out = os.path.join(OUT_DIR, "01_results_table.png")
    fig.savefig(out); plt.close(fig)
    print(f"  saved {out}")


def save_boxplot(data, pats):
    labels = [lab for k, lab in ARMS if k in data]
    vals = [np.array([data[k][p]["full_vol"] for p in pats]) for k, _ in ARMS if k in data]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(vals, labels=labels, showmeans=True, widths=0.6)
    for i, v in enumerate(vals, 1):
        ax.scatter(np.full_like(v, i) + np.random.uniform(-0.06, 0.06, len(v)),
                   v, s=26, alpha=0.6, color="#31456b", zorder=3)
    ax.set_ylabel("Full-volume Dice", fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=12)
    fig.suptitle("Per-patient full-volume Dice by approach")
    out = os.path.join(OUT_DIR, "02_dice_boxplot.png")
    fig.savefig(out); plt.close(fig)
    print(f"  saved {out}")


def save_per_patient_bars(data, pats):
    keys = [k for k, _ in ARMS if k in data]
    labels = [lab for k, lab in ARMS if k in data]
    x = np.arange(len(pats))
    w = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(max(11, len(pats) * 0.62), 6))
    for i, (k, lab) in enumerate(zip(keys, labels)):
        ax.bar(x + i * w, [data[k][p]["full_vol"] for p in pats], width=w, label=lab)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(pats, fontsize=10)
    ax.set_xlabel("patient", fontsize=13)
    ax.set_ylabel("Full-volume Dice", fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11, ncol=len(keys), loc="lower center")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-patient full-volume Dice — all approaches")
    out = os.path.join(OUT_DIR, "03_per_patient_dice.png")
    fig.savefig(out); plt.close(fig)
    print(f"  saved {out}")


# ==========================================================================
#  QUALITATIVE FIGURES  (one per approach; needs fold-correct predictions)
# ==========================================================================
def save_qualitative(preds_by_patient, arm_key, arm_label, n_slices=4):
    rows = list(preds_by_patient.keys())
    fig, axes = plt.subplots(len(rows), n_slices,
                             figsize=(4.2 * n_slices, 4.2 * len(rows)),
                             squeeze=False)
    pred_label = PRED_LABELS[arm_key]
    for r, p in enumerate(rows):
        sl, gts, arms = preds_by_patient[p]
        masks, vol = arms[pred_label]
        picks = _pick_slices(gts, n_slices)
        for c in range(n_slices):
            ax = axes[r][c]
            if c < len(picks):
                si = picks[c]
                ax.imshow(sl[si]["img"], cmap="gray", interpolation="nearest")
                if gts[si].any():
                    ax.contour(gts[si], levels=[0.5], colors="red", linewidths=1.3)
                m = masks[si]
                if m.any():
                    ax.imshow(np.ma.masked_where(~m, m), cmap=_PRED_CMAP,
                              alpha=0.45, vmin=0, vmax=1)
                ax.set_title(f"slice {sl[si]['slice']} · Dice {_dice(m, gts[si]):.2f}",
                             fontsize=13)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"Patient {p}\nvol Dice {vol:.3f}",
                              fontsize=12, fontweight="bold")
    fig.suptitle(f"{arm_label} — liver segmentation\n"
                 f"prediction = cyan,  ground truth = red contour", y=1.0)
    fig.text(0.5, 0.006,
             "Displayed slices are an area-weighted representative sample; "
             "per-slice Dice need not equal the volumetric Dice.",
             ha="center", fontsize=9, style="italic", color="0.35")
    fig.tight_layout(rect=[0, 0.025, 1, 0.94])
    out = os.path.join(OUT_DIR, f"qualitative_{_safe(arm_label)}.png")
    fig.savefig(out); plt.close(fig)
    print(f"  saved {out}")


# ==========================================================================
def main():
    demo = sys.argv[1:] if len(sys.argv) > 1 else DEMO_PATIENTS
    demo = [p for p in demo if p in cd.ALL20]

    print(f"Output folder: {OUT_DIR}")
    print("Quantitative figures ...")
    data, pats = _load_results()
    save_metrics_table(data, pats)
    save_boxplot(data, pats)
    save_per_patient_bars(data, pats)

    print(f"Qualitative panels (demo patients {demo}) ...")
    predictor = Predictor()
    preds_by_patient = {p: predictor.predict(p) for p in demo}
    for k, lab in ARMS:
        save_qualitative(preds_by_patient, k, lab)

    print(f"\nDone. All figures in: {OUT_DIR}")


if __name__ == "__main__":
    main()
