"""
Aggregate the three arms into the comparison.

Reads comparison/results/{nmf,cnn,nmf_cnn}.json (whichever exist), then:
  - per-arm summary (mean +/- std, median, min, #>=0.70) for full-volume & liver-only Dice
  - paired Wilcoxon signed-rank between every pair of arms (on the shared 20 patients)
  - writes results/table.md, results/summary.md, results/per_patient.csv
  - writes figures results/figs/box_full.png and results/figs/per_patient_full.png

    python comparison/evaluate.py
"""
import os
import sys
import json
import csv

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_data as cd  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARMS = [("nmf", "NMF-only"), ("cnn", "CNN-only"),
        ("nmf_cnn", "NMF+CNN (ROI)"), ("nmf_ch", "NMF+CNN (ch)")]
FIG_DIR = os.path.join(cd.RESULTS_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def load_arm(key):
    path = os.path.join(cd.RESULTS_DIR, f"{key}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        blob = json.load(fh)
    return blob["per_patient"]


def vec(per, patients, field):
    return np.array([per[p][field] for p in patients], float)


def summarize(v):
    v = np.asarray(v, float)
    finite = v[np.isfinite(v)]
    return dict(mean=float(np.nanmean(v)), std=float(np.nanstd(v)),
                median=float(np.nanmedian(v)), min=float(np.nanmin(v)),
                ge70=int((finite >= 0.70).sum()), n=int(len(finite)))


def wilcoxon(a, b):
    """Paired Wilcoxon signed-rank; returns (stat, p, median_diff) or None."""
    try:
        from scipy.stats import wilcoxon as wx
    except Exception:
        return None
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    diff = a - b
    if np.allclose(diff, 0):
        return (float("nan"), 1.0, 0.0)
    try:
        stat, p = wx(a, b)
        return (float(stat), float(p), float(np.median(diff)))
    except Exception as e:
        print(f"  wilcoxon failed: {e}")
        return None


def main():
    present = [(k, name) for k, name in ARMS if load_arm(k) is not None]
    if not present:
        print("No result JSONs found in", cd.RESULTS_DIR)
        print("Run arm_nmf.py / arm_cnn.py / arm_nmf_cnn.py first.")
        return
    data = {k: load_arm(k) for k, _ in present}

    # shared patient set (intersection across present arms), in ALL20 order
    common = [p for p in cd.ALL20 if all(p in data[k] for k, _ in present)]
    print(f"Arms present: {[n for _, n in present]}   shared patients: {len(common)}")

    full = {k: vec(data[k], common, "full_vol") for k, _ in present}
    liver = {k: vec(data[k], common, "liver_vol") for k, _ in present}

    # ---- summary table ----
    lines = ["# Arm comparison (5-fold patient-level CV, CHAOS T2-SPIR)\n",
             f"Shared patients: n = {len(common)}\n",
             "| Arm | Full-vol Dice | Liver-only Dice | median (full) | min | #>=0.70 |",
             "|-----|---------------|-----------------|---------------|-----|---------|"]
    for k, name in present:
        sf, sl = summarize(full[k]), summarize(liver[k])
        lines.append(f"| {name} | {sf['mean']:.3f} +/- {sf['std']:.3f} "
                     f"| {sl['mean']:.3f} +/- {sl['std']:.3f} "
                     f"| {sf['median']:.3f} | {sf['min']:.3f} | {sf['ge70']}/{sf['n']} |")

    # ---- paired significance ----
    lines += ["\n## Paired Wilcoxon signed-rank (full-volume Dice)\n",
              "| Comparison | median diff | p-value |",
              "|------------|-------------|---------|"]
    keys = [k for k, _ in present]
    names = {k: n for k, n in present}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            res = wilcoxon(full[a], full[b])
            if res is None:
                continue
            _, p, md = res
            lines.append(f"| {names[a]} vs {names[b]} | {md:+.3f} | {p:.4f} |")

    table_md = "\n".join(lines) + "\n"
    with open(os.path.join(cd.RESULTS_DIR, "table.md"), "w") as fh:
        fh.write(table_md)
    with open(os.path.join(cd.RESULTS_DIR, "summary.md"), "w") as fh:
        fh.write(table_md)
    print("\n" + table_md)

    # ---- per-patient CSV ----
    with open(os.path.join(cd.RESULTS_DIR, "per_patient.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["patient"] + [f"{k}_full" for k in keys] + [f"{k}_liver" for k in keys]
        w.writerow(header)
        for idx, p in enumerate(common):
            row = [p] + [f"{full[k][idx]:.4f}" for k in keys] + \
                  [f"{liver[k][idx]:.4f}" for k in keys]
            w.writerow(row)

    # ---- figures ----
    _boxplot(full, present, "Full-volume Dice by arm",
             os.path.join(FIG_DIR, "box_full.png"))
    _per_patient_bars(full, present, common,
                      os.path.join(FIG_DIR, "per_patient_full.png"))
    print(f"figures -> {FIG_DIR}")


def _boxplot(full, present, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [n for _, n in present]
    vals = [full[k][np.isfinite(full[k])] for k, _ in present]
    ax.boxplot(vals, labels=labels, showmeans=True)
    for i, v in enumerate(vals, 1):
        ax.scatter(np.full_like(v, i) + np.random.uniform(-0.05, 0.05, len(v)),
                   v, s=14, alpha=0.6, color="tab:blue")
    ax.set_ylabel("Dice"); ax.set_title(title); ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _per_patient_bars(full, present, patients, path):
    keys = [k for k, _ in present]
    x = np.arange(len(patients))
    w = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(max(8, len(patients) * 0.5), 4))
    for i, (k, name) in enumerate(present):
        ax.bar(x + i * w, full[k], width=w, label=name)
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(patients, rotation=0, fontsize=7)
    ax.set_ylabel("Full-vol Dice"); ax.set_xlabel("patient"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
