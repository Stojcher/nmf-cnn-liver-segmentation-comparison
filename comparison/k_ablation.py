# -*- coding: utf-8 -*-
"""
k-ablation for the NMF-only pipeline (reviewer Comment 4, Part A).

Runs the identical NMF-only 5-fold cross-validation for several numbers of NMF
components k, and reports Dice as a function of k. k = 5 reproduces the published
result (full-volume 0.743) and validates the driver.

The number of components enters only through cache_patients(k=...): the per-pixel
NMF-context features are always 7-dimensional (permutation-invariant cluster
descriptors), so changing k alters the cluster assignments but not the classifier
input dimensionality -- the rest of the pipeline is unchanged. Only the NMF-only
arm is ablated; the channel-fusion CNN arm would require retraining per k
(infeasible on CPU), which is stated as a limitation.

Each NEW k rebuilds the feature cache (preprocessing + NMF for 20 patients,
~30-40 min) and then runs the CV (~a few minutes). k = 5 just loads the existing
cache.

    python comparison/k_ablation.py                 # k in {3,4,5,6,8}
    python comparison/k_ablation.py --ks 5          # validate (reproduce 0.743)
    python comparison/k_ablation.py --ks 3,4,6,8
"""
import os
import sys
import gc
import time
import json
import argparse

import numpy as np

try:                                   # clean console output on cp1252 (Windows)
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_data as cd            # noqa: E402
import liver_nmf_segmentation as L  # noqa: E402


def run_k(k):
    """NMF-only 5-fold CV with k NMF components -> per-k summary + per-patient."""
    cache, names, _ = L.cache_patients(L.ALL20, k=k)      # builds the cache if new
    folds = L.make_folds(L.ALL20, L.N_FOLDS)
    per = {}
    for i, test_p in enumerate(folds):
        train_p = [p for p in L.ALL20 if p not in test_p]
        pix, gate, ac, gthr, refk = L.train_pipeline(cache, names, train_p)
        for p in test_p:
            per[p] = L.patient_dice(pix, gate, cache, p, ac, gthr, refk)
        print(f"    fold {i+1}/{len(folds)} done (test {test_p})", flush=True)
        del pix, gate
        gc.collect()

    full = np.array([per[p][0] for p in L.ALL20])
    liv = np.array([per[p][1] for p in L.ALL20])
    return dict(k=k, full_mean=float(full.mean()), full_std=float(full.std()),
                full_median=float(np.median(full)), full_min=float(full.min()),
                liver_mean=float(np.nanmean(liv)), liver_std=float(np.nanstd(liv)),
                n_ge70=int((full >= 0.70).sum()),
                per_patient={p: [float(per[p][0]), float(per[p][1])] for p in per})


def summarize(results, out_dir, make_fig=True):
    results = sorted(results, key=lambda r: r["k"])
    lines = [
        "# NMF component-count ablation (NMF-only pipeline)\n",
        "5-fold patient-level cross-validation, varying the number of NMF components "
        "k. k = 5 is the value used in the paper (reproduces the published 0.743). "
        "Only the NMF-only arm is ablated.\n",
        "| k | full-volume Dice | liver-only Dice | median | min | #>=0.70 |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        star = "  (used)" if r["k"] == 5 else ""
        lines.append(
            f"| {r['k']}{star} | {r['full_mean']:.3f} ± {r['full_std']:.3f} | "
            f"{r['liver_mean']:.3f} ± {r['liver_std']:.3f} | {r['full_median']:.3f} | "
            f"{r['full_min']:.3f} | {r['n_ge70']}/20 |")
    spread = max(r["full_mean"] for r in results) - min(r["full_mean"] for r in results)
    lines.append(f"\n_Full-volume mean Dice varies by {spread:.3f} across "
                 f"k = {min(r['k'] for r in results)}–{max(r['k'] for r in results)}._\n")
    md = "\n".join(lines) + "\n"
    path = os.path.join(out_dir, "k_ablation.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
    print(f"wrote {path}")

    # machine-readable
    with open(os.path.join(out_dir, "k_ablation.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    if not make_fig:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = os.path.join(out_dir, "figs")
    os.makedirs(figdir, exist_ok=True)
    ks = [r["k"] for r in results]
    mean = [r["full_mean"] for r in results]
    std = [r["full_std"] for r in results]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.errorbar(ks, mean, yerr=std, fmt="-o", color="#1b6ca8", capsize=4, lw=2)
    if 5 in ks:
        ax.axvline(5, ls="--", color="#888888", lw=1, label="k = 5 (used)")
        ax.legend(fontsize=9)
    ax.set_xlabel("Number of NMF components (k)")
    ax.set_ylabel("Full-volume Dice")
    ax.set_title("NMF-only performance vs. number of components")
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(figdir, "k_ablation.png")
    fig.savefig(fig_path, dpi=300)
    print(f"wrote {fig_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="3,4,5,6", help="comma list of k values")
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",") if x.strip()]

    results = []
    t0 = time.time()
    for k in ks:
        print(f"\n=== k = {k} ===", flush=True)
        t = time.time()
        r = run_k(k)
        r["seconds"] = time.time() - t
        results.append(r)
        print(f"  k={k}: full-vol {r['full_mean']:.3f} ± {r['full_std']:.3f} "
              f"(median {r['full_median']:.3f}, min {r['full_min']:.3f}), "
              f"liver {r['liver_mean']:.3f}, >=0.70: {r['n_ge70']}/20  "
              f"[{r['seconds']/60:.1f} min]", flush=True)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    summarize(results, cd.RESULTS_DIR, make_fig=not a.no_fig)


if __name__ == "__main__":
    main()
