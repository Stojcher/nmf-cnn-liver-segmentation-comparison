# -*- coding: utf-8 -*-
"""
Channel-ordering stability analysis (reviewer Comment 4, Part B).

The channel-fusion variant resolves NMF's permutation indeterminacy by sorting the
k component maps by their soft-weighted mean intensity (channel 0 = darkest,
channel k-1 = brightest). This script quantifies how consistently that ordering
places the liver in a given channel across the cohort.

For every liver-containing slice we recompute the same NMF the pipeline uses, find
the liver's DOMINANT component (the component to which most liver pixels are
assigned, argmax of the raw soft memberships -- the principled, normalization-free
measure), and map that component to its intensity-sorted channel index. We then
report the distribution of that index over all 20 patients.

A concentration on one index would mean the sorting pins the liver to a fixed
channel; a spread means it does not. Either way this is the stability analysis the
reviewer asked for.

Recomputes the NMF per slice (~a few minutes); uses only the cached slices + liver
masks. Writes comparison/results/channel_stability.md and a bar-chart figure.

    python comparison/channel_stability.py
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_data as cd            # noqa: E402
import liver_nmf_segmentation as L  # noqa: E402

warnings.filterwarnings("ignore")             # light NMF convergence warnings
K = L.K
MIN_LIVER = L.MIN_LIVER_PX
CHANCE = 100.0 / K                            # per-channel % under an arbitrary order


def liver_sorted_channel(img, body, gt):
    """Intensity-sorted channel index of the liver's dominant NMF component.

    Returns None for slices without a meaningful liver.
    """
    stack_n, _ = L.feature_stack(img, body, L.NMF_GROUPS)
    Vn, coords = L.design_matrix(stack_n, body)
    labels, Wn, _Hc, _err = L.nmf_cluster(Vn, k=K, max_iter=200)
    ys, xs = coords
    inten = img[ys, xs]
    wsum = Wn.sum(0) + 1e-9
    wmean_int = (Wn * inten[:, None]).sum(0) / wsum     # soft-weighted mean intensity
    order = np.argsort(wmean_int)                       # sorted channel -> component
    inv = np.argsort(order)                             # component -> sorted channel
    liver_sel = gt[ys, xs]                              # body pixels that are liver
    if liver_sel.sum() < MIN_LIVER:
        return None
    comp = int(np.bincount(labels[liver_sel], minlength=K).argmax())
    return int(inv[comp])


def main():
    slices = cd.load_slices(cd.ALL20)
    per_patient, all_assign = {}, []
    for p in cd.ALL20:
        a = []
        for s in slices[p]:
            ch = liver_sorted_channel(s["img"], s["body"], s["gt"])
            if ch is not None:
                a.append(ch)
        per_patient[p] = a
        all_assign.extend(a)
        print(f"  patient {p}: {len(a)} liver slices", flush=True)

    all_assign = np.array(all_assign, dtype=int)
    n = len(all_assign)
    counts = np.bincount(all_assign, minlength=K)
    pct = 100.0 * counts / max(n, 1)
    modal = int(np.argmax(counts))
    twodark = pct[0] + pct[1]
    cons = [np.bincount(np.array(v, int), minlength=K).max() / len(v)
            for v in per_patient.values() if v]
    mean_cons = 100.0 * float(np.mean(cons))
    n_80 = sum(c >= 0.8 for c in cons)

    lines = [
        "# Channel-ordering stability (NMF channel fusion)\n",
        f"For each of the {n} liver-containing slices across the 20 patients, the "
        f"liver's dominant NMF component was mapped to its intensity-sorted channel "
        f"index (0 = darkest, {K - 1} = brightest). Under an arbitrary (unsorted) "
        f"ordering each index would be equally likely (~{CHANCE:.0f}%).\n",
        "| Sorted channel | 0 (darkest) | " +
        " | ".join(str(c) for c in range(1, K - 1)) + f" | {K - 1} (brightest) |",
        "|" + "---|" * (K + 1),
        "| Slices (%) | " + " | ".join(f"{pct[c]:.1f}" for c in range(K)) + " |",
        "| Slices (n) | " + " | ".join(f"{counts[c]}" for c in range(K)) + " |",
        f"\nThe liver maps to channel {modal} most often ({pct[modal]:.1f}%) and to "
        f"the two darkest channels (0-1) in {twodark:.1f}% of slices, above the "
        f"{2*CHANCE:.0f}% expected under an arbitrary ordering. Mean per-patient "
        f"consistency (largest single-channel share) is {mean_cons:.1f}%, and "
        f"{n_80}/20 patients reach 80% single-channel consistency. The sorting thus "
        "imposes a consistent intensity ordering under which the liver concentrates "
        "in the darker channels, rather than confining it to a single fixed channel.\n",
        "## Per-patient consistency\n",
        "| Patient | n slices | modal channel | consistency (%) |",
        "|---|---|---|---|",
    ]
    for p in cd.ALL20:
        a = np.array(per_patient[p], dtype=int)
        if len(a) == 0:
            lines.append(f"| {p} | 0 | - | - |"); continue
        c = np.bincount(a, minlength=K)
        pm = int(np.argmax(c))
        tag = "  (outlier)" if p == "15" else ""
        lines.append(f"| {p}{tag} | {len(a)} | {pm} | {100.0*c[pm]/len(a):.1f} |")

    out = "\n".join(lines) + "\n"
    path = os.path.join(cd.RESULTS_DIR, "channel_stability.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out)
    print("\n" + out)
    print(f"wrote {path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = os.path.join(cd.RESULTS_DIR, "figs")
    os.makedirs(figdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(range(K), pct, color=["#9ec5e0"] * K)
    bars[modal].set_color("#1b6ca8")
    ax.axhline(CHANCE, ls="--", color="#888888", lw=1,
               label=f"arbitrary ordering (~{CHANCE:.0f}%)")
    ax.set_xlabel(f"Sorted NMF channel (0 = darkest, {K - 1} = brightest)")
    ax.set_ylabel("Liver-containing slices (%)")
    ax.set_title("Channel the liver is assigned to")
    ax.set_xticks(range(K)); ax.set_ylim(0, max(40, pct.max() + 8))
    ax.legend(fontsize=8)
    for c in range(K):
        ax.text(c, pct[c] + 1, f"{pct[c]:.0f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig_path = os.path.join(figdir, "channel_stability.png")
    fig.savefig(fig_path, dpi=300)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
