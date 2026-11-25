# -*- coding: utf-8 -*-
"""
Effect sizes + bootstrap confidence intervals for the paired arm comparisons.

Addresses the "underpowered / N=20" limitation by reporting, for each
pairwise comparison, not just the Wilcoxon p-value but the MAGNITUDE of the
difference and its uncertainty:

    * mean paired Dice difference (A - B) with a 95% bootstrap CI (BCa),
    * median paired difference with a 95% percentile-bootstrap CI (robust),
    * paired Wilcoxon p (reproduces Table 2),
    * Cohen's d_z (paired standardized effect size).

A non-significant p alone is ambiguous ("no effect" vs "undetected"); the CI
distinguishes them -- a tight CI around 0 is positive evidence the effect is
small, a wide CI honestly shows the sample cannot rule one out.

Reads comparison/results/per_patient.csv; writes effect_size_table.md and prints.
Full-volume Dice is the primary metric (pass --metric liver for liver-only).

    python comparison/effect_sizes.py
    python comparison/effect_sizes.py --metric liver --boot 20000
"""
import os
import sys
import csv
import argparse

try:                                   # Windows cp1252 console can't encode some glyphs
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from scipy.stats import wilcoxon, bootstrap

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "results", "per_patient.csv")
OUT_PATH = os.path.join(HERE, "results", "effect_size_table.md")

# approach -> CSV column stem (metric suffix appended: _full / _liver)
COL = {"NMF-only": "nmf", "CNN-only": "cnn",
       "NMF+CNN (ROI)": "nmf_cnn", "NMF+CNN (channels)": "nmf_ch"}

# "A vs B" reported as (A - B), matching the sign convention of Table 2.
PRIMARY = [("CNN-only", "NMF+CNN (ROI)"),
           ("CNN-only", "NMF+CNN (channels)"),
           ("NMF+CNN (ROI)", "NMF+CNN (channels)")]
CONTEXT = [("NMF-only", "CNN-only"),
           ("NMF-only", "NMF+CNN (ROI)"),
           ("NMF-only", "NMF+CNN (channels)")]


def load(metric):
    pats, data = [], {name: [] for name in COL}
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pats.append(r["patient"])
            for name, stem in COL.items():
                data[name].append(float(r[f"{stem}_{metric}"]))
    return pats, {k: np.asarray(v, float) for k, v in data.items()}


def _ci(d, stat, method, n_resamples, rng):
    if len(d) < 2 or np.allclose(d, d[0]):
        return float("nan"), float("nan")
    try:
        res = bootstrap((d,), stat, confidence_level=0.95, n_resamples=n_resamples,
                        method=method, random_state=rng)
        return float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception:
        return float("nan"), float("nan")


def analyze(a, b, n_resamples, rng):
    d = a - b
    sd = float(np.std(d, ddof=1))
    mlo, mhi = _ci(d, np.mean, "BCa", n_resamples, rng)
    med_lo, med_hi = _ci(d, np.median, "percentile", n_resamples, rng)
    try:
        p = float(wilcoxon(a, b).pvalue)
    except ValueError:                       # all-zero differences
        p = float("nan")
    return dict(n=len(d), mean=float(np.mean(d)), mlo=mlo, mhi=mhi,
                med=float(np.median(d)), med_lo=med_lo, med_hi=med_hi,
                p=p, dz=(float(np.mean(d)) / sd if sd > 0 else float("nan")))


def table(comparisons, data, keep_idx, n_resamples, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for A, B in comparisons:
        a, b = data[A][keep_idx], data[B][keep_idx]
        s = analyze(a, b, n_resamples, rng)
        rows.append((f"{A} vs {B}", s))
    return rows


def fmt_rows(rows):
    out = ["| Comparison | n | mean diff (A-B) | 95% CI (BCa) | median diff | "
           "95% CI | Wilcoxon p | Cohen's dz |",
           "|---|---|---|---|---|---|---|---|"]
    for name, s in rows:
        out.append(
            f"| {name} | {s['n']} | {s['mean']:+.3f} | "
            f"[{s['mlo']:+.3f}, {s['mhi']:+.3f}] | {s['med']:+.3f} | "
            f"[{s['med_lo']:+.3f}, {s['med_hi']:+.3f}] | {s['p']:.4f} | {s['dz']:+.2f} |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["full", "liver"], default="full",
                    help="Dice metric to analyze (default: full-volume)")
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pats, data = load(a.metric)
    all_idx = np.arange(len(pats))
    no15 = np.array([i for i, p in enumerate(pats) if p != "15"])

    lines = [f"# Effect sizes and 95% confidence intervals ({a.metric}-volume Dice)\n",
             f"Paired per-patient differences over n={len(pats)} patients; "
             f"{a.boot} bootstrap resamples. Positive mean/median = the first "
             "approach scores higher. A tight CI around 0 is evidence of a small "
             "true effect; a wide CI shows the sample cannot rule one out.\n",
             "\n## CNN-arm comparisons (the non-significant results under scrutiny)\n"]
    lines += fmt_rows(table(PRIMARY, data, all_idx, a.boot, a.seed))
    lines += ["\n## CNN-arm comparisons, excluding the p15 outlier "
              f"(n={len(no15)})\n"]
    lines += fmt_rows(table(PRIMARY, data, no15, a.boot, a.seed + 1))
    lines += ["\n## NMF-only vs CNN-based (context: the large, significant gap)\n"]
    lines += fmt_rows(table(CONTEXT, data, all_idx, a.boot, a.seed + 2))

    text = "\n".join(lines) + "\n"
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
