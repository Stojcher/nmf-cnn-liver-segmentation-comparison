"""
Arm 1 (NMF-only): reproduce the existing pipeline's 5-fold CV and dump the
PER-PATIENT Dice vector needed for the paired statistical tests.

We do NOT edit liver_nmf_segmentation.py.  Its cross_validate() only prints a
summary, so here we import its building blocks and run the identical fold loop
ourselves to capture per-patient numbers.  This must reproduce the published
full-volume 0.743 / liver-only 0.777.
"""
import os
import sys
import gc
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_data as cd            # noqa: E402
import liver_nmf_segmentation as L  # noqa: E402


def run():
    cache, names, _ = L.cache_patients(L.ALL20)
    folds = L.make_folds(L.ALL20, L.N_FOLDS)
    print("Folds:")
    for i, f in enumerate(folds):
        print(f"  fold {i+1}: {f}")

    per = {}
    for i, test_p in enumerate(folds):
        train_p = [p for p in L.ALL20 if p not in test_p]
        pix, gate, ac, gthr, k = L.train_pipeline(cache, names, train_p)
        for p in test_p:
            full, liv = L.patient_dice(pix, gate, cache, p, ac, gthr, k)
            per[p] = (float(full), float(liv))
        print(f"  fold {i+1} done: test {test_p}  "
              f"full-vol mean {np.mean([per[p][0] for p in test_p]):.3f}", flush=True)
        del pix, gate
        gc.collect()

    full = np.array([per[p][0] for p in L.ALL20])
    liv = np.array([per[p][1] for p in L.ALL20])
    print("\n=== NMF-only 5-fold CV (n=20) ===")
    print(f"  full-volume Dice : {full.mean():.3f} +/- {full.std():.3f} "
          f"(median {np.median(full):.3f}, min {full.min():.3f})")
    print(f"  liver-only  Dice : {np.nanmean(liv):.3f} +/- {np.nanstd(liv):.3f}")
    print(f"  patients >= 0.70 : {(full >= 0.70).sum()}/20")

    path = os.path.join(cd.RESULTS_DIR, "nmf.json")
    with open(path, "w") as fh:
        json.dump(dict(arm="nmf", cfg=dict(method="NMF+HGB+gate+refine"),
                       per_patient={p: dict(full_vol=per[p][0], liver_vol=per[p][1])
                                    for p in per}), fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    run()
