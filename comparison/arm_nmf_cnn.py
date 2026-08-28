"""
Arm 3 (NMF+CNN cascade): NMF localizes a coarse liver ROI, the SAME U-Net as the
CNN-only arm segments within it.  The only difference from arm_cnn.py is the ROI
crop supplied via roi_provider -> the delta isolates NMF localization.

    python comparison/arm_nmf_cnn.py                 # full 5-fold CV
    python comparison/arm_nmf_cnn.py --smoke         # 1 fold, few epochs
    python comparison/arm_nmf_cnn.py --epochs 30 --folds 0,1
    python comparison/arm_nmf_cnn.py --fresh         # ignore cached models
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cnn_core   # noqa: E402
import nmf_roi     # noqa: E402


def roi_provider(slices, patient):
    return nmf_roi.roi_for_patient(slices, patient=patient)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--folds", type=str, default=None)
    ap.add_argument("--input-size", type=int, default=None)
    ap.add_argument("--fresh", action="store_true")
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = {}
    folds = None
    arm = "nmf_cnn"
    if a.smoke:
        cfg["epochs"] = a.epochs or 5
        folds = [0]
        arm = "nmf_cnn_smoke"      # keep smoke models/results separate from the real run
    else:
        if a.epochs:
            cfg["epochs"] = a.epochs
        if a.folds:
            folds = [int(x) for x in a.folds.split(",")]
    if a.input_size:
        cfg["input_size"] = a.input_size

    cnn_core.run_cv(roi_provider=roi_provider, cfg=cfg, arm_name=arm,
                    folds_to_run=folds, fresh=a.fresh)


if __name__ == "__main__":
    main()
