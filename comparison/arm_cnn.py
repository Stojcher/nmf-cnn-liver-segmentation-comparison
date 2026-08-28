"""
Arm 2 (CNN-only): supervised U-Net trained on ground-truth masks, no NMF.
Full body-masked frame -> U-Net.  Same 5-fold CV, same metrics as the other arms.

    python comparison/arm_cnn.py                 # full 5-fold CV
    python comparison/arm_cnn.py --smoke         # 1 fold, few epochs (timing/sanity)
    python comparison/arm_cnn.py --epochs 30 --folds 0,1
    python comparison/arm_cnn.py --fresh         # ignore cached fold models
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cnn_core  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1 fold, few epochs, for timing/sanity")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--folds", type=str, default=None,
                    help="comma-separated fold indices, e.g. 0,1")
    ap.add_argument("--input-size", type=int, default=None)
    ap.add_argument("--fresh", action="store_true", help="ignore cached models")
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = {}
    folds = None
    arm = "cnn"
    if a.smoke:
        cfg["epochs"] = a.epochs or 5
        folds = [0]
        arm = "cnn_smoke"          # keep smoke models/results separate from the real run
    else:
        if a.epochs:
            cfg["epochs"] = a.epochs
        if a.folds:
            folds = [int(x) for x in a.folds.split(",")]
    if a.input_size:
        cfg["input_size"] = a.input_size

    cnn_core.run_cv(roi_provider=None, cfg=cfg, arm_name=arm,
                    folds_to_run=folds, fresh=a.fresh)


if __name__ == "__main__":
    main()
