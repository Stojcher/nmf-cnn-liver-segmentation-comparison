"""
Arm 4 (NMF+CNN, input-channel fusion): the SAME U-Net as CNN-only, but the input
is the T2 slice PLUS k canonically-ordered NMF component maps as extra channels
(in_ch = 1 + k).  No ROI crop (full body-masked frame).  The only difference from
the CNN-only arm is the extra NMF channels -> the delta isolates NMF's value as a
per-pixel feature prior (an alternative to the ROI cascade).

    python comparison/arm_nmf_ch.py                 # full 5-fold CV
    python comparison/arm_nmf_ch.py --smoke         # 1 fold, few epochs
    python comparison/arm_nmf_ch.py --epochs 30 --folds 0,1
    python comparison/arm_nmf_ch.py --fresh
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cnn_core       # noqa: E402
import nmf_channels   # noqa: E402


def channel_provider(slices, patient):
    return nmf_channels.channels_for_patient(slices, patient=patient)


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
    arm = "nmf_ch"
    if a.smoke:
        cfg["epochs"] = a.epochs or 5
        folds = [0]
        arm = "nmf_ch_smoke"
    else:
        if a.epochs:
            cfg["epochs"] = a.epochs
        if a.folds:
            folds = [int(x) for x in a.folds.split(",")]
    if a.input_size:
        cfg["input_size"] = a.input_size

    cnn_core.run_cv(roi_provider=None, channel_provider=channel_provider,
                    cfg=cfg, arm_name=arm, folds_to_run=folds, fresh=a.fresh)


if __name__ == "__main__":
    main()
