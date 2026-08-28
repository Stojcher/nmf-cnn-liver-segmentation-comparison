"""
Qualitative side-by-side overlays: for each patient, show the predicted liver
mask (cyan) vs ground truth (red contour) for ALL FOUR arms on representative
slices.  Complements the quantitative table/figures from evaluate.py.

    python comparison/visualize_arms.py            # default patients (15, 10, 22)
    python comparison/visualize_arms.py 15 3 39    # specific patients
    python comparison/visualize_arms.py 15 --slices 6 --no-show   # save only

Each patient is predicted with the SAME fold model it was a *test* case in
(leakage-free): CNN arms reload the cached models/<arm>_fold<i>.pt; the NMF-only
arm retrains its light pipeline on that fold's training patients (a few min the
first time a fold is needed).  Figures are saved to results/figs/overlay_patient_<p>.png.
"""
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_data as cd            # noqa: E402
import cnn_core                     # noqa: E402
import nmf_roi                      # noqa: E402
import nmf_channels                 # noqa: E402
import liver_nmf_segmentation as L  # noqa: E402

import matplotlib.pyplot as plt     # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

_PRED_CMAP = ListedColormap(["cyan"])          # solid cyan prediction overlay
DEFAULT_PATIENTS = ["15", "10", "22"]
# CNN arms: (result/model name, display label, uses_roi, uses_channels)
CNN_ARMS = [
    ("cnn",     "CNN-only",      False, False),
    ("nmf_cnn", "NMF+CNN (ROI)", True,  False),
    ("nmf_ch",  "NMF+CNN (ch)",  False, True),
]


def _dice(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    d = a.sum() + b.sum()
    return 2 * np.logical_and(a, b).sum() / d if d else (1.0 if a.sum() == 0 else 0.0)


class Predictor:
    """Caches per-fold NMF pipelines and CNN models so multiple patients are cheap."""

    def __init__(self, log=print):
        self.log = log
        self.log("loading feature cache + all slices ...")
        self.slices = cd.load_slices(cd.ALL20)
        self.cache, self.names, _ = L.cache_patients(cd.ALL20)
        self.folds = cd.make_folds()
        self.cfg = dict(cnn_core.DEFAULT_CFG)
        self._pipe = {}        # fold -> NMF pipeline tuple
        self._models = {}      # (arm, fold) -> UNet

    def fold_of(self, p):
        for i, f in enumerate(self.folds):
            if p in f:
                return i
        raise ValueError(f"patient {p} not in any fold")

    def _nmf_pipeline(self, i):
        if i not in self._pipe:
            train_p = [p for p in cd.ALL20 if p not in self.folds[i]]
            self.log(f"  training NMF pipeline for fold {i+1} ...")
            self._pipe[i] = L.train_pipeline(self.cache, self.names, train_p)
        return self._pipe[i]

    def _cnn_model(self, arm, i, in_ch):
        key = (arm, i)
        if key not in self._models:
            import torch
            mpath = os.path.join(cd.MODELS_DIR, f"{arm}_fold{i}.pt")
            if not os.path.exists(mpath):
                self._models[key] = None
            else:
                m = cnn_core.UNet(features=tuple(self.cfg["features"]), in_ch=in_ch)
                m.load_state_dict(torch.load(mpath, map_location="cpu"))
                m.eval()
                self._models[key] = m
        return self._models[key]

    def predict(self, p):
        """Return dict: slices list, gts, and per-arm {label: (masks_by_pos, vol_dice)}."""
        i = self.fold_of(p)
        sl = self.slices[p]
        gts = [s["gt"] for s in sl]
        out = {}

        # --- NMF-only ---
        pix, gate, ac, gthr, k = self._nmf_pipeline(i)
        nmf_masks = L.final_masks(pix, gate, self.cache, p, ac, gthr, k)
        by_slice = {s_idx: m for (s_idx, m, _gt) in nmf_masks}
        masks = [by_slice.get(s["slice"], np.zeros(s["shape"], bool)) for s in sl]
        out["NMF-only"] = (masks, cd.volumetric_dice(masks, gts)[0])

        # --- CNN arms ---
        for arm, label, uses_roi, uses_ch in CNN_ARMS:
            bbox = nmf_roi.roi_for_patient(sl, patient=p) if uses_roi else None
            extras = nmf_channels.channels_for_patient(sl, patient=p) if uses_ch else None
            in_ch = 1 + (extras[0].shape[-1] if extras is not None else 0)
            model = self._cnn_model(arm, i, in_ch)
            if model is None:
                self.log(f"  [skip] no cached model for {label} (models/{arm}_fold{i}.pt)")
                continue
            preds, _ = cnn_core.predict_patient(model, sl, bbox, self.cfg, extras=extras)
            out[label] = (preds, cd.volumetric_dice(preds, gts)[0])
        return sl, gts, out


def _pick_slices(gts, n):
    """Pick n representative liver slices, weighted by liver AREA so the sample
    reflects where the liver mass is (the slices that dominate the volumetric Dice)
    rather than being evenly spaced by slice index. Selection depends only on the
    ground truth, so the same slices are shown across all approaches."""
    liver = [i for i, g in enumerate(gts) if g.sum() > 0]
    if not liver:
        return []
    if len(liver) <= n:
        return liver
    areas = np.array([gts[i].sum() for i in liver], float)
    cum = np.cumsum(areas); cum /= cum[-1]              # 0..1 over the liver mass
    targets = (np.arange(n) + 0.5) / n                 # n even area-quantiles
    picks = []
    for t in targets:
        j = min(int(np.searchsorted(cum, t)), len(liver) - 1)
        if liver[j] not in picks:
            picks.append(liver[j])
    if len(picks) < n:                                  # dedup left us short ->
        for j in np.argsort(-areas):                    # fill from largest-liver slices
            if liver[j] not in picks:
                picks.append(liver[j])
            if len(picks) >= n:
                break
    return sorted(picks)


def visualize(p, predictor, n_slices=6, show=True):
    sl, gts, arms = predictor.predict(p)
    picks = _pick_slices(gts, n_slices)
    if not picks:
        print(f"patient {p}: no liver slices to show")
        return

    labels = ["Input + GT"] + list(arms.keys())        # first column = reference
    ncol, nrow = len(labels), len(picks)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.6 * nrow),
                             squeeze=False)

    for r, si in enumerate(picks):
        img, gt = sl[si]["img"], gts[si]
        for c, label in enumerate(labels):
            ax = axes[r][c]
            ax.imshow(img, cmap="gray", interpolation="nearest")
            if gt.any():
                ax.contour(gt, levels=[0.5], colors="red", linewidths=1.0)
            if label != "Input + GT":
                mask = arms[label][0][si]
                if mask.any():
                    ax.imshow(np.ma.masked_where(~mask, mask), cmap=_PRED_CMAP,
                              alpha=0.45, vmin=0, vmax=1)
                ax.set_title(f"Dice {_dice(mask, gt):.2f}", fontsize=8, pad=2)
            if r == 0:
                ax.set_title((label if label == "Input + GT"
                              else f"{label}\nDice {_dice(arms[label][0][si], gt):.2f}"),
                             fontsize=8, pad=3)
            if c == 0:
                ax.set_ylabel(f"slice {sl[si]['slice']}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

    dice_str = "  |  ".join(f"{lab} {arms[lab][1]:.3f}" for lab in arms)
    fig.suptitle(f"Patient {p} — liver: prediction (cyan) vs GT (red)\n"
                 f"volumetric Dice:  {dice_str}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = os.path.join(cd.RESULTS_DIR, "figs", f"overlay_patient_{p}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"patient {p}:  {dice_str}\n  saved -> {out}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patients", nargs="*", default=DEFAULT_PATIENTS,
                    help="patient ids (default: 15 10 22)")
    ap.add_argument("--slices", type=int, default=6, help="slices per patient")
    ap.add_argument("--no-show", action="store_true", help="save only, no window")
    a = ap.parse_args()
    pats = a.patients if a.patients else DEFAULT_PATIENTS

    predictor = Predictor()
    for p in pats:
        if p not in cd.ALL20:
            print(f"[skip] patient {p} not in the labeled set")
            continue
        visualize(p, predictor, n_slices=a.slices, show=not a.no_show)


if __name__ == "__main__":
    main()
