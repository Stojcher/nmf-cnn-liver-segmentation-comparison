"""
Shared data layer for the three-arm comparison (NMF, CNN, NMF+CNN cascade).

Everything downstream reuses the ALREADY-BUILT feature cache (.feat_cache/) from
the NMF pipeline, so all three arms share *identical* preprocessing (body mask ->
N4 -> NL-means -> normalize).  Each cached slice stores, at the body-mask pixels:
    X[:, 0]  = normalized intensity  ([0,1])   <- we reconstruct the CNN input from this
    coords   = (ys, xs) body-pixel locations
    shape    = native slice shape (H, W)
    gt       = per-slice liver ground-truth mask (native shape)
    slice    = original slice index

This module does NOT modify liver_nmf_segmentation.py; it only imports from it.
"""
import os
import sys

import numpy as np

# --- make the project root importable (parent of this file's folder) ----------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import liver_nmf_segmentation as L   # noqa: E402  (the existing pipeline, imported only)

ALL20 = list(L.ALL20)
N_FOLDS = L.N_FOLDS

# results / caches live under comparison/
COMPARISON_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(COMPARISON_DIR, "results")
MODELS_DIR = os.path.join(COMPARISON_DIR, "models")
ROI_CACHE_DIR = os.path.join(COMPARISON_DIR, ".roi_cache")
for _d in (RESULTS_DIR, MODELS_DIR, ROI_CACHE_DIR):
    os.makedirs(_d, exist_ok=True)


def load_slices(patients=None):
    """Reconstruct per-patient CNN-ready slices from the shared feature cache.

    Returns dict: patient -> list of per-slice dicts with keys
        slice : int                    original slice index
        img   : (H,W) float32 [0,1]    body-masked normalized intensity
        gt    : (H,W) bool             liver ground truth
        body  : (H,W) bool             body mask (pixels that carry features)
        shape : (H, W)
    """
    if patients is None:
        patients = ALL20
    cache, _names, _is_nmf = L.cache_patients(patients)
    out = {}
    for p in patients:
        slices = []
        for r in cache[p]:
            ys, xs = r["coords"]
            img = np.zeros(r["shape"], np.float32)
            img[ys, xs] = r["X"][:, 0]          # normalized intensity channel
            body = np.zeros(r["shape"], bool)
            body[ys, xs] = True
            slices.append(dict(slice=int(r["slice"]), img=img,
                               gt=r["gt"].astype(bool), body=body,
                               shape=tuple(r["shape"])))
        out[p] = slices
    return out


def make_folds(patients=None, k=None, seed=0):
    """SAME patient-level folds the NMF arm uses (imported, not reimplemented)."""
    if patients is None:
        patients = ALL20
    if k is None:
        k = N_FOLDS
    return L.make_folds(patients, k, seed)


def volumetric_dice(preds, gts):
    """Volumetric Dice matching liver_nmf_segmentation.patient_dice.

    preds, gts: parallel lists of native-shape bool arrays for ONE patient.
    Returns (full_vol, liver_vol):
        full_vol  = Dice over ALL slices (false positives on non-liver slices count)
        liver_vol = Dice over slices whose GT contains liver
    """
    vi = vd = vi_l = vd_l = 0
    for m, gt in zip(preds, gts):
        m = np.asarray(m, bool)
        gt = np.asarray(gt, bool)
        inter = int(np.logical_and(m, gt).sum())
        tot = int(m.sum() + gt.sum())
        vi += inter
        vd += tot
        if gt.sum() > 0:
            vi_l += inter
            vd_l += tot
    full = (2 * vi / vd) if vd else 1.0
    liv = (2 * vi_l / vd_l) if vd_l else float("nan")
    return float(full), float(liv)
