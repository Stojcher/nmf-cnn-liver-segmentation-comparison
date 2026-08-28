"""
NMF ROI provider  --  the "NMF" ingredient of the NMF+CNN cascade arm.

Unsupervised, training-free liver LOCALIZATION (no ground truth, no fitted model),
so it is leakage-free across CV folds and cheap on CPU.  For each patient we:
    1. reconstruct the (body-masked, normalized) slice from the shared cache,
    2. recompute the SAME NMF the pipeline uses (feature_stack on gauss+position ->
       design_matrix -> nmf_cluster, all imported from liver_nmf_segmentation),
    3. pick the liver-like cluster per slice with a soft prior
       (upper-mid abdomen position * mid-dark intensity * moderate size),
    4. union the picked pixels across the volume and take a GENEROUS padded
       bounding box -> one ROI per patient.

The ROI only has to *contain* the liver comfortably; the CNN does the fine
segmentation inside it.  If localization fails, we fall back to the whole-body
bounding box so the liver is never lost.

ROIs are cached to comparison/.roi_cache/ so repeated runs are instant.
"""
import os
import sys
import pickle

import numpy as np
from scipy import ndimage as ndi

# make sibling modules importable whether run as a script or as comparison.nmf_roi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_data as cd            # noqa: E402  (also puts project root on sys.path)
import liver_nmf_segmentation as L  # noqa: E402


# --- soft-prior hyper-parameters (deliberately loose; ROI is meant to be generous)
_POS_MU, _POS_SIG = 0.45, 0.18      # liver sits in the upper-middle rows
_INT_MU, _INT_SIG = 0.35, 0.22      # liver is mid-dark on T2-SPIR (normalized)
_SIZE_MU, _SIZE_SIG = 0.12, 0.10    # a moderate fraction of the body
# ROI half-size as a fraction of the frame; box is centered on the NMF-estimated
# liver centroid.  Generous enough to contain the liver, tight enough to localize
# (~0.32 x 0.30 half-sizes -> ~38% of the frame area).
_ROI_HH, _ROI_HW = 0.32, 0.30
_MIN_CC = 40                        # ignore per-slice picks smaller than this


def _pick_liver_cluster(labels, coords, shape, img):
    """Return a boolean mask (native shape) of the most liver-like NMF cluster."""
    ys, xs = coords
    H, W = shape
    npix = len(ys)
    best, best_score = None, -np.inf
    for j in np.unique(labels):
        sel = labels == j
        n = int(sel.sum())
        if n < 50:
            continue
        cy = ys[sel].mean() / (H - 1)
        mi = float(img[ys[sel], xs[sel]].mean())
        size = n / npix
        pos = np.exp(-((cy - _POS_MU) ** 2) / (2 * _POS_SIG ** 2))
        inten = np.exp(-((mi - _INT_MU) ** 2) / (2 * _INT_SIG ** 2))
        siz = np.exp(-((size - _SIZE_MU) ** 2) / (2 * _SIZE_SIG ** 2))
        score = pos * inten * siz
        if score > best_score:
            best_score, best = score, j
    mask = np.zeros(shape, bool)
    if best is not None:
        s = labels == best
        mask[ys[s], xs[s]] = True
    return mask


def _localize_slice(img, body):
    """NMF-cluster liver localization for one slice -> coarse boolean mask.

    Uses the SAME features (gauss+position) and k as the pipeline, but a lighter
    NMF (fewer iterations): ROI localization only needs to be coarse, and this
    keeps the one-time ROI pass fast on CPU.
    """
    if body.sum() < L.MIN_LIVER_PX:
        return np.zeros_like(body)
    stack_n, _ = L.feature_stack(img, body, L.NMF_GROUPS)
    Vn, coords = L.design_matrix(stack_n, body)
    labels, _W, _H, _err = L.nmf_cluster(Vn, k=L.K, max_iter=200)
    return _pick_liver_cluster(labels, coords, img.shape, img)


def _largest_cc(mask):
    lbl, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(np.ones_like(lbl), lbl, np.arange(1, n + 1))
    return lbl == (int(np.argmax(sizes)) + 1)


def _weighted_median(values, weights):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights, float)[order]
    cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


def _box_from_center(cy, cx, shape):
    H, W = shape
    hh, hw = int(_ROI_HH * H), int(_ROI_HW * W)
    r0, r1 = int(round(cy)) - hh, int(round(cy)) + hh
    c0, c1 = int(round(cx)) - hw, int(round(cx)) + hw
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(H - 1, r1), min(W - 1, c1)
    return r0, r1, c0, c1


def roi_for_patient(slices, patient=None, use_cache=True):
    """One localizing ROI bbox (r0,r1,c0,c1) for a patient's volume.

    The box is centered on the size-weighted median of the per-slice NMF liver
    centroids (robust to the noisy per-slice picks) and sized to a fixed generous
    fraction of the frame.  All slices of a CHAOS patient share the same shape, so
    one bbox applies to the whole volume.
    """
    cache_path = None
    if use_cache and patient is not None:
        cache_path = os.path.join(cd.ROI_CACHE_DIR, f"roi_{patient}_k{L.K}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)

    shape = slices[0]["shape"]
    body_union = np.zeros(shape, bool)
    cys, cxs, ws = [], [], []
    for s in slices:
        body_union |= s["body"]
        cc = _largest_cc(_localize_slice(s["img"], s["body"]))
        n = int(cc.sum())
        if n < _MIN_CC:
            continue
        ys, xs = np.where(cc)
        cys.append(ys.mean())
        cxs.append(xs.mean())
        ws.append(n)

    if ws:
        cy = _weighted_median(cys, ws)
        cx = _weighted_median(cxs, ws)
    else:                                  # localization failed -> body centroid
        ys, xs = np.where(body_union)
        cy, cx = (ys.mean(), xs.mean()) if len(ys) else (shape[0] / 2, shape[1] / 2)

    bbox = _box_from_center(cy, cx, shape)

    if cache_path is not None:
        with open(cache_path, "wb") as fh:
            pickle.dump(bbox, fh)
    return bbox


def all_rois(slices_by_patient, verbose=True):
    """Return dict patient -> ROI bbox for every patient in the given dict."""
    rois = {}
    for p, slices in slices_by_patient.items():
        rois[p] = roi_for_patient(slices, patient=p)
        if verbose:
            r0, r1, c0, c1 = rois[p]
            H, W = slices[0]["shape"]
            frac = (r1 - r0 + 1) * (c1 - c0 + 1) / (H * W)
            print(f"  ROI patient {p}: rows[{r0},{r1}] cols[{c0},{c1}] "
                  f"({frac*100:.0f}% of frame)", flush=True)
    return rois


if __name__ == "__main__":
    # Standalone: compute + report ROIs for all patients (also warms the cache).
    sl = cd.load_slices(cd.ALL20)
    all_rois(sl)
    print("ROI cache ready.")
