"""
NMF component maps as extra CNN input channels (input-channel fusion variant).

For each slice we recompute the SAME NMF the pipeline uses (gauss+position
features, k components) and turn the per-pixel soft memberships W (n_pixels x k)
into k image-space channel maps.

NMF components have an arbitrary order from fit to fit, so raw channel j is not
comparable across slices/patients.  We CANONICALIZE the ordering by sorting the k
components by their soft-weighted mean intensity (dark -> bright): channel 0
always highlights the darkest tissue component, channel k-1 the brightest.  This
gives the CNN semantically stable channels (the liver is mid-dark on T2-SPIR).

Maps are cached to comparison/.nmf_maps_cache/ (float16) so runs are fast.
Nothing existing is modified; we only import from liver_nmf_segmentation.
"""
import os
import sys
import pickle

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_data as cd            # noqa: E402
import liver_nmf_segmentation as L  # noqa: E402

MAPS_CACHE_DIR = os.path.join(cd.COMPARISON_DIR, ".nmf_maps_cache")
os.makedirs(MAPS_CACHE_DIR, exist_ok=True)


def _maps_for_slice(img, body, k):
    """(H, W, k) canonically-ordered NMF component maps, [0,1], background 0."""
    H, W = img.shape
    maps = np.zeros((H, W, k), np.float32)
    if body.sum() < L.MIN_LIVER_PX:
        return maps
    stack_n, _ = L.feature_stack(img, body, L.NMF_GROUPS)
    Vn, coords = L.design_matrix(stack_n, body)
    labels, Wn, _Hc, _err = L.nmf_cluster(Vn, k=k, max_iter=200)
    ys, xs = coords
    inten = img[ys, xs]

    # canonical order: soft-weighted mean intensity per component (ascending)
    wsum = Wn.sum(0) + 1e-9
    wmean_int = (Wn * inten[:, None]).sum(0) / wsum
    order = np.argsort(wmean_int)

    for ch, j in enumerate(order):
        col = Wn[:, j]
        lo, hi = float(col.min()), float(col.max())
        if hi > lo:
            maps[ys, xs, ch] = (col - lo) / (hi - lo)
    return maps


def channels_for_patient(slices, patient=None, k=None, use_cache=True):
    """List of (H, W, k) NMF channel maps aligned with `slices` (a patient's list)."""
    if k is None:
        k = L.K
    cache_path = None
    if use_cache and patient is not None:
        cache_path = os.path.join(MAPS_CACHE_DIR, f"maps_{patient}_k{k}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fh:
                return [m.astype(np.float32) for m in pickle.load(fh)]

    maps = [_maps_for_slice(s["img"], s["body"], k) for s in slices]

    if cache_path is not None:
        with open(cache_path, "wb") as fh:
            pickle.dump([m.astype(np.float16) for m in maps], fh)
    return maps


if __name__ == "__main__":
    # Warm the cache for all patients (also reports timing).
    import time
    sl = cd.load_slices(cd.ALL20)
    for p in cd.ALL20:
        t = time.time()
        m = channels_for_patient(sl[p], patient=p)
        print(f"  maps patient {p}: {len(m)} slices, {m[0].shape}  ({time.time()-t:.0f}s)",
              flush=True)
    print("NMF channel-map cache ready.")
