"""
================================================================================
  T2 MRI LIVER SEGMENTATION  --  NMF + light supervision  (SINGLE-FILE PIPELINE)
================================================================================

  >>>  RUN THIS FILE.  <<<

  Two modes:
      python liver_nmf_segmentation.py          # segment held-out patients + show plots
      python liver_nmf_segmentation.py cv        # 5-fold cross-validation

  Method (NMF is central throughout):
      DICOM  ->  preprocess (body mask, N4 bias correction, NL-means, normalize)
             ->  per-pixel features  =  BASE (intensity/Gaussian/texture/position)
                                        + NMF cluster-context features
             ->  pixel classifier (HistGradientBoosting)
             ->  slice-level liver-presence gate (RandomForest)  -> suppress false positives
             ->  boundary refinement (intensity gating)
             ->  liver mask

  Validated 5-fold CV (all 20 CHAOS Train_Sets patients):
      full-volume Dice 0.743 +/- 0.128,  liver-only 0.777 +/- 0.123.

  This is a self-contained consolidation of the development pipeline (the
  original step1..step23 scripts are archived under development/). It reuses the
  feature cache .feat_cache/ ; first run on uncached patients extracts them
  (~2-3 min each, one-time), later runs load in seconds.

  Dataset: CHAOS T2-SPIR. Train_Sets/MR/<patient>/T2SPIR/{DICOM_anon,Ground}.
  GT label 63 = liver. (Test_Sets has no masks, hence CV on Train_Sets.)
================================================================================
"""

import os
import sys
import glob
import time
import pickle
import numpy as np
import pydicom
from PIL import Image
from scipy import ndimage as ndi
from skimage import morphology, restoration, util
import SimpleITK as sitk
from sklearn.decomposition import NMF
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

# ==========================================================================
#  CONFIGURATION
# ==========================================================================
DATASET_PATH = r"D:\Code\Liver segmentation\Train_Sets\MR"
CACHE_DIR    = r"D:\Code\Liver segmentation\.feat_cache"

LIVER_LO, LIVER_HI = 55, 70        # CHAOS MR liver label band (nominal 63)
CLIP_LO_PCT, CLIP_HI_PCT = 1.0, 99.0
BODY_THR_FRAC = 0.08               # inclusive body mask (tissue vs air)
K = 5                              # NMF components
MIN_LIVER_PX = 150                 # below this, a slice has "no liver"

BASE_GROUPS = ("intensity", "gauss", "texture", "position")   # classifier features
NMF_GROUPS  = ("gauss", "position")                           # features NMF clusters on
PROB_THR = 0.7                     # pixel-probability threshold (tuned on train)
MAX_TRAIN_ROWS = 600_000
BG_PER_FG = 3.0
KS = [1.5, 2.0, 2.5, 3.0, 4.0]     # refinement strengths searched (on train)

# Train/test split for the segment+plot mode (honest held-out demo).
TRAIN = ["1", "2", "3", "5", "8", "20", "21", "22", "31", "32", "33", "34"]
TEST  = ["10", "13", "15", "19"]
# All labeled patients, for cross-validation.
ALL20 = ["1", "2", "3", "5", "8", "10", "13", "15", "19", "20",
         "21", "22", "31", "32", "33", "34", "36", "37", "38", "39"]
N_FOLDS = 5


# ==========================================================================
#  1. DATA LOADING
# ==========================================================================
def load_t2_volume(dataset_path, patient):
    """Load a patient's T2-SPIR volume + liver GT, slices sorted by position."""
    base = os.path.join(dataset_path, patient, "T2SPIR")
    dicom_dir, ground_dir = os.path.join(base, "DICOM_anon"), os.path.join(base, "Ground")
    dcm_files = sorted(glob.glob(os.path.join(dicom_dir, "*.dcm")))
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files in {dicom_dir}")

    slices = [(f, pydicom.dcmread(f)) for f in dcm_files]

    def sort_key(item):
        ipp = getattr(item[1], "ImagePositionPatient", None)
        return float(ipp[2]) if ipp is not None else float(getattr(item[1], "InstanceNumber", 0))
    slices.sort(key=sort_key)

    vol, gt = [], []
    for f, ds in slices:
        vol.append(ds.pixel_array.astype(np.float32))
        stem = os.path.splitext(os.path.basename(f))[0]
        gt_path = os.path.join(ground_dir, stem + ".png")
        gt.append(np.array(Image.open(gt_path)) if os.path.exists(gt_path)
                  else np.zeros_like(vol[-1], dtype=np.uint8))

    volume = np.stack(vol).astype(np.float32)
    gt_raw = np.stack(gt).astype(np.uint8)
    liver_mask = (gt_raw >= LIVER_LO) & (gt_raw <= LIVER_HI)

    ds0 = slices[0][1]
    ps = getattr(ds0, "PixelSpacing", [1.0, 1.0])
    spacing = (float(getattr(ds0, "SliceThickness", 1.0)), float(ps[0]), float(ps[1]))
    return dict(volume=volume, liver_mask=liver_mask, gt_raw=gt_raw, spacing=spacing)


# ==========================================================================
#  2. PREPROCESSING  (body mask -> N4 bias -> NL-means denoise -> normalize)
# ==========================================================================
def body_mask_incl(slice_img, frac=BODY_THR_FRAC):
    """Inclusive tissue-vs-air mask (retains the dim liver everywhere)."""
    if slice_img.max() <= 0:
        return np.zeros_like(slice_img, dtype=bool)
    hi = np.percentile(slice_img, 99.5)
    mask = slice_img > frac * hi
    mask = morphology.remove_small_objects(mask, 64)
    mask = morphology.binary_closing(mask, morphology.disk(5))
    mask = ndi.binary_fill_holes(mask)
    lbl, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(np.ones_like(lbl), lbl, np.arange(1, n + 1))
    mask = lbl == (np.argmax(sizes) + 1)
    return ndi.binary_fill_holes(mask)


def n4_correct(slice_img, mask, shrink=2):
    """SimpleITK N4 bias-field correction inside the body mask."""
    if mask.sum() == 0:
        return slice_img.astype(np.float32)
    img = sitk.GetImageFromArray(slice_img.astype(np.float32))
    m = sitk.GetImageFromArray(mask.astype(np.uint8))
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50])
    _ = corrector.Execute(sitk.Shrink(img, [shrink, shrink]), sitk.Shrink(m, [shrink, shrink]))
    corrected = img / sitk.Exp(corrector.GetLogBiasFieldAsImage(img))
    out = sitk.GetArrayFromImage(corrected).astype(np.float32)
    out[~mask] = 0.0
    return out


def denoise(slice_img, mask):
    """Edge-preserving non-local-means denoising inside the body."""
    if mask.sum() == 0:
        return slice_img
    vmax = slice_img.max() if slice_img.max() > 0 else 1.0
    norm = slice_img / vmax
    sigma = np.mean(restoration.estimate_sigma(norm))
    out = restoration.denoise_nl_means(norm, h=0.8 * sigma, sigma=sigma,
                                       patch_size=5, patch_distance=6, fast_mode=True)
    out = out * vmax
    out[~mask] = 0.0
    return out.astype(np.float32)


def normalize(slice_img, mask):
    """Percentile-clip inside body, scale to [0,1]. Background stays 0."""
    out = np.zeros_like(slice_img, dtype=np.float32)
    if mask.sum() == 0:
        return out
    lo, hi = np.percentile(slice_img[mask], [CLIP_LO_PCT, CLIP_HI_PCT])
    if hi <= lo:
        hi = lo + 1.0
    out = (np.clip(slice_img, lo, hi) - lo) / (hi - lo)
    out[~mask] = 0.0
    return out.astype(np.float32)


def preprocess_incl(slice_img):
    """Full per-slice preprocessing -> (normalized [0,1] image, body mask)."""
    mask = body_mask_incl(slice_img)
    return normalize(denoise(n4_correct(slice_img, mask), mask), mask), mask


# ==========================================================================
#  3. NMF FEATURES
# ==========================================================================
def _local_stats(img, size=7):
    mean = ndi.uniform_filter(img, size=size)
    var = np.clip(ndi.uniform_filter(img * img, size=size) - mean * mean, 0, None)
    return mean, np.sqrt(var)


def _local_entropy(img, mask, disk_radius=5):
    from skimage.filters.rank import entropy
    img8 = util.img_as_ubyte(np.clip(img, 0, 1))
    return entropy(img8, morphology.disk(disk_radius), mask=mask).astype(np.float32)


def _scale01(x, mask):
    """Min-max scale within mask to [0,1]; background -> 0."""
    out = np.zeros_like(x, dtype=np.float32)
    vals = x[mask]
    if vals.size == 0:
        return out
    lo, hi = vals.min(), vals.max()
    if hi <= lo:
        return out
    out[mask] = (x[mask] - lo) / (hi - lo)
    return out


def feature_stack(norm_slice, mask, groups):
    """(H,W,F) stack of non-negative [0,1] feature maps for the given groups."""
    H, W = norm_slice.shape
    feats, names = [], []

    def add(arr, name):
        feats.append(_scale01(arr, mask)); names.append(name)

    if "intensity" in groups:
        add(norm_slice, "intensity")
    if "gauss" in groups:
        for s in (1, 2, 4):
            add(ndi.gaussian_filter(norm_slice, sigma=s), f"gauss_s{s}")
    if "texture" in groups:
        lmean, lstd = _local_stats(norm_slice, size=7)
        add(lmean, "local_mean"); add(lstd, "local_std")
        add(_local_entropy(norm_slice, mask, 5), "entropy")
    if "position" in groups:
        rr, cc = np.mgrid[0:H, 0:W].astype(np.float32)
        add(rr / (H - 1), "pos_row"); add(cc / (W - 1), "pos_col")
        cy, cx = (H - 1) / 2, (W - 1) / 2
        add(np.hypot(rr - cy, cc - cx), "dist_center")
    return np.stack(feats, axis=-1).astype(np.float32), names


def design_matrix(stack, mask):
    """Flatten stack to V=(n_body_pixels, F); return V and pixel coords."""
    ys, xs = np.where(mask)
    return stack[ys, xs, :], (ys, xs)


def nmf_cluster(V, k=K, random_state=0, max_iter=1000):
    """NMF on V; return per-pixel argmax cluster labels (+ W, H, err)."""
    init = "nndsvda" if k <= V.shape[1] else "random"
    model = NMF(n_components=k, init=init, solver="cd", beta_loss="frobenius",
                max_iter=max_iter, tol=1e-4, random_state=random_state)
    W = model.fit_transform(V)
    H = model.components_
    h_norm = np.linalg.norm(H, axis=1, keepdims=True)
    h_norm[h_norm == 0] = 1.0
    W_n = W * h_norm.T                 # resolve NMF scale ambiguity
    return np.argmax(W_n, axis=1), W_n, H / h_norm, model.reconstruction_err_


def nmf_context(labels, W, coords, norm, k):
    """7 permutation-invariant per-pixel features describing the pixel's cluster."""
    ys, xs = coords
    H, Wd = norm.shape
    npix = len(ys)
    inten = norm[ys, xs]
    f = np.zeros((npix, 7), dtype=np.float32)
    for j in range(k):
        sel = labels == j
        if not sel.any():
            continue
        cy, cx = ys[sel].mean(), xs[sel].mean()
        f[sel, 0] = inten[sel].mean()                       # cluster mean intensity
        f[sel, 1] = sel.sum() / npix                        # cluster size fraction
        f[sel, 2] = cy / (H - 1)                            # centroid row
        f[sel, 3] = cx / (Wd - 1)                           # centroid col
        f[sel, 4] = inten[sel].std()                        # homogeneity
        f[sel, 5] = np.hypot(ys[sel] - cy, xs[sel] - cx) / H  # dist to centroid
    Wn = W / (W.sum(1, keepdims=True) + 1e-9)
    f[:, 6] = Wn.max(1)                                     # soft-membership strength
    names = ["nmf_clust_meanI", "nmf_clust_size", "nmf_clust_cr",
             "nmf_clust_cc", "nmf_clust_std", "nmf_dist_cent", "nmf_wmax"]
    return f, names


# ==========================================================================
#  4. FEATURE CACHE  (preprocess + NMF per slice, computed once, saved to disk)
# ==========================================================================
def build_patient(patient, k=K):
    """Per-slice records: features X (BASE+NMF), label y, gt, coords, shape."""
    data = load_t2_volume(DATASET_PATH, patient)
    vol, liver = data["volume"], data["liver_mask"]
    records, base_names, nmf_names = [], None, None
    for i in range(vol.shape[0]):
        norm, mask = preprocess_incl(vol[i])
        if mask.sum() < MIN_LIVER_PX:
            continue
        stack_b, base_names = feature_stack(norm, mask, BASE_GROUPS)
        Xb, coords = design_matrix(stack_b, mask)
        stack_n, _ = feature_stack(norm, mask, NMF_GROUPS)
        Vn, _ = design_matrix(stack_n, mask)
        labels, W, _, _ = nmf_cluster(Vn, k=k)
        Xn, nmf_names = nmf_context(labels, W, coords, norm, k)
        X = np.hstack([Xb, Xn]).astype(np.float32)
        y = liver[i][coords[0], coords[1]].astype(np.int8)
        records.append(dict(slice=i, X=X, y=y, coords=coords, shape=norm.shape, gt=liver[i]))
    is_nmf = np.array([False] * len(base_names) + [True] * len(nmf_names))
    return records, base_names + nmf_names, is_nmf


def cache_patients(patients, k=K):
    """Load per-patient features from disk cache, building (once) if missing."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    out, names, is_nmf = {}, None, None
    for p in patients:
        fpath = os.path.join(CACHE_DIR, f"patient_{p}_k{k}.pkl")
        if os.path.exists(fpath):
            with open(fpath, "rb") as fh:
                blob = pickle.load(fh)
            out[p], names, is_nmf = blob["records"], blob["names"], blob["is_nmf"]
            print(f"  loaded patient {p} from cache", flush=True)
            continue
        t = time.time()
        out[p], names, is_nmf = build_patient(p, k=k)
        with open(fpath, "wb") as fh:
            pickle.dump(dict(records=out[p], names=names, is_nmf=is_nmf), fh)
        print(f"  built+cached patient {p}: {len(out[p])} slices ({time.time()-t:.0f}s)", flush=True)
    return out, names, is_nmf


# ==========================================================================
#  5. SUPERVISED PIXEL CLASSIFIER
# ==========================================================================
def assemble_train(cache, patients, rng):
    """Pool training pixels with balanced background subsampling."""
    Xs, ys = [], []
    for p in patients:
        for r in cache[p]:
            X, y = r["X"], r["y"]
            fg = np.where(y == 1)[0]
            bg = np.where(y == 0)[0]
            if len(fg) == 0:
                idx = rng.choice(bg, size=min(len(bg), 200), replace=False)
            else:
                nbg = min(len(bg), int(BG_PER_FG * len(fg)))
                idx = np.concatenate([fg, rng.choice(bg, size=nbg, replace=False)])
            Xs.append(X[idx]); ys.append(y[idx])
    X = np.vstack(Xs); y = np.concatenate(ys)
    if len(y) > MAX_TRAIN_ROWS:
        sub = rng.choice(len(y), size=MAX_TRAIN_ROWS, replace=False)
        X, y = X[sub], y[sub]
    return X, y


def train_clf(X, y):
    w = np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0)
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08,
                                         max_depth=None, l2_regularization=1.0,
                                         random_state=0)
    clf.fit(X, y, sample_weight=w)
    return clf


# ==========================================================================
#  6. SEGMENTATION UTILS + GATE + REFINEMENT
# ==========================================================================
def dice(pred, gt):
    d = pred.sum() + gt.sum()
    return 2 * np.logical_and(pred, gt).sum() / d if d > 0 else (1.0 if pred.sum() == 0 else 0.0)


def postprocess_mask(raw):
    """Clean a raw mask -> closing, largest connected component, fill holes."""
    if raw.sum() == 0:
        return raw.astype(bool)
    m = morphology.binary_closing(raw, morphology.disk(3))
    lbl, n = ndi.label(m)
    if n == 0:
        return np.zeros_like(raw, dtype=bool)
    sizes = ndi.sum(np.ones_like(lbl), lbl, np.arange(1, n + 1))
    m = lbl == (np.argmax(sizes) + 1)
    m = ndi.binary_fill_holes(m)
    m = morphology.binary_opening(m, morphology.disk(2))
    return ndi.binary_fill_holes(m).astype(bool)


def slice_features(prob, mask, body_n, norm_pos):
    """Summaries of the pixel-probability map -> features for the presence gate."""
    return [prob.max(), np.percentile(prob, 95), prob.mean(),
            (prob > 0.5).mean(), (prob > 0.7).mean(), mask.sum() / body_n, norm_pos]


def refine(mask, intens, k):
    """Boundary refinement: drop pixels brighter than median+k*MAD (liver is mid-dark)."""
    if mask.sum() == 0:
        return mask
    vals = intens[mask]
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-6
    refined = ndi.binary_fill_holes(mask & (intens <= med + k * 1.4826 * mad))
    lbl, n = ndi.label(refined)
    if n == 0:
        return mask
    sizes = ndi.sum(np.ones_like(lbl), lbl, np.arange(1, n + 1))
    refined = ndi.binary_fill_holes(lbl == (np.argmax(sizes) + 1))
    return refined if refined.sum() >= MIN_LIVER_PX else mask     # never nuke


def build_rows(clf, cache, patients, ac):
    """Per slice: pixel mask, intensity map, gate features, gt (aligned to cache order)."""
    rows = []
    for p in patients:
        recs = cache[p]
        idxs = [r["slice"] for r in recs]
        lo, hi = min(idxs), max(idxs)
        for r in recs:
            prob = clf.predict_proba(r["X"][:, ac])[:, 1]
            ys, xs = r["coords"]
            pm = np.zeros(r["shape"], bool); pm[ys, xs] = prob >= PROB_THR
            pm = postprocess_mask(pm)
            if pm.sum() < MIN_LIVER_PX:
                pm = np.zeros(r["shape"], bool)
            intens = np.zeros(r["shape"], np.float32); intens[ys, xs] = r["X"][:, 0]
            npos = (r["slice"] - lo) / max(hi - lo, 1)
            rows.append(dict(mask=pm, intens=intens, gt=r["gt"], shape=r["shape"],
                             feat=slice_features(prob, pm, len(prob), npos)))
    return rows


def score(rows, gprob, gthr, k=None):
    """Score rows with gate threshold gthr (+ optional refine k). Returns
    (all_mean, all_vol, liver_mean, liver_vol)."""
    allp, livp = [], []
    vi_a = vd_a = vi_l = vd_l = 0
    for i, r in enumerate(rows):
        m = r["mask"] if gprob[i] >= gthr else np.zeros_like(r["mask"])
        if k is not None and m.sum() > 0:
            m = refine(m, r["intens"], k)
        gt = r["gt"]; d = dice(m, gt)
        allp.append(d); vi_a += np.logical_and(m, gt).sum(); vd_a += m.sum() + gt.sum()
        if gt.sum() > 0:
            livp.append(d); vi_l += np.logical_and(m, gt).sum(); vd_l += m.sum() + gt.sum()
    return (np.mean(allp), (2 * vi_a / vd_a if vd_a else 0),
            np.mean(livp), (2 * vi_l / vd_l if vd_l else 0))


# ==========================================================================
#  7. FULL PIPELINE (train classifier + gate; choose gate-thr & refine-k on TRAIN)
# ==========================================================================
def train_pipeline(cache, names, train_patients):
    ac = np.arange(len(names))
    rng = np.random.default_rng(0)
    pix = train_clf(*assemble_train(cache, train_patients, rng))
    tr = build_rows(pix, cache, train_patients, ac)
    gate = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  n_jobs=-1, random_state=0)
    gate.fit(np.array([r["feat"] for r in tr]),
             np.array([int(r["gt"].sum() > 0) for r in tr]))
    g_tr = gate.predict_proba(np.array([r["feat"] for r in tr]))[:, 1]
    gate_thr = max(np.linspace(0.2, 0.8, 13), key=lambda t: score(tr, g_tr, t)[0])
    refine_k = max(KS, key=lambda k: score(tr, g_tr, gate_thr, k)[1])
    return pix, gate, ac, gate_thr, refine_k


def final_masks(pix, gate, cache, patient, ac, gate_thr, refine_k):
    """Final (gate + refine) liver mask per slice for one patient."""
    recs = cache[patient]
    rows = build_rows(pix, cache, [patient], ac)
    gprob = gate.predict_proba(np.array([r["feat"] for r in rows]))[:, 1]
    out = []
    for k, r in enumerate(rows):
        m = r["mask"] if gprob[k] >= gate_thr else np.zeros_like(r["mask"])
        if m.sum() > 0:
            m = refine(m, r["intens"], refine_k)
        out.append((recs[k]["slice"], m, r["gt"]))
    return out


# ==========================================================================
#  8. CROSS-VALIDATION (5-fold, patient-level, leakage-free)
# ==========================================================================
def make_folds(patients, k, seed=0):
    rng = np.random.default_rng(seed)
    p = list(patients); rng.shuffle(p)
    return [sorted(p[i::k], key=lambda x: int(x)) for i in range(k)]


def patient_dice(pix, gate, cache, p, ac, gthr, k):
    full_vol = liv_vol = None
    rows = build_rows(pix, cache, [p], ac)
    gp = gate.predict_proba(np.array([r["feat"] for r in rows]))[:, 1]
    vi = vd = vi_l = vd_l = 0
    for i, r in enumerate(rows):
        m = r["mask"] if gp[i] >= gthr else np.zeros_like(r["mask"])
        if m.sum() > 0:
            m = refine(m, r["intens"], k)
        gt = r["gt"]
        vi += np.logical_and(m, gt).sum(); vd += m.sum() + gt.sum()
        if gt.sum() > 0:
            vi_l += np.logical_and(m, gt).sum(); vd_l += m.sum() + gt.sum()
    full_vol = 2 * vi / vd if vd else 1.0
    liv_vol = 2 * vi_l / vd_l if vd_l else float("nan")
    return full_vol, liv_vol


def cross_validate():
    import gc
    cache, names, _ = cache_patients(ALL20)
    folds = make_folds(ALL20, N_FOLDS)
    print("\n5-fold patient-level CV. Folds:")
    for i, f in enumerate(folds):
        print(f"  fold {i+1}: {f}")
    per = {}
    print("\nRunning folds ...")
    for i, test_p in enumerate(folds):
        train_p = [p for p in ALL20 if p not in test_p]
        pix, gate, ac, gthr, k = train_pipeline(cache, names, train_p)
        for p in test_p:
            per[p] = patient_dice(pix, gate, cache, p, ac, gthr, k)
        print(f"  fold {i+1}: test {test_p}  full-vol mean "
              f"{np.mean([per[p][0] for p in test_p]):.3f}", flush=True)
        del pix, gate                      # free per-fold memory (4GB-friendly)
        gc.collect()

    full = np.array([per[p][0] for p in ALL20])
    liv = np.array([per[p][1] for p in ALL20])
    print("\n=== 5-FOLD CV (n=20) ===")
    print(f"  full-volume Dice : {full.mean():.3f} +/- {full.std():.3f} "
          f"(median {np.median(full):.3f}, min {full.min():.3f})")
    print(f"  liver-only  Dice : {np.nanmean(liv):.3f} +/- {np.nanstd(liv):.3f}")
    print(f"  patients >= 0.70 full-vol : {(full >= 0.70).sum()}/20")
    print("\nPer-patient full-volume Dice:")
    for p in sorted(ALL20, key=lambda x: int(x)):
        print(f"  patient {p:>3}: {per[p][0]:.3f}")


# ==========================================================================
#  9. SEGMENT + VISUALIZE (held-out demo)
# ==========================================================================
def plot_patient(pix, gate, cache, patient, ac, gate_thr, refine_k):
    import matplotlib.pyplot as plt
    vol = load_t2_volume(DATASET_PATH, patient)["volume"]
    masks = final_masks(pix, gate, cache, patient, ac, gate_thr, refine_k)
    liver = [(s, m, gt) for (s, m, gt) in masks if gt.sum() > 0]
    if not liver:
        return
    pick = [liver[i] for i in np.linspace(0, len(liver) - 1, min(9, len(liver))).astype(int)]
    cols = 3
    rows_n = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(5 * cols, 5 * rows_n))
    axes = np.atleast_1d(axes).ravel()
    for ax, (s, m, gt) in zip(axes, pick):
        ax.imshow(vol[s], cmap="gray")
        ax.imshow(np.ma.masked_where(~m, m), cmap="cool", alpha=0.4, vmin=0, vmax=1)
        ax.contour(gt, levels=[0.5], colors="red", linewidths=1.0)
        ax.set_title(f"slice {s}  Dice={dice(m, gt):.2f}", fontsize=10)
        ax.axis("off")
    for ax in axes[len(pick):]:
        ax.axis("off")
    fig.suptitle(f"Patient {patient} (held-out): liver (cyan) vs GT (red)", fontsize=13)
    fig.tight_layout()
    plt.show()


def segment_and_plot():
    cache, names, _ = cache_patients(TRAIN + TEST)
    pix, gate, ac, gate_thr, refine_k = train_pipeline(cache, names, TRAIN)
    print(f"gate_threshold={gate_thr:.2f}  refine_k={refine_k}")

    te = build_rows(pix, cache, TEST, ac)
    g_te = gate.predict_proba(np.array([r["feat"] for r in te]))[:, 1]
    s = score(te, g_te, gate_thr, refine_k)
    print("\n=== HELD-OUT (test patients) ===")
    print(f"  full-volume Dice : {s[1]:.3f}")
    print(f"  liver-only  Dice : {s[3]:.3f}")
    print("\nPlotting each held-out patient ...")
    for p in TEST:
        plot_patient(pix, gate, cache, p, ac, gate_thr, refine_k)
    print("Done.")


# ==========================================================================
#  MAIN
# ==========================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "cv":
        cross_validate()                 # 5-fold CV over all 20 patients
    else:
        segment_and_plot()               # train + plot segmentation on held-out patients
