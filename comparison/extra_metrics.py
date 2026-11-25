# -*- coding: utf-8 -*-
"""
Additional segmentation metrics beyond Dice (reviewer R2-C6).

For all four arms we regenerate the per-patient HELD-OUT predictions and report,
per patient and aggregated over the 20 patients:

    full-volume Dice      -> to VERIFY we reproduce Table 1 (0.743/0.855/0.862/0.868)
    IoU (Jaccard)         -> overlap
    sensitivity (recall)  -> TP / (TP+FN)   (under-segmentation)
    precision             -> TP / (TP+FP)   (over-segmentation)
    HD95 (mm)             -> 95th-percentile symmetric surface distance (boundary)
    ASSD (mm)             -> average symmetric surface distance (boundary)

HOW THE PREDICTIONS ARE OBTAINED (no information is recomputed differently from
the published runs):
  * CNN arms (cnn, nmf_cnn, nmf_ch): the CACHED fold models comparison/models/
    <arm>_fold{i}.pt are reloaded and inference is re-run with the exact same
    predict_patient() used in the published CV. NO CNN RETRAINING. If a cached
    model is missing the script stops with a clear message (it will not retrain).
  * NMF-only (nmf): there is no object cache for the classifier/gate, so the
    NMF-only 5-fold pipeline is re-fit per fold (train_pipeline) and masks are
    produced with final_masks() -- the same gate+refine logic patient_dice scores.

Overlap/clinical metrics are POOLED over all of a patient's slices, exactly like
the published volumetric Dice (false positives on non-liver slices count).

Boundary metrics are 3D: each retained slice's mask is placed at its ORIGINAL
slice index in a full-depth volume (dropped end-slices contain no liver and stay
empty), and symmetric surface distances are computed in millimetres using the
DICOM voxel spacing. The surface-distance routine mirrors the standard
binary-erosion-border + Euclidean-distance-transform algorithm (as in medpy).

Edge cases: a patient's ground truth always contains liver, so sensitivity and
the boundary metrics are defined whenever the prediction is non-empty. If a
prediction is entirely empty for a patient, precision and the boundary metrics
are reported as NaN for that patient and excluded from the aggregate (the count
is reported).

Deterministic and fully offline. Writes results/extra_metrics.{md,json}.

    python comparison/extra_metrics.py                 # all four arms
    python comparison/extra_metrics.py --arms cnn,nmf_ch
"""
import os
import sys
import json
import time
import argparse
import warnings

import numpy as np
from scipy import ndimage as ndi

try:                                     # clean console on Windows cp1252
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_data as cd            # noqa: E402
import cnn_core                     # noqa: E402  (UNet, predict_patient, DEFAULT_CFG, DEVICE)
import nmf_roi                      # noqa: E402  (ROI provider)
import nmf_channels                 # noqa: E402  (channel-map provider)
import liver_nmf_segmentation as L  # noqa: E402  (NMF-only pipeline, spacing, constants)
import torch                        # noqa: E402

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)   # light NMF (max_iter=200)

ORDER = ["nmf", "cnn", "nmf_cnn", "nmf_ch"]
DISPLAY = {"nmf": "NMF-only", "cnn": "CNN-only",
           "nmf_cnn": "NMF+CNN (ROI)", "nmf_ch": "NMF+CNN (channel fusion)"}
PUBLISHED = {"nmf": 0.743, "cnn": 0.855, "nmf_cnn": 0.862, "nmf_ch": 0.868}  # Table 1 full-vol


# ==========================================================================
#  METRICS
# ==========================================================================
def _surface_distances_mm(a, b, spacing):
    """All symmetric surface-to-surface distances (mm) between 3D bool masks a,b.

    border(x) = x AND NOT erode(x)  (6-connectivity). For each border voxel of one
    mask we take the Euclidean distance (respecting anisotropic `spacing`) to the
    nearest border voxel of the other mask, in both directions.
    """
    struct = ndi.generate_binary_structure(a.ndim, 1)
    a_border = a ^ ndi.binary_erosion(a, structure=struct, iterations=1)
    b_border = b ^ ndi.binary_erosion(b, structure=struct, iterations=1)
    # distance_transform_edt gives, for each voxel, the distance to the nearest
    # ZERO voxel; so on ~border the zeros ARE the border voxels.
    dt_to_b = ndi.distance_transform_edt(~b_border, sampling=spacing)
    dt_to_a = ndi.distance_transform_edt(~a_border, sampling=spacing)
    return np.concatenate([dt_to_b[a_border], dt_to_a[b_border]])


def patient_metrics(preds, gts, slice_idx, spacing, depth, shape):
    """All metrics for one patient from parallel per-slice bool masks."""
    inter = pred_sum = gt_sum = 0
    for m, g in zip(preds, gts):
        m = np.asarray(m, bool); g = np.asarray(g, bool)
        inter += int(np.logical_and(m, g).sum())
        pred_sum += int(m.sum())
        gt_sum += int(g.sum())

    denom = pred_sum + gt_sum
    dice = (2.0 * inter / denom) if denom else 1.0            # == volumetric_dice full
    union = pred_sum + gt_sum - inter
    iou = (inter / union) if union else 1.0
    sens = (inter / gt_sum) if gt_sum else float("nan")       # recall
    prec = (inter / pred_sum) if pred_sum else float("nan")   # NaN if nothing predicted

    # ---- 3D boundary metrics (mm) ----
    H, W = shape
    pv = np.zeros((depth, H, W), bool)
    gv = np.zeros((depth, H, W), bool)
    for z, m, g in zip(slice_idx, preds, gts):
        pv[z] = np.asarray(m, bool)
        gv[z] = np.asarray(g, bool)
    if pv.any() and gv.any():
        d = _surface_distances_mm(pv, gv, spacing)
        hd95, assd, hd = float(np.percentile(d, 95)), float(d.mean()), float(d.max())
    else:
        hd95 = assd = hd = float("nan")

    return dict(dice=dice, iou=iou, sens=sens, prec=prec,
                hd95=hd95, assd=assd, hd=hd, pred_empty=(pred_sum == 0))


# ==========================================================================
#  GEOMETRY (voxel spacing + full depth + slice shape), one DICOM read/patient
# ==========================================================================
_geom = {}
def geom(p):
    if p not in _geom:
        data = L.load_t2_volume(L.DATASET_PATH, p)
        vol = data["volume"]
        _geom[p] = dict(spacing=tuple(float(s) for s in data["spacing"]),   # (sz,sy,sx) mm
                        depth=int(vol.shape[0]),
                        shape=(int(vol.shape[1]), int(vol.shape[2])))
    return _geom[p]


# ==========================================================================
#  ARM RUNNERS
# ==========================================================================
def run_cnn_arm(arm, roi_provider=None, channel_provider=None, log=print):
    """Reload cached fold models and re-run inference (no retraining)."""
    cfg = dict(cnn_core.DEFAULT_CFG)
    slices = cd.load_slices(cd.ALL20)
    rois = {p: roi_provider(slices[p], p) for p in cd.ALL20} if roi_provider else None
    chans = {p: channel_provider(slices[p], p) for p in cd.ALL20} if channel_provider else None
    in_ch = 1 + (chans[cd.ALL20[0]][0].shape[-1] if chans else 0)
    folds = cd.make_folds()
    out = {}
    for i, test_p in enumerate(folds):
        mpath = os.path.join(cd.MODELS_DIR, f"{arm}_fold{i}.pt")
        if not os.path.exists(mpath):
            raise FileNotFoundError(
                f"Missing cached model {mpath}. Run comparison/arm_*.py for '{arm}' "
                f"first; this script never retrains the CNN.")
        model = cnn_core.UNet(features=tuple(cfg["features"]), in_ch=in_ch).to(cnn_core.DEVICE)
        model.load_state_dict(torch.load(mpath, map_location=cnn_core.DEVICE))
        for p in test_p:
            preds, gts = cnn_core.predict_patient(
                model, slices[p], (rois.get(p) if rois else None), cfg,
                extras=(chans.get(p) if chans else None))
            g = geom(p)
            slice_idx = [s["slice"] for s in slices[p]]
            out[p] = patient_metrics(preds, gts, slice_idx, g["spacing"],
                                     g["depth"], preds[0].shape)
            # self-check: our pooled Dice must equal the published volumetric_dice
            vd_full, _ = cd.volumetric_dice(preds, gts)
            assert abs(vd_full - out[p]["dice"]) < 1e-9, (p, vd_full, out[p]["dice"])
            log(f"    [{arm}] patient {p}: dice {out[p]['dice']:.3f}  "
                f"hd95 {out[p]['hd95']:.1f}mm  assd {out[p]['assd']:.2f}mm")
        del model
    return out


def run_nmf_arm(log=print):
    """Re-fit the NMF-only pipeline per fold and regenerate masks via final_masks."""
    cache, names, _ = L.cache_patients(L.ALL20)
    folds = L.make_folds(L.ALL20, L.N_FOLDS)
    out = {}
    for i, test_p in enumerate(folds):
        train_p = [p for p in L.ALL20 if p not in test_p]
        pix, gate, ac, gthr, rk = L.train_pipeline(cache, names, train_p)
        for p in test_p:
            fm = L.final_masks(pix, gate, cache, p, ac, gthr, rk)     # [(slice, pred, gt)]
            slice_idx = [int(z) for (z, _m, _g) in fm]
            preds = [np.asarray(m, bool) for (_z, m, _g) in fm]
            gts = [np.asarray(g, bool) for (_z, _m, g) in fm]
            g = geom(p)
            out[p] = patient_metrics(preds, gts, slice_idx, g["spacing"],
                                     g["depth"], preds[0].shape)
            log(f"    [nmf] patient {p}: dice {out[p]['dice']:.3f}  "
                f"hd95 {out[p]['hd95']:.1f}mm  assd {out[p]['assd']:.2f}mm")
        log(f"  nmf fold {i+1}/{len(folds)} done (test {test_p})", flush=True)
    return out


# ==========================================================================
#  AGGREGATE + REPORT
# ==========================================================================
def _agg(values):
    v = np.array(values, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std()), int(v.size)


def summarize(results, out_dir):
    rows_md = [
        "# Additional segmentation metrics (held-out, 5-fold CV, n=20)\n",
        "Overlap/clinical metrics are pooled per patient over all slices (as the "
        "published volumetric Dice). Boundary metrics (HD95, ASSD) are 3D symmetric "
        "surface distances in millimetres, using DICOM voxel spacing. Values are "
        "mean +/- population SD over the 20 patients.\n",
        "| Approach | Dice | IoU | Sensitivity | Precision | HD95 (mm) | ASSD (mm) |",
        "|---|---|---|---|---|---|---|",
    ]
    blob = {}
    for arm in ORDER:
        if arm not in results:
            continue
        per = results[arm]
        pats = list(per.keys())
        dice_m, dice_s, _ = _agg([per[p]["dice"] for p in pats])
        iou_m, iou_s, _ = _agg([per[p]["iou"] for p in pats])
        se_m, se_s, _ = _agg([per[p]["sens"] for p in pats])
        pr_m, pr_s, prn = _agg([per[p]["prec"] for p in pats])
        hd_m, hd_s, hdn = _agg([per[p]["hd95"] for p in pats])
        as_m, as_s, asn = _agg([per[p]["assd"] for p in pats])

        def pm(m, s, n, total=len(pats)):
            tag = f" (n={n})" if n < total else ""
            return f"{m:.3f} +/- {s:.3f}{tag}"

        def pmd(m, s, n, total=len(pats)):
            tag = f" (n={n})" if n < total else ""
            return f"{m:.2f} +/- {s:.2f}{tag}"

        rows_md.append(
            f"| {DISPLAY[arm]} | {pm(dice_m, dice_s, len(pats))} | "
            f"{pm(iou_m, iou_s, len(pats))} | {pm(se_m, se_s, len(pats))} | "
            f"{pm(pr_m, pr_s, prn)} | {pmd(hd_m, hd_s, hdn)} | {pmd(as_m, as_s, asn)} |")

        blob[arm] = dict(
            display=DISPLAY[arm], n=len(pats),
            dice=[dice_m, dice_s], iou=[iou_m, iou_s], sensitivity=[se_m, se_s],
            precision=[pr_m, pr_s, prn], hd95_mm=[hd_m, hd_s, hdn],
            assd_mm=[as_m, as_s, asn],
            per_patient={p: per[p] for p in pats})

    # ---- Dice reproduction check ----
    rows_md.append("\n## Dice reproduction check (vs. Table 1)\n")
    rows_md.append("| Approach | this script | published | match |")
    rows_md.append("|---|---|---|---|")
    all_ok = True
    for arm in ORDER:
        if arm not in results:
            continue
        m = np.mean([results[arm][p]["dice"] for p in results[arm]])
        ref = PUBLISHED[arm]
        ok = abs(m - ref) < 1e-3
        all_ok = all_ok and ok
        rows_md.append(f"| {DISPLAY[arm]} | {m:.3f} | {ref:.3f} | "
                       f"{'OK' if ok else 'MISMATCH'} |")
    rows_md.append(f"\n_Reproduction: {'ALL MATCH' if all_ok else 'MISMATCH - investigate'}._\n")

    md = "\n".join(rows_md) + "\n"
    with open(os.path.join(out_dir, "extra_metrics.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    with open(os.path.join(out_dir, "extra_metrics.json"), "w") as fh:
        json.dump(blob, fh, indent=2)
    print("\n" + md)
    print(f"wrote {os.path.join(out_dir, 'extra_metrics.md')}")
    print(f"wrote {os.path.join(out_dir, 'extra_metrics.json')}")
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ORDER),
                    help="comma list from: nmf,cnn,nmf_cnn,nmf_ch")
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]

    t0 = time.time()
    results = {}
    for arm in arms:
        print(f"\n=== {DISPLAY.get(arm, arm)} ({arm}) ===", flush=True)
        t = time.time()
        if arm == "cnn":
            results[arm] = run_cnn_arm("cnn")
        elif arm == "nmf_cnn":
            results[arm] = run_cnn_arm(
                "nmf_cnn", roi_provider=lambda s, p: nmf_roi.roi_for_patient(s, patient=p))
        elif arm == "nmf_ch":
            results[arm] = run_cnn_arm(
                "nmf_ch",
                channel_provider=lambda s, p: nmf_channels.channels_for_patient(s, patient=p))
        elif arm == "nmf":
            results[arm] = run_nmf_arm()
        else:
            raise SystemExit(f"unknown arm '{arm}'")
        print(f"  [{arm}] done in {(time.time()-t)/60:.1f} min", flush=True)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    summarize(results, cd.RESULTS_DIR)


if __name__ == "__main__":
    main()
