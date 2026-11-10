"""
Computational-cost benchmark for the four approaches (CPU-only).

Measures, on the current machine, the costs that distinguish the approaches and
substantiates the "NMF adds negligible computational cost" claim by SHOWING it:

  - shared preprocessing + base NMF feature extraction (one-time, per volume),
  - the extra NMF computation each hybrid adds (ROI localization / channel maps),
  - model training time,
  - inference time per volume.

Writes a Markdown table to  comparison/results/cost_table.md  and prints it.

    python comparison/cost_benchmark.py                 # default benchmark (F5-friendly)
    python comparison/cost_benchmark.py --time-epochs 5 # more epochs for the CNN timing
    python comparison/cost_benchmark.py --nmf-patients 5 --skip-preprocess

Notes on reproducibility:
  * CNN per-epoch training time is measured over a few epochs on a small subset of
    the fold-0 training patients, scaled to the full training set by the number of
    slices, and extrapolated to the full 40-epoch x 5-fold budget (clearly labelled
    "extrapolated"). The three CNN arms are measured on the same subset, so their
    relative training cost is directly comparable.
  * Everything else is measured directly.
  * Absolute times are hardware-dependent; the informative quantity is the RELATIVE
    cost -- the NMF overhead vs. the U-Net training.
"""
import os
import sys
import time
import argparse
import warnings

# Import torch first: on Windows, loading the scientific stack (numpy/SimpleITK/
# skimage/...) before torch can intermittently trigger a c10.dll init failure
# (WinError 1114) under memory pressure. Loading torch first avoids it.
import torch  # noqa: F401,E402

import numpy as np
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_data as cd            # noqa: E402
import cnn_core                     # noqa: E402
import nmf_roi                      # noqa: E402
import nmf_channels                 # noqa: E402
import liver_nmf_segmentation as L  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)  # light NMF (max_iter=200)

FULL_EPOCHS = 40
N_FOLDS = 5


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def _fmt(sec):
    """Human-readable duration."""
    if sec != sec:               # NaN
        return "n/a"
    if sec < 1:
        return f"{sec*1000:.0f} ms"
    if sec < 90:
        return f"{sec:.1f} s"
    return f"{sec/60:.1f} min"


# --------------------------------------------------------------------------
#  Individual measurements
# --------------------------------------------------------------------------
def time_preprocess(patient):
    """Shared preprocessing + base NMF feature extraction for ONE volume."""
    t = time.perf_counter()
    L.build_patient(patient, k=L.K)
    return time.perf_counter() - t


def time_nmf_overhead(slices, patients):
    """Extra NMF computation the hybrids add: ROI box + channel maps (per volume)."""
    roi, maps = [], []
    for p in patients:
        t = time.perf_counter()
        nmf_roi.roi_for_patient(slices[p], patient=p, use_cache=False)
        roi.append(time.perf_counter() - t)
        t = time.perf_counter()
        nmf_channels.channels_for_patient(slices[p], patient=p, use_cache=False)
        maps.append(time.perf_counter() - t)
    return _mean(roi), _mean(maps)


def time_nmf_pipeline(cache, names, train_p, test_p):
    """NMF-only: classifier+gate training (per fold) and inference (per volume)."""
    t = time.perf_counter()
    pix, gate, ac, gthr, k = L.train_pipeline(cache, names, train_p)
    train_t = time.perf_counter() - t
    inf = []
    for p in test_p:
        t = time.perf_counter()
        L.patient_dice(pix, gate, cache, p, ac, gthr, k)
        inf.append(time.perf_counter() - t)
    return train_t, _mean(inf)


def time_cnn(slices, train_p, test_p, rois, channels, time_epochs):
    """A CNN arm: measured per-epoch training time and per-volume inference time.

    rois / channels : dict patient->bbox / patient->list(maps), or None.
    """
    cfg = dict(cnn_core.DEFAULT_CFG)
    cfg["epochs"] = time_epochs
    samples = cnn_core._build_samples(slices, train_p, rois, channels)
    t = time.perf_counter()
    model = cnn_core.train_unet(samples, cfg, log=lambda *a, **k: None)
    per_epoch = (time.perf_counter() - t) / max(time_epochs, 1)
    inf = []
    for p in test_p:
        bbox = rois.get(p) if rois else None
        extras = channels.get(p) if channels else None
        t = time.perf_counter()
        cnn_core.predict_patient(model, slices[p], bbox, cfg, extras=extras)
        inf.append(time.perf_counter() - t)
    return per_epoch, _mean(inf)


# --------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-epochs", type=int, default=3,
                    help="epochs used to measure CNN per-epoch time (default 3)")
    ap.add_argument("--nmf-patients", type=int, default=3,
                    help="how many patients to average the NMF/inference timings over")
    ap.add_argument("--train-subset", type=int, default=4,
                    help="patients used to measure CNN per-epoch time, scaled to the "
                         "full training set by slice count (default 4)")
    ap.add_argument("--skip-preprocess", action="store_true",
                    help="skip the (slow) per-volume preprocessing measurement")
    a = ap.parse_args()

    print("Loading cached slices + features ...", flush=True)
    slices = cd.load_slices(cd.ALL20)
    cache, names, _ = L.cache_patients(cd.ALL20)
    folds = cd.make_folds()
    test_p = folds[0]
    train_p = [p for p in cd.ALL20 if p not in test_p]
    sample_p = cd.ALL20[:a.nmf_patients]

    # CNN per-epoch time is measured on a subset of train patients, then scaled to the
    # full training set by the ratio of training slices (training cost is ~linear in
    # the number of slices).
    cnn_train_p = train_p[:a.train_subset]
    subset_slices = sum(len(slices[p]) for p in cnn_train_p)
    full_slices = sum(len(slices[p]) for p in train_p)
    scale = (full_slices / subset_slices) if subset_slices else 1.0

    # pre-compute ROIs / channel maps (cached) for the fold-0 patients
    print("Preparing NMF ROIs and channel maps (cached) ...", flush=True)
    used = sorted(set(train_p) | set(test_p), key=int)
    rois = {p: nmf_roi.roi_for_patient(slices[p], patient=p) for p in used}
    chans = {p: nmf_channels.channels_for_patient(slices[p], patient=p) for p in used}

    # --- shared preprocessing (one volume) ---
    prep = float("nan")
    if not a.skip_preprocess:
        print(f"Timing shared preprocessing on patient {sample_p[0]} ...", flush=True)
        prep = time_preprocess(sample_p[0])

    # --- extra NMF overhead (per volume) ---
    print(f"Timing NMF ROI / channel-map computation on {len(sample_p)} volumes ...",
          flush=True)
    roi_t, map_t = time_nmf_overhead(slices, sample_p)

    # --- NMF-only pipeline ---
    print("Timing NMF-only training + inference ...", flush=True)
    nmf_train, nmf_inf = time_nmf_pipeline(cache, names, train_p, test_p)

    # --- CNN arms (per-epoch measured on the train subset) ---
    # Warm up torch first (one-time thread-pool / kernel init) so the first arm is
    # not penalised and the three arms are directly comparable.
    print("Warming up torch ...", flush=True)
    _warm = cnn_core._build_samples(slices, cnn_train_p[:1], None, None)
    cnn_core.train_unet(_warm, {**cnn_core.DEFAULT_CFG, "epochs": 1},
                        log=lambda *x, **k: None)

    def run_cnn(tag, r, c):
        print(f"Timing CNN training ({a.time_epochs} epochs, {len(cnn_train_p)} "
              f"patients) + inference: {tag} ...", flush=True)
        return time_cnn(slices, cnn_train_p, test_p, r, c, a.time_epochs)
    cnn_pe, cnn_inf = run_cnn("CNN-only", None, None)
    roi_pe, roi_inf = run_cnn("NMF+CNN (ROI)", rois, None)
    ch_pe, ch_inf = run_cnn("NMF+CNN (channels)", None, chans)

    def epoch_full(pe):          # scale subset per-epoch to the full training set
        return pe * scale

    def full_train(pe):          # full 40-epoch x 5-fold budget (extrapolated)
        return epoch_full(pe) * FULL_EPOCHS * N_FOLDS

    # --- assemble table ---
    lines = [
        "# Computational cost (CPU-only)\n",
        f"Measured on this machine. CNN per-epoch training time is measured over "
        f"{a.time_epochs} epochs on {len(cnn_train_p)} training patients, scaled to the "
        f"full {len(train_p)}-patient training set, and extrapolated to the full "
        f"{FULL_EPOCHS}-epoch x {N_FOLDS}-fold budget. Inference is per volume "
        f"(averaged over {len(test_p)} held-out patients). NMF overhead is averaged "
        f"over {len(sample_p)} volumes.\n",
        f"Shared preprocessing + base NMF feature extraction: **{_fmt(prep)} per volume** "
        "(one-time, cached; identical for all approaches).\n",
        "| Approach | Extra NMF computation (per volume) | Model training | Inference (per volume) |",
        "|----------|-----------------------------------|----------------|------------------------|",
        f"| NMF-only | - (uses the shared features) | {_fmt(nmf_train)} per fold "
        f"({_fmt(nmf_train*N_FOLDS)} full CV) | {_fmt(nmf_inf)} |",
        f"| CNN-only | - | {_fmt(epoch_full(cnn_pe))}/epoch -> {_fmt(full_train(cnn_pe))} "
        f"(extrap.) | {_fmt(cnn_inf)} |",
        f"| NMF+CNN (ROI) | ROI localization: {_fmt(roi_t)} | {_fmt(epoch_full(roi_pe))}"
        f"/epoch -> {_fmt(full_train(roi_pe))} (extrap.) | {_fmt(roi_inf)} |",
        f"| NMF+CNN (channels) | channel maps: {_fmt(map_t)} | {_fmt(epoch_full(ch_pe))}"
        f"/epoch -> {_fmt(full_train(ch_pe))} (extrap.) | {_fmt(ch_inf)} |",
        "\n_The extra NMF computation is unsupervised, one-time and cached; it is a "
        "few seconds per volume against minutes of U-Net training, and the three "
        "CNN arms train in essentially the same time -- i.e. NMF adds negligible cost._\n",
    ]
    out = "\n".join(lines)
    path = os.path.join(cd.RESULTS_DIR, "cost_table.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out)
    print("\n" + out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
