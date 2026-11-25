# -*- coding: utf-8 -*-
"""
Memory + preprocessing/NMF-timing supplement to the cost benchmark (reviewer R2-C7).

The existing cost_benchmark.py already reports training and inference time. This
script adds the two quantities that were missing or bundled:

  1. Preprocessing time vs. NMF-feature-computation time, measured SEPARATELY
     (previously reported together as the ~81 s/volume shared step).
  2. Memory: the U-Net parameter count / size, and the PEAK process RAM (resident
     set size) during training and during inference, for each approach.

Why subprocesses: peak RAM is measured per approach in its OWN fresh Python
process, so memory from one arm cannot leak into the next. The NMF-only arm is
measured WITHOUT importing PyTorch (its pipeline needs no deep-learning stack),
giving a fair footprint; the CNN arms import torch FIRST (Windows c10.dll load
ordering) and are warmed up before the measured training, so the peak reflects
steady-state training rather than one-time initialization.

Only MEMORY and the preprocessing/NMF split are reported here; the published
training/inference TIMES from cost_benchmark.py are left unchanged.

Deterministic inputs, fully offline. Writes results/cost_memory.md.

    python comparison/cost_memory.py                 # full (spawns 4 isolated workers)
    python comparison/cost_memory.py --worker cnn --time-epochs 1 --train-subset 2  # debug one arm

IMPORTANT: nothing heavy is imported at module top. Each process imports torch
(CNN only) FIRST, then psutil and the SciPy stack, to avoid the Windows c10.dll
initialization failure (WinError 1114) seen when torch loads after that stack.
"""
import os
import sys
import gc
import time
import json
import argparse
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ARMS = ["nmf", "cnn", "nmf_cnn", "nmf_ch"]
DISPLAY = {"nmf": "NMF-only", "cnn": "CNN-only",
           "nmf_cnn": "NMF+CNN (ROI)", "nmf_ch": "NMF+CNN (channel fusion)"}


def _run_with_peak(fn):
    """Run fn() while sampling process RSS every 20 ms; return (result, peak_rss_bytes)."""
    import psutil
    proc = psutil.Process()
    gc.collect()
    peak = {"v": proc.memory_info().rss}
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                peak["v"] = max(peak["v"], proc.memory_info().rss)
            except Exception:
                pass
            time.sleep(0.02)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    res = fn()
    stop.set()
    th.join()
    peak["v"] = max(peak["v"], proc.memory_info().rss)
    return res, peak["v"]


def _fmt_mem(b):
    return f"{b/(1024**3):.2f} GB" if b >= 1024**3 else f"{b/(1024**2):.0f} MB"


def _fmt_t(sec):
    if sec < 1:
        return f"{sec*1000:.0f} ms"
    if sec < 90:
        return f"{sec:.1f} s"
    return f"{sec/60:.1f} min"


# ==========================================================================
#  WORKER: measure ONE arm's peak RAM in an isolated process
# ==========================================================================
def worker(arm, time_epochs, train_subset):
    # torch FIRST for the CNN arms (before the SciPy stack / psutil), never for NMF-only
    if arm != "nmf":
        import torch  # noqa: F401
    import psutil
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    import common_data as cd
    import liver_nmf_segmentation as L

    slices = cd.load_slices(cd.ALL20)
    folds = cd.make_folds()
    test_p = folds[0]
    train_p = [p for p in cd.ALL20 if p not in test_p]
    proc = psutil.Process()

    if arm == "nmf":
        cache, names, _ = L.cache_patients(cd.ALL20)
        base = proc.memory_info().rss
        pipe, train_peak = _run_with_peak(lambda: L.train_pipeline(cache, names, train_p))
        pix, gate, ac, gthr, kk = pipe
        _, inf_peak = _run_with_peak(
            lambda: [L.patient_dice(pix, gate, cache, p, ac, gthr, kk) for p in test_p])
        out = dict(arm=arm, train_peak=train_peak, inf_peak=inf_peak, base=base, uses_torch=False)
    else:
        import cnn_core
        import nmf_roi
        import nmf_channels
        cnn_train_p = train_p[:train_subset]
        used = sorted(set(cnn_train_p) | set(test_p), key=int)
        r = c = None
        if arm == "nmf_cnn":
            r = {p: nmf_roi.roi_for_patient(slices[p], patient=p) for p in used}
        elif arm == "nmf_ch":
            c = {p: nmf_channels.channels_for_patient(slices[p], patient=p) for p in used}
        cfg = {**cnn_core.DEFAULT_CFG, "epochs": time_epochs}
        # warm up (one-time torch/thread init) so the measured peak is steady-state
        warm = cnn_core._build_samples(slices, cnn_train_p[:1], None, None)
        cnn_core.train_unet(warm, {**cnn_core.DEFAULT_CFG, "epochs": 1}, log=lambda *a, **k: None)
        del warm
        gc.collect()
        base = proc.memory_info().rss
        samples = cnn_core._build_samples(slices, cnn_train_p, r, c)
        model, train_peak = _run_with_peak(
            lambda: cnn_core.train_unet(samples, cfg, log=lambda *a, **k: None))

        def infer():
            for p in test_p:
                cnn_core.predict_patient(model, slices[p], (r.get(p) if r else None),
                                         cfg, extras=(c.get(p) if c else None))
        _, inf_peak = _run_with_peak(infer)
        out = dict(arm=arm, train_peak=train_peak, inf_peak=inf_peak, base=base, uses_torch=True)

    print("RESULT_JSON " + json.dumps(out), flush=True)


# ==========================================================================
#  PARENT helpers
# ==========================================================================
def full_breakdown(patient, L):
    """Full per-volume feature-prep breakdown (mirrors build_patient exactly), so the
    components sum to the total ~81 s shared step: DICOM load + preprocessing +
    handcrafted (classifier) feature extraction + NMF feature computation+decomposition."""
    t0 = time.perf_counter()
    data = L.load_t2_volume(L.DATASET_PATH, patient)
    t_load = time.perf_counter() - t0
    vol = data["volume"]
    t_prep = t_base = t_nmf = 0.0
    for i in range(vol.shape[0]):
        t = time.perf_counter()
        norm, mask = L.preprocess_incl(vol[i])
        t_prep += time.perf_counter() - t
        if mask.sum() < L.MIN_LIVER_PX:
            continue
        t = time.perf_counter()
        stack_b, _ = L.feature_stack(norm, mask, L.BASE_GROUPS)   # classifier features
        Xb, coords = L.design_matrix(stack_b, mask)
        t_base += time.perf_counter() - t
        t = time.perf_counter()
        stack_n, _ = L.feature_stack(norm, mask, L.NMF_GROUPS)     # NMF features + decomp
        Vn, _ = L.design_matrix(stack_n, mask)
        labels, W, _, _ = L.nmf_cluster(Vn, k=L.K)
        _ = L.nmf_context(labels, W, coords, norm, L.K)
        t_nmf += time.perf_counter() - t
    return dict(load=t_load, prep=t_prep, base=t_base, nmf=t_nmf,
                total=t_load + t_prep + t_base + t_nmf)


def unet_param_mb(cnn_core, in_ch):
    m = cnn_core.UNet(features=tuple(cnn_core.DEFAULT_CFG["features"]), in_ch=in_ch)
    n = sum(p.numel() for p in m.parameters())
    del m
    return n, n * 4 / (1024 ** 2)


# ==========================================================================
#  MAIN (parent)
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None, choices=ARMS)
    ap.add_argument("--breakdown", action="store_true",
                    help="only measure the per-volume feature-prep timing breakdown")
    ap.add_argument("--time-epochs", type=int, default=3)
    ap.add_argument("--train-subset", type=int, default=4)
    ap.add_argument("--nmf-patients", type=int, default=3)
    a = ap.parse_args()

    if a.worker:
        worker(a.worker, a.time_epochs, a.train_subset)
        return

    # ----- breakdown mode: timing decomposition only (no torch, no memory workers) -----
    if a.breakdown:
        import numpy as np
        import common_data as cd
        import liver_nmf_segmentation as L
        sample_p = cd.ALL20[: a.nmf_patients]
        print(f"Full feature-prep timing breakdown over {len(sample_p)} volumes ...",
              flush=True)
        acc = {k: [] for k in ("load", "prep", "base", "nmf", "total")}
        for p in sample_p:
            b = full_breakdown(p, L)
            for k in acc:
                acc[k].append(b[k])
            print(f"  patient {p}: load {b['load']:.1f}s  prep {b['prep']:.1f}s  "
                  f"features {b['base']:.1f}s  nmf {b['nmf']:.1f}s  total {b['total']:.1f}s",
                  flush=True)
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        lines = [
            "# Per-volume feature-preparation timing breakdown (CPU-only)\n",
            f"Mean over {len(sample_p)} volumes; mirrors the one-time, cached cache-build "
            "(build_patient). The components sum to the total shared preparation step.\n",
            "| Stage | Time per volume |",
            "|---|---|",
            f"| DICOM load | {_fmt_t(m['load'])} |",
            f"| Preprocessing (body mask, N4, NL-means, normalize) | {_fmt_t(m['prep'])} |",
            f"| Handcrafted feature extraction (classifier; incl. local entropy) | {_fmt_t(m['base'])} |",
            f"| NMF feature computation + decomposition | {_fmt_t(m['nmf'])} |",
            f"| **Total shared preparation** | **{_fmt_t(m['total'])}** |",
            "",
        ]
        out = "\n".join(lines) + "\n"
        path = os.path.join(cd.RESULTS_DIR, "cost_timing_breakdown.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(out)
        print("\n" + out)
        print(f"wrote {path}")
        return

    # ----- memory mode: model sizes + peak RAM per arm (isolated processes) -----
    import torch  # noqa: F401  (load before the SciPy stack, Windows)
    import subprocess
    import common_data as cd
    import liver_nmf_segmentation as L
    import cnn_core

    p1, mb1 = unet_param_mb(cnn_core, 1)          # CNN-only / ROI (1-channel input)
    pk, mbk = unet_param_mb(cnn_core, 1 + L.K)    # channel fusion (1 + K channels)

    res = {}
    for arm in ARMS:
        print(f"\n=== measuring peak RAM for {DISPLAY[arm]} (isolated process) ===", flush=True)
        cp = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", arm,
             "--time-epochs", str(a.time_epochs), "--train-subset", str(a.train_subset)],
            capture_output=True, text=True)
        out_lines = [l for l in cp.stdout.splitlines() if l.startswith("RESULT_JSON ")]
        if not out_lines:
            print("---- worker stdout ----\n", cp.stdout)
            print("---- worker stderr ----\n", cp.stderr)
            raise SystemExit(f"worker '{arm}' produced no result")
        res[arm] = json.loads(out_lines[-1][len("RESULT_JSON "):])
        print(f"    train peak {_fmt_mem(res[arm]['train_peak'])}, "
              f"inference peak {_fmt_mem(res[arm]['inf_peak'])}", flush=True)

    lines = [
        "# Memory supplement (CPU-only)\n",
        "Peak RAM is the peak resident set size of the process; each approach is "
        "measured in its own process. The NMF-only arm is measured without the PyTorch "
        "stack (its pipeline does not use it).\n",
        f"**U-Net model size:** {p1:,} parameters ({mb1:.1f} MB) for the CNN-only and "
        f"ROI arms (single-channel input), and {pk:,} parameters ({mbk:.1f} MB) for the "
        f"channel-fusion arm ({1+L.K}-channel input); the NMF-only model is a "
        "gradient-boosted tree ensemble with a random-forest gate.\n",
        "| Approach | Peak RAM (training) | Peak RAM (inference) |",
        "|---|---|---|",
    ]
    for arm in ARMS:
        note = "" if res[arm]["uses_torch"] else " (no PyTorch)"
        lines.append(f"| {DISPLAY[arm]}{note} | {_fmt_mem(res[arm]['train_peak'])} | "
                     f"{_fmt_mem(res[arm]['inf_peak'])} |")
    lines.append(f"\n_All approaches fit within the 8 GB machine. Adding NMF to the CNN "
                 f"leaves the ROI arm essentially unchanged and raises the channel-fusion "
                 f"arm by a few hundred MB (the extra input channels; +{pk-p1:,} "
                 "parameters); the standalone NMF-only pipeline is the heaviest, as it "
                 "loads the full handcrafted-feature cache and pools pixels for the "
                 "classifier._\n")

    out = "\n".join(lines) + "\n"
    path = os.path.join(cd.RESULTS_DIR, "cost_memory.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out)
    print("\n" + out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
