# Computational cost (CPU-only)

Measured on this machine. CNN per-epoch training time is measured over 3 epochs on 4 training patients, scaled to the full 16-patient training set, and extrapolated to the full 40-epoch x 5-fold budget. Inference is per volume (averaged over 4 held-out patients). NMF overhead is averaged over 3 volumes.

Shared preprocessing + base NMF feature extraction: **80.9 s per volume** (one-time, cached; identical for all approaches).

| Approach | Extra NMF computation (per volume) | Model training | Inference (per volume) |
|----------|-----------------------------------|----------------|------------------------|
| NMF-only | - (uses the shared features) | 2.7 min per fold (13.4 min full CV) | 9.3 s |
| CNN-only | - | 37.8 s/epoch -> 126.0 min (extrap.) | 1.9 s |
| NMF+CNN (ROI) | ROI localization: 17.4 s | 31.9 s/epoch -> 106.3 min (extrap.) | 1.7 s |
| NMF+CNN (channels) | channel maps: 17.2 s | 34.2 s/epoch -> 114.0 min (extrap.) | 2.7 s |

_The extra NMF computation is unsupervised, one-time and cached; it is a few seconds per volume against minutes of U-Net training, and the three CNN arms train in essentially the same time -- i.e. NMF adds negligible cost._