# Computational cost (CPU-only)

Measured on this machine. CNN per-epoch training time is measured over 3 epochs on 4 training patients, scaled to the full 16-patient training set, and extrapolated to the full 40-epoch x 5-fold budget. Inference is per volume (averaged over 4 held-out patients). NMF overhead is averaged over 3 volumes.

Shared preprocessing + base NMF feature extraction: **40.8 s per volume** (one-time, cached; identical for all approaches).

| Approach | Extra NMF computation (per volume) | Model training | Inference (per volume) |
|----------|-----------------------------------|----------------|------------------------|
| NMF-only | - (uses the shared features) | 1.6 min per fold (7.9 min full CV) | 4.9 s |
| CNN-only | - | 23.4 s/epoch -> 78.0 min (extrap.) | 1.2 s |
| NMF+CNN (ROI) | ROI localization: 9.1 s | 21.2 s/epoch -> 70.7 min (extrap.) | 1.1 s |
| NMF+CNN (channels) | channel maps: 9.0 s | 23.8 s/epoch -> 79.4 min (extrap.) | 1.8 s |

_The extra NMF computation is unsupervised, one-time and cached; it is a few seconds per volume against minutes of U-Net training, and the three CNN arms train in essentially the same time -- i.e. NMF adds negligible cost._
