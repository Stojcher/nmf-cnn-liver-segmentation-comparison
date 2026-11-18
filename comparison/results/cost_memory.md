# Memory and preprocessing/NMF timing supplement (CPU-only)

Preprocessing and NMF-feature computation times are measured separately on one volume (previously reported together). Peak RAM is the peak resident set size of the process; each approach is measured in its own process, and the NMF-only arm is measured without the PyTorch stack (its pipeline does not use it).

**Shared per-volume feature preparation:** preprocessing 18.1 s + NMF feature computation 15.4 s (one-time, cached; identical for all approaches).

**U-Net model size:** 1,942,289 parameters (7.4 MB) for the CNN-only and ROI arms (single-channel input), and 1,943,009 parameters (7.4 MB) for the channel-fusion arm (6-channel input); the NMF-only model is a gradient-boosted tree ensemble with a random-forest gate.

| Approach | Peak RAM (training) | Peak RAM (inference) |
|---|---|---|
| NMF-only (no PyTorch) | 2.11 GB | 1.36 GB |
| CNN-only | 928 MB | 835 MB |
| NMF+CNN (ROI) | 908 MB | 814 MB |
| NMF+CNN (channel fusion) | 1.17 GB | 1.02 GB |

_Peak RAM is the absolute process footprint (the CNN arms load the full deep-learning stack). The channel-fusion arm adds only 720 parameters over the single-channel U-Net, and the three CNN arms use comparable memory -- i.e. NMF adds negligible memory cost._

