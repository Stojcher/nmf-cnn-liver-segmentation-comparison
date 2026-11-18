# Table 5 (consolidated): computational cost and memory per approach (CPU-only)

Auto-assembled by make_cost_table.py from cost_table.md, cost_memory.md and cost_timing_breakdown.md. Columns map 1:1 to the reviewer's request.

| Approach | Preprocessing (per vol) | NMF computation (per vol) | Training | Inference (per vol) | Peak RAM (training / inference) |
|---|---|---|---|---|---|
| NMF-only | 16.9 s | 17.4 s | 1.6 min per fold (7.9 min full CV) | 4.9 s | 2.11 GB / 1.36 GB |
| CNN-only | 16.9 s | – | 23.4 s/epoch → 78.0 min (extrap.) | 1.2 s | 928 MB / 835 MB |
| NMF+CNN (ROI) | 16.9 s | 9.1 s | 21.2 s/epoch → 70.7 min (extrap.) | 1.1 s | 908 MB / 814 MB |
| NMF+CNN (channel fusion) | 16.9 s | 9.0 s | 23.8 s/epoch → 79.4 min (extrap.) | 1.8 s | 1.17 GB / 1.02 GB |

Preprocessing (body mask, N4, denoising, normalization) is shared by all approaches, computed once per volume and cached (~16.9 s). NMF computation is one-time and cached: the NMF-only pipeline's feature NMF (~17.4 s, max_iter=1000) versus the lighter NMF the hybrids use for the ROI/channel maps (max_iter=200). Training and inference times are extrapolated/averaged as in cost_table.md. Peak RAM is the peak resident-set size (NMF-only measured without the PyTorch stack).

U-Net model size: 1,942,289 parameters (7.4 MB) for the CNN-only and ROI arms (single-channel input), and 1,943,009 parameters (7.4 MB) for the channel-fusion arm (6-channel input); the NMF-only model is a gradient-boosted tree ensemble with a random-forest gate.

