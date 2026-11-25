# Per-volume feature-preparation timing breakdown (CPU-only)

Mean over 2 volumes; mirrors the one-time, cached cache-build (build_patient). The components sum to the total shared preparation step.

| Stage | Time per volume |
|---|---|
| DICOM load | 113 ms |
| Preprocessing (body mask, N4, NL-means, normalize) | 16.9 s |
| Handcrafted feature extraction (classifier; incl. local entropy) | 1.2 s |
| NMF feature computation + decomposition | 17.4 s |
| **Total shared preparation** | **35.7 s** |

