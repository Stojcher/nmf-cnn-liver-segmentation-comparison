# Additional segmentation metrics (held-out, 5-fold CV, n=20)

Overlap/clinical metrics are pooled per patient over all slices (as the published volumetric Dice). Boundary metrics (HD95, ASSD) are 3D symmetric surface distances in millimetres, using DICOM voxel spacing. Values are mean +/- population SD over the 20 patients.

| Approach | Dice | IoU | Sensitivity | Precision | HD95 (mm) | ASSD (mm) |
|---|---|---|---|---|---|---|
| NMF-only | 0.743 +/- 0.128 | 0.604 +/- 0.134 | 0.831 +/- 0.167 | 0.692 +/- 0.096 | 51.77 +/- 20.22 | 13.52 +/- 6.29 |
| CNN-only | 0.855 +/- 0.179 | 0.773 +/- 0.174 | 0.851 +/- 0.193 | 0.888 +/- 0.077 | 47.35 +/- 58.25 | 8.41 +/- 8.66 |
| NMF+CNN (ROI) | 0.862 +/- 0.170 | 0.784 +/- 0.179 | 0.846 +/- 0.190 | 0.919 +/- 0.043 | 31.72 +/- 33.93 | 6.82 +/- 7.47 |
| NMF+CNN (channel fusion) | 0.868 +/- 0.139 | 0.785 +/- 0.153 | 0.857 +/- 0.167 | 0.906 +/- 0.028 | 23.61 +/- 29.47 | 5.36 +/- 5.61 |

## Dice reproduction check (vs. Table 1)

| Approach | this script | published | match |
|---|---|---|---|
| NMF-only | 0.743 | 0.743 | OK |
| CNN-only | 0.855 | 0.855 | OK |
| NMF+CNN (ROI) | 0.862 | 0.862 | OK |
| NMF+CNN (channel fusion) | 0.868 | 0.868 | OK |

_Reproduction: ALL MATCH._

