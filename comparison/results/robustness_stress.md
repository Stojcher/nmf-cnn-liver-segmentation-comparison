# Synthetic robustness stress test

Three CNN-based approaches; corruption injected on the raw image and the full pipeline (preprocessing + NMF + U-Net) re-run; cached fold models, no retraining. Cells are **mean Dice (worst-case Dice)** over the evaluated patients.


## Harness check (clean baseline)

| Approach | clean mean, this run | published (Table 1) |
|---|---|---|
| CNN-only | 0.855 (n=20) | 0.855 |
| NMF+CNN (ROI) | 0.862 (n=20) | 0.862 |
| NMF+CNN (channels) | 0.868 (n=20) | 0.868 |

## noise

| Approach | sev 0 | sev 0.02 | sev 0.05 | sev 0.1 | sev 0.15 |
|---|---|---|---|---|---|
| CNN-only | 0.855 (0.084) | 0.856 (0.085) | 0.855 (0.105) | 0.841 (0.136) | 0.815 (0.085) |
| NMF+CNN (ROI) | 0.862 (0.153) | 0.866 (0.179) | 0.867 (0.339) | 0.811 (0.058) | 0.725 (0.060) |
| NMF+CNN (channels) | 0.868 (0.278) | 0.862 (0.306) | 0.868 (0.345) | 0.815 (0.273) | 0.715 (0.109) |

_Paired Wilcoxon vs CNN-only (median Dice diff, p) at each severity > 0 — noise:_

| Comparison | sev 0.02 | sev 0.05 | sev 0.1 | sev 0.15 |
|---|---|---|---|---|
| NMF+CNN (ROI) vs CNN-only | +0.014, p=0.245 | +0.008, p=0.546 | -0.011, p=0.312 | -0.045, p=0.058 |
| NMF+CNN (channels) vs CNN-only | -0.002, p=0.841 | -0.002, p=0.927 | -0.008, p=0.202 | -0.079, p=0.000 |

## bias

| Approach | sev 0 | sev 0.1 | sev 0.2 | sev 0.3 | sev 0.4 |
|---|---|---|---|---|---|
| CNN-only | 0.855 (0.084) | 0.853 (0.081) | 0.854 (0.084) | 0.853 (0.081) | 0.849 (0.077) |
| NMF+CNN (ROI) | 0.862 (0.153) | 0.866 (0.216) | 0.832 (0.000) | 0.868 (0.193) | 0.867 (0.190) |
| NMF+CNN (channels) | 0.868 (0.278) | 0.866 (0.294) | 0.863 (0.342) | 0.869 (0.390) | 0.856 (0.256) |

_Paired Wilcoxon vs CNN-only (median Dice diff, p) at each severity > 0 — bias:_

| Comparison | sev 0.1 | sev 0.2 | sev 0.3 | sev 0.4 |
|---|---|---|---|---|
| NMF+CNN (ROI) vs CNN-only | +0.016, p=0.312 | +0.008, p=0.869 | +0.019, p=0.245 | +0.015, p=0.058 |
| NMF+CNN (channels) vs CNN-only | -0.000, p=0.648 | -0.000, p=0.956 | +0.001, p=0.927 | -0.004, p=0.729 |

## motion

| Approach | sev 0 | sev 0.1 | sev 0.2 | sev 0.3 | sev 0.4 |
|---|---|---|---|---|---|
| CNN-only | 0.855 (0.084) | 0.758 (0.030) | 0.647 (0.051) | 0.542 (0.046) | 0.423 (0.021) |
| NMF+CNN (ROI) | 0.862 (0.153) | 0.745 (0.019) | 0.578 (0.000) | 0.360 (0.000) | 0.187 (0.000) |
| NMF+CNN (channels) | 0.868 (0.278) | 0.719 (0.257) | 0.630 (0.346) | 0.540 (0.162) | 0.455 (0.134) |

_Paired Wilcoxon vs CNN-only (median Dice diff, p) at each severity > 0 — motion:_

| Comparison | sev 0.1 | sev 0.2 | sev 0.3 | sev 0.4 |
|---|---|---|---|---|
| NMF+CNN (ROI) vs CNN-only | +0.017, p=0.312 | +0.016, p=0.546 | -0.145, p=0.005 | -0.281, p=0.001 |
| NMF+CNN (channels) vs CNN-only | -0.026, p=0.027 | -0.036, p=0.143 | -0.020, p=0.596 | +0.048, p=0.133 |
