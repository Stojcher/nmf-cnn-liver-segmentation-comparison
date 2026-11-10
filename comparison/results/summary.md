# Arm comparison (5-fold patient-level CV, CHAOS T2-SPIR)

Shared patients: n = 20

| Arm | Full-vol Dice | Liver-only Dice | median (full) | min | #>=0.70 |
|-----|---------------|-----------------|---------------|-----|---------|
| NMF-only | 0.743 +/- 0.128 | 0.777 +/- 0.123 | 0.780 | 0.282 | 17/20 |
| CNN-only | 0.855 +/- 0.179 | 0.866 +/- 0.181 | 0.894 | 0.084 | 19/20 |
| NMF+CNN (ROI) | 0.862 +/- 0.170 | 0.874 +/- 0.169 | 0.908 | 0.153 | 19/20 |
| NMF+CNN (ch) | 0.868 +/- 0.139 | 0.874 +/- 0.139 | 0.908 | 0.278 | 19/20 |

## Paired Wilcoxon signed-rank (full-volume Dice)

Holm-Bonferroni adjusts for the 6 pairwise comparisons; significance is assessed on the adjusted p-values.

| Comparison | median diff | p-value | Holm-adj. p |
|------------|-------------|---------|-------------|
| NMF-only vs CNN-only | -0.110 | 0.0006 | 0.0023 |
| NMF-only vs NMF+CNN (ROI) | -0.117 | 0.0001 | 0.0005 |
| NMF-only vs NMF+CNN (ch) | -0.115 | 0.0000 | 0.0000 |
| CNN-only vs NMF+CNN (ROI) | -0.012 | 0.4749 | 1.0000 |
| CNN-only vs NMF+CNN (ch) | -0.002 | 0.5459 | 1.0000 |
| NMF+CNN (ROI) vs NMF+CNN (ch) | +0.007 | 0.8408 | 1.0000 |
