# Effect sizes and 95% confidence intervals (full-volume Dice)

Paired per-patient differences over n=20 patients; 10000 bootstrap resamples. Positive mean/median = the first approach scores higher. A tight CI around 0 is evidence of a small true effect; a wide CI shows the sample cannot rule one out.


## CNN-arm comparisons (the non-significant results under scrutiny)

| Comparison | n | mean diff (A-B) | 95% CI (BCa) | median diff | 95% CI | Wilcoxon p | Cohen's dz |
|---|---|---|---|---|---|---|---|
| CNN-only vs NMF+CNN (ROI) | 20 | -0.008 | [-0.025, +0.013] | -0.012 | [-0.026, +0.016] | 0.4749 | -0.17 |
| CNN-only vs NMF+CNN (channels) | 20 | -0.013 | [-0.049, +0.001] | -0.002 | [-0.010, +0.007] | 0.5459 | -0.27 |
| NMF+CNN (ROI) vs NMF+CNN (channels) | 20 | -0.006 | [-0.030, +0.013] | +0.007 | [-0.014, +0.015] | 0.8408 | -0.11 |

## CNN-arm comparisons, excluding the p15 outlier (n=19)

| Comparison | n | mean diff (A-B) | 95% CI (BCa) | median diff | 95% CI | Wilcoxon p | Cohen's dz |
|---|---|---|---|---|---|---|---|
| CNN-only vs NMF+CNN (ROI) | 19 | -0.004 | [-0.022, +0.016] | -0.007 | [-0.025, +0.024] | 0.6794 | -0.10 |
| CNN-only vs NMF+CNN (channels) | 19 | -0.004 | [-0.016, +0.005] | +0.001 | [-0.009, +0.007] | 0.7983 | -0.15 |
| NMF+CNN (ROI) vs NMF+CNN (channels) | 19 | +0.001 | [-0.019, +0.017] | +0.010 | [-0.012, +0.016] | 0.5412 | +0.02 |

## NMF-only vs CNN-based (context: the large, significant gap)

| Comparison | n | mean diff (A-B) | 95% CI (BCa) | median diff | 95% CI | Wilcoxon p | Cohen's dz |
|---|---|---|---|---|---|---|---|
| NMF-only vs CNN-only | 20 | -0.112 | [-0.146, -0.059] | -0.110 | [-0.145, -0.094] | 0.0006 | -1.15 |
| NMF-only vs NMF+CNN (ROI) | 20 | -0.120 | [-0.152, -0.078] | -0.117 | [-0.156, -0.086] | 0.0001 | -1.38 |
| NMF-only vs NMF+CNN (channels) | 20 | -0.125 | [-0.159, -0.099] | -0.115 | [-0.147, -0.092] | 0.0000 | -1.81 |
