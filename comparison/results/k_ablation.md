# NMF component-count ablation (NMF-only pipeline)

5-fold patient-level cross-validation, varying the number of NMF components k. k = 5 is the value used in the paper (reproduces the published 0.743). Only the NMF-only arm is ablated.

| k | full-volume Dice | liver-only Dice | median | min | #>=0.70 |
|---|---|---|---|---|---|
| 3 | 0.746 ± 0.123 | 0.780 ± 0.108 | 0.800 | 0.379 | 16/20 |
| 4 | 0.760 ± 0.114 | 0.783 ± 0.106 | 0.802 | 0.361 | 16/20 |
| 5  (used) | 0.743 ± 0.128 | 0.777 ± 0.123 | 0.780 | 0.282 | 17/20 |
| 6 | 0.723 ± 0.121 | 0.762 ± 0.118 | 0.755 | 0.277 | 14/20 |

_Full-volume mean Dice varies by 0.037 across k = 3–6._

