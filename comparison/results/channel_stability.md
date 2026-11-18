# Channel-ordering stability (NMF channel fusion)

For each of the 401 liver-containing slices across the 20 patients, the liver's dominant NMF component was mapped to its intensity-sorted channel index (0 = darkest, 4 = brightest). Under an arbitrary (unsorted) ordering each index would be equally likely (~20%).

| Sorted channel | 0 (darkest) | 1 | 2 | 3 | 4 (brightest) |
|---|---|---|---|---|---|
| Slices (%) | 26.9 | 23.4 | 8.0 | 17.0 | 24.7 |
| Slices (n) | 108 | 94 | 32 | 68 | 99 |

The liver maps to channel 0 most often (26.9%) and to the two darkest channels (0-1) in 50.4% of slices, above the 40% expected under an arbitrary ordering. Mean per-patient consistency (largest single-channel share) is 39.7%, and 0/20 patients reach 80% single-channel consistency. The sorting thus imposes a consistent intensity ordering under which the liver concentrates in the darker channels, rather than confining it to a single fixed channel.

## Per-patient consistency

| Patient | n slices | modal channel | consistency (%) |
|---|---|---|---|
| 1 | 24 | 2 | 25.0 |
| 2 | 19 | 0 | 57.9 |
| 3 | 16 | 4 | 43.8 |
| 5 | 24 | 0 | 37.5 |
| 8 | 26 | 0 | 42.3 |
| 10 | 21 | 4 | 38.1 |
| 13 | 18 | 1 | 27.8 |
| 15  (outlier) | 17 | 0 | 58.8 |
| 19 | 21 | 1 | 52.4 |
| 20 | 18 | 1 | 38.9 |
| 21 | 18 | 4 | 33.3 |
| 22 | 19 | 1 | 36.8 |
| 31 | 17 | 4 | 35.3 |
| 32 | 28 | 0 | 35.7 |
| 33 | 22 | 1 | 27.3 |
| 34 | 19 | 3 | 42.1 |
| 36 | 20 | 0 | 50.0 |
| 37 | 16 | 0 | 31.2 |
| 38 | 18 | 4 | 44.4 |
| 39 | 20 | 4 | 35.0 |
