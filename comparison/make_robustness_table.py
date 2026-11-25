# -*- coding: utf-8 -*-
"""
Generate the motion robustness table (paper Table 4) from the full-run CSV.

Reads comparison/results/robustness_stress.csv and writes a Markdown table of
mean (worst-case) volumetric Dice per approach, per severity, for one
perturbation (default: motion). No recompute -- it only aggregates saved scores.

    python comparison/make_robustness_table.py                 # motion (Table 4)
    python comparison/make_robustness_table.py --perturbation noise
"""
import os
import csv
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "results", "robustness_stress.csv")

ARM_ORDER = ["CNN-only", "NMF+CNN (ROI)", "NMF+CNN (channels)"]
LABEL = {"CNN-only": "CNN-only", "NMF+CNN (ROI)": "NMF+CNN (ROI)",
         "NMF+CNN (channels)": "NMF+CNN (channel fusion)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perturbation", default="motion",
                    choices=["motion", "noise", "bias"])
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(CSV_PATH, newline="", encoding="utf-8"))
            if r["perturbation"] == a.perturbation]
    sevs = sorted({float(r["severity"]) for r in rows})

    def cell(arm, s):
        vals = [float(r["dice_full"]) for r in rows
                if r["approach"] == arm and float(r["severity"]) == s]
        return f"{np.mean(vals):.3f} ({np.min(vals):.3f})"

    hdr = ["Approach", "Clean"] + [f"{s:g}" for s in sevs if s > 0]
    lines = ["| " + " | ".join(hdr) + " |",
             "|" + "---|" * len(hdr)]
    for arm in ARM_ORDER:
        cells = [LABEL[arm], cell(arm, 0.0)] + [cell(arm, s) for s in sevs if s > 0]
        lines.append("| " + " | ".join(cells) + " |")

    caption = (f"Table 4. Mean and worst-case (minimum) volumetric Dice under "
               f"synthetic {a.perturbation} corruption, as a function of severity. "
               f"Each cell is mean (worst-case) across the 20 patients.")
    out = caption + "\n\n" + "\n".join(lines) + "\n"
    path = os.path.join(HERE, "results", f"table4_{a.perturbation}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
