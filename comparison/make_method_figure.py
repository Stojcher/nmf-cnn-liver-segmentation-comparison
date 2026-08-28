# -*- coding: utf-8 -*-
"""Block-diagram figure of the four liver-segmentation approaches."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

# colours by stage type
C_PRE = "#cfe3f7"   # shared preprocessing
C_FEAT = "#d6efe9"  # feature extraction / resize / crop (shared-ish)
C_NMF = "#c9e8c9"   # NMF (unsupervised)
C_ML = "#e6d6f2"    # classical ML (classifier / gate / refine)
C_CNN = "#fbe0c3"   # U-Net
C_POST = "#f4edc4"  # threshold / post-process
C_OUT = "#dcdcdc"   # output mask

BW, BH, GAP, X0 = 2.35, 1.5, 0.55, 0.5

fig, ax = plt.subplots(figsize=(15.5, 8.6))
ax.set_xlim(0, 21); ax.set_ylim(0, 13.2); ax.axis("off")


def box(x, y, text, fc, h=BH, fs=8.6):
    ax.add_patch(FancyBboxPatch((x, y), BW, h,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 fc=fc, ec="#333333", lw=1.2))
    ax.text(x + BW / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, wrap=True, zorder=3)


def arrow(x1, x2, y):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.6))


def arrow2(x1, y1p, x2, y2p):
    ax.annotate("", xy=(x2, y2p), xytext=(x1, y1p),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.5))


def pipeline(y, label, items):
    ax.text(X0, y + BH + 0.32, label, ha="left", va="bottom",
            fontsize=12, fontweight="bold", color="#1f3864")
    x = X0
    centers = []
    for text, fc in items:
        box(x, y, text, fc)
        centers.append((x, x + BW))
        x += BW + GAP
    for i in range(len(items) - 1):
        arrow(centers[i][1], centers[i + 1][0], y + BH / 2)


fig.suptitle("Four liver-segmentation approaches (shared preprocessing and evaluation)",
             fontsize=15, fontweight="bold", y=0.98)

# Row 1 (custom): base features + NMF cluster-context features are computed in
# parallel and CONCATENATED -> classifier -> candidate mask (threshold 0.7 +
# morphology) -> presence gate -> refinement.  (Matches build_patient / build_rows.)
y1 = 11.0
cy = y1 + BH / 2
ax.text(X0, y1 + BH + 0.32, "1.  NMF-only", ha="left", va="bottom",
        fontsize=12, fontweight="bold", color="#1f3864")
xp = [X0 + i * (BW + GAP) for i in range(7)]
hh = 0.70
box(xp[0], y1, "T2-SPIR slice\n+ Preprocessing", C_PRE)
box(xp[1], y1 + BH - hh, "Base per-pixel\nfeatures", C_FEAT, h=hh, fs=7.9)       # upper stream
box(xp[1], y1, "NMF (k = 5)\n→ cluster context", C_NMF, h=hh, fs=7.9)       # lower stream
box(xp[2], y1, "Gradient-boosting\npixel classifier", C_ML)
box(xp[3], y1, "Candidate mask\n(threshold 0.7\n+ morphology)", C_POST)
box(xp[4], y1, "Presence gate\n(random forest)", C_ML)
box(xp[5], y1, "Intensity-gating\nrefinement", C_ML)
box(xp[6], y1, "Liver\nmask", C_OUT)
yu, yl = y1 + BH - hh / 2, y1 + hh / 2
arrow2(xp[0] + BW, cy, xp[1], yu); arrow2(xp[0] + BW, cy, xp[1], yl)   # preproc -> both streams
arrow2(xp[1] + BW, yu, xp[2], cy); arrow2(xp[1] + BW, yl, xp[2], cy)   # streams -> classifier
for i in range(2, 6):
    arrow(xp[i] + BW, xp[i + 1], cy)

pipeline(8.4, "2.  CNN-only", [
    ("T2-SPIR slice\n+ Preprocessing", C_PRE),
    ("Resize\n128 × 128", C_FEAT),
    ("2D U-Net", C_CNN),
    ("Threshold 0.5\n+ post-process", C_POST),
    ("Liver\nmask", C_OUT),
])

pipeline(5.8, "3.  NMF+CNN  —  ROI cascade", [
    ("T2-SPIR slice\n+ Preprocessing", C_PRE),
    ("NMF ROI\nlocalizer\n(unsupervised)", C_NMF),
    ("Crop to ROI\n+ resize", C_FEAT),
    ("2D U-Net", C_CNN),
    ("Paste back\n+ post-process", C_POST),
    ("Liver\nmask", C_OUT),
])

pipeline(3.2, "4.  NMF+CNN  —  channel fusion", [
    ("T2-SPIR slice\n+ Preprocessing", C_PRE),
    ("NMF component\nmaps (k)  \u2295  T2\n\u2192 6 channels", C_NMF),
    ("2D U-Net\n(6-channel input)", C_CNN),
    ("Threshold 0.5\n+ post-process", C_POST),
    ("Liver\nmask", C_OUT),
])

# legend
legend_items = [
    Patch(fc=C_PRE, ec="#333", label="Shared preprocessing"),
    Patch(fc=C_FEAT, ec="#333", label="Features / resize / crop"),
    Patch(fc=C_NMF, ec="#333", label="NMF (unsupervised)"),
    Patch(fc=C_ML, ec="#333", label="Classical ML"),
    Patch(fc=C_CNN, ec="#333", label="U-Net (shared, arms 2\u20134)"),
    Patch(fc=C_POST, ec="#333", label="Threshold / post-process"),
    Patch(fc=C_OUT, ec="#333", label="Output"),
]
ax.legend(handles=legend_items, loc="lower center", ncol=7, fontsize=9,
          frameon=False, bbox_to_anchor=(0.5, -0.02))

ax.text(10.5, 1.55, "All approaches use the same preprocessing and the same 5-fold "
        "patient-level cross-validation with volumetric Dice;\narms 2\u20134 share the "
        "identical U-Net and training procedure.",
        ha="center", va="center", fontsize=9.5, style="italic", color="#444444")

import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results final", "method_overview.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)
