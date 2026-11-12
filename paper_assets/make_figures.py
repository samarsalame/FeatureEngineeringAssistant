"""
Build the paper figures from the GUI screenshots and the benchmark results.

Usage:  python paper_assets/make_figures.py      (after capture_gui.py and benchmark.py)
Output: paper_assets/figures/*.png  (300 dpi)
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
SHOTS = os.path.join(HERE, "screenshots")
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(PROJECT, "FinalFeatureEng"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont

# ---- palette (reference categorical slots, light mode) + chrome ----------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]  # secondary encoding: identity never colour-only
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


# ---- 1. composite screenshot panels -------------------------------------
def _font(size):
    for f in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def side_by_side(files, out, height=900, gap=30):
    imgs = [Image.open(os.path.join(SHOTS, f)).convert("RGB") for f in files]
    imgs = [im.resize((int(im.width * height / im.height), height), Image.LANCZOS) for im in imgs]
    pad_top = 70
    W = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new("RGB", (W, height + pad_top), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for i, im in enumerate(imgs):
        canvas.paste(im, (x, pad_top))
        draw.text((x + 4, 8), f"({chr(97 + i)})", fill=INK, font=_font(48))
        x += im.width + gap
    canvas.save(os.path.join(FIG, out))
    print("saved", out, canvas.size)


def stacked(files, out, crop=None, gap=30):
    imgs = [Image.open(os.path.join(SHOTS, f)).convert("RGB") for f in files]
    if crop:
        imgs = [im.crop(crop) for im in imgs]
    pad_left = 90
    H = sum(im.height for im in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new("RGB", (max(im.width for im in imgs) + pad_left, H), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for i, im in enumerate(imgs):
        canvas.paste(im, (pad_left, y))
        draw.text((4, y + 4), f"({chr(97 + i)})", fill=INK, font=_font(48))
        y += im.height + gap
    canvas.save(os.path.join(FIG, out))
    print("saved", out, canvas.size)


side_by_side(["02a_profile_numeric.png", "02b_profile_categorical.png"], "fig_profiles.png")
# Latent Features Explorer: controls + plot area only (drop app sidebar/header)
stacked(["06c_latent_explain.png", "06d_latent_importance.png"], "fig_latent_explorer.png",
        crop=(232, 156, 1432, 888))
side_by_side(["05a_feature_creation_popup.png", "07_pruning.png"], "fig_creation_pruning.png")
side_by_side(["10a_confusion_matrix.png", "10b_roc_pr.png"], "fig_confusion_roc.png", height=600)
for f in ["01_load_preview", "03_cleaning", "06c_latent_explain", "06d_latent_importance",
          "08_feature_selection", "09_evaluate_impact", "04_control_settings"]:
    Image.open(os.path.join(SHOTS, f + ".png")).convert("RGB").save(os.path.join(FIG, f + ".png"))


# ---- 2. benchmark dot plot (small multiples: F1 | ROC-AUC) ---------------
summ = pd.read_csv(os.path.join(RES, "summary.csv"))
models = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
configs = ["Raw", "Latent only", "Raw + Latent", "Selected"]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
for ax, (met, label) in zip(axes, [("f1", "Weighted F1-score"), ("roc_auc", "ROC-AUC")]):
    for j, cfg in enumerate(configs):
        sub = summ[summ.config == cfg].set_index("model").loc[models]
        ypos = np.arange(len(models)) + (j - 1.5) * 0.18
        ax.errorbar(sub[f"{met}_mean"], ypos, xerr=sub[f"{met}_ci"], fmt=MARKERS[j],
                    color=SERIES[j], ms=5, lw=1.2, capsize=0,
                    mec="white", mew=0.6, label=cfg)
    ax.set_xlabel(label)
    ax.grid(axis="y", visible=False)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, color=INK)
axes[0].invert_yaxis()
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.55, 1.06), handletextpad=0.3, columnspacing=1.2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_benchmark.png"))
plt.close(fig)
print("saved fig_benchmark.png")


# ---- 3. selection stability ---------------------------------------------
freq = pd.read_csv(os.path.join(RES, "selection_frequency.csv"), index_col=0)["frequency"]
top = freq.head(25)[::-1]
is_lat = [n.startswith("latent_") for n in top.index]
fig, ax = plt.subplots(figsize=(4.6, 4.4))
ax.barh(range(len(top)), top.values, height=0.72,
        color=[SERIES[1] if l else SERIES[0] for l in is_lat],
        hatch=None, edgecolor="white", linewidth=0.6)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top.index, color=INK)
ax.set_xlim(0, 1.0)
ax.set_xlabel("Selection frequency across 30 outer training folds")
ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=SERIES[0], label="Original feature"),
                   Patch(color=SERIES[1], label="Latent feature")],
          loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_selection_stability.png"))
plt.close(fig)
print("saved fig_selection_stability.png")


# ---- 4. latent-to-original correlation map (the GUI session's own features) --
post = pd.read_csv(os.path.join(RES, "gui_post_creation.csv"))
lat = post[[c for c in post.columns if c.startswith("latent_")]]
X = post.drop(columns=list(lat.columns) + ["diagnosis"])
corr = pd.DataFrame({c: X.corrwith(lat[c]) for c in lat.columns}).T  # 16 x 30
div = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
fig, ax = plt.subplots(figsize=(6.6, 3.6))
im = ax.imshow(corr.values, cmap=div, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(corr.shape[1]))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=7, color=INK)
ax.set_yticks(range(corr.shape[0]))
ax.set_yticklabels(corr.index, fontsize=7, color=INK)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb.set_label("Pearson r", color=INK2)
cb.outline.set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_latent_correlation.png"))
plt.close(fig)
corr.to_csv(os.path.join(RES, "latent_original_correlation.csv"))
print("saved fig_latent_correlation.png")
