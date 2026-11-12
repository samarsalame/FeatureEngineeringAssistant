"""Architecture diagram of the Feature Engineering Assistant (Fig. 1 of the SoftwareX paper)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "fig_architecture.png")

INK, INK2, EDGE = "#0b0b0b", "#52514e", "#898781"
BLUE, BLUE_BG = "#2a78d6", "#e8f1fc"
ORANGE, ORANGE_BG = "#eb6834", "#fdeee7"
GREY_BG = "#f0efec"
plt.rcParams.update({"font.family": "Times New Roman"})

fig, ax = plt.subplots(figsize=(8.6, 4.9))
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")


def box(x, y, w, h, title, sub, fc, ec, title_color=INK, tsize=7.6, ssize=6.0, sub_drop=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=tsize, weight="bold",
            color=title_color, linespacing=1.1)
    ax.text(x + w / 2, y + h - sub_drop, sub, ha="center", va="top", fontsize=ssize, color=INK2,
            linespacing=1.3)


def arrow(x1, y1, x2, y2, color=EDGE, style="-|>", lw=1.0, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


# GUI shell
ax.add_patch(FancyBboxPatch((0.8, 0.8), 98.4, 60.4, boxstyle="round,pad=0.2,rounding_size=1.5",
                            fc="white", ec=EDGE, lw=0.9, ls="--"))
ax.text(2.5, 59.4, "Tkinter GUI (Main.py): sidebar actions, tabs, pop-ups, background threads",
        fontsize=8, color=INK2, va="top", style="italic")

# pipeline modules (top row)
mods = [
    ("Load &\nprofile", "dataset.py\ncolumn_\nprofile.py"),
    ("Data\ncleaning", "data_pre-\nprocessing.py"),
    ("Feature\ncreation", "PCA · ICA\n· autoencoder"),
    ("Manual\npruning", "Pruning.py"),
    ("Feature\nselection", "MI · RF · RFE\nK by CV"),
    ("Evaluate\nimpact", "k-fold CV\nt-test · d"),
    ("Report\nexport", "export_\nreport.py"),
]
w, h, y = 12.2, 17.5, 36.5
xs = [2.6 + i * 13.8 for i in range(len(mods))]
for x, (t, s) in zip(xs, mods):
    box(x, y, w, h, t, s, GREY_BG, EDGE)
for i in range(len(xs) - 1):
    arrow(xs[i] + w + 0.25, y + h / 2, xs[i + 1] - 0.25, y + h / 2, color=INK2, lw=1.1)

# shared state
sx, sy, sw, sh = 6, 15.5, 88, 11.5
ax.add_patch(FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0.25,rounding_size=1.2",
                            fc=BLUE_BG, ec=BLUE, lw=1.4))
ax.text(50, sy + sh - 2.2, "Shared dataset state (dataset.py)", ha="center", va="top",
        fontsize=9.2, weight="bold", color=BLUE)
ax.text(50, sy + sh - 6.6,
        "CURRENT_DF (working data)  ·  ORIGINAL_DF (after loading)  ·  POST_CREATION_DF (after feature creation)"
        "\nBANNED_FEATURES (pruned columns)  ·  TASK_TYPE  ·  metadata (types, roles, missing, unique)",
        ha="center", va="top", fontsize=7, color=INK, linespacing=1.35)

# read/write arrows module <-> state
for x in xs:
    cx = x + w / 2
    arrow(cx - 1.1, y - 0.3, cx - 1.1, sy + sh + 0.35, color=BLUE, lw=0.9)          # write
    arrow(cx + 1.1, sy + sh + 0.35, cx + 1.1, y - 0.3, color=BLUE, lw=0.9)          # read
ax.text(xs[0] + w / 2 + 2.2, (y + sy + sh) / 2, "read /\nwrite", fontsize=6, color=BLUE, ha="left", va="center")

# bottom row: explorer + snapshots, read-only
box(10, 2.3, 38, 10.2, "Latent Features Explorer (see_latent.py)",
    "distributions · latent heatmap · correlation with\noriginal features · RF importance",
    ORANGE_BG, ORANGE, tsize=7.8, ssize=6.2, sub_drop=4.6)
box(52, 2.3, 38, 10.2, "Snapshots → stage datasets",
    "baseline · pruned · post-creation · selected,\nre-evaluated with the same CV model",
    GREY_BG, EDGE, tsize=7.8, ssize=6.2, sub_drop=4.6)
arrow(29, sy - 0.3, 29, 12.9, color=ORANGE, lw=1.0)
arrow(71, sy - 0.3, 71, 12.9, color=INK2, lw=1.0)
ax.text(50, sy + sh - 2.2, "", fontsize=1)

fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT)
