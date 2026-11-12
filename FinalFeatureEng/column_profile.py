# Enhanced column_profile.py with seaborn plots (Colab style)
# Automated profiling reports are triggered from the PLOTS TAB
# This file ONLY handles column-level seaborn visualizations.

import tkinter as tk
from tkinter import messagebox
from typing import Optional

import seaborn as sns
import pandas as pd
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import dataset

# =============================
# NEW IMPORTS (optional)
# =============================
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except:
    WORDCLOUD_AVAILABLE = False

matplotlib.use("TkAgg")
sns.set_theme(style="whitegrid", palette="muted")


# -------------------------------------------------------------------------
# Helper: read metadata for column
# -------------------------------------------------------------------------
def _get_column_meta(col_name: str) -> Optional[dict]:
    meta = dataset.get_current_meta()
    if not meta:
        return None
    for c in meta.get("columns", []):
        if c.get("name") == col_name:
            return c
    return None


def _infer_role(series: pd.Series, col_meta: Optional[dict]) -> str:
    if col_meta and col_meta.get("role"):
        return str(col_meta["role"])
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


# -------------------------------------------------------------------------
# COLUMN PROFILING WINDOW (with seaborn high-level EDA)
# -------------------------------------------------------------------------
def show_column_profile(parent: tk.Tk | tk.Toplevel, col_name: str) -> None:
    df = dataset.get_current_df()
    if df is None:
        messagebox.showerror("No dataset", "Please load a dataset first.")
        return

    if col_name not in df.columns:
        messagebox.showerror("Column not found", f"Column '{col_name}' not in dataframe.")
        return

    s = df[col_name]
    col_meta = _get_column_meta(col_name)

    non_null_count = int(s.notna().sum())
    total = len(s)
    n_unique = int(s.nunique(dropna=True))
    n_missing = total - non_null_count

    dtype = str(s.dtype)
    role = _infer_role(s, col_meta)

    is_numeric = pd.api.types.is_numeric_dtype(s)

    # Numeric summary
    if is_numeric:
        s_numeric = pd.to_numeric(s, errors="coerce").dropna()
        if len(s_numeric) > 0:
            min_val = s_numeric.min()
            max_val = s_numeric.max()
            mean_val = s_numeric.mean()
            std_val = s_numeric.std()
        else:
            min_val = max_val = mean_val = std_val = None
    else:
        s_numeric = None
        min_val = max_val = mean_val = std_val = None

    # ---------------------------------------------------------------------
    # Window layout
    # ---------------------------------------------------------------------
    win = tk.Toplevel(parent)
    win.title(f"Profile — {col_name}")
    win.configure(bg="#111317")
    win.geometry("900x650")
    win.minsize(800, 550)

    # SUMMARY FRAME
    summary_frame = tk.Frame(win, bg="#111317", padx=14, pady=12)
    summary_frame.pack(fill="x")

    tk.Label(summary_frame, text=col_name, fg="white", bg="#111317",
             font=("Segoe UI", 13, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 6))

    def row(label, value, r):
        tk.Label(summary_frame, text=label, bg="#111317", fg="#9da5b4",
                 font=("Segoe UI", 9)).grid(row=r, column=0, sticky="w")
        tk.Label(summary_frame, text=value, bg="#111317", fg="#d7dae0",
                 font=("Segoe UI", 9)).grid(row=r, column=1, sticky="w")

    row("Role:", role, 1)
    row("Dtype:", dtype, 2)
    row("Non-null count:", f"{non_null_count} / {total}", 3)
    row("Missing:", f"{n_missing} ({(n_missing/total)*100:.2f}%)", 4)
    row("Unique values:", str(n_unique), 5)

    if is_numeric:
        row("Min:", "–" if min_val is None else f"{min_val:.6g}", 6)
        row("Max:", "–" if max_val is None else f"{max_val:.6g}", 7)
        row("Mean:", "–" if mean_val is None else f"{mean_val:.6g}", 8)
        row("Std:", "–" if std_val is None else f"{std_val:.6g}", 9)

    # ---------------------------------------------------------------------
    # PLOTTING AREA
    # ---------------------------------------------------------------------
    plot_frame = tk.Frame(win, bg="#111317")
    plot_frame.pack(fill="both", expand=True, padx=16, pady=(4, 16))

    fig = Figure(figsize=(8.2, 4.8), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas_widget = canvas.get_tk_widget()

    # Controls
    controls_frame = tk.Frame(plot_frame, bg="#111317", pady=4)
    controls_frame.pack(fill="x", side="top")

    tk.Label(controls_frame, text="Plot:", bg="#111317", fg="#d7dae0",
             font=("Segoe UI", 9, "bold"))

    # =====================================================================
    # NUMERIC PLOTS (unchanged)
    # =====================================================================
    if is_numeric:
        plot_var = tk.StringVar(value="Histogram")

        for name in ["Histogram", "KDE", "Box", "Violin", "Scatter", "Line"]:
            tk.Radiobutton(
                controls_frame,
                text=name,
                variable=plot_var,
                value=name,
                bg="#111317", fg="#d7dae0",
                activebackground="#111317",
                selectcolor="#111317",
                command=lambda: update_numeric_plot()
            ).pack(side="left", padx=4)

        canvas_widget.pack(fill="both", expand=True, side="top")

        def update_numeric_plot():
            fig.clear()
            ax = fig.add_subplot(111)

            if s_numeric is None or len(s_numeric) == 0:
                ax.text(0.5, 0.5, "No numeric values", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_axis_off()
                fig.tight_layout(); canvas.draw(); return

            p = plot_var.get()

            if p == "Histogram":
                sns.histplot(s_numeric, kde=True, ax=ax)

            elif p == "KDE":
                sns.kdeplot(s_numeric, fill=True, ax=ax)

            elif p == "Box":
                sns.boxplot(x=s_numeric, ax=ax)

            elif p == "Violin":
                sns.violinplot(x=s_numeric, ax=ax)

            elif p == "Scatter":
                sns.scatterplot(x=list(range(len(s_numeric))), y=s_numeric, ax=ax, s=20)
                ax.set_xlabel("Index")

            elif p == "Line":
                ax.plot(s_numeric.values, linewidth=1.3)
                ax.set_xlabel("Index")

            ax.set_title(f"{col_name} — {p}")
            fig.tight_layout(); canvas.draw()

        update_numeric_plot()

    # =====================================================================
    # CATEGORICAL PLOTS (extends original with 2 new options)
    # =====================================================================
    else:
        plot_var = tk.StringVar(value="Bar")
        vc = s.astype("string").value_counts().head(12)

        # original options
        cat_plot_list = ["Bar", "Horizontal", "Pie", "Table"]

        # =============================
        # NEW OPTIONS
        # =============================
        cat_plot_list.append("Top-N Categories")
        if WORDCLOUD_AVAILABLE:
            cat_plot_list.append("WordCloud")

        for name in cat_plot_list:
            tk.Radiobutton(
                controls_frame,
                text=name,
                variable=plot_var,
                value=name,
                bg="#111317", fg="#d7dae0",
                activebackground="#111317",
                selectcolor="#111317",
                command=lambda: update_cat_plot()
            ).pack(side="left", padx=4)

        canvas_widget.pack(fill="both", expand=True, side="top")

        def update_cat_plot():
            fig.clear()
            ax = fig.add_subplot(111)

            if vc.empty:
                ax.text(0.5, 0.5, "No categorical values.", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_axis_off()
                fig.tight_layout(); canvas.draw(); return

            cats = vc.index.tolist()
            counts = vc.values
            p = plot_var.get()

            # =============================
            # ORIGINAL PLOTS
            # =============================
            if p == "Bar":
                sns.barplot(x=cats, y=counts, ax=ax)
                ax.set_xticklabels(cats, rotation=45, ha="right")
                ax.set_ylabel("Count")

            elif p == "Horizontal":
                sns.barplot(x=counts, y=cats, ax=ax)
                ax.set_xlabel("Count")

            elif p == "Pie":
                wedges, _, autotxt = ax.pie(
                    counts,
                    autopct="%1.1f%%",
                    startangle=90,
                    normalize=True,
                    colors=sns.color_palette("muted", len(counts)),
                )
                for t in autotxt:
                    t.set_fontsize(8)
                ax.axis("equal")

            elif p == "Table":
                ax.axis("off")
                table_data = [[c, int(n)] for c, n in zip(cats, counts)]
                tbl = ax.table(
                    cellText=table_data,
                    colLabels=["Category", "Count"],
                    loc="center",
                    cellLoc="left",
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(9)
                tbl.scale(1, 1.3)
                fig.subplots_adjust(top=0.85)

            # =============================
            # NEW: Top-N Categories
            # =============================
            elif p == "Top-N Categories":
                top_vc = s.astype("string").value_counts().head(20)
                sns.barplot(x=top_vc.index, y=top_vc.values, ax=ax)
                ax.set_xticklabels(top_vc.index, rotation=45, ha="right")
                ax.set_ylabel("Count")

            # =============================
            # NEW: WordCloud
            # =============================
            elif p == "WordCloud":
                if not WORDCLOUD_AVAILABLE:
                    ax.text(
                        0.5, 0.5,
                        "Install wordcloud library to enable WordCloud.",
                        ha="center", va="center",
                        transform=ax.transAxes,
                    )
                    ax.axis("off")
                else:
                    text = " ".join(s.astype("string").dropna())
                    wc = WordCloud(width=900, height=450, background_color="white")
                    img = wc.generate(text)
                    ax.imshow(img, interpolation="bilinear")
                    ax.axis("off")

            ax.set_title(f"{col_name} — {p}")
            fig.tight_layout()
            canvas.draw()

        update_cat_plot()

    win.transient(parent)
    win.grab_set()

