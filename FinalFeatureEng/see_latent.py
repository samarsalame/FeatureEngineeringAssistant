# ===============================================================
# see_latent.py
# Latent Features Explorer Tab
#
# Works with:
#   - PCA latent_* (latent_pca_*)
#   - ICA latent_* (latent_ica_*)
#   - Deep tabular autoencoder latent_* (latent_auto_*)
#
# Provides:
#   - Distribution plots for each latent feature
#   - Correlation heatmap between latent features
#   - Correlation of one latent feature with original features
#   - RF importance comparison (latent vs original)
# ===============================================================

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer

import dataset

matplotlib.use("TkAgg")
sns.set_theme(style="darkgrid")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _get_df():
    return dataset.get_current_df()


def _get_latent_columns(df: pd.DataFrame):
    """All columns whose name starts with 'latent_'."""
    return [c for c in df.columns if str(c).startswith("latent_")]


def _get_original_numeric_columns(df: pd.DataFrame):
    """Numeric columns that are NOT latent_*."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if not str(c).startswith("latent_")]


# ---------------------------------------------------------------------
# Main entry: add_latent_tab
# ---------------------------------------------------------------------
def add_latent_tab(notebook: ttk.Notebook, card_bg: str = "#20232a") -> tk.Frame:
    """
    Creates a new 'Latent Features' tab inside the given ttk.Notebook.

    Usage in main2.py:
        import see_latent
        latent_tab = see_latent.add_latent_tab(notebook, card_bg)
    """
    tab = tk.Frame(notebook, bg=card_bg)
    notebook.add(tab, text="Latent Features")

    tab.grid_rowconfigure(2, weight=1)
    tab.grid_columnconfigure(0, weight=1)

    # -----------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------
    header_frame = tk.Frame(tab, bg=card_bg)
    header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 0))
    header_frame.grid_columnconfigure(0, weight=1)

    lbl_title = tk.Label(
        header_frame,
        text="Latent Features Explorer",
        bg=card_bg,
        fg="white",
        font=("Segoe UI", 11, "bold"),
    )
    lbl_title.grid(row=0, column=0, sticky="w")

    chip = tk.Label(
        header_frame,
        text="PCA · ICA · Deep Autoencoder",
        bg="#111317",
        fg="#9da5b4",
        font=("Segoe UI", 8),
        padx=8,
        pady=3,
    )
    chip.grid(row=0, column=1, sticky="e")

    lbl_sub = tk.Label(
        tab,
        text="Inspect latent distributions, correlations and their relationship with original features.",
        bg=card_bg,
        fg="#9da5b4",
        font=("Segoe UI", 9),
    )
    lbl_sub.grid(row=1, column=0, sticky="w", padx=10, pady=(4, 6))

    # -----------------------------------------------------------------
    # Controls + Plot area
    # -----------------------------------------------------------------
    main_pane = tk.Frame(tab, bg=card_bg)
    main_pane.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
    main_pane.grid_rowconfigure(0, weight=1)
    main_pane.grid_columnconfigure(1, weight=1)

    controls = tk.Frame(main_pane, bg=card_bg)
    controls.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
    plot_frame = tk.Frame(main_pane, bg=card_bg)
    plot_frame.grid(row=0, column=1, sticky="nsew")

    # -----------------------------------------------------------------
    # Matplotlib Figure
    # -----------------------------------------------------------------
    fig = Figure(figsize=(7.0, 4.5), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)

    # -----------------------------------------------------------------
    # Controls: latent selector, plot types, buttons
    # -----------------------------------------------------------------
    # latent feature dropdown
    tk.Label(
        controls,
        text="Latent feature:",
        bg=card_bg,
        fg="#d7dae0",
        font=("Segoe UI", 9, "bold"),
    ).grid(row=0, column=0, sticky="w", pady=(2, 2))

    latent_var = tk.StringVar()
    latent_combo = ttk.Combobox(
        controls,
        textvariable=latent_var,
        state="readonly",
        width=26,
    )
    latent_combo.grid(row=1, column=0, sticky="w", pady=(0, 6))

    def refresh_latent_list():
        df = _get_df()
        if df is None:
            return
        cols = _get_latent_columns(df)
        if not cols:
            messagebox.showinfo(
                "No latent features",
                "No columns starting with 'latent_' were found.\n"
                "Try running Feature Creation first.",
            )
            latent_combo["values"] = []
            latent_var.set("")
            return

        # Sort so autoencoder features appear first, then PCA, then ICA
        cols_sorted = sorted(
            cols,
            key=lambda c: (0 if c.startswith("latent_auto_") else
                           1 if c.startswith("latent_pca_") else
                           2 if c.startswith("latent_ica_") else 3, c),
        )
        latent_combo["values"] = cols_sorted
        latent_var.set(cols_sorted[0])

    btn_refresh = tk.Button(
        controls,
        text="Refresh list",
        command=refresh_latent_list,
        bg="#4C8BF5",
        fg="white",
        relief="flat",
        font=("Segoe UI", 8, "bold"),
        padx=6,
        pady=3,
        cursor="hand2",
    )
    btn_refresh.grid(row=1, column=1, padx=(6, 0), sticky="w")

    # --- single-latent plot type ---
    tk.Label(
        controls,
        text="Single latent plot:",
        bg=card_bg,
        fg="#d7dae0",
        font=("Segoe UI", 9, "bold"),
    ).grid(row=2, column=0, sticky="w", pady=(10, 2))

    plot_type = tk.StringVar(value="hist")

    rb_hist = tk.Radiobutton(
        controls, text="Histogram", variable=plot_type, value="hist",
        bg=card_bg, fg="#d7dae0", selectcolor=card_bg, anchor="w"
    )
    rb_kde = tk.Radiobutton(
        controls, text="KDE", variable=plot_type, value="kde",
        bg=card_bg, fg="#d7dae0", selectcolor=card_bg, anchor="w"
    )
    rb_scatter = tk.Radiobutton(
        controls, text="Scatter vs index", variable=plot_type, value="scatter",
        bg=card_bg, fg="#d7dae0", selectcolor=card_bg, anchor="w"
    )
    rb_hist.grid(row=3, column=0, sticky="w")
    rb_kde.grid(row=4, column=0, sticky="w")
    rb_scatter.grid(row=5, column=0, sticky="w")

    def plot_single_latent():
        df = _get_df()
        if df is None:
            return
        col = latent_var.get()
        if not col:
            messagebox.showwarning("No feature selected", "Please select a latent feature.")
            return
        if col not in df.columns:
            messagebox.showerror("Missing column", f"Column '{col}' not found in dataframe.")
            return

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            messagebox.showinfo("No data", f"Column '{col}' has no numeric values.")
            return

        fig.clear()
        ax = fig.add_subplot(111)
        p = plot_type.get()

        if p == "hist":
            sns.histplot(s, kde=True, ax=ax)
            ax.set_ylabel("Count")
        elif p == "kde":
            sns.kdeplot(s, fill=True, ax=ax)
            ax.set_ylabel("Density")
        else:  # scatter
            ax.scatter(range(len(s)), s, s=10)
            ax.set_xlabel("Index")
            ax.set_ylabel(col)

        ax.set_title(f"{col} — {p}")
        fig.tight_layout()
        canvas.draw()

    btn_plot_single = tk.Button(
        controls,
        text="Plot selected latent",
        command=plot_single_latent,
        bg="#8C53FF",
        fg="white",
        relief="flat",
        font=("Segoe UI", 9, "bold"),
        padx=8,
        pady=4,
        cursor="hand2",
    )
    btn_plot_single.grid(row=6, column=0, columnspan=2, sticky="w", pady=(6, 10))

    # --- correlation heatmap ---
    tk.Label(
        controls,
        text="Latent correlations:",
        bg=card_bg,
        fg="#d7dae0",
        font=("Segoe UI", 9, "bold"),
    ).grid(row=7, column=0, sticky="w", pady=(4, 2))

    def plot_latent_heatmap():
        df = _get_df()
        if df is None:
            return
        latent_cols = _get_latent_columns(df)
        if len(latent_cols) < 2:
            messagebox.showinfo(
                "Too few latent features",
                "Need at least 2 latent_* columns for a correlation heatmap.",
            )
            return

        sub = df[latent_cols].select_dtypes(include=[np.number]).copy()
        if sub.empty:
            messagebox.showinfo("No numeric data", "Latent columns are not numeric.")
            return

        corr = sub.corr()

        fig.clear()
        ax = fig.add_subplot(111)
        sns.heatmap(
            corr,
            ax=ax,
            annot=False,
            cmap="viridis",
            square=True,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Correlation between latent features")
        fig.tight_layout()
        canvas.draw()

    btn_heatmap = tk.Button(
        controls,
        text="Latent Corr Heatmap",
        command=plot_latent_heatmap,
        bg="#00B894",
        fg="white",
        relief="flat",
        font=("Segoe UI", 9, "bold"),
        padx=8,
        pady=4,
        cursor="hand2",
    )
    btn_heatmap.grid(row=8, column=0, columnspan=2, sticky="w", pady=(2, 8))

    # --- explainability: correlation with original features ---
    tk.Label(
        controls,
        text="Explainability (corr with original):",
        bg=card_bg,
        fg="#d7dae0",
        font=("Segoe UI", 9, "bold"),
    ).grid(row=9, column=0, sticky="w", pady=(4, 2))

    def plot_explainability():
        df = _get_df()
        if df is None:
            return
        col = latent_var.get()
        if not col:
            messagebox.showwarning("No feature selected", "Please select a latent feature.")
            return
        if col not in df.columns:
            messagebox.showerror("Missing column", f"Column '{col}' not found in dataframe.")
            return

        orig_cols = _get_original_numeric_columns(df)
        if not orig_cols:
            messagebox.showinfo(
                "No original numeric features",
                "No non-latent numeric columns were found.",
            )
            return

        sub = df[orig_cols + [col]].dropna()
        if sub.empty:
            messagebox.showinfo(
                "No complete rows",
                "No rows where latent + original features are all non-missing.",
            )
            return

        corr = sub[orig_cols].corrwith(sub[col])
        corr = corr.dropna()
        if corr.empty:
            messagebox.showinfo("No correlation", "Could not compute correlations.")
            return

        corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
        top = corr.head(20)

        fig.clear()
        ax = fig.add_subplot(111)
        ax.barh(top.index[::-1], top.values[::-1])
        ax.set_xlabel("Correlation with " + col)
        ax.set_title(f"Original features most correlated with {col}")
        fig.tight_layout()
        canvas.draw()

    btn_explain = tk.Button(
        controls,
        text="Explain selected latent",
        command=plot_explainability,
        bg="#FF7A59",
        fg="white",
        relief="flat",
        font=("Segoe UI", 9, "bold"),
        padx=8,
        pady=4,
        cursor="hand2",
    )
    btn_explain.grid(row=10, column=0, columnspan=2, sticky="w", pady=(2, 8))

    # --- feature importance comparison (latent vs original) ---
    tk.Label(
        controls,
        text="Importance (RF, optional target):",
        bg=card_bg,
        fg="#d7dae0",
        font=("Segoe UI", 9, "bold"),
    ).grid(row=11, column=0, sticky="w", pady=(4, 2))

    target_var = tk.StringVar()
    target_combo = ttk.Combobox(
        controls,
        textvariable=target_var,
        state="readonly",
        width=26,
    )
    target_combo.grid(row=12, column=0, sticky="w", pady=(0, 4))

    def refresh_targets():
        df = _get_df()
        if df is None:
            return
        candidates = [c for c in df.columns if not str(c).startswith("latent_")]
        target_combo["values"] = candidates
        if candidates:
            target_var.set(candidates[-1])  # often target is last
        else:
            target_var.set("")

    btn_refresh_targets = tk.Button(
        controls,
        text="Refresh targets",
        command=refresh_targets,
        bg="#444444",
        fg="white",
        relief="flat",
        font=("Segoe UI", 8, "bold"),
        padx=6,
        pady=3,
        cursor="hand2",
    )
    btn_refresh_targets.grid(row=12, column=1, padx=(6, 0), sticky="w")

    def plot_importance():
        df = _get_df()
        if df is None:
            return
        target = target_var.get()
        if not target:
            messagebox.showwarning(
                "No target selected",
                "Select a target column (or set one in Control Settings).",
            )
            return
        if target not in df.columns:
            messagebox.showerror("Missing column", f"Target '{target}' not found in dataframe.")
            return

        y = df[target]
        X = df.drop(columns=[target])

        # only numeric predictors (original + latent)
        X = X.select_dtypes(include=[np.number])
        if X.empty:
            messagebox.showinfo(
                "No numeric predictors",
                "No numeric predictor columns were found.",
            )
            return

        imputer = SimpleImputer(strategy="median")
        X_imp = imputer.fit_transform(X)

        # choose classifier vs regressor
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
            model = RandomForestRegressor(n_estimators=200, random_state=42)
        else:
            y_clean = y.astype("category").cat.codes
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            y = y_clean

        try:
            model.fit(X_imp, y)
        except Exception as e:
            messagebox.showerror("Model error", f"Could not fit RandomForest:\n{e}")
            return

        importances = pd.Series(model.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)
        top = importances.head(25)

        fig.clear()
        ax = fig.add_subplot(111)

        # Color latent_* vs original features differently
        colors = [
            "#FF7A59" if name.startswith("latent_") else "#4C8BF5"
            for name in top.index
        ]
        ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
        ax.set_xlabel("RandomForest importance")
        ax.set_title(
            f"Feature importance vs target '{target}'\n"
            "(orange = latent, blue = original)"
        )
        fig.tight_layout()
        canvas.draw()

    btn_importance = tk.Button(
        controls,
        text="Compare importance",
        command=plot_importance,
        bg="#FF5EAA",
        fg="white",
        relief="flat",
        font=("Segoe UI", 9, "bold"),
        padx=8,
        pady=4,
        cursor="hand2",
    )
    btn_importance.grid(row=13, column=0, columnspan=2, sticky="w", pady=(4, 4))

    # initial (safe) population of lists
    refresh_latent_list()
    refresh_targets()

    return tab
