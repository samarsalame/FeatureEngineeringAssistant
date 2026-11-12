import tkinter as tk
from tkinter import ttk, messagebox

import threading, traceback

import dataset      # must be in same folder
import column_profile  # <-- NEW: for column-level popup
from feature_creation import run_feature_creation
from feature_creation import generate_autoencoder_latent
from feature_selection import run_feature_selection
import datareport
import Pruning  # for pruning popup
import see_latent
import evaluate_impact  
import export_report
import data_preprocessing
import dataset_downloads

# ------------------------------------------------------------------
# GLOBAL APP SETTINGS
# ------------------------------------------------------------------
APP_SETTINGS = {
    "target": None,
    "test_size": 0.2,
    "model_type": "Random Forest",
    "task_type": None,  # "regression", "classification", or None (auto-detect)
    "top_k": None,      # NEW: fixed K for feature selection (None => auto CV)
}

class ToolTip:
    """Very small tooltip helper for Tkinter widgets."""

    def __init__(self, widget, text, wraplength=260):
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self.tipwindow = None

        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event=None):
        if self.tipwindow is not None:
            return

        # Position: just below and slightly right of the widget
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)   # no window decorations
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#111317",
            foreground="#e5e7eb",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 8),
            wraplength=self.wraplength,
            padx=6,
            pady=4,
        )
        label.pack(ipadx=1)

    def _hide(self, event=None):
        if self.tipwindow is not None:
            self.tipwindow.destroy()
            self.tipwindow = None


# --- Main window ---
root = tk.Tk()
root.title("Feature Engineering Assistant")
root.geometry("1100x650")  # width x height
root.config(bg="#050816")  # slightly deeper background

# --- Overall layout: sidebar (left) + main content (right) ---
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)

# ========== LEFT SIDEBAR ==========
sidebar = tk.Frame(root, bg="#111317", width=220)
sidebar.grid(row=0, column=0, sticky="ns")
sidebar.grid_propagate(False)  # Fix width

app_title = tk.Label(
    sidebar,
    text="Feature Engineering\nAssistant",
    bg="#111317",
    fg="white",
    font=("Segoe UI", 14, "bold"),
    justify="left"
)
app_title.pack(padx=16, pady=(16, 4), anchor="w")

subtitle_label = tk.Label(
    sidebar,
    text="Craft features, run\nexperiments, export.",
    bg="#111317",
    fg="#9da5b4",
    font=("Segoe UI", 9),
    justify="left"
)
subtitle_label.pack(padx=16, pady=(0, 16), anchor="w")

# thin divider line
divider = tk.Frame(sidebar, bg="#22252e", height=1)
divider.pack(fill="x", padx=14, pady=(0, 8))

# --- Hover colors for each button background ---
HOVER_COLORS = {
    "#4C8BF5": "#3b78f0",  # blue
    "#8C53FF": "#7a3aff",  # purple
    "#FF7A59": "#ff6a3c",  # orange
    "#00B894": "#00a583",  # teal/green
    "#FFC312": "#e6b200",  # yellow
    "#FF5EAA": "#ff479c",  # pink
}
EVAL_THREAD = None
EVAL_CANCEL_EVENT = threading.Event()
LAST_EVAL_SIGNATURE = None
LAST_EVAL_RESULTS = None

# Helper to make colorful "modern" flat buttons with icon + hover
def make_nav_button(parent, text, bg_color, fg_color="#ffffff", icon=None):
    base_bg = bg_color
    hover_bg = HOVER_COLORS.get(bg_color, bg_color)

    label_text = f"{icon}  {text}" if icon else text

    btn = tk.Button(
        parent,
        text=label_text,
        anchor="w",
        relief="flat",
        padx=14,
        pady=8,
        bg=base_bg,
        fg=fg_color,
        activebackground=hover_bg,
        activeforeground=fg_color,
        font=("Segoe UI", 10, "bold"),
        bd=0,
        highlightthickness=0,
        cursor="hand2"
    )
    btn.pack(fill="x", padx=14, pady=4)

    # simple hover effect
    def on_enter(_event):
        btn.configure(bg=hover_bg)

    def on_leave(_event):
        btn.configure(bg=base_bg)

    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

    return btn

# Colorful sidebar buttons (with icons 🎨)
# ORDER (as requested):
# Load Dataset → Feature Creation → Transformations → Feature Selection → Evaluate Impact → Export Report
btn_load = make_nav_button(sidebar, "Load Dataset",       "#4C8BF5", icon="📂")
btn_clean = make_nav_button(sidebar, "Data Cleaning",     "#16A34A", icon="🧹")
btn_feature_creation = make_nav_button(sidebar, "Feature Creation",  "#8C53FF", icon="✨")
btn_transformations = make_nav_button(sidebar, "Data Pruning",    "#00B894", icon="🧪")
btn_feature_selection = make_nav_button(sidebar, "Feature Selection", "#FF7A59", icon="🎯")
btn_eval_impact = make_nav_button(sidebar, "Evaluate Impact",        "#FFC312", icon="📊")
btn_export_report = make_nav_button(sidebar, "Export Report",        "#FF5EAA", icon="📤")

footer_label = tk.Label(
    sidebar,
    text="v0.1 • Prototype UI",
    bg="#111317",
    fg="#7f848e",
    font=("Segoe UI", 9)
)
footer_label.pack(side="bottom", pady=10, padx=16, anchor="w")

# ========== RIGHT MAIN AREA ==========
main_bg = "#050816"
panel_bg = "#181a1f"
card_bg = "#20232a"

main_area = tk.Frame(root, bg=main_bg)
main_area.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

main_area.grid_rowconfigure(1, weight=1)
main_area.grid_columnconfigure(0, weight=1)

# Top bar
top_bar = tk.Frame(main_area, bg=main_bg)
top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
top_bar.grid_columnconfigure(0, weight=1)

header_label = tk.Label(
    top_bar,
    text="Workspace",
    bg=main_bg,
    fg="white",
    font=("Segoe UI", 13, "bold")
)
header_label.grid(row=0, column=0, sticky="w")

subtitle_label = tk.Label(
    top_bar,
    text="Preview your dataset, configure controls, plot features and inspect results.",
    bg=main_bg,
    fg="#9da5b4",
    font=("Segoe UI", 9)
)
subtitle_label.grid(row=1, column=0, sticky="w", pady=(2, 0))

status_pill = tk.Label(
    top_bar,
    text="● No dataset loaded",
    bg="#111827",
    fg="#9da5b4",
    font=("Segoe UI", 9),
    padx=10,
    pady=4
)
status_pill.grid(row=0, column=1, rowspan=2, sticky="e", padx=(0, 4))

# Card-style container for the tabs
card = tk.Frame(
    main_area,
    bg=card_bg,
    bd=0,
    relief="flat",
    highlightthickness=1,
    highlightbackground="#2f333d"
)
card.grid(row=1, column=0, sticky="nsew")
card.grid_rowconfigure(0, weight=1)
card.grid_columnconfigure(0, weight=1)

# Notebook style
style = ttk.Style()
style.theme_use("default")

style.configure(
    "TNotebook",
    background=card_bg,
    borderwidth=0
)
style.configure(
    "TNotebook.Tab",
    background=card_bg,
    foreground="white",
    padding=(14, 8),
    font=("Segoe UI", 10, "bold")
)
style.map(
    "TNotebook.Tab",
    background=[("selected", "#4C8BF5"), ("active", "#4C8BF5")],
    foreground=[("selected", "white"), ("active", "white")]
)

# Dark Treeview style (for column summary table)
style.configure(
    "Dark.Treeview",
    background="#1e2127",
    fieldbackground="#1e2127",
    foreground="#d7dae0",
    rowheight=22,
    borderwidth=0
)
style.map(
    "Dark.Treeview",
    background=[("selected", "#4C8BF5")],
    foreground=[("selected", "white")]
)
style.configure(
    "Dark.Treeview.Heading",
    background="#111317",
    foreground="white",
    font=("Segoe UI", 9, "bold")
)

notebook = ttk.Notebook(card, style="TNotebook")
notebook.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

# ==========================================================
# TAB 1: DATASET PREVIEW TABLE
# ==========================================================
tab_preview = tk.Frame(notebook, bg=card_bg)
notebook.add(tab_preview, text="Dataset Preview Table")
latent_tab = see_latent.add_latent_tab(notebook, card_bg)



# we will have:
# row 0: header + chip
# row 1: sub text
# row 2: stat cards
# row 3: column summary table
# row 4: text preview
tab_preview.grid_rowconfigure(3, weight=1)
tab_preview.grid_columnconfigure(0, weight=1)

# header row with chip
preview_header_frame = tk.Frame(tab_preview, bg=card_bg)
preview_header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 0))
preview_header_frame.grid_columnconfigure(0, weight=1)

preview_header = tk.Label(
    preview_header_frame,
    text="Dataset Preview",
    bg=card_bg,
    fg="white",
    font=("Segoe UI", 11, "bold")
)
preview_header.grid(row=0, column=0, sticky="w")

preview_chip = tk.Label(
    preview_header_frame,
    text="rows · schema",
    bg="#111317",
    fg="#9da5b4",
    font=("Segoe UI", 8),
    padx=8,
    pady=3
)
preview_chip.grid(row=0, column=1, sticky="e")

preview_sub = tk.Label(
    tab_preview,
    text="First N rows and basic schema will appear here after loading a dataset.",
    bg=card_bg,
    fg="#9da5b4",
    font=("Segoe UI", 9)
)
preview_sub.grid(row=1, column=0, sticky="w", padx=10, pady=(6, 8))

# --- NEW: stat cards row: rows / cols / dupes ---
stats_frame = tk.Frame(tab_preview, bg=card_bg)
stats_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 6))
stats_frame.grid_columnconfigure(0, weight=1)
stats_frame.grid_columnconfigure(1, weight=1)
stats_frame.grid_columnconfigure(2, weight=1)

def make_stat_card(parent, title, value="–"):
    card = tk.Frame(parent, bg="#111317", padx=12, pady=8)
    title_lbl = tk.Label(
        card,
        text=title,
        bg="#111317",
        fg="#9da5b4",
        font=("Segoe UI", 9)
    )
    title_lbl.pack(anchor="w")
    value_lbl = tk.Label(
        card,
        text=value,
        bg="#111317",
        fg="white",
        font=("Segoe UI", 12, "bold")
    )
    value_lbl.pack(anchor="w")
    return card, value_lbl

card_rows, lbl_rows_value = make_stat_card(stats_frame, "Rows")
card_cols, lbl_cols_value = make_stat_card(stats_frame, "Columns")
card_dups, lbl_dups_value = make_stat_card(stats_frame, "Duplicate rows")

card_rows.grid(row=0, column=0, sticky="ew", padx=(0, 4))
card_cols.grid(row=0, column=1, sticky="ew", padx=4)
card_dups.grid(row=0, column=2, sticky="ew", padx=(4, 0))

# --- COLUMN SUMMARY AREA: table (left) + details (right) ---
columns_pane = ttk.Panedwindow(tab_preview, orient="horizontal")
columns_pane.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 8))

# LEFT: compact table
columns_frame = tk.Frame(columns_pane, bg=card_bg)
columns_frame.grid_rowconfigure(0, weight=1)
columns_frame.grid_columnconfigure(0, weight=1)

columns_tree = ttk.Treeview(
    columns_frame,
    columns=("name", "role", "dtype", "missing", "unique"),
    show="headings",
    style="Dark.Treeview"
)
columns_tree.heading("name", text="Name", anchor="w")
columns_tree.heading("role", text="Role", anchor="w")
columns_tree.heading("dtype", text="Dtype", anchor="w")
columns_tree.heading("missing", text="Missing (%)", anchor="center")
columns_tree.heading("unique", text="Unique", anchor="center")

columns_tree.column("name", width=200, anchor="w", stretch=True)
columns_tree.column("role", width=130, anchor="w", stretch=False)
columns_tree.column("dtype", width=80, anchor="w", stretch=False)
columns_tree.column("missing", width=90, anchor="center", stretch=False)
columns_tree.column("unique", width=80, anchor="center", stretch=False)

columns_tree.grid(row=0, column=0, sticky="nsew")
tree_scroll_y = tk.Scrollbar(columns_frame, orient="vertical", command=columns_tree.yview)
tree_scroll_y.grid(row=0, column=1, sticky="ns")
columns_tree.configure(yscrollcommand=tree_scroll_y.set)

# RIGHT: column details card
col_detail_frame = tk.Frame(columns_pane, bg=card_bg, padx=10, pady=8)
col_detail_frame.grid_columnconfigure(1, weight=1)

detail_title = tk.Label(
    col_detail_frame,
    text="Column details",
    bg=card_bg,
    fg="#9da5b4",
    font=("Segoe UI", 9, "bold")
)
detail_title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

# name
lbl_det_name = tk.Label(
    col_detail_frame,
    text="Name:",
    bg=card_bg,
    fg="#9da5b4",
    font=("Segoe UI", 9)
)
val_det_name = tk.Label(
    col_detail_frame,
    text="–",
    bg=card_bg,
    fg="#d7dae0",
    font=("Segoe UI", 9, "bold")
)
lbl_det_name.grid(row=1, column=0, sticky="w")
val_det_name.grid(row=1, column=1, sticky="w")

# role
lbl_det_role = tk.Label(
    col_detail_frame,
    text="Role:",
    bg=card_bg,
    fg="#9da5b4",
    font=("Segoe UI", 9)
)
val_det_role = tk.Label(
    col_detail_frame,
    text="–",
    bg=card_bg,
    fg="#d7dae0",
    font=("Segoe UI", 9)
)
lbl_det_role.grid(row=2, column=0, sticky="w", pady=(4, 0))
val_det_role.grid(row=2, column=1, sticky="w", pady=(4, 0))

# dtype
lbl_det_dtype = tk.Label(
    col_detail_frame,
    text="Dtype:",
    bg=card_bg,
    fg="#9da5b4",
    font=("Segoe UI", 9)
)
val_det_dtype = tk.Label(
    col_detail_frame,
    text="–",
    bg=card_bg,
    fg="#d7dae0",
    font=("Segoe UI", 9)
)
lbl_det_dtype.grid(row=3, column=0, sticky="w", pady=(4, 0))
val_det_dtype.grid(row=3, column=1, sticky="w", pady=(4, 0))

# missing
lbl_det_missing = tk.Label(
    col_detail_frame,
    text="Missing:",
    bg=card_bg,
    fg="#9da5b4",
    font=("Segoe UI", 9)
)
val_det_missing = tk.Label(
    col_detail_frame,
    text="–",
    bg=card_bg,
    fg="#d7dae0",
    font=("Segoe UI", 9)
)
lbl_det_missing.grid(row=4, column=0, sticky="w", pady=(4, 0))
val_det_missing.grid(row=4, column=1, sticky="w", pady=(4, 0))

# unique
lbl_det_unique = tk.Label(
    col_detail_frame,
    text="Unique:",
    bg=card_bg,
    fg="#9da5b4",
    font=("Segoe UI", 9)
)
val_det_unique = tk.Label(
    col_detail_frame,
    text="–",
    bg=card_bg,
    fg="#d7dae0",
    font=("Segoe UI", 9)
)
lbl_det_unique.grid(row=5, column=0, sticky="w", pady=(4, 0))
val_det_unique.grid(row=5, column=1, sticky="w", pady=(4, 0))

# example
lbl_det_example = tk.Label(
    col_detail_frame,
    text="Example:",
    bg=card_bg,
    fg="#9da5b4",
    font=("Segoe UI", 9)
)
val_det_example = tk.Label(
    col_detail_frame,
    text="–",
    bg=card_bg,
    fg="#d7dae0",
    font=("Segoe UI", 9),
    wraplength=230,
    justify="left"
)
lbl_det_example.grid(row=6, column=0, sticky="nw", pady=(4, 0))
val_det_example.grid(row=6, column=1, sticky="w", pady=(4, 0))

def update_preview_ui_from_meta(
    meta,
    lbl_rows_value,
    lbl_cols_value,
    lbl_dups_value,
    columns_tree
):
    if not meta:
        return

    # --- Fill top stat cards ---
    n_rows = (
        meta.get("n_rows")
        or meta.get("rows")
        or meta.get("num_rows")
        or None
    )
    n_cols = (
        meta.get("n_cols")
        or meta.get("cols")
        or meta.get("num_cols")
        or None
    )
    # <-- small tweak: also support key used in dataset._summarize_dataframe
    n_dups = (
        meta.get("n_dup_rows")
        or meta.get("n_duplicates")
        or meta.get("duplicate_rows")
        or meta.get("n_dups")
        or 0
    )

    # Fallback: infer cols from columns list if needed
    if n_cols is None:
        cols_list = meta.get("columns") or meta.get("cols") or []
        try:
            n_cols = len(cols_list)
        except TypeError:
            n_cols = "–"

    lbl_rows_value.config(text=str(n_rows) if n_rows is not None else "–")
    lbl_cols_value.config(text=str(n_cols) if n_cols is not None else "–")
    lbl_dups_value.config(text=str(n_dups))

    # --- Fill the columns table ---
    # clear old rows
    for item in columns_tree.get_children():
        columns_tree.delete(item)

    cols_meta = meta.get("columns") or meta.get("cols") or []

    for col in cols_meta:
        # Try a few sensible key names so it works with your dataset module
        name = (
            col.get("name")
            or col.get("col_name")
            or col.get("column")
            or "–"
        )
        role = col.get("role", "feature")
        dtype = (
            col.get("dtype")
            or col.get("type")
            or col.get("data_type")
            or "–"
        )
        # <-- small tweak: include 'pct_missing' used in dataset._summarize_dataframe
        missing = (
            col.get("missing_pct")
            or col.get("pct_missing")
            or col.get("missing_percent")
            or col.get("missing")
            or 0
        )
        unique = (
            col.get("unique")
            or col.get("n_unique")
            or col.get("unique_count")
            or "–"
        )

        columns_tree.insert(
            "",
            "end",
            values=(name, role, dtype, missing, unique),
        )

# Selection: update right-side details from the table selection
def on_columns_select(event):
    selected = columns_tree.selection()
    if not selected:
        return
    item_id = selected[0]
    values = columns_tree.item(item_id, "values")
    if len(values) != 5:
        return

    name, role, dtype, missing, unique = values
    val_det_name.config(text=name or "–")
    val_det_role.config(text=role or "–")
    val_det_dtype.config(text=dtype or "–")
    val_det_missing.config(text=str(missing))
    val_det_unique.config(text=str(unique))
    # example stays as-is for now; you can hook it from meta later

columns_tree.bind("<<TreeviewSelect>>", on_columns_select)

# NEW: double-click handler to open Column Profile popup
def on_columns_double_click(event):
    item_id = columns_tree.identify_row(event.y)
    if not item_id:
        return
    values = columns_tree.item(item_id, "values")
    if not values:
        return
    col_name = values[0]
    if not col_name or col_name == "–":
        return
    column_profile.show_column_profile(root, col_name)

columns_tree.bind("<Double-1>", on_columns_double_click)

# add both sides to the Panedwindow
columns_pane.add(columns_frame, weight=3)
columns_pane.add(col_detail_frame, weight=2)

# Text preview (your original preview_text)
preview_container = tk.Frame(tab_preview, bg=card_bg)
preview_container.grid(row=4, column=0, sticky="nsew", padx=10, pady=(0,10))
preview_container.grid_rowconfigure(0, weight=1)
preview_container.grid_columnconfigure(0, weight=1)

preview_text = tk.Text(
    preview_container,
    wrap="none",
    bg="#1e2127",
    fg="#d7dae0",
    insertbackground="white",
    relief="flat",
    font=("Consolas", 9),
    borderwidth=0,
    highlightthickness=1,
    highlightbackground="#2f333d"
)
preview_text.grid(row=0, column=0, sticky="nsew")

scroll_y = tk.Scrollbar(preview_container, orient="vertical", command=preview_text.yview)
scroll_y.grid(row=0, column=1, sticky="ns")
scroll_x = tk.Scrollbar(preview_container, orient="horizontal", command=preview_text.xview)
scroll_x.grid(row=1, column=0, sticky="ew")

preview_text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

# Placeholder content
preview_text.insert("end", ">>> Dataset preview table will appear here...\n\n")
preview_text.insert("end", "Columns:\n")
preview_text.insert("end", "  - feature_1 (float)\n")
preview_text.insert("end", "  - feature_2 (int)\n")
preview_text.insert("end", "  - feature_3 (category)\n")
preview_text.insert("end", "  - target (float)\n\n")
preview_text.insert("end", "Sample rows (mocked):\n")
for i in range(1, 21):
    preview_text.insert("end", f"Row {i:02d}: 0.{i}   {i}   class_{i%3}   {i*0.5}\n")

# ==========================================================
# TAB 2: PLOTS
# ==========================================================
tab_plots = tk.Frame(notebook, bg=card_bg)
notebook.add(tab_plots, text="Plots")

tab_plots.grid_rowconfigure(2, weight=1)
tab_plots.grid_columnconfigure(0, weight=1)

plots_header_frame = tk.Frame(tab_plots, bg=card_bg)
plots_header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 0))
plots_header_frame.grid_columnconfigure(0, weight=1)

plots_header = tk.Label(
    plots_header_frame,
    text="Plots",
    bg=card_bg,
    fg="white",
    font=("Segoe UI", 11, "bold")
)
plots_header.grid(row=0, column=0, sticky="w")

plots_chip = tk.Label(
    plots_header_frame,
    text="viz · insights",
    bg="#111317",
    fg="#9da5b4",
    font=("Segoe UI", 8),
    padx=8,
    pady=3
)
plots_chip.grid(row=0, column=1, sticky="e")

plots_sub = tk.Label(
    tab_plots,
    text="Visualizations of distributions, correlations and feature importance will appear here.",
    bg=card_bg,
    fg="#9da5b4",
    font=("Segoe UI", 9)
)
plots_sub.grid(row=2, column=0, sticky="w", padx=10, pady=(6, 8))

# AUTOMATED REPORT BUTTONS (Colorful, Big Font)
report_buttons_frame = tk.Frame(tab_plots, bg=card_bg)
report_buttons_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))

def make_report_button(text, color, command):
    btn = tk.Button(
        report_buttons_frame,
        text=text,
        command=command,
        bg=color,
        fg="white",
        relief="flat",
        font=("Segoe UI", 11, "bold"),
        padx=14,
        pady=8,
        cursor="hand2",
        activebackground=color,
        activeforeground="white",
        bd=0,
        highlightthickness=0
    )
    btn.pack(side="left", padx=6)
    return btn

# --- FINAL TWO REPORT BUTTONS ---
btn_ydata = make_report_button(
    "YData Profile",
    "#8C53FF",
    datareport.generate_ydata_report
)

btn_skimpy = make_report_button(
    "Skimpy Summary (PNG)",
    "#FF7A59",
    datareport.generate_skimpy_png
)

# Warning note under the skimpy button
skimpy_warning = tk.Label(
    report_buttons_frame,
    text="⚠️ Skimpy, and y-dataProfiling may not support all datasets",
    bg=card_bg,
    fg="#ffb347",   # warm warning color
    font=("Segoe UI", 8, "italic")
)
skimpy_warning.pack(side="left", padx=6)

# >>> FIXED LINE: move plot container to row 3  
plots_container = tk.Frame(tab_plots, bg=card_bg)
plots_container.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0,10))
plots_container.grid_rowconfigure(0, weight=1)
plots_container.grid_columnconfigure(0, weight=1)

# Placeholder "plot area"
plots_placeholder = tk.Label(
    plots_container,
    text="(Plot canvas placeholder)\n\nLater you can embed matplotlib/other plots here.",
    bg="#1e2127",
    fg="#d7dae0",
    font=("Segoe UI", 10),
    justify="center",
    borderwidth=0,
    highlightthickness=1,
    highlightbackground="#2f333d"
)
plots_placeholder.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

# ==========================================================
# TAB 3: CONTROL SETTINGS
# ==========================================================
tab_controls = tk.Frame(notebook, bg=card_bg)
notebook.add(tab_controls, text="Control Settings")

tab_controls.grid_rowconfigure(2, weight=1)
tab_controls.grid_columnconfigure(1, weight=1)

controls_header_frame = tk.Frame(tab_controls, bg=card_bg)
controls_header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(8, 0))
controls_header_frame.grid_columnconfigure(0, weight=1)

controls_header = tk.Label(
    controls_header_frame,
    text="Control Settings",
    bg=card_bg,
    fg="white",
    font=("Segoe UI", 11, "bold")
)
controls_header.grid(row=0, column=0, sticky="w")

controls_chip = tk.Label(
    controls_header_frame,
    text="config · model",
    bg="#111317",
    fg="#9da5b4",
    font=("Segoe UI", 8),
    padx=8,
    pady=3
)
controls_chip.grid(row=0, column=1, sticky="e")

controls_sub = tk.Label(
    tab_controls,
    text="Configure transformations, feature generation and model parameters.",
    bg=card_bg,
    fg="#9da5b4",
    font=("Segoe UI", 9)
)
controls_sub.grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(6, 8))

controls_container = tk.Frame(tab_controls, bg=card_bg)
controls_container.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))
controls_container.grid_columnconfigure(1, weight=1)

# Target
lbl_target = tk.Label(
    controls_container,
    text="Target column:",
    bg=card_bg,
    fg="#d7dae0",
    font=("Segoe UI", 9)
)
lbl_target.grid(row=0, column=0, sticky="w", padx=4, pady=4)

entry_target = tk.Entry(
    controls_container,
    bg="#1e2127",
    fg="#d7dae0",
    insertbackground="white",
    relief="flat",
    font=("Segoe UI", 9),
    borderwidth=0,
    highlightthickness=1,
    highlightbackground="#2f333d",
    highlightcolor="#4C8BF5"
)
entry_target.grid(row=0, column=1, sticky="ew", padx=4, pady=4)

# Test size
lbl_test_size = tk.Label(
    controls_container,
    text="Test size (0–1):",
    bg=card_bg,
    fg="#d7dae0",
    font=("Segoe UI", 9)
)
lbl_test_size.grid(row=1, column=0, sticky="w", padx=4, pady=4)

entry_test_size = tk.Entry(
    controls_container,
    bg="#1e2127",
    fg="#d7dae0",
    insertbackground="white",
    relief="flat",
    font=("Segoe UI", 9),
    borderwidth=0,
    highlightthickness=1,
    highlightbackground="#2f333d",
    highlightcolor="#4C8BF5"
)
entry_test_size.insert(0, "0.2")
entry_test_size.grid(row=1, column=1, sticky="ew", padx=4, pady=4)

# Model type
lbl_model = tk.Label(
    controls_container,
    text="Model type:",
    bg=card_bg,
    fg="#d7dae0",
    font=("Segoe UI", 9)
)
lbl_model.grid(row=2, column=0, sticky="w", padx=4, pady=4)

model_var = tk.StringVar(value="Random Forest")
combo_model = ttk.Combobox(
    controls_container,
    textvariable=model_var,
    values=["Random Forest", "XGBoost", "LightGBM", "Logistic Regression", "Linear Regression"],
    state="readonly"
)
combo_model.grid(row=2, column=1, sticky="ew", padx=4, pady=4)

# Helper text for each model
_MODEL_HELP_TEXT = {
    "Random Forest": "Robust default for most tabular problems; handles non-linearities and mixed features well.",
    "XGBoost": "Boosted trees; strong performance on complex tabular data, but may need tuning and is a bit slower.",
    "LightGBM": "Great for large, wide datasets with many features; optimized gradient-boosted trees.",
    "Logistic Regression": "Good baseline for classification; fast and interpretable, assumes roughly linear decision boundary.",
    "Linear Regression": "Good baseline for regression; works best when relationships are mostly linear."
}

lbl_model_help = tk.Label(
    controls_container,
    text=_MODEL_HELP_TEXT.get(model_var.get(), ""),
    bg=card_bg,
    fg="#9da5b4",
    font=("Segoe UI", 8),
    wraplength=360,
    justify="left"
)
lbl_model_help.grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 6))

def _on_model_change(event):
    text = _MODEL_HELP_TEXT.get(model_var.get(), "")
    lbl_model_help.config(text=text)

combo_model.bind("<<ComboboxSelected>>", _on_model_change)

# Task type (row 4)
lbl_task_type = tk.Label(
    controls_container,
    text="Task type:",
    bg=card_bg,
    fg="#d7dae0",
    font=("Segoe UI", 9)
)
lbl_task_type.grid(row=4, column=0, sticky="w", padx=4, pady=4)

task_type_var = tk.StringVar(value="Auto (detect)")
combo_task_type = ttk.Combobox(
    controls_container,
    textvariable=task_type_var,
    values=["Auto (detect)", "Regression", "Classification"],
    state="readonly"
)
combo_task_type.grid(row=4, column=1, sticky="ew", padx=4, pady=4)

# Apply Settings BUTTON – row 5 (ABOVE K + Auto-K)
btn_apply = tk.Button(
    controls_container,
    text="Apply Settings",
    bg="#4C8BF5",
    fg="white",
    relief="flat",
    font=("Segoe UI", 10, "bold"),
    padx=10,
    pady=6,
    cursor="hand2",
    activebackground="#3b78f0",
    activeforeground="white",
    borderwidth=0,
    highlightthickness=1,
    highlightbackground="#3b78f0"
)
btn_apply.grid(row=5, column=0, sticky="w", padx=4, pady=(6, 4))


# K row – row 6 (between Apply and Auto-K)
lbl_k = tk.Label(
    controls_container,
    text="Number of selected features (K):",
    bg=card_bg,
    fg="#d7dae0",
    font=("Segoe UI", 9)
)
lbl_k.grid(row=6, column=0, sticky="w", padx=4, pady=4)

lbl_k_value = tk.Label(
    controls_container,
    text="Auto (not selected yet)",
    bg=card_bg,
    fg="#9da5b4",
    font=("Segoe UI", 9)
)
lbl_k_value.grid(row=6, column=1, sticky="w", padx=4, pady=4)

# Auto-select K BUTTON – row 7 (BELOW Apply)
btn_auto_k = tk.Button(
    controls_container,
    text="Auto-select K using CV",
    bg="#4b5563",
    fg="white",
    relief="flat",
    font=("Segoe UI", 9, "bold"),
    padx=10,
    pady=4,
    cursor="hand2",
    activebackground="#374151",
    activeforeground="white",
    borderwidth=0,
    highlightthickness=1,
    highlightbackground="#374151"
)
btn_auto_k.grid(row=7, column=0, sticky="w", padx=4, pady=(6, 4))

ToolTip(
    btn_auto_k,
    (
        "Automatically chooses the number of features K using cross-validation.\n\n"
        "Under the hood:\n"
        "• Ranks features by combining Mutual Information, model-based importance\n"
        "  and RFE ranking.\n"
        "• Tries several candidate K values (e.g. 5, 10, 15, 20, 25).\n"
        "• For each K, fits the selected model and evaluates it with k-fold CV:\n"
        "    - Regression: minimizes RMSE\n"
        "    - Classification: maximizes F1 (weighted)\n"
        "• Picks the K that gives the best average CV score and stores it."
    )
)

# Progress bar for Auto-K (initially hidden)
auto_k_progress = ttk.Progressbar(
    controls_container,
    mode="indeterminate",
    length=220,
)
auto_k_progress.grid(row=8, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 4))
auto_k_progress.grid_remove()  # hide until used

# Note at the end – row 8
controls_note = tk.Label(
    controls_container,
    text="Press on Feature Selection button to apply model.",
    bg=card_bg,
    fg="#9da5b4",
    font=("Segoe UI", 8, "italic")
)
controls_note.grid(row=9, column=0, columnspan=2, sticky="w", padx=4, pady=(8, 4))

# Hover for Apply button
def _apply_btn_enter(event):
    btn_apply.config(bg="#3b78f0")

def _apply_btn_leave(event):
    btn_apply.config(bg="#4C8BF5")

btn_apply.bind("<Enter>", _apply_btn_enter)
btn_apply.bind("<Leave>", _apply_btn_leave)


# ==========================================================
# TAB 4: RESULTS PANEL
# ==========================================================
tab_results = tk.Frame(notebook, bg=card_bg)
notebook.add(tab_results, text="Results Panel")

tab_results.grid_rowconfigure(3, weight=1)
tab_results.grid_columnconfigure(0, weight=1)

results_header_frame = tk.Frame(tab_results, bg=card_bg)
results_header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 0))
results_header_frame.grid_columnconfigure(0, weight=1)

results_header = tk.Label(
    results_header_frame,
    text="Results & Impact",
    bg=card_bg,
    fg="white",
    font=("Segoe UI", 11, "bold")
)
results_header.grid(row=0, column=0, sticky="w")

results_chip = tk.Label(
    results_header_frame,
    text="metrics · importance",
    bg="#111317",
    fg="#9da5b4",
    font=("Segoe UI", 8),
    padx=8,
    pady=3
)
results_chip.grid(row=0, column=1, sticky="e")
btn_cancel_eval = tk.Button(
    results_header_frame,
    text="Cancel",
    bg="#4C8BF5",
    fg="white",
    relief="flat",
    font=("Segoe UI", 9),
    padx=10,
    pady=3,
    state="disabled",  # disabled by default
)
btn_cancel_eval.grid(row=0, column=2, sticky="e", padx=(8, 0))

results_sub = tk.Label(
    tab_results,
    text="Here you’ll see metrics, feature importance and before/after comparisons.",
    bg=card_bg,
    fg="#9da5b4",
    font=("Segoe UI", 9)
)
results_sub.grid(row=1, column=0, sticky="w", padx=10, pady=(6, 8))

# --- NEW: small button bar for extra visualizations ---
results_controls_frame = tk.Frame(tab_results, bg=card_bg)
results_controls_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 6))

btn_confmat = tk.Button(
    results_controls_frame,
    text="Confusion Matrix",
    bg="#4b5563",
    fg="white",
    relief="flat",
    font=("Segoe UI", 9),
    padx=10,
    pady=4,
    cursor="hand2",
    command=lambda: evaluate_impact.show_confusion_matrices(root),
)
btn_confmat.pack(side="left", padx=(0, 8))

btn_rocpr = tk.Button(
    results_controls_frame,
    text="ROC & PR Curves",
    bg="#4b5563",
    fg="white",
    relief="flat",
    font=("Segoe UI", 9),
    padx=10,
    pady=4,
    cursor="hand2",
    command=lambda: evaluate_impact.show_roc_pr_curves(root),
)
btn_rocpr.pack(side="left", padx=(0, 8))

results_container = tk.Frame(tab_results, bg=card_bg)
results_container.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0,10))

results_container.grid_rowconfigure(0, weight=1)
results_container.grid_columnconfigure(0, weight=1)

results_text = tk.Text(
    results_container,
    wrap="none",              # so lines don't break mid-row
    bg="#1e2127",
    fg="#d7dae0",
    insertbackground="white",
    relief="flat",
    font=("Consolas", 9),     # monospaced for aligned table
    borderwidth=0,
    highlightthickness=1,
    highlightbackground="#2f333d"
)

results_text.grid(row=0, column=0, sticky="nsew")

results_scroll_y = tk.Scrollbar(results_container, orient="vertical", command=results_text.yview)
results_scroll_y.grid(row=0, column=1, sticky="ns")
results_text.configure(yscrollcommand=results_scroll_y.set)

# Placeholder results content
results_text.insert("end", "Model comparison (mock):\n\n")
results_text.insert("end", "• Baseline model (no feature engineering)\n")
results_text.insert("end", "    - RMSE: 0.812\n    - R²:   0.62\n\n")
results_text.insert("end", "• After feature engineering\n")
results_text.insert("end", "    - RMSE: 0.694\n    - R²:   0.74\n\n")
results_text.insert("end", "Feature importance (top 5):\n")
results_text.insert("end", "    1. feature_3_encoded      0.31\n")
results_text.insert("end", "    2. feature_1_log_scaled   0.24\n")
results_text.insert("end", "    3. interaction_f1_f2      0.17\n")
results_text.insert("end", "    4. feature_2_binned       0.15\n")
results_text.insert("end", "    5. feature_1              0.13\n\n")
results_text.insert("end", "Notes:\n- All values above are placeholders.\n- This panel can later host plots and interactive elements.\n")



tab_export = ttk.Frame(notebook)
notebook.add(tab_export, text="Export Current Dataset")
export_frame = tk.Frame(tab_export, padx=20, pady=20)
export_frame.pack(fill="both", expand=True)

title_label = tk.Label(
    export_frame,
    text="Export Your Dataset",
    font=("Segoe UI", 16, "bold"),
    pady=10
)
title_label.pack()

btn_cleaned = tk.Button(
    export_frame,
    text="⬇ Download Current Dataset",
    command=dataset_downloads.download_cleaned,
    bg="#16A34A",
    fg="white",
    font=("Segoe UI", 11, "bold"),
    padx=12, pady=8
)
btn_cleaned.pack(pady=10)


# ==========================================================
# ==== SETTINGS + PIPELINE CALLBACKS =======================
# ==========================================================
def on_apply_settings():
    """Validate and save settings into APP_SETTINGS."""
    if dataset.CURRENT_DF is None:
        messagebox.showwarning("No dataset", "Load a dataset first.")
        return

    df = dataset.CURRENT_DF

    target = entry_target.get().strip()
    if not target:
        messagebox.showerror("Invalid target", "Please enter a target column.")
        return

    if target not in df.columns:
        messagebox.showerror("Invalid target", f"Target '{target}' not found in dataset.")
        return

    try:
        test_size_str = entry_test_size.get().strip()
        test_size = float(test_size_str)
        if not (0 < test_size < 1):
            raise ValueError()
    except ValueError:
        messagebox.showerror("Invalid test size", "Test size must be a number between 0 and 1.")
        return

    APP_SETTINGS["target"] = target
    APP_SETTINGS["test_size"] = test_size
    APP_SETTINGS["model_type"] = model_var.get()

    # NEW: save task type choice
    choice = task_type_var.get()
    choice_lower = choice.lower()

    if choice_lower.startswith("regres"):
        task_type = "regression"
    elif choice_lower.startswith("class"):
        task_type = "classification"
    else:
        task_type = None  # Auto-detect: fall back to heuristic

    APP_SETTINGS["task_type"] = task_type

    # Propagate to dataset module so the rest of the pipeline can use it
    if hasattr(dataset, "set_task_type"):
        dataset.set_task_type(task_type)
    else:
        # Fallback if helper is not available for some reason
        dataset.TASK_TYPE = task_type

    messagebox.showinfo("Settings Saved", "Your choices have been saved successfully.")

import threading  # make sure this is at the top of Main.py

def on_auto_select_k():
    """Automatically choose K using the same CV-based theory as run_feature_selection."""
    if dataset.CURRENT_DF is None:
        messagebox.showwarning("No dataset", "Load a dataset first.")
        return

    target = APP_SETTINGS.get("target")
    if not target:
        messagebox.showerror(
            "Target required",
            "Please set a target column and click 'Apply Settings' before auto-selecting K.",
        )
        notebook.select(tab_controls)
        return

    model_type = APP_SETTINGS.get("model_type", "Random Forest")

    # UI: show progress bar, disable buttons
    btn_auto_k.config(state="disabled")
    btn_apply.config(state="disabled")
    auto_k_progress.grid()     # show
    auto_k_progress.start(10)  # animate

    def worker():
        try:
            results = run_feature_selection(
                target=target,
                top_k=None,         # let it tune K via CV
                model_type=model_type,
            )
            error = None
        except Exception as e:
            results = None
            error = e

        # back to main thread to touch the UI
        def finish():
            auto_k_progress.stop()
            auto_k_progress.grid_remove()
            btn_auto_k.config(state="normal")
            btn_apply.config(state="normal")

            if error is not None:
                messagebox.showerror("Auto-select K Error", str(error))
                return

            k_used = results.get("k_used")
            if k_used is None:
                messagebox.showerror(
                    "Auto-select K Error",
                    "Could not determine K from feature selection results.",
                )
                return

            APP_SETTINGS["top_k"] = int(k_used)
            lbl_k_value.config(text=f"Auto-selected K = {k_used} features")

            messagebox.showinfo(
                "K selected",
                f"Automatically chosen K = {k_used} features.\n"
                "This value will be used the next time you run Feature Selection."
            )

        root.after(0, finish)

    threading.Thread(target=worker, daemon=True).start()


# ==========================================================
# ==== BUTTON CALLBACKS (SIDEBAR + OTHER ACTIONS) ==========
# ==========================================================
def on_load_dataset():
    # show the preview tab
    notebook.select(tab_preview)

    # Step 1: load dataset and fill the text preview
    dataset.open_dataset(preview_text)

    # Step 2: get metadata and update stat cards + column table
    meta = dataset.get_current_meta()
    update_preview_ui_from_meta(
        meta,
        lbl_rows_value,
        lbl_cols_value,
        lbl_dups_value,
        columns_tree,
    )

    # update status pill
    if meta:
        status_pill.config(text="● Dataset loaded", fg="#4ade80")
def on_data_cleaning():
    """Run full data cleaning on the current dataset and refresh the preview."""
    if dataset.CURRENT_DF is None:
        messagebox.showwarning("No dataset", "Load a dataset first.")
        return

    # Show preview tab while cleaning
    notebook.select(tab_preview)

    try:
        # This will update dataset.CURRENT_DF, CURRENT_META and the preview text
        data_preprocessing.run_data_cleaning(preview_text)
    except Exception:
        traceback.print_exc()
        messagebox.showerror(
            "Data Cleaning Error",
            "Something went wrong during data cleaning. Check the console for details.",
        )
        return

    # Refresh the summary cards + columns table from the new meta
    meta = dataset.get_current_meta()
    update_preview_ui_from_meta(
        meta,
        lbl_rows_value,
        lbl_cols_value,
        lbl_dups_value,
        columns_tree,
    )


# ============================================================
# FEATURE CREATION WITH PROGRESS POPUP
# ============================================================
def start_feature_creation_with_progress(
    use_pca=True,
    use_ica=True,
    use_auto=True,
    pca_comp=5,
    ica_comp=5,
    auto_dim=3,
    scaling="StandardScaler",
    target_column=None,        # <--- NEW
):
    """Run feature creation in a background thread and show a
    determinate progress bar that updates at major phases.

    target_column (optional):
        Name of the target column to EXCLUDE from PCA/ICA/AE.
        If None, falls back to APP_SETTINGS['target'] when available.
    """
    if dataset.CURRENT_DF is None:
        messagebox.showwarning("No dataset", "Load a dataset first.")
        return

    # Decide which column we will actually try to exclude
    effective_target = target_column or APP_SETTINGS.get("target")

    if effective_target and effective_target not in dataset.CURRENT_DF.columns:
        messagebox.showerror(
            "Invalid target column",
            f"Column '{effective_target}' was not found in the dataset. "
            "It will not be excluded from feature creation.",
        )
        effective_target = None

    # ---- POPUP WINDOW ----
    loading = tk.Toplevel(root)
    loading.title("Creating Latent Features...")
    loading.geometry("380x160")
    loading.configure(bg="#1e2127")
    loading.grab_set()

    lbl_status = tk.Label(
        loading,
        text="Starting feature creation...",
        bg="#1e2127",
        fg="white",
        font=("Segoe UI", 10)
    )
    lbl_status.pack(pady=(12, 6))

    pb = ttk.Progressbar(
        loading,
        orient="horizontal",
        length=320,
        mode="determinate",
        maximum=100
    )
    pb.pack(pady=(0, 12))
    pb["value"] = 0

    # ---- PROGRESS UPDATE HELPER ----
    def set_progress(val, text=None):
        def _inner():
            pb["value"] = val
            if text is not None:
                lbl_status.config(text=text)
            loading.update_idletasks()
        root.after(0, _inner)

    # ---- BACKGROUND WORKER ----
    def worker():
        try:
            # 1 — PREPARE INPUT
            set_progress(5, "Preparing data...")
            original_df = dataset.CURRENT_DF.copy()

            # Separate target column (if requested and available)
            target_series = None
            if effective_target and effective_target in original_df.columns:
                target_series = original_df[effective_target]
                df_for_creation = original_df.drop(columns=[effective_target])
            else:
                df_for_creation = original_df

            # 2 — RUN FEATURE CREATION (on df WITHOUT target)
            set_progress(20, "Running PCA / ICA / Autoencoder...")
            result = run_feature_creation(
                df_for_creation,
                use_pca=use_pca,
                use_ica=use_ica,
                use_autoencoder=use_auto,
                pca_components=pca_comp,
                ica_components=ica_comp,
                auto_latent_dim=auto_dim,
                scaling=scaling
            )

            df_no_target_new = result["df_new"]
            latent = result["latent_df"]
            summary = result["summary"]

            # 2b — re-attach target column so CURRENT_DF still has it
            if target_series is not None:
                # safety: in case something named the same sneaks in
                if effective_target in df_no_target_new.columns:
                    df_no_target_new = df_no_target_new.drop(columns=[effective_target])
                df_new = df_no_target_new.copy()
                df_new[effective_target] = target_series.values
                summary["excluded_target"] = effective_target
            else:
                df_new = df_no_target_new
                summary["excluded_target"] = None

            # 3 — UPDATE DATASET
            set_progress(80, "Updating dataset and UI...")
            dataset.CURRENT_DF = df_new
            dataset.POST_CREATION_DF = df_new.copy()
            dataset.BANNED_FEATURES = []
            dataset.CURRENT_META = dataset._summarize_dataframe(df_new)

            # 4 — UI FINISH
            def finish():
                set_progress(100, "Done.")
                loading.destroy()

                update_preview_ui_from_meta(
                    dataset.CURRENT_META,
                    lbl_rows_value,
                    lbl_cols_value,
                    lbl_dups_value,
                    columns_tree,
                )

                notebook.select(tab_results)
                results_text.delete("1.0", "end")

                results_text.insert("end", "=== FEATURE CREATION COMPLETE ===\n\n")
                results_text.insert("end", f"Scaling used: {summary['scaling']}\n\n")
                results_text.insert("end", f"Total latent features: {summary['total_latent']}\n\n")

                if summary.get("excluded_target"):
                    results_text.insert(
                        "end",
                        f"Target column excluded from latent generators: "
                        f"{summary['excluded_target']}\n\n"
                    )

                if latent.empty:
                    results_text.insert("end", "⚠ No latent features were created.\n")
                else:
                    for col in latent.columns:
                        results_text.insert("end", f" - {col}\n")

            root.after(0, finish)

        except Exception:
            err = traceback.format_exc()

            def show_error():
                loading.destroy()
                messagebox.showerror("Feature Creation Error", err)

            root.after(0, show_error)

    # ---- START THREAD ----
    threading.Thread(target=worker, daemon=True).start()


# ============================================================
# ADVANCED FEATURE CREATION POPUP (Option 2)
# ============================================================
def open_feature_creation_popup():
    popup = tk.Toplevel(root)
    popup.title("Feature Creation")
    popup.geometry("520x540")
    popup.config(bg="#1e2127")
    popup.grab_set()

    # --- TITLE ---
    tk.Label(
        popup,
        text="Feature Creation",
        bg="#1e2127",
        fg="white",
        font=("Segoe UI", 15, "bold")
    ).pack(pady=(12, 4))

    tk.Label(
        popup,
        text="Choose how deep the latent feature generator should explore.\nHover cards to see explanations.",
        bg="#1e2127",
        fg="#9da5b4",
        font=("Segoe UI", 9)
    ).pack()

# --- Target column (to avoid leakage) ---
    target_frame = tk.Frame(popup, bg="#1e2127")
    target_frame.pack(pady=(10, 0), padx=20, fill="x")

    tk.Label(
        target_frame,
        text="Target column:",
        bg="#1e2127",
        fg="#d7dae0",
        font=("Segoe UI", 9)
    ).grid(row=0, column=0, sticky="w", padx=(0, 6))

    entry_target_popup = tk.Entry(
        target_frame,
        bg="#1e2127",
        fg="#d7dae0",
        insertbackground="white",
        relief="flat",
        font=("Segoe UI", 9),
        borderwidth=0,
        highlightthickness=1,
        highlightbackground="#2f333d",
        highlightcolor="#4C8BF5",
    )
    entry_target_popup.grid(row=0, column=1, sticky="ew")
    target_frame.grid_columnconfigure(1, weight=1)

    # pre-fill with global target if available
    if APP_SETTINGS.get("target"):
        entry_target_popup.insert(0, APP_SETTINGS["target"])

    # --------------------------------------------------------
    # CARD FACTORY (Hover-expand explanation)
    # --------------------------------------------------------
    def make_card(parent, title, subtitle, expl):
        frame = tk.Frame(
            parent,
            bg="#111317",
            highlightthickness=1,
            highlightbackground="#2a2d34",
            padx=14, pady=10
        )

        lbl_title = tk.Label(
            frame,
            text=title,
            bg="#111317",
            fg="white",
            font=("Segoe UI", 11, "bold")
        )
        lbl_title.pack(anchor="w")

        lbl_sub = tk.Label(
            frame,
            text=subtitle,
            bg="#111317",
            fg="#9da5b4",
            font=("Segoe UI", 9)
        )
        lbl_sub.pack(anchor="w", pady=(0, 4))

        # Expanded description (initially hidden)
        lbl_expl = tk.Label(
            frame,
            text=expl,
            bg="#111317",
            fg="#d7dae0",
            font=("Segoe UI", 9),
            wraplength=420,
            justify="left"
        )
        lbl_expl.pack(anchor="w", pady=(4, 0))
        lbl_expl.pack_forget()

        # Hover effects
        def on_enter(_event):
            frame.config(bg="#1b1f27")
            lbl_title.config(bg="#1b1f27")
            lbl_sub.config(bg="#1b1f27")
            lbl_expl.config(bg="#1b1f27")
            lbl_expl.pack(anchor="w", pady=(4, 0))

        def on_leave(_event):
            frame.config(bg="#111317")
            lbl_title.config(bg="#111317")
            lbl_sub.config(bg="#111317")
            lbl_expl.config(bg="#111317")
            lbl_expl.pack_forget()

        frame.bind("<Enter>", on_enter)
        frame.bind("<Leave>", on_leave)
        lbl_title.bind("<Enter>", on_enter)
        lbl_title.bind("<Leave>", on_leave)
        lbl_sub.bind("<Enter>", on_enter)
        lbl_sub.bind("<Leave>", on_leave)

        return frame

    # --------------------------------------------------------
    # CARD CONTAINER
    # --------------------------------------------------------
    cards_frame = tk.Frame(popup, bg="#1e2127")
    cards_frame.pack(pady=16)

    # --- PCA Card ---
    card_pca = make_card(
        cards_frame,
        "PCA — Principal Component Analysis",
        "Linear latent structure",
        "PCA finds directions of maximum variance. "
        "It reduces dimensionality by extracting linear combinations "
        "of your features. Works well for most datasets, fast and stable."
    )
    card_pca.pack(fill="x", padx=20, pady=6)

    # --- ICA Card ---
    card_ica = make_card(
        cards_frame,
        "ICA — Independent Component Analysis",
        "Separates independent signals",
        "ICA identifies statistically independent components. "
        "Useful when your data is a mixture of underlying sources.\n"
        "May struggle with collinearity but can reveal structure PCA misses."
    )
    card_ica.pack(fill="x", padx=20, pady=6)

    # --- Autoencoder Card ---
    card_auto = make_card(
        cards_frame,
        "Deep Autoencoder",
        "Nonlinear latent structure",
        "Learns nonlinear transformations of numeric and categorical features. "
        "Can uncover complex patterns and perform powerful compression. "
        "Slower but very expressive, recommended for rich, messy datasets."
    )
    card_auto.pack(fill="x", padx=20, pady=6)

    # --------------------------------------------------------
    # MODE BUTTONS
    # --------------------------------------------------------
    tk.Label(
        popup,
        text="Choose a mode:",
        bg="#1e2127",
        fg="#9da5b4",
        font=("Segoe UI", 10, "bold")
    ).pack(pady=(14, 6))

    modes = tk.Frame(popup, bg="#1e2127")
    modes.pack()

    def apply_fast():
        target_col = entry_target_popup.get().strip() or None
        popup.destroy()
        start_feature_creation_with_progress(
            True, False, False, 4, 0, 0, "StandardScaler",
            target_column=target_col,
        )

    def apply_balanced():
        target_col = entry_target_popup.get().strip() or None
        popup.destroy()
        start_feature_creation_with_progress(
            True, True, False, 5, 5, 0, "StandardScaler",
            target_column=target_col,
        )

    def apply_deep():
        target_col = entry_target_popup.get().strip() or None
        popup.destroy()
        start_feature_creation_with_progress(
            True, True, True, 5, 5, 6, "MinMaxScaler",
            target_column=target_col,
        )

    tk.Button(
        modes,
        text="⚡ Fast Mode",
        command=apply_fast,
        bg="#4C8BF5",
        fg="white",
        relief="flat",
        font=("Segoe UI", 10, "bold"),
        padx=14, pady=8,
        cursor="hand2"
    ).grid(row=0, column=0, padx=6)

    tk.Button(
        modes,
        text="🎯 Balanced",
        command=apply_balanced,
        bg="#8C53FF",
        fg="white",
        relief="flat",
        font=("Segoe UI", 10, "bold"),
        padx=14, pady=8,
        cursor="hand2"
    ).grid(row=0, column=1, padx=6)

    tk.Button(
        modes,
        text="🤖 Deep Mode",
        command=apply_deep,
        bg="#FF7A59",
        fg="white",
        relief="flat",
        font=("Segoe UI", 10, "bold"),
        padx=14, pady=8,
        cursor="hand2"
    ).grid(row=0, column=2, padx=6)


def on_feature_creation(
    use_pca=True,
    use_ica=True,
    use_auto=True,
    pca_comp=5,
    ica_comp=5,
    auto_dim=3,
    scaling="StandardScaler",
    target_column=None,   # <--- NEW
):
    """Backward-compatible wrapper that now just forwards to
    start_feature_creation_with_progress."""
    start_feature_creation_with_progress(
        use_pca,
        use_ica,
        use_auto,
        pca_comp,
        ica_comp,
        auto_dim,
        scaling,
        target_column=target_column,
    )


def on_transformations():
    """Open manual pruning popup without changing tabs."""
    def refresh_ui():
        meta = dataset.get_current_meta()
        update_preview_ui_from_meta(
            meta,
            lbl_rows_value,
            lbl_cols_value,
            lbl_dups_value,
            columns_tree,
        )
    Pruning.open_pruning_window(root, refresh_ui)

def on_feature_selection():
    if dataset.CURRENT_DF is None:
        messagebox.showwarning("No dataset", "Load a dataset first.")
        return

    target = APP_SETTINGS["target"]
    if not target:
        messagebox.showerror("Target required", "Please set target in Control Settings.")
        notebook.select(tab_controls)
        return

    model_type = APP_SETTINGS.get("model_type", "Random Forest")

    # Decide whether to use fixed K from settings or internal auto-CV
    top_k_setting = APP_SETTINGS.get("top_k")  # None => keep auto behavior inside run_feature_selection

    # Show busy cursor while selection runs
    root.config(cursor="watch")
    root.update_idletasks()

    try:
        results = run_feature_selection(
        target=target,
        top_k=None,          # K will be chosen via CV over k_candidates
        model_type=model_type,
    )

        selected = results["selected"]

        # Use preprocessed X_all from feature_selection
        X = results["X_all"].copy()
        y = results["y"]
        X[target] = y
        df_new = X[selected + [target]]

        dataset.CURRENT_DF = df_new
        dataset.CURRENT_META = dataset._summarize_dataframe(df_new)

        update_preview_ui_from_meta(
            dataset.CURRENT_META,
            lbl_rows_value,
            lbl_cols_value,
            lbl_dups_value,
            columns_tree,
        )

        notebook.select(tab_results)
        results_text.delete("1.0", "end")
        results_text.insert("end", "FEATURE SELECTION COMPLETED\n\n")
        results_text.insert("end", f"Model used for importance: {results['model_type']}\n")
        results_text.insert(
            "end",
            f"Task type: {'Regression' if results['is_regression'] else 'Classification'}\n\n",
        )
        results_text.insert("end", f"Selected {len(selected)} features:\n\n")
        for f in selected:
            results_text.insert("end", f" - {f}\n")


        # NEW: show K source
        k_used = results.get("k_used")
        if k_used is not None:
            if APP_SETTINGS.get("top_k") is None:
                msg = f"\nK was chosen automatically via cross-validation: K = {k_used}\n"
            else:
                msg = f"\nK was fixed from Control Settings: K = {k_used}\n"
            results_text.insert("end", msg)

        # Optional: show rankings
        mi_rank = results["mi_rank"]
        model_rank = results["model_rank"]

        results_text.insert("end", "\nTop 10 by Mutual Information:\n")
        for name, score in mi_rank.head(10).items():
            results_text.insert("end", f"  • {name}: {score:.4f}\n")

        results_text.insert("end", f"\nTop 10 by {results['model_type']} importance:\n")
        for name, score in model_rank.head(10).items():
            results_text.insert("end", f"  • {name}: {score:.4f}\n")

    except Exception as e:
        messagebox.showerror("Feature Selection Error", str(e))
    finally:
        root.config(cursor="")


def on_cancel_eval():
    global EVAL_CANCEL_EVENT
    if EVAL_THREAD is not None and EVAL_THREAD.is_alive():
        EVAL_CANCEL_EVENT.set()
        results_text.insert(
            "end",
            "\nCancellation requested... evaluation will stop after the current fold.\n"
        )
def _build_eval_signature_for_ui():
    """Small signature to detect if anything relevant changed
    since the last Evaluate Impact run."""
    df = dataset.CURRENT_DF
    orig = getattr(dataset, "ORIGINAL_DF", None)
    post = getattr(dataset, "POST_CREATION_DF", None)
    banned = getattr(dataset, "BANNED_FEATURES", [])

    target = APP_SETTINGS.get("target")
    model_type = APP_SETTINGS.get("model_type", "Random Forest")

    return (
        target,
        model_type,
        df.shape if df is not None else None,
        orig.shape if orig is not None else None,
        post.shape if post is not None else None,
        tuple(sorted(list(banned))),
    )

def on_eval_impact():
    global EVAL_THREAD, EVAL_CANCEL_EVENT
    global LAST_EVAL_SIGNATURE, LAST_EVAL_RESULTS

    if dataset.CURRENT_DF is None:
        messagebox.showwarning("No dataset", "Load a dataset first.")
        return

    target = APP_SETTINGS.get("target")
    if not target:
        messagebox.showerror(
            "Target required",
            "Please set a target column in the Control Settings tab.",
        )
        notebook.select(tab_controls)
        return

    model_type = APP_SETTINGS.get("model_type", "Random Forest")
    n_splits = evaluate_impact.DEFAULT_FOLDS

    current_sig = _build_eval_signature_for_ui()

    # 1) If an evaluation is already running, don't start a second one
    if EVAL_THREAD is not None and EVAL_THREAD.is_alive():
        messagebox.showinfo(
            "Evaluation running",
            "An evaluation is already running. Use the Cancel button to stop it.",
        )
        return

    notebook.select(tab_results)

    # 2) If nothing changed AND we have cached results, just re-render them
    if LAST_EVAL_SIGNATURE == current_sig and LAST_EVAL_RESULTS is not None:
        results_text.delete("1.0", "end")
        evaluate_impact.render_results_to_text_widget(LAST_EVAL_RESULTS, results_text)
        return

    # 3) Otherwise, start a new background evaluation
    EVAL_CANCEL_EVENT = threading.Event()

    results_text.delete("1.0", "end")
    results_text.insert("end", "Running impact evaluation...\n\n")
    results_text.insert("end", "You can keep using the UI. "
                               "Click 'Cancel' to stop the run.\n")

    root.config(cursor="watch")
    btn_cancel_eval.config(state="normal")

    def worker(sig_snapshot):
        global LAST_EVAL_SIGNATURE, LAST_EVAL_RESULTS, EVAL_THREAD
        try:
            results = evaluate_impact.run_impact_evaluation_cached(
                target=target,
                model_type=model_type,
                n_splits=n_splits,
                cancel_event=EVAL_CANCEL_EVENT,
            )
        except RuntimeError as e:
            # likely cancelled
            def _ui_cancelled():
                results_text.insert("end", "\nEvaluation cancelled by user.\n")
                btn_cancel_eval.config(state="disabled")
                root.config(cursor="")
            root.after(0, _ui_cancelled)
            EVAL_THREAD = None
            return
        except Exception as e:
            def _ui_error():
                messagebox.showerror("Evaluation Error", str(e))
                btn_cancel_eval.config(state="disabled")
                root.config(cursor="")
            root.after(0, _ui_error)
            EVAL_THREAD = None
            return

        def _ui_done():
            global LAST_EVAL_SIGNATURE, LAST_EVAL_RESULTS
            LAST_EVAL_SIGNATURE = sig_snapshot
            LAST_EVAL_RESULTS = results

            results_text.delete("1.0", "end")
            evaluate_impact.render_results_to_text_widget(results, results_text)

            btn_cancel_eval.config(state="disabled")
            root.config(cursor="")

        root.after(0, _ui_done)
        EVAL_THREAD = None

    # start worker thread
    EVAL_THREAD = threading.Thread(target=worker, args=(current_sig,), daemon=True)
    EVAL_THREAD.start()

def on_export_report():
    notebook.select(tab_results)
    # Generate a DOCX report using the current pipeline state
    export_report.generate_pipeline_report(APP_SETTINGS)


# Attach the commands to the existing buttons
btn_load.config(command=on_load_dataset)
btn_clean.config(command=on_data_cleaning)
btn_feature_creation.config(command=open_feature_creation_popup)
btn_transformations.config(command=on_transformations)
btn_feature_selection.config(command=on_feature_selection)
btn_eval_impact.config(command=on_eval_impact)
btn_cancel_eval.config(command=on_cancel_eval)
btn_export_report.config(command=on_export_report)

btn_apply.config(command=on_apply_settings)
btn_auto_k.config(command=on_auto_select_k) 


if __name__ == "__main__":
    root.mainloop()