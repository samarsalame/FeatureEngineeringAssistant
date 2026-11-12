# dataset.py
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Dict, Any
import pandas as pd
import numpy as np

CURRENT_DF: pd.DataFrame | None = None
CURRENT_META: Dict[str, Any] | None = None

# NEW: snapshots for Evaluate Impact
ORIGINAL_DF: pd.DataFrame | None = None      # dataset right after loading
POST_CREATION_DF: pd.DataFrame | None = None # dataset right after feature creation

CURRENT_DF = None
CURRENT_META = None
ORIGINAL_DF = None
POST_CREATION_DF = None

# New: track manually removed features
BANNED_FEATURES = []
CURRENT_META = None


def _detect_column_role(s: pd.Series) -> str:
    """Heuristic to classify column role."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"

    if s.dtype == "object":
        try:
            pd.to_datetime(
                s.dropna().sample(min(50, len(s.dropna()))),
                errors="raise",
            )
            return "datetime"
        except Exception:
            pass

    if pd.api.types.is_bool_dtype(s):
        return "boolean"

    if pd.api.types.is_numeric_dtype(s):
        unique_ratio = s.nunique(dropna=True) / max(len(s), 1)
        return "numeric" if unique_ratio > 0.3 else "numeric (low-cardinality)"

    nunique = s.nunique(dropna=True)
    if nunique == len(s.dropna()):
        return "id-like"
    if nunique <= 50:
        return "categorical"
    return "categorical (high-cardinality)"


def _summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Return dataset-level + column-level summary."""
    n_rows, n_cols = df.shape
    n_dup_rows = df.duplicated().sum()

    missing_counts = df.isna().sum()
    missing_percent = (missing_counts / max(n_rows, 1)) * 100

    columns_info = []
    for col in df.columns:
        s = df[col]
        role = _detect_column_role(s)
        col_info = {
            "name": col,
            "dtype": str(s.dtype),
            "role": role,
            "n_missing": int(missing_counts[col]),
            "pct_missing": float(round(missing_percent[col], 2)),
            "n_unique": int(s.nunique(dropna=True)),
            "example": s.dropna().iloc[0] if s.dropna().size > 0 else None,
        }
        columns_info.append(col_info)

    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "n_dup_rows": int(n_dup_rows),
        "columns": columns_info,
    }


def _render_summary_to_text_widget(
    preview_text: tk.Text,
    df: pd.DataFrame,
    meta: Dict[str, Any],
    file_path: str,
):
    """Fill the preview_text widget with an overview + head()."""
    preview_text.delete("1.0", "end")

    preview_text.insert("end", f"Loaded file: {file_path}\n")
    preview_text.insert("end", f"Rows: {meta['n_rows']}  |  Columns: {meta['n_cols']}\n")
    preview_text.insert("end", f"Duplicate rows: {meta['n_dup_rows']}\n\n")

    preview_text.insert("end", "Column summary:\n")
    preview_text.insert(
        "end",
        "name | role | dtype | missing (# / %) | unique | example\n"
    )
    preview_text.insert("end", "-" * 90 + "\n")

    for c in meta["columns"]:
        preview_text.insert(
            "end",
            f"{c['name']} | {c['role']} | {c['dtype']} | "
            f"{c['n_missing']} ({c['pct_missing']}%) | "
            f"{c['n_unique']} | {c['example']}\n"
        )

    preview_text.insert("end", "\nFirst 10 rows:\n\n")
    preview_text.insert("end", df.head(10).to_string(index=False))
    preview_text.insert("end", "\n")


def open_dataset(preview_text_widget: tk.Text):
    """
    Entry point for 'Load Dataset' button.
    Loads file, builds metadata, stores df/meta globally,
    renders summary into preview_text_widget.
    """
    global CURRENT_DF, CURRENT_META
    global ORIGINAL_DF, POST_CREATION_DF, BANNED_FEATURES, TASK_TYPE
    file_path = filedialog.askopenfilename(
        title="Select dataset",
        filetypes=[
            ("Data files", "*.csv *.xlsx *.xls *.parquet"),
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx *.xls"),
            ("Parquet files", "*.parquet"),
            ("All files", "*.*"),
        ],
    )
    if not file_path:
        return

    try:
        lower = file_path.lower()
        if lower.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        elif lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path, engine="python")
    except Exception as e:
        messagebox.showerror("Error", f"Could not load dataset:\n{e}")
        return

    if df.empty:
        messagebox.showwarning("Warning", "Loaded dataset is empty.")
        return

    df.replace(
        to_replace=["NA", "N/A", "na", "n/a", "null", "NULL", "?"],
        value=np.nan,
        inplace=True,
    )

    meta = _summarize_dataframe(df)
    ORIGINAL_DF = df.copy()      # baseline snapshot
    POST_CREATION_DF = None      # reset later-stage snapshot
    BANNED_FEATURES = []         # reset banned columns list
    CURRENT_DF = df
    CURRENT_META = meta
    TASK_TYPE = None             # NEW: reset user choice when new data is loaded

    _render_summary_to_text_widget(preview_text_widget, df, meta, file_path)

    messagebox.showinfo(
        "Dataset loaded",
        f"Dataset loaded successfully with {meta['n_rows']} rows "
        f"and {meta['n_cols']} columns.",
    )


def get_current_df():
    return CURRENT_DF


def get_current_meta():
    return CURRENT_META
# NEW: helpers for task type
def set_task_type(task_type: str | None):
    """
    Store user-chosen task type globally.

    task_type should be "regression", "classification", or None for auto-detection.
    """
    global TASK_TYPE
    if task_type is None:
        TASK_TYPE = None
    else:
        task_type = str(task_type).lower()
        if task_type in ("regression", "classification"):
            TASK_TYPE = task_type
        else:
            TASK_TYPE = None


def get_task_type() -> str | None:
    return TASK_TYPE
