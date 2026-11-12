# data_preprocessing.py
import re
import tkinter as tk
from tkinter import messagebox
from typing import Dict, Any, List

import numpy as np
import pandas as pd

import dataset


def _describe_changes(
    before: pd.DataFrame,
    after: pd.DataFrame,
    empty_cols: List[str],
    id_like_cols: List[str],
    high_missing_cols: List[str],
    constant_cols: List[str],
    dup_rows_dropped: int,
) -> str:
    """Build a human-readable summary of cleaning operations."""
    lines: List[str] = []
    lines.append("Data cleaning summary:\n")
    lines.append(
        f"Rows: {before.shape[0]} → {after.shape[0]} "
        f"(dropped {before.shape[0] - after.shape[0]} rows: {dup_rows_dropped} duplicate)\n"
    )
    lines.append(f"Columns: {before.shape[1]} → {after.shape[1]}\n\n")

    if dup_rows_dropped > 0:
        lines.append(f"- Removed {dup_rows_dropped} duplicate rows.\n")
    if empty_cols:
        lines.append(f"- Dropped {len(empty_cols)} fully empty columns.\n")
    if high_missing_cols:
        lines.append(f"- Dropped {len(high_missing_cols)} columns with >60% missing values.\n")
    if constant_cols:
        lines.append(f"- Dropped {len(constant_cols)} constant / near-constant columns.\n")
    if id_like_cols:
        lines.append(f"- Dropped {len(id_like_cols)} ID-like columns (almost unique per row).\n")

    lines.append("- Filled remaining missing numeric values with column median.\n")
    lines.append("- Filled remaining missing categorical values with mode (most frequent).\n\n")

    return "".join(lines)


def _is_id_like_name(col_name: str) -> bool:
    """
    Heuristic for ID-like columns based on the column name only.

    Examples that return True:
        'id', 'ID', 'customer_id', 'userId', 'OrderID'
    Examples that should return False:
        'width', 'valid_flag', 'grid'
    """
    name = str(col_name).strip()
    lower = name.lower()

    # Exact 'id'
    if lower == "id":
        return True

    # Split on non-alphanumeric to find tokens like 'customer', 'id', etc.
    tokens = re.split(r"[^a-z0-9]+", lower)
    tokens = [t for t in tokens if t]  # remove empty strings

    # Token 'id' on its own, e.g. 'customer id'
    if "id" in tokens:
        return True

    # Things like 'customerid', 'userid', 'orderid'
    # but avoid common non-ID words that end with 'id'
    skip = {"grid", "valid", "invalid"}
    for t in tokens:
        if t.endswith("id") and len(t) <= 8 and t not in skip:
            return True

    return False


def run_data_cleaning(preview_text: tk.Text) -> None:
    """
    Clean the currently loaded dataset in-place and update preview.

    Steps:
    - Drop fully empty columns.
    - Drop duplicate rows.
    - Drop columns with >60% missing values.
    - Drop constant / near-constant columns.
    - Drop ID-like columns (name suggests ID and values are almost unique per row).
    - Impute remaining missing numeric values with median.
    - Impute remaining missing categorical values with mode.
    - Recompute metadata and re-render the dataset preview.
    """
    if dataset.CURRENT_DF is None:
        messagebox.showwarning("No dataset", "Please load a dataset before running data cleaning.")
        return

    df_before = dataset.CURRENT_DF.copy()

    # 1) Drop fully empty columns
    empty_cols = [c for c in df_before.columns if df_before[c].isna().all()]
    df = df_before.drop(columns=empty_cols, errors="ignore")

    # 2) Drop duplicate rows
    dup_rows = int(df.duplicated().sum())
    if dup_rows > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    # 3) Drop columns with very high missing ratio (> 60%)
    missing_ratio = df.isna().mean()
    high_missing_cols = missing_ratio[missing_ratio > 0.60].index.tolist()
    df = df.drop(columns=high_missing_cols, errors="ignore")

    # 4) Drop constant / near-constant columns (only 1 unique non-NaN value)
    nunique = df.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    df = df.drop(columns=constant_cols, errors="ignore")

    # 5) Drop ID-like columns: name looks like an ID AND very high cardinality
    id_like_cols: List[str] = []
    n_rows = df.shape[0]

    if n_rows > 0:
        for col in df.columns:
            # Only consider columns whose names look ID-like
            if not _is_id_like_name(col):
                continue

            unique_vals = df[col].nunique(dropna=True)

            # Drop only if values are almost unique per row
            if unique_vals / n_rows > 0.9:
                id_like_cols.append(col)

    df = df.drop(columns=id_like_cols, errors="ignore")

    # 6) Impute missing values: numeric -> median, categorical -> mode
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in num_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    for col in cat_cols:
        if df[col].isna().any():
            mode_series = df[col].mode(dropna=True)
            if not mode_series.empty:
                mode_val = mode_series.iloc[0]
                df[col] = df[col].fillna(mode_val)

    # Update global dataset state (keep ORIGINAL_DF as the raw snapshot)
    dataset.CURRENT_DF = df

    # Recompute metadata and re-render preview
    try:
        meta = dataset._summarize_dataframe(df)  # type: ignore[attr-defined]
    except Exception:
        # Fallback: simple meta if private helper not available
        meta = {
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "n_dup_rows": int(df.duplicated().sum()),
            "columns": [],
        }

    dataset.CURRENT_META = meta

    # Clear the preview text and show updated summary + cleaning log
    preview_text.configure(state="normal")
    preview_text.delete("1.0", "end")

    try:
        dataset._render_summary_to_text_widget(
            preview_text, df, meta, file_path="[cleaned in memory]"
        )  # type: ignore[attr-defined]
    except Exception:
        # Minimal fallback if render helper is not available
        preview_text.insert(
            "end",
            f"Cleaned dataset (in-memory). Rows: {df.shape[0]}, Columns: {df.shape[1]}\n\n",
        )
        preview_text.insert("end", df.head(10).to_string(index=False))
        preview_text.insert("end", "\n\n")

    summary_text = _describe_changes(
        before=df_before,
        after=df,
        empty_cols=empty_cols,
        id_like_cols=id_like_cols,
        high_missing_cols=high_missing_cols,
        constant_cols=constant_cols,
        dup_rows_dropped=dup_rows,
    )
    preview_text.insert("end", "\n" + "-" * 80 + "\n\n")
    preview_text.insert("end", summary_text)

    preview_text.configure(state="disabled")

    messagebox.showinfo(
        "Data cleaning complete",
        "Data cleaning finished successfully. The current dataset has been updated.",
    )
