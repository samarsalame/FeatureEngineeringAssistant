from __future__ import annotations

from typing import Dict, Any, List, Optional
import os
import datetime

import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd

import dataset
import evaluate_impact as impact_mod

try:
    from docx import Document
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


# ---------------------------------------------------------------------
# Small helpers to read current app state
# ---------------------------------------------------------------------
def _get_meta() -> Dict[str, Any]:
    """Return dataset metadata, recomputing if needed."""
    if dataset.CURRENT_META is not None:
        return dataset.CURRENT_META

    if dataset.CURRENT_DF is None:
        return {}

    # fall back to computing summary from CURRENT_DF
    try:
        return dataset._summarize_dataframe(dataset.CURRENT_DF)
    except Exception:
        return {}


def _latent_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Count latent features by family."""
    if df is None:
        return {"total": 0, "pca": 0, "ica": 0, "auto": 0}

    cols = list(df.columns)
    total = sum(str(c).startswith("latent_") for c in cols)
    pca = sum(str(c).startswith("latent_pca_") for c in cols)
    ica = sum(str(c).startswith("latent_ica_") for c in cols)
    auto = sum(str(c).startswith("latent_auto_") for c in cols)

    return {
        "total": total,
        "pca": pca,
        "ica": ica,
        "auto": auto,
    }


def _safe_shape(df: Optional[pd.DataFrame]) -> str:
    if df is None:
        return "–"
    try:
        return f"{df.shape[0]} rows × {df.shape[1]} columns"
    except Exception:
        return "–"


def _collect_impact_results() -> Optional[Dict[str, Any]]:
    """
    Return last evaluation results from evaluate_impact, if any.
    We only read the in-memory cache; we do NOT re-run evaluation here,
    to avoid long blocking runs when exporting.
    """
    res = getattr(impact_mod, "_LAST_RESULTS", None)
    if not res:
        return None
    if not res.get("stages"):
        return None
    return res


# ---------------------------------------------------------------------
# DOCX building helpers
# ---------------------------------------------------------------------
def _apply_base_style(doc: Document) -> None:
    """Set a simple, clean base style."""
    style = doc.styles["Normal"]
    style.font.name = "Segoe UI"
    style.font.size = Pt(10)


def _add_title(doc: Document, title: str) -> None:
    p = doc.add_paragraph()
    run = p.add_run(title)
    run.bold = True
    run.font.size = Pt(18)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _add_small_caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(9)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _add_h1(doc: Document, text: str) -> None:
    doc.add_heading(text, level=1)


def _add_h2(doc: Document, text: str) -> None:
    doc.add_heading(text, level=2)


def _add_kv_table(doc: Document, rows: List[tuple]) -> None:
    """
    Add a 2-col key/value table. rows: list of (key, value) strings.
    """
    table = doc.add_table(rows=len(rows), cols=2)
    table.style = "Light List"

    for i, (k, v) in enumerate(rows):
        key_cell = table.cell(i, 0)
        val_cell = table.cell(i, 1)

        key_run = key_cell.paragraphs[0].add_run(str(k))
        key_run.bold = True

        val_cell.paragraphs[0].add_run(str(v))


def _format_metric(mdict: Optional[Dict[str, float]]) -> str:
    if not mdict:
        return "n/a"
    mean = mdict.get("mean")
    std = mdict.get("std")
    if mean is None:
        return "n/a"
    if std is None or std == 0:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def _add_impact_section(doc: Document, impact_results: Dict[str, Any]) -> None:
    stages = impact_results.get("stages", [])
    main_metric = impact_results.get("main_metric")

    if not stages:
        doc.add_paragraph("No evaluation stages were found.")
        return

    # Determine task type and key metric columns (like render_results_to_text_widget)
    task_type = None
    for st in stages:
        if "task_type" in st:
            task_type = st["task_type"]
            break

    if task_type == "regression":
        metric_cols = ["rmse", "mae", "r2"]
        main_name = "RMSE"
    else:
        metric_cols = ["accuracy", "f1", "precision", "recall", "roc_auc"]
        main_name = "F1-score"

    n_cols = 2 + len(metric_cols) + 1  # Stage, #features, metrics..., Δ
    table = doc.add_table(rows=len(stages) + 1, cols=n_cols)
    table.style = "Light List Accent 1"

    # Header row
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Stage"
    hdr_cells[1].text = "# Features"
    for j, m in enumerate(metric_cols, start=2):
        hdr_cells[j].text = m
    hdr_cells[n_cols - 1].text = "Δ main"

    # Body
    for i, st in enumerate(stages, start=1):
        cells = table.rows[i].cells
        name = st.get("name", "Stage")
        n_feat = st.get("n_features", 0)

        cells[0].text = str(name)
        cells[1].text = str(n_feat)

        if "metrics" in st:
            metrics = st["metrics"]
            for j, m in enumerate(metric_cols, start=2):
                cells[j].text = _format_metric(metrics.get(m))

            delta = st.get("delta_main")
            if delta is None:
                delta_str = "–"
            else:
                if main_metric == "rmse":
                    direction = "↓" if delta < 0 else "↑"
                else:
                    direction = "↑" if delta > 0 else "↓"
                delta_str = f"{direction} {delta:+.4f}"
        else:
            for j in range(2, 2 + len(metric_cols)):
                cells[j].text = "error"
            delta_str = "n/a"

        cells[n_cols - 1].text = delta_str

    # Overall improvement summary
    doc.add_paragraph()
    try:
        first = next(
            st for st in stages
            if "metrics" in st and main_metric in st["metrics"]
        )
        last = next(
            st for st in reversed(stages)
            if "metrics" in st and main_metric in st["metrics"]
        )
        base = first["metrics"][main_metric]["mean"]
        final = last["metrics"][main_metric]["mean"]

        if main_metric == "rmse":
            abs_change = base - final
            rel_change = abs_change / base if base != 0 else 0.0
            text = (
                f"Overall, the pipeline reduced {main_name} "
                f"from {base:.4f} to {final:.4f} "
                f"({abs_change:+.4f} absolute, {rel_change*100:+.1f}% relative)."
            )
        else:
            abs_change = final - base
            rel_change = abs_change / base if base != 0 else 0.0
            text = (
                f"Overall, the pipeline improved {main_name} "
                f"from {base:.4f} to {final:.4f} "
                f"({abs_change:+.4f} absolute, {rel_change*100:+.1f}% relative)."
            )

        doc.add_paragraph(text)
    except Exception:
        pass

    doc.add_paragraph(
        "All values are computed via k-fold cross-validation using the same "
        "model configuration across all stages."
    )


# ---------------------------------------------------------------------
# Main entry point called from Main.py
# ---------------------------------------------------------------------
def generate_pipeline_report(app_settings: Dict[str, Any]) -> None:
    """
    Entry point for the 'Export Report' button.

    app_settings: the APP_SETTINGS dict from Main.py
    """
    if dataset.CURRENT_DF is None:
        messagebox.showwarning("No dataset", "Load a dataset before exporting a report.")
        return

    if not HAS_DOCX:
        messagebox.showerror(
            "python-docx missing",
            "The 'python-docx' package is required to export a DOCX report.\n\n"
            "Install it with:\n\n    pip install python-docx"
        )
        return

    # Ask user for output path
    file_path = filedialog.asksaveasfilename(
        title="Save pipeline report",
        defaultextension=".docx",
        filetypes=[("Word Document", "*.docx")],
    )
    if not file_path:
        return

    # --- Collect state ---
    meta = _get_meta()

    cur_df = dataset.CURRENT_DF
    orig_df = getattr(dataset, "ORIGINAL_DF", None)
    post_df = getattr(dataset, "POST_CREATION_DF", None)
    banned = list(getattr(dataset, "BANNED_FEATURES", []))

    # latent summary based on current dataset
    latent_info = _latent_counts(cur_df)

    target = app_settings.get("target")
    test_size = app_settings.get("test_size")
    model_type = app_settings.get("model_type", "Random Forest")

    # impact evaluation (if run previously)
    impact_results = _collect_impact_results()

    # Determine "active features" as current columns minus target
    if cur_df is not None:
        current_cols = list(cur_df.columns)
        if target in current_cols:
            active_features = [c for c in current_cols if c != target]
        else:
            active_features = current_cols
    else:
        active_features = []

    # -----------------------------------------------------------------
    # Build DOCX
    # -----------------------------------------------------------------
    doc = Document()
    _apply_base_style(doc)

    # Title + caption
    _add_title(doc, "Feature Engineering Assistant — Pipeline Report")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    _add_small_caption(doc, f"Generated on {now}")

    # 1) Dataset section
    _add_h1(doc, "1. Dataset Overview")

    n_rows = meta.get("n_rows") or "–"
    n_cols = meta.get("n_cols") or "–"
    n_dup = meta.get("n_dup_rows") or meta.get("n_duplicates") or 0

    _add_kv_table(
        doc,
        [
            ("Rows", n_rows),
            ("Columns", n_cols),
            ("Duplicate rows", n_dup),
            ("Original dataset shape", _safe_shape(orig_df)),
            ("Post feature-creation shape", _safe_shape(post_df)),
            ("Current dataset shape", _safe_shape(cur_df)),
        ],
    )

    doc.add_paragraph()
    doc.add_paragraph(
        "Column-level roles are inferred heuristically (numeric, categorical, "
        "datetime, id-like, etc.) based on the raw pandas dtypes and basic "
        "distribution statistics."
    )

    # Optional: small table of first few columns + roles
    cols_meta = meta.get("columns") or []
    if cols_meta:
        _add_h2(doc, "1.1 Column Roles (sample)")
        k = min(10, len(cols_meta))
        table = doc.add_table(rows=k + 1, cols=5)
        table.style = "Light List"

        hdr = table.rows[0].cells
        hdr[0].text = "Name"
        hdr[1].text = "Role"
        hdr[2].text = "Dtype"
        hdr[3].text = "Missing %"
        hdr[4].text = "Unique values"

        for i, cinfo in enumerate(cols_meta[:k], start=1):
            row = table.rows[i].cells
            row[0].text = str(cinfo.get("name", ""))
            row[1].text = str(cinfo.get("role", ""))
            row[2].text = str(cinfo.get("dtype", ""))
            row[3].text = str(cinfo.get("pct_missing", ""))
            row[4].text = str(cinfo.get("n_unique", ""))

    # 2) Feature creation section
    _add_h1(doc, "2. Latent Feature Creation")

    doc.add_paragraph(
        "Latent features are generated from numeric and categorical inputs via "
        "a combination of PCA, ICA and a deep tabular autoencoder. All latent "
        "columns are prefixed with 'latent_' to keep them clearly separated "
        "from the original schema."
    )

    _add_kv_table(
        doc,
        [
            ("Total latent features (current dataset)", latent_info["total"]),
            ("PCA-derived (latent_pca_*)", latent_info["pca"]),
            ("ICA-derived (latent_ica_*)", latent_info["ica"]),
            ("Autoencoder-derived (latent_auto_*)", latent_info["auto"]),
        ],
    )

    doc.add_paragraph(
        "Each latent block is normalized and checked to ensure it only contains "
        "flat numeric values with unique column names and no nested objects."
    )

    # 3) Manual pruning section
    _add_h1(doc, "3. Manual Transformations & Pruning")

    doc.add_paragraph(
        "The manual pruning panel keeps track of columns that were explicitly "
        "removed by the user. These are stored in a global 'banned features' "
        "list and excluded from subsequent stages of the pipeline."
    )

    _add_kv_table(
        doc,
        [
            ("Number of banned features", len(banned)),
        ],
    )

    if banned:
        doc.add_paragraph("Banned feature names:")
        for col in banned:
            doc.add_paragraph(f"• {col}", style="List Bullet")
    else:
        doc.add_paragraph("No features have been manually banned in this session.")

    # 4) Current feature set + model configuration
    _add_h1(doc, "4. Model Configuration & Active Features")

    _add_kv_table(
        doc,
        [
            ("Target column", target or "– not set –"),
            ("Test size", test_size if test_size is not None else "–"),
            ("Model type", model_type),
            ("Current #features (excluding target)", len(active_features)),
        ],
    )

    if active_features:
        doc.add_paragraph("Current active feature names (post pipeline):")
        # keep it readable; if many, put them in a compact bullet list
        max_show = 80
        to_show = active_features[:max_show]
        for name in to_show:
            doc.add_paragraph(f"• {name}", style="List Bullet")
        if len(active_features) > max_show:
            doc.add_paragraph(
                f"... and {len(active_features) - max_show} more features.",
                style=None,
            )

    # 5) Impact evaluation section
    _add_h1(doc, "5. Impact Evaluation Across Pipeline Stages")

    if impact_results is None:
        doc.add_paragraph(
            "No cached impact evaluation was found for this session.\n\n"
            "To include quantitative before/after metrics, run 'Evaluate Impact' "
            "from the sidebar before exporting the report."
        )
    else:
        doc.add_paragraph(
            "This section summarizes how model performance evolves as we move "
            "from raw features to feature creation, pruning and the final "
            "feature-selected dataset."
        )
        _add_impact_section(doc, impact_results)

    # Save document
    try:
        doc.save(file_path)
    except Exception as e:
        messagebox.showerror(
            "Save failed",
            f"Could not save report to:\n{file_path}\n\nError: {e}",
        )
        return

    messagebox.showinfo("Report exported", f"Report saved to:\n{file_path}")
