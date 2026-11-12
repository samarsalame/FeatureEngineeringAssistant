from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from scipy.stats import ttest_rel

import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import dataset
from feature_selection import _prepare_supervised_Xy, _build_model

DEFAULT_FOLDS = 5

# simple in-memory cache
_LAST_SIGNATURE: Optional[Tuple] = None
_LAST_RESULTS: Optional[Dict[str, Any]] = None


def _detect_task_type(y: pd.Series) -> str:
    """
    Decide whether to treat the problem as regression or classification.

    Priority:
    1) User-chosen task type from dataset.TASK_TYPE (or dataset.get_task_type()).
    2) Fallback heuristic based on y type and number of unique values.
    """
    # 1) Check user override
    user_choice = None
    if hasattr(dataset, "get_task_type"):
        try:
            user_choice = dataset.get_task_type()
        except Exception:
            user_choice = getattr(dataset, "TASK_TYPE", None)
    else:
        user_choice = getattr(dataset, "TASK_TYPE", None)

    if isinstance(user_choice, str):
        user_choice_lower = user_choice.lower()
        if user_choice_lower in ("regression", "classification"):
            return user_choice_lower

    # 2) Fallback: original heuristic (only when user hasn't specified)
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 15:
        return "regression"
    return "classification"


def _build_cv_splitter(y: np.ndarray, task_type: str, n_splits: int):
    n_splits = max(2, int(n_splits))

    if task_type == "classification":
        _, counts = np.unique(y, return_counts=True)
        max_splits = counts.min()
        n_splits = min(n_splits, max_splits)
        if n_splits < 2:
            return KFold(n_splits=2, shuffle=True, random_state=42)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # regression
    n_samples = len(y)
    n_splits = min(n_splits, max(2, n_samples))
    return KFold(n_splits=n_splits, shuffle=True, random_state=42)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size between two paired samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = b - a
    return diff.mean() / (diff.std(ddof=1) + 1e-9)


def _evaluate_single_stage(
    df: pd.DataFrame,
    target: str,
    model_type: str,
    n_splits: int,
    cancel_event=None,
) -> Dict[str, Any]:
    """
    Run k-fold CV on a single dataframe and return aggregate metrics,
    plus per-fold main metric scores and outputs (for plots/tests).

    cancel_event: optional threading.Event; if set(), we abort.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Evaluation cancelled")

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataframe for evaluation.")

    # Build X, y using the same helper as feature_selection
    X, y = _prepare_supervised_Xy(df, target)
    y_series = pd.Series(y).reset_index(drop=True)
    y_array = np.asarray(y_series).ravel()

    if X.shape[0] != len(y_array):
        raise ValueError("X and y have mismatched number of rows during evaluation.")

    task_type = _detect_task_type(y_series)
    splitter = _build_cv_splitter(y_array, task_type, n_splits)

    fold_metrics: List[Dict[str, float]] = []
    fold_main_scores: List[float] = []
    fold_outputs: List[Dict[str, Any]] = []

    for train_idx, test_idx in splitter.split(X, y_array):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Evaluation cancelled")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]

        mdl = _build_model(model_type, is_regression=(task_type == "regression"))
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)

        metrics: Dict[str, float] = {}

        if task_type == "regression":
            # Older sklearn may not support squared=False, so compute RMSE manually
            mse = mean_squared_error(y_test, y_pred)
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            metrics["rmse"] = rmse
            metrics["mae"] = mae
            metrics["r2"] = r2

            main_score = rmse
            y_proba = None  # not applicable for regression

        else:
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            metrics["accuracy"] = float(acc)
            metrics["f1"] = float(f1)
            metrics["precision"] = float(prec)
            metrics["recall"] = float(rec)

            main_score = float(f1)
            y_proba = None

            # ROC-AUC + probabilities if available
            try:
                proba = mdl.predict_proba(X_test)
                if proba is not None:
                    classes = np.unique(y_test)
                    if len(classes) == 2:
                        # binarize against the model's positive class so that
                        # string labels (e.g. "M"/"B") are supported
                        auc_val = roc_auc_score(y_test == mdl.classes_[1], proba[:, 1])
                    else:
                        auc_val = roc_auc_score(y_test, proba, multi_class="ovr")
                    metrics["roc_auc"] = float(auc_val)

                    # store probability of positive class for binary ROC/PR plots
                    if proba.shape[1] == 2:
                        y_proba = proba[:, 1]
            except Exception:
                # if model doesn't support predict_proba, just skip ROC-AUC
                pass

        fold_metrics.append(metrics)
        fold_main_scores.append(main_score)
        fold_outputs.append(
            {
                "y_true": y_test.copy(),
                "y_pred": y_pred.copy(),
                "y_proba": y_proba,
            }
        )

    agg: Dict[str, Dict[str, float]] = {}
    if fold_metrics:
        metric_names = fold_metrics[0].keys()
        for m in metric_names:
            vals = [fm[m] for fm in fold_metrics if m in fm]
            if not vals:
                continue
            agg[m] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    return {
        "n_features": int(X.shape[1]),
        "task_type": task_type,
        "metrics": agg,
        "fold_main_scores": np.asarray(fold_main_scores, dtype=float),
        "fold_outputs": fold_outputs,
    }




def _safe_df_copy(obj: Any) -> Optional[pd.DataFrame]:
    if obj is None:
        return None
    try:
        return obj.copy()
    except Exception:
        return None


def _build_signature(
    target: str,
    model_type: str,
    n_splits: int,
) -> Tuple:
    """A cheap signature that changes when relevant state changes."""
    cur = dataset.CURRENT_DF
    orig = getattr(dataset, "ORIGINAL_DF", None)
    post = getattr(dataset, "POST_CREATION_DF", None)
    banned = getattr(dataset, "BANNED_FEATURES", [])

    return (
        target,
        model_type,
        int(n_splits),
        cur.shape if cur is not None else None,
        orig.shape if orig is not None else None,
        post.shape if post is not None else None,
        tuple(sorted(list(banned))),
    )


def run_impact_evaluation(
    target: str,
    model_type: str = "Random Forest",
    n_splits: int = DEFAULT_FOLDS,
    cancel_event=None,
) -> Dict[str, Any]:
    """
    Evaluate how much feature engineering helped.

    For each pipeline snapshot (baseline, pruning, feature creation,
    current dataset) we:

      * Ensure the target column is present (re-attaching it from a
        reference dataframe when possible).
      * Run k-fold CV with the same model configuration.
      * Aggregate metrics, deltas and significance tests.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Evaluation cancelled")

    if dataset.CURRENT_DF is None:
        raise ValueError("No dataset loaded. Please load a dataset first.")

    # Take safe copies of all snapshots
    original_df = _safe_df_copy(getattr(dataset, "ORIGINAL_DF", None))
    if original_df is None:
        original_df = _safe_df_copy(dataset.CURRENT_DF)

    post_creation = _safe_df_copy(getattr(dataset, "POST_CREATION_DF", None))
    current_df = _safe_df_copy(dataset.CURRENT_DF)

    # We will use this as the reference for adding back the target
    reference_df = None
    if current_df is not None and target in current_df.columns:
        reference_df = current_df
    elif original_df is not None and target in original_df.columns:
        reference_df = original_df

    # Never treat the target as banned
    banned = list(getattr(dataset, "BANNED_FEATURES", [])) or []
    banned = [c for c in banned if c != target]

    def _ensure_target(
        df: Optional[pd.DataFrame],
        target: str,
        ref: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        """
        Make sure `target` is a column of df. If it's missing but ref has it
        and row counts match, copy it over.
        """
        if df is None:
            return None
        if target in df.columns:
            return df
        if ref is not None and target in ref.columns and len(ref) == len(df):
            df2 = df.copy()
            df2[target] = ref[target].values
            return df2
        # Could not safely re-attach
        return df

    # Apply _ensure_target to each snapshot BEFORE building stages
    original_df = _ensure_target(original_df, target, reference_df)
    post_creation = _ensure_target(post_creation, target, reference_df)
    current_df = _ensure_target(current_df, target, reference_df)

    def _has_target(df: Optional[pd.DataFrame]) -> bool:
        return df is not None and target in df.columns

    stages: List[Tuple[str, pd.DataFrame]] = []

    # 1) Baseline (raw features)
    if _has_target(original_df):
        stages.append(("Baseline (raw features)", original_df))

    # 2) Transformations (pruned columns)
    if banned and _has_target(original_df):
        trans_df = original_df.drop(
            columns=[c for c in banned if c in original_df.columns],
            errors="ignore",
        )
        trans_df = _ensure_target(trans_df, target, reference_df)
        if _has_target(trans_df):
            stages.append(("Transformations (pruned columns)", trans_df))

    # 3) Feature Creation snapshot
    if post_creation is not None:
        df_fc = post_creation
        if banned:
            df_fc = df_fc.drop(
                columns=[c for c in banned if c in df_fc.columns],
                errors="ignore",
            )
        df_fc = _ensure_target(df_fc, target, reference_df)
        if _has_target(df_fc):
            stages.append(("Feature Creation (latent + pruning)", df_fc))

    # 4) Current dataset (after feature selection)
    if _has_target(current_df):
        stages.append(("Feature Selection (current dataset)", current_df))

    if not stages:
        raise ValueError(
            f"Could not assemble any stages for evaluation: "
            f"none of the snapshots contain the target column '{target}'."
        )

    # --------- Run CV on each stage ---------
    stage_results: List[Dict[str, Any]] = []

    for name, df_stage in stages:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Evaluation cancelled")

        if target not in df_stage.columns:
            # Safety: if we still don't have the target, record an error row
            stage_results.append(
                {
                    "name": name,
                    "error": f"Target '{target}' not found in this stage (skipped).",
                    "n_features": int(df_stage.shape[1]),
                }
            )
            continue

        try:
            res = _evaluate_single_stage(
                df_stage, target, model_type, n_splits, cancel_event=cancel_event
            )
            res["name"] = name
            stage_results.append(res)
        except Exception as e:
            stage_results.append(
                {
                    "name": name,
                    "error": str(e),
                    "n_features": int(df_stage.shape[1]),
                }
            )

    # --------- Pick main metric (rmse for regression, f1 for classification) ---------
    main_metric = None
    for st in stage_results:
        if "metrics" not in st:
            continue
        task_type = st.get("task_type")
        if task_type == "regression":
            main_metric = "rmse"
        elif task_type == "classification":
            main_metric = "f1"
        if main_metric is not None:
            break

    # --------- Compute deltas vs previous stage ---------
    prev_val = None
    for st in stage_results:
        delta = None
        if main_metric and "metrics" in st:
            metric_info = st["metrics"].get(main_metric)
            if metric_info is not None:
                cur_val = float(metric_info["mean"])
                if prev_val is not None:
                    delta = cur_val - prev_val
                prev_val = cur_val
        st["delta_main"] = delta

    # --------- Paired t-test + Cohen's d between consecutive stages ---------
    significance_rows: List[Dict[str, Any]] = []
    if main_metric is not None:
        for i in range(1, len(stage_results)):
            a_stage = stage_results[i - 1]
            b_stage = stage_results[i]
            a_scores = a_stage.get("fold_main_scores")
            b_scores = b_stage.get("fold_main_scores")
            if a_scores is None or b_scores is None:
                continue
            a_scores = np.asarray(a_scores, dtype=float)
            b_scores = np.asarray(b_scores, dtype=float)
            if a_scores.shape != b_scores.shape or a_scores.size < 2:
                continue

            t_stat, p_val = ttest_rel(a_scores, b_scores)
            effect = _cohens_d(a_scores, b_scores)

            significance_rows.append(
                {
                    "from": a_stage.get("name", f"Stage {i}"),
                    "to": b_stage.get("name", f"Stage {i+1}"),
                    "p_value": float(p_val),
                    "effect_size": float(effect),
                }
            )

    return {
        "stages": stage_results,
        "main_metric": main_metric,
        "significance": significance_rows,
    }




def run_impact_evaluation_cached(
    target: str,
    model_type: str = "Random Forest",
    n_splits: int = DEFAULT_FOLDS,
    cancel_event=None,
) -> Dict[str, Any]:
    """
    Wrapper with simple caching: if nothing changed (by signature),
    return last results instead of recomputing.
    """
    global _LAST_SIGNATURE, _LAST_RESULTS

    sig = _build_signature(target, model_type, n_splits)

    if _LAST_SIGNATURE == sig and _LAST_RESULTS is not None:
        return _LAST_RESULTS

    results = run_impact_evaluation(
        target=target,
        model_type=model_type,
        n_splits=n_splits,
        cancel_event=cancel_event,
    )

    _LAST_SIGNATURE = sig
    _LAST_RESULTS = results
    return results



def render_results_to_text_widget(results: Dict[str, Any], text_widget) -> None:
    text_widget.delete("1.0", "end")

    stages = results.get("stages", [])
    main_metric = results.get("main_metric")

    if not stages:
        text_widget.insert("end", "No evaluation results to display.\n")
        return

    text_widget.insert("end", "EVALUATE IMPACT — Feature Engineering Pipeline\n\n")

    # Detect task type from any stage that has it
    task_type = None
    for st in stages:
        if "task_type" in st:
            task_type = st["task_type"]
            break

    if task_type == "regression":
        metric_cols = ["rmse", "r2"]
        main_name = "RMSE"
    else:
        metric_cols = ["accuracy", "f1"]
        main_name = "F1-score"

    header = f"{'Stage':30} | {'#features':10} | "
    for m in metric_cols:
        header += f"{m:>14} | "
    header += "Δ vs prev\n"
    text_widget.insert("end", header)
    text_widget.insert("end", "-" * len(header) + "\n")

    def fmt_metric(mdict: Dict[str, float] | None) -> str:
        if not mdict:
            return "n/a"
        mean = mdict.get("mean")
        std = mdict.get("std")
        if mean is None:
            return "n/a"
        if std is None or std == 0:
            return f"{mean:.4f}"
        return f"{mean:.4f} ± {std:.4f}"

    # ---- Per-stage rows ----
    for st in stages:
        name = st.get("name", "Stage")
        n_feat = st.get("n_features", 0)
        stage_error = st.get("error")

        if "metrics" in st and st["metrics"]:
            metrics = st["metrics"]
            vals = [fmt_metric(metrics.get(m)) for m in metric_cols]
            delta = st.get("delta_main")
            if delta is None:
                delta_str = "–"
            else:
                if main_metric == "rmse":
                    # lower is better
                    direction = "↓" if delta < 0 else "↑"
                else:
                    # higher is better
                    direction = "↑" if delta > 0 else "↓"
                delta_str = f"{direction} {delta:+.4f}"
        else:
            vals = ["error"] * len(metric_cols)
            delta_str = "n/a"

        row = f"{name[:30]:30} | {n_feat:10d} | "
        for v in vals:
            row += f"{v:>14} | "
        row += f"{delta_str:>9}\n"
        text_widget.insert("end", row)

        # Show underlying error message for this stage, if any
        if stage_error:
            text_widget.insert("end", f"    ↳ Error: {stage_error}\n")

    text_widget.insert("end", "\n")

    # ---- Overall summary from first to last stage ----
    try:
        if main_metric is not None:
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

            abs_change = final - base
            rel_change = abs_change / base if base != 0 else 0.0

            if main_metric == "rmse":
                # lower is better
                if final < base:
                    verb = "reduced"
                elif final > base:
                    verb = "increased"
                else:
                    verb = "kept"
            else:
                # higher is better (F1, accuracy, etc.)
                if final > base:
                    verb = "improved"
                elif final < base:
                    verb = "reduced"
                else:
                    verb = "kept"

            text_widget.insert(
                "end",
                f"Overall, the pipeline {verb} {main_name} "
                f"from {base:.4f} to {final:.4f} "
                f"({abs_change:+.4f} absolute, {rel_change*100:+.1f}% relative).\n"
            )
    except Exception:
        # If we can't compute a summary (e.g. no valid metrics), just skip it
        pass

    # ---- Statistical significance table (paired t-test + Cohen's d) ----
    signif = results.get("significance") or []
    if signif:
        text_widget.insert(
            "end",
            "\nStatistical Significance (paired t-test on per-fold "
            f"{main_name}):\n"
        )
        text_widget.insert("end", "-" * 80 + "\n")
        text_widget.insert(
            "end",
            f"{'From':25s} → {'To':25s} | {'p-value':>10s} | {'Cohen d':>9s}\n"
        )
        text_widget.insert("end", "-" * 80 + "\n")
        for row in signif:
            text_widget.insert(
                "end",
                f"{row['from']:25s} → {row['to']:25s} | "
                f"{row['p_value']:10.4g} | {row['effect_size']:9.3f}\n"
            )
        text_widget.insert("end", "\n")

    text_widget.insert(
        "end",
        "Each row is based on k-fold cross-validation using the same model "
        "configuration across stages.\n"
    )


# ---------- Visualization helpers for the Results Panel ----------


def show_confusion_matrices(root) -> None:
    """
    Open a Tkinter window showing the confusion matrix aggregated over folds
    for the final classification stage.
    """
    if _LAST_RESULTS is None:
        messagebox.showwarning("No Results", "Please run Evaluate Impact first.")
        return

    stages = _LAST_RESULTS.get("stages", [])
    if not stages:
        messagebox.showwarning("No Results", "No stages found in evaluation.")
        return

    final_stage = None
    for st in reversed(stages):
        if st.get("task_type") == "classification" and st.get("fold_outputs"):
            final_stage = st
            break

    if final_stage is None:
        messagebox.showinfo(
            "Not Classification",
            "Confusion matrix is only available for classification tasks.",
        )
        return

    folds = final_stage["fold_outputs"]
    if not folds:
        messagebox.showwarning(
            "No Fold Data", "No per-fold outputs stored. Please re-run evaluation."
        )
        return

    y_true_all = np.concatenate([np.asarray(f["y_true"]) for f in folds])
    y_pred_all = np.concatenate([np.asarray(f["y_pred"]) for f in folds])

    labels = np.unique(y_true_all)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)

    win = tk.Toplevel(root)
    win.title("Confusion Matrix — Final Stage")

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title("Confusion Matrix (aggregated over CV folds)")

    max_val = cm.max() if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if max_val and cm[i, j] > max_val / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


def show_roc_pr_curves(root) -> None:
    """
    Open a Tkinter window showing ROC and Precision–Recall curves
    for the final classification stage (binary only).
    """
    if _LAST_RESULTS is None:
        messagebox.showwarning("No Results", "Please run Evaluate Impact first.")
        return

    stages = _LAST_RESULTS.get("stages", [])
    if not stages:
        messagebox.showwarning("No Results", "No stages found in evaluation.")
        return

    final_stage = None
    for st in reversed(stages):
        if st.get("task_type") == "classification" and st.get("fold_outputs"):
            final_stage = st
            break

    if final_stage is None:
        messagebox.showinfo(
            "Not Classification",
            "ROC and PR curves are only available for classification tasks.",
        )
        return

    folds = final_stage["fold_outputs"]
    if not folds:
        messagebox.showwarning(
            "No Fold Data", "No per-fold outputs stored. Please re-run evaluation."
        )
        return

    y_true_list: List[np.ndarray] = []
    y_proba_list: List[np.ndarray] = []

    for f in folds:
        y_true = np.asarray(f["y_true"])
        y_proba = f.get("y_proba")
        if y_proba is None:
            messagebox.showinfo(
                "No Probabilities",
                "Current model does not expose predict_proba; ROC/PR curves are unavailable.",
            )
            return
        y_true_list.append(y_true)
        y_proba_list.append(np.asarray(y_proba))

    y_true_all = np.concatenate(y_true_list)
    y_proba_all = np.concatenate(y_proba_list)

    labels = np.unique(y_true_all)
    if labels.size != 2:
        messagebox.showinfo(
            "Multiclass Not Supported",
            "ROC/PR visualization currently supports binary classification only.",
        )
        return

    pos_label = labels[1]

    fpr, tpr, _ = roc_curve(y_true_all, y_proba_all, pos_label=pos_label)
    roc_auc_val = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(
        y_true_all, y_proba_all, pos_label=pos_label
    )

    win = tk.Toplevel(root)
    win.title("ROC & Precision–Recall — Final Stage")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # ROC curve
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], linestyle="--")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve (AUC = {roc_auc_val:.3f})")

    # Precision–Recall curve
    axes[1].plot(recall, precision)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall Curve")

    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
