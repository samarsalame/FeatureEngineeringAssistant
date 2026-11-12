# ===============================================================
# feature_selection.py
# Supervised feature selection module with optional CV-based K tuning
# ===============================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

from xgboost import XGBClassifier, XGBRegressor

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGBM = True
except ImportError:  # pragma: no cover
    _HAS_LGBM = False

import dataset


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def _get_current_df() -> pd.DataFrame | None:
    """Small wrapper to support either dataset.get_current_df() or dataset.CURRENT_DF."""
    if hasattr(dataset, "get_current_df"):
        return dataset.get_current_df()
    return getattr(dataset, "CURRENT_DF", None)


def _prepare_supervised_Xy(df: pd.DataFrame, target: str):
    """Split df into design matrix X and target y with preprocessing.

    - Numeric columns -> StandardScaler
    - Categorical columns -> OneHotEncoder
    - latent_* columns are appended as-is (not scaled, and only once)
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataset.")

    # y as 1D Series
    y = df[target]
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = pd.Series(y).reset_index(drop=True)

    # All features except target
    X = df.drop(columns=[target]).copy()

    # Separate latent_* so we don't preprocess them twice
    latent_cols = [c for c in X.columns if c.startswith("latent_")]
    X_tabular = X.drop(columns=latent_cols)

    numeric_cols = X_tabular.select_dtypes(include=[np.number]).columns
    cat_cols = X_tabular.select_dtypes(include=["object", "category"]).columns

    transformers = []
    if len(numeric_cols) > 0:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if len(cat_cols) > 0:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

    if not transformers and not latent_cols:
        raise ValueError(
            "No numeric, categorical, or latent feature columns found after removing the target."
        )

    X_processed = None
    feature_names: list[str] = []

    if transformers:
        preprocessor = ColumnTransformer(transformers)
        X_processed = preprocessor.fit_transform(X_tabular)

        # numeric feature names
        if len(numeric_cols) > 0:
            feature_names.extend(list(numeric_cols))

        # one-hot cat feature names
        if len(cat_cols) > 0:
            cat_encoder: OneHotEncoder = preprocessor.named_transformers_["cat"]  # type: ignore[index]
            cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
            feature_names.extend(list(cat_feature_names))

        # Sparse -> dense if needed
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()

        X_processed = pd.DataFrame(X_processed, columns=feature_names)
    else:
        # No numeric/cat, start from empty frame
        X_processed = pd.DataFrame(index=X.index)

    # Append latent_* columns exactly once
    if latent_cols:
        X_latent = X[latent_cols].reset_index(drop=True)
        X_processed = pd.concat([X_processed.reset_index(drop=True), X_latent], axis=1)

    # Final safety: drop any duplicate column names if they still exist
    X_processed = X_processed.loc[:, ~X_processed.columns.duplicated()].reset_index(drop=True)

    return X_processed, y


def _build_model(model_type: str, is_regression: bool):
    model_type = (model_type or "Random Forest").strip()

    if is_regression:
        if model_type == "Random Forest":
            return RandomForestRegressor(n_estimators=200, random_state=42)
        if model_type == "XGBoost":
            return XGBRegressor(n_estimators=200, max_depth=5, random_state=42)
        if model_type == "LightGBM":
            if not _HAS_LGBM:
                raise ImportError(
                    "LightGBM is not installed. Please install it or choose another model."
                )
            return LGBMRegressor(n_estimators=200, random_state=42)
        # Linear / Logistic fall back to Linear for regression
        return LinearRegression()

    # classification
    if model_type == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=42)
    if model_type == "XGBoost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=5,
            eval_metric="logloss",
            random_state=42,
        )
    if model_type == "LightGBM":
        if not _HAS_LGBM:
            raise ImportError(
                "LightGBM is not installed. Please install it or choose another model."
            )
        return LGBMClassifier(n_estimators=200, random_state=42)
    # Linear / Logistic -> Logistic for classification
    return LogisticRegression(max_iter=2000)


def _compute_model_importance(model, X: pd.DataFrame) -> pd.Series:
    feature_names = X.columns
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    elif hasattr(model, "coef_"):
        coefs = np.asarray(model.coef_)
        if coefs.ndim == 1:
            scores = np.abs(coefs)
        else:
            scores = np.mean(np.abs(coefs), axis=0)
    else:
        raise ValueError(
            "Selected model does not support feature_importances_ or coef_."
        )
    return pd.Series(scores, index=feature_names).sort_values(ascending=False)


def _infer_task_type(y: pd.Series) -> bool:
    """Return True if regression, False if classification (if user did not choose)."""
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 15:
        return True
    return False


def _build_cv(is_regression: bool, y: np.ndarray, n_splits: int = 5):
    if is_regression:
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # classification
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


# -----------------------------------------------------------
# Public API
# -----------------------------------------------------------

def run_feature_selection(
    target: str,
    top_k: int | None = None,
    model_type: str = "Random Forest",
    k_candidates: list[int] | None = None,
    cv_folds: int = 5,
):
    """Supervised feature selection.

    Parameters
    ----------
    target : str
        Name of the target column.
    top_k : int or None
        If an integer, select exactly this many top features (legacy behavior).
        If None, K is chosen automatically by cross-validation over ``k_candidates``.
    model_type : str
        Base model used both for ranking and CV scoring.
    k_candidates : list[int] or None
        Candidate values of K when ``top_k`` is None. If None, a default
        set based on the number of features will be used.
    cv_folds : int
        Number of CV folds used when tuning K.
    """
    df = _get_current_df()
    if df is None:
        raise ValueError("No dataset loaded.")

    X, y = _prepare_supervised_Xy(df, target)

    # --- Force y to be a 1-D numpy array for sklearn ---
    y_array = np.asarray(y).ravel()

    # Determine regression vs classification:
    # 1) Prefer user choice from dataset.TASK_TYPE
    # 2) Fallback to heuristic if not set
    user_choice = getattr(dataset, "TASK_TYPE", None)
    if hasattr(dataset, "get_task_type"):
        try:
            user_choice = dataset.get_task_type()
        except Exception:
            pass

    if isinstance(user_choice, str) and user_choice.lower() in ("regression", "classification"):
        is_regression = (user_choice.lower() == "regression")
    else:
        is_regression = _infer_task_type(y)

    # 1) Mutual Information scores
    if is_regression:
        mi_vals = mutual_info_regression(X, y_array)
    else:
        mi_vals = mutual_info_classif(X, y_array)
    mi_scores = pd.Series(mi_vals, index=X.columns)

    # 2) Model-based feature ranking
    base_model = _build_model(model_type, is_regression)
    base_model.fit(X, y_array)
    model_importance = _compute_model_importance(base_model, X)

    # 3) RFE ranking (use n_features_to_select=1 just to get a full ranking_)
    rfe_base = LinearRegression() if is_regression else LogisticRegression(max_iter=2000)
    rfe = RFE(rfe_base, n_features_to_select=1)
    rfe.fit(X, y_array)
    rfe_rank = pd.Series(rfe.ranking_, index=X.columns).sort_values()

    # Aggregate rankings into a single ordering
    rank_df = pd.DataFrame({
        "mi": mi_scores.rank(ascending=False),
        "model": model_importance.rank(ascending=False),
        "rfe": rfe_rank.rank(ascending=True),  # RFE: 1 is best
    })
    rank_df["avg_rank"] = rank_df.mean(axis=1)
    global_order = rank_df.sort_values("avg_rank").index.tolist()

    n_features_total = len(global_order)

    # Decide K
    if top_k is None:
        # Build default candidate list if not provided
        if k_candidates is None:
            # e.g., [5, 10, 15, 20, 25] but never exceeding n_features_total
            base_candidates = [5, 10, 15, 20, 25]
            k_candidates = sorted({k for k in base_candidates if k <= n_features_total})
            if not k_candidates:
                k_candidates = [min(5, n_features_total)]

        # Cross-validate each K
        cv = _build_cv(is_regression, y_array, n_splits=cv_folds)
        best_k = None
        best_score = -np.inf

        scoring = "neg_root_mean_squared_error" if is_regression else "f1_weighted"

        k_scores: dict[int, float] = {}

        for k in k_candidates:
            selected_feats = global_order[:k]
            X_k = X[selected_feats]

            scores = cross_val_score(
                _build_model(model_type, is_regression),
                X_k,
                y_array,
                cv=cv,
                scoring=scoring,
            )
            mean_score = float(scores.mean())
            k_scores[k] = mean_score

            if mean_score > best_score:
                best_score = mean_score
                best_k = k

        if best_k is None:
            best_k = min(15, n_features_total)

        top_k_final = best_k
    else:
        # clamp to available features
        top_k_final = max(1, min(int(top_k), n_features_total))
        k_scores = {}

    final_selected = global_order[:top_k_final]

    return {
        "selected": final_selected,
        "mi_rank": mi_scores.sort_values(ascending=False),
        "model_rank": model_importance.sort_values(ascending=False),
        "rfe_rank": rfe_rank,
        "X_all": X,
        "y": y,                    # keep as original Series for UI
        "model_type": model_type,
        "is_regression": is_regression,
        "k_used": top_k_final,
        "k_scores": k_scores,      # may be empty if top_k was fixed
    }
