"""
Leakage-free benchmark of the Feature Engineering Assistant pipeline on the
Wisconsin Diagnostic Breast Cancer data (data.csv).

Every learned step - scaling, PCA, ICA, the autoencoder, and supervised feature
selection (the app's own run_feature_selection, incl. its CV-based choice of K) -
is fitted on the training part of each outer fold only and then applied to the
held-out part.

Protocol: 10-fold stratified CV repeated 3 times (30 outer folds).
Configurations (same latent settings as the GUI "Deep mode"):
    Raw            30 cleaned original features
    Latent only    16 latent features (PCA 5 + ICA 5 + AE 6)
    Raw + Latent   46 features
    Selected       top-K of Raw + Latent, K tuned by inner 5-fold CV
Models: Logistic Regression, Random Forest, XGBoost, LightGBM (app defaults).
Statistics: Nadeau-Bengio corrected resampled t-test vs. Raw.

Usage:  python paper_assets/benchmark.py
"""
import os
import sys
import json
import time
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "FinalFeatureEng"))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef

import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense

import dataset
from feature_selection import run_feature_selection, _build_model

SEED = 42
N_SPLITS, N_REPEATS = 10, 3
MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
CONFIGS = ["Raw", "Latent only", "Raw + Latent", "Selected"]


# ------------------------------------------------------------------ data
df = pd.read_csv(os.path.join(PROJECT, "data.csv"))
# identical to the app's Data Cleaning result on this file:
# drops the empty trailing column and the ID-like 'id' column
df = df.drop(columns=[c for c in df.columns if c == "id" or df[c].isna().all()])
y_all = (df["diagnosis"] == "M").astype(int).to_numpy()
X_raw = df.drop(columns=["diagnosis"]).reset_index(drop=True)
print("data:", X_raw.shape, "malignant =", int(y_all.sum()))


# ------------------------------------------------- latent features (fit on train)
def autoencoder_latent(Xtr, Xte, latent_dim=6, seed=0):
    """Same architecture/training as feature_creation._generate_tabular_autoencoder_latent."""
    tf.keras.utils.set_random_seed(seed)
    inp = Input(shape=(Xtr.shape[1],))
    h = Dense(128, activation="relu")(inp)
    h = Dense(64, activation="relu")(h)
    z = Dense(latent_dim, activation="linear")(h)
    d = Dense(64, activation="relu")(z)
    d = Dense(128, activation="relu")(d)
    out = Dense(Xtr.shape[1], activation="linear")(d)
    ae = Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")
    ae.fit(Xtr, Xtr, epochs=20, batch_size=32, verbose=0,
           callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3,
                                                       restore_best_weights=True)])
    enc = Model(inp, z)
    return enc.predict(Xtr, verbose=0), enc.predict(Xte, verbose=0)


def make_latent(Xtr, Xte, seed):
    sc = MinMaxScaler().fit(Xtr)  # GUI Deep mode uses MinMaxScaler
    Str, Ste = sc.transform(Xtr), sc.transform(Xte)
    pca = PCA(n_components=5, random_state=seed).fit(Str)
    ica = FastICA(n_components=5, random_state=seed, max_iter=1000).fit(Str)
    a_tr, a_te = autoencoder_latent(Str, Ste, 6, seed)
    blocks_tr = [pca.transform(Str), ica.transform(Str), a_tr]
    blocks_te = [pca.transform(Ste), ica.transform(Ste), a_te]
    names = ([f"latent_pca_{i+1}" for i in range(5)] + [f"latent_ica_{i+1}" for i in range(5)]
             + [f"latent_auto_{i+1}" for i in range(6)])
    Ltr = pd.DataFrame(np.hstack(blocks_tr), columns=names, index=Xtr.index)
    Lte = pd.DataFrame(np.hstack(blocks_te), columns=names, index=Xte.index)
    return Ltr, Lte, pca.explained_variance_ratio_


# ------------------------------------------------------------- main loop
dataset.set_task_type("classification")
rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
rows, sel_log, evr_log = [], [], []
t_start = time.time()

for fold, (tr, te) in enumerate(rskf.split(X_raw, y_all)):
    t0 = time.time()
    Xtr, Xte = X_raw.iloc[tr], X_raw.iloc[te]
    ytr, yte = y_all[tr], y_all[te]

    t_lat = time.time()
    Ltr, Lte, evr = make_latent(Xtr, Xte, seed=SEED + fold)
    t_lat = time.time() - t_lat
    evr_log.append(evr)

    feats = {
        "Raw": (Xtr, Xte),
        "Latent only": (Ltr, Lte),
        "Raw + Latent": (pd.concat([Xtr, Ltr], axis=1), pd.concat([Xte, Lte], axis=1)),
    }

    # app's supervised selection, run on the training fold only
    t_sel = time.time()
    train_df = feats["Raw + Latent"][0].copy()
    train_df["diagnosis"] = ytr
    dataset.CURRENT_DF = train_df.reset_index(drop=True)
    res = run_feature_selection(target="diagnosis", top_k=None, model_type="Random Forest")
    selected = res["selected"]
    t_sel = time.time() - t_sel
    sel_log.append({"fold": fold, "k": res["k_used"], "selected": selected,
                    "k_scores": res["k_scores"]})
    feats["Selected"] = (feats["Raw + Latent"][0][selected], feats["Raw + Latent"][1][selected])

    for cfg in CONFIGS:
        A, B = feats[cfg]
        for m in MODELS:
            mdl = make_pipeline(StandardScaler(), _build_model(m, is_regression=False))
            mdl.fit(A, ytr)
            proba = mdl.predict_proba(B)[:, 1]
            pred = (proba >= 0.5).astype(int)
            rows.append({
                "fold": fold, "config": cfg, "model": m, "n_features": A.shape[1],
                "accuracy": accuracy_score(yte, pred),
                "f1": f1_score(yte, pred, average="weighted"),
                "roc_auc": roc_auc_score(yte, proba),
                "mcc": matthews_corrcoef(yte, pred),
            })
    print(f"fold {fold+1:2d}/30  K={res['k_used']:2d}  latent {t_lat:.1f}s  "
          f"selection {t_sel:.1f}s  total {time.time()-t0:.1f}s")

total_time = time.time() - t_start
res_df = pd.DataFrame(rows)
res_df.to_csv(os.path.join(OUT, "fold_results.csv"), index=False)


# ------------------------------------------------------------- statistics
def corrected_ttest(diff, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test."""
    diff = np.asarray(diff, float)
    J = len(diff)
    var = np.var(diff, ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = diff.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return float(t), float(p)


n_test = len(y_all) / N_SPLITS
n_train = len(y_all) - n_test
summary = []
for m in MODELS:
    base = res_df[(res_df.model == m) & (res_df.config == "Raw")].sort_values("fold")
    for cfg in CONFIGS:
        cur = res_df[(res_df.model == m) & (res_df.config == cfg)].sort_values("fold")
        row = {"model": m, "config": cfg, "n_features": float(cur.n_features.mean())}
        for met in ["accuracy", "f1", "roc_auc", "mcc"]:
            v = cur[met].to_numpy()
            row[f"{met}_mean"] = v.mean()
            row[f"{met}_sd"] = v.std(ddof=1)
            # corrected 95% CI half-width
            se = np.sqrt((1 / len(v) + n_test / n_train) * np.var(v, ddof=1))
            row[f"{met}_ci"] = stats.t.ppf(0.975, len(v) - 1) * se
            if cfg != "Raw":
                d = v - base[met].to_numpy()
                t, p = corrected_ttest(d, n_train, n_test)
                row[f"{met}_delta"] = d.mean()
                row[f"{met}_p"] = p
        summary.append(row)
sum_df = pd.DataFrame(summary)
sum_df.to_csv(os.path.join(OUT, "summary.csv"), index=False)

# selection stability
all_sel = [f for s in sel_log for f in s["selected"]]
freq = pd.Series(all_sel).value_counts() / len(sel_log)
freq.to_csv(os.path.join(OUT, "selection_frequency.csv"), header=["frequency"])
k_counts = pd.Series([s["k"] for s in sel_log]).value_counts().sort_index()

# Jaccard stability between fold selections
sets = [set(s["selected"]) for s in sel_log]
jac = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]

meta = {
    "n_samples": int(len(y_all)), "n_malignant": int(y_all.sum()),
    "n_features_raw": int(X_raw.shape[1]),
    "outer_folds": N_SPLITS * N_REPEATS, "total_seconds": round(total_time, 1),
    "pca_explained_variance_mean": np.mean(evr_log, axis=0).round(4).tolist(),
    "pca_cumulative_5": float(np.mean(np.sum(evr_log, axis=1)).round(4)),
    "k_distribution": {int(k): int(v) for k, v in k_counts.items()},
    "selection_jaccard_mean": float(np.mean(jac)), "selection_jaccard_sd": float(np.std(jac)),
    "latent_share_in_selection": float(np.mean(
        [np.mean([f.startswith("latent_") for f in s["selected"]]) for s in sel_log])),
    "top_selected": freq.head(25).round(3).to_dict(),
    "versions": {"tensorflow": tf.__version__},
}
import sklearn, xgboost, lightgbm
meta["versions"].update({"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                         "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__})
with open(os.path.join(OUT, "benchmark_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

pd.set_option("display.width", 200)
print(sum_df[["model", "config", "n_features", "f1_mean", "f1_sd", "f1_p",
              "roc_auc_mean", "roc_auc_p", "mcc_mean"]].round(4).to_string())
print(json.dumps({k: v for k, v in meta.items() if k != "top_selected"}, indent=1))
print("total", round(total_time, 1), "s")
