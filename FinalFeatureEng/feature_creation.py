# ===============================================================
# feature_creation.py  (FINAL, STABLE, COLLISION-FREE VERSION)
# Deep Autoencoder + PCA + ICA for Tabular Feature Engineering
#
# Includes:
# - Robust mixed-type handling (numeric/string/categorical)
# - PCA / ICA with safe fallbacks
# - Deep Tabular Autoencoder with categorical embeddings
# - Latent block normalization (no nested objects)
# - Column name de-duplication & collision prevention
# - Compatibility with dataset.py and Latent Explorer
# - Legacy autoencoder preserved for Main.py
# ===============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA

import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten


# ===============================================================
# HELPERS
# ===============================================================

def _make_scaler(name: str):
    if name == "MinMaxScaler":
        return MinMaxScaler()
    if name == "RobustScaler":
        return RobustScaler()
    return StandardScaler()


def _make_unique(names):
    """
    Ensures column names are unique (prefix_1, prefix_2, prefix_3, ...),
    WITHOUT relying on deprecated pandas internals.
    """
    seen = {}
    final = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            final.append(str(name))
        else:
            seen[name] += 1
            final.append(f"{name}_{seen[name]}")
    return final


def _prepare_mixed_inputs(df: pd.DataFrame, scaling: str, max_cat_cardinality=500):
    df = df.copy()

    # 1) numeric columns
    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    X_num = df[numeric_cols].copy() if numeric_cols else None

    # 2) detect numeric-like strings
    non_num_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_cols = []

    for col in non_num_cols:
        s = df[col].astype(str)
        stripped = (
            s.str.replace(",", "", regex=False)
             .str.replace("$", "", regex=False)
             .str.replace("%", "", regex=False)
        )
        parsed = pd.to_numeric(stripped, errors="coerce")

        if parsed.notna().mean() > 0.90:
            if X_num is None:
                X_num = pd.DataFrame(index=df.index)
            X_num[col] = parsed
            numeric_cols.append(col)
        else:
            cat_cols.append(col)

    # 3) scale numeric
    if X_num is not None and X_num.shape[1] > 0:
        X_num = X_num.astype(float).replace([np.inf, -np.inf], np.nan)
        X_num = X_num.fillna(X_num.median())
        X_num_scaled = _make_scaler(scaling).fit_transform(X_num)
    else:
        numeric_cols = []
        X_num_scaled = None

    # 4) encode categoricals
    cat_codes = {}
    cat_card = {}

    for col in cat_cols:
        s = df[col].astype(str)

        vc = s.value_counts(dropna=False)
        if len(vc) > max_cat_cardinality:
            keep = set(vc.head(max_cat_cardinality - 1).index)
            s = s.where(s.isin(keep), "__OTHER__")

        cat = pd.Categorical(s)
        codes = cat.codes + 1   # shift negative to 0
        vocab = int(codes.max()) + 1
        if vocab <= 1:
            continue

        cat_codes[col] = codes.reshape(-1, 1)
        cat_card[col] = vocab

    cat_cols = [c for c in cat_cols if c in cat_codes]

    return X_num_scaled, numeric_cols, cat_cols, cat_codes, cat_card


# ===============================================================
# SAFE LATENT BLOCK NORMALIZATION
# ===============================================================

def _normalize_latent_block(block, prefix: str):
    """
    Ensures latent outputs contain ONLY numeric columns,
    with no nested objects, and unique prefix-based names.
    """
    if block is None:
        return pd.DataFrame()

    if isinstance(block, np.ndarray):
        block = pd.DataFrame(block)

    if isinstance(block, pd.Series):
        block = block.to_frame()

    if not isinstance(block, pd.DataFrame):
        try:
            block = pd.DataFrame(block)
        except:
            return pd.DataFrame()

    # expand nested objects
    expand_cols = []
    for col in block.columns:
        if block[col].apply(
            lambda x: isinstance(x, (dict, list, np.ndarray, pd.Series, pd.DataFrame))
        ).any():
            expand_cols.append(col)

    for col in expand_cols:
        expanded = pd.DataFrame(list(block[col]))
        expanded.columns = [f"{col}_{i}" for i in range(expanded.shape[1])]
        block = block.drop(columns=[col])
        block = pd.concat([block, expanded], axis=1)

    # convert all to numeric
    for col in block.columns:
        block[col] = pd.to_numeric(block[col], errors="coerce")

    block = block.fillna(0)

    # prefix column names
    block.columns = [f"{prefix}_{i+1}" for i in range(block.shape[1])]

    return block


# ===============================================================
# TABULAR AUTOENCODER
# ===============================================================

def _generate_tabular_autoencoder_latent(
    X_num_scaled, numeric_cols, cat_cols, cat_codes, cat_card, latent_dim=8
):
    if latent_dim <= 0:
        return pd.DataFrame()

    # ----------------------------
    # inputs
    # ----------------------------
    inputs = []
    encoded = []

    if X_num_scaled is not None and X_num_scaled.shape[1] > 0:
        inp_num = Input(shape=(X_num_scaled.shape[1],), name="num_in")
        inputs.append(inp_num)
        encoded.append(inp_num)

    for col in cat_cols:
        vocab = cat_card[col]
        emb_dim = min(50, max(4, vocab // 2))
        inp = Input(shape=(1,), name=f"cat_in_{col}")
        emb = Flatten()(Embedding(vocab, emb_dim)(inp))
        inputs.append(inp)
        encoded.append(emb)

    if not encoded:
        return pd.DataFrame()

    x = Concatenate()(encoded) if len(encoded) > 1 else encoded[0]
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    bottleneck = Dense(latent_dim, activation="linear")(x)

    d = Dense(64, activation="relu")(bottleneck)
    d = Dense(128, activation="relu")(d)

    outputs = {}
    losses = {}
    targets = {}

    # numeric reconstruction
    if X_num_scaled is not None and X_num_scaled.shape[1] > 0:
        out = Dense(X_num_scaled.shape[1], activation="linear", name="num_out")(d)
        outputs["num_out"] = out
        losses["num_out"] = "mse"
        targets["num_out"] = X_num_scaled

    # categorical reconstruction
    for col in cat_cols:
        vocab = cat_card[col]
        out = Dense(vocab, activation="softmax", name=f"cat_out_{col}")(d)
        outputs[f"cat_out_{col}"] = out
        losses[f"cat_out_{col}"] = "sparse_categorical_crossentropy"
        targets[f"cat_out_{col}"] = cat_codes[col].reshape(-1)

    # dict outputs so the dict-keyed losses/targets line up (required by Keras 3)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss=losses)

    # prepare feed
    feed = {}
    if X_num_scaled is not None and X_num_scaled.shape[1] > 0:
        feed["num_in"] = X_num_scaled
    for col in cat_cols:
        feed[f"cat_in_{col}"] = cat_codes[col]

    # train
    try:
        model.fit(
            feed, targets,
            epochs=20,
            batch_size=32,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss",
                    patience=3,
                    restore_best_weights=True,
                )
            ],
        )

        encoder = Model(inputs, bottleneck)
        latent = encoder.predict(feed, verbose=0)
        return _normalize_latent_block(latent, "latent_auto")

    except Exception as e:
        print("⚠ Autoencoder failed:", e)
        if X_num_scaled is None:
            return pd.DataFrame()

        n = min(latent_dim, X_num_scaled.shape[1])
        fb = PCA(n_components=n).fit_transform(X_num_scaled)
        return _normalize_latent_block(fb, "latent_auto")


# ===============================================================
# LEGACY AUTOENCODER (for Main.py compatibility)
# ===============================================================

def generate_autoencoder_latent(X_scaled, latent_dim=3):
    """Legacy numeric-only autoencoder kept because Main.py imports it."""
    if X_scaled is None or latent_dim <= 0:
        return pd.DataFrame()

    try:
        dim = X_scaled.shape[1]
    except:
        return pd.DataFrame()

    inp = Input(shape=(dim,))
    x = Dense(32, activation="relu")(inp)
    bottleneck = Dense(latent_dim, activation="linear")(x)
    d = Dense(32, activation="relu")(bottleneck)
    out = Dense(dim, activation="linear")(d)

    model = Model(inp, out)
    encoder = Model(inp, bottleneck)
    model.compile(optimizer="adam", loss="mse")

    try:
        model.fit(X_scaled, X_scaled, epochs=20, batch_size=32, verbose=0)
        latent = encoder.predict(X_scaled, verbose=0)
        return _normalize_latent_block(latent, "latent_legacy")
    except:
        return _normalize_latent_block(
            PCA(n_components=min(latent_dim, dim)).fit_transform(X_scaled),
            "latent_legacy",
        )


# ===============================================================
# MAIN PUBLIC API
# ===============================================================

def run_feature_creation(
    df: pd.DataFrame,
    use_pca=True,
    use_ica=True,
    use_autoencoder=True,
    pca_components=5,
    ica_components=5,
    auto_latent_dim=8,
    scaling="StandardScaler",
):
    latent_blocks = []

    # ----------------------------
    # prepare inputs
    # ----------------------------
    X_num_scaled, numeric_cols, cat_cols, cat_codes, cat_card = \
        _prepare_mixed_inputs(df, scaling)

    # ----------------------------
    # PCA
    # ----------------------------
    if use_pca and X_num_scaled is not None and pca_components > 0:
        try:
            n = min(pca_components, X_num_scaled.shape[1])
            out = PCA(n_components=n).fit_transform(X_num_scaled)
            latent_blocks.append(_normalize_latent_block(out, "latent_pca"))
        except Exception as e:
            print("⚠ PCA failed:", e)

    # ----------------------------
    # ICA
    # ----------------------------
    if use_ica and X_num_scaled is not None and ica_components > 0:
        try:
            n = min(ica_components, X_num_scaled.shape[1])
            out = FastICA(n_components=n).fit_transform(X_num_scaled)
        except Exception:
            print("⚠ ICA fallback used.")
            out = PCA(n_components=n, whiten=True).fit_transform(X_num_scaled)

        latent_blocks.append(_normalize_latent_block(out, "latent_ica"))

    # ----------------------------
    # Autoencoder
    # ----------------------------
    if use_autoencoder:
        ae = _generate_tabular_autoencoder_latent(
            X_num_scaled, numeric_cols, cat_cols, cat_codes, cat_card, auto_latent_dim
        )
        latent_blocks.append(ae)

    # ----------------------------
    # concatenate latent blocks
    # ----------------------------
    clean = [
        blk.reset_index(drop=True)
        for blk in latent_blocks
        if isinstance(blk, pd.DataFrame) and blk.shape[1] > 0
    ]

    if clean:
        latent_df = pd.concat(clean, axis=1)
    else:
        latent_df = pd.DataFrame(index=df.index)

    # ------------------------------------
    # UNIQUE COLUMN NAMES (CRITICAL FIX!)
    # ------------------------------------
    latent_df.columns = _make_unique(latent_df.columns)

    # ------------------------------------
    # Prevent collisions with original df
    # ------------------------------------
    orig = set(df.columns)
    new_names = []
    for col in latent_df.columns:
        new_col = col
        counter = 1
        while new_col in orig:
            counter += 1
            new_col = f"{col}_{counter}"
        orig.add(new_col)
        new_names.append(new_col)

    latent_df.columns = new_names

    # force numeric & no NaNs
    latent_df = latent_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # final safety check
    for col in latent_df.columns:
        if latent_df[col].apply(lambda x: isinstance(x, (dict, list, pd.DataFrame, pd.Series, np.ndarray))).any():
            raise ValueError(f"Malformed latent column detected: {col}")

    # ----------------------------
    # final combine
    # ----------------------------
    df_new = pd.concat(
        [df.reset_index(drop=True), latent_df.reset_index(drop=True)],
        axis=1
    )

    summary = {
        "numeric_cols_used": numeric_cols,
        "categorical_cols_used": cat_cols,
        "total_latent": latent_df.shape[1],
        "used_pca": use_pca,
        "used_ica": use_ica,
        "used_autoencoder": use_autoencoder,
        "scaling": scaling,
    }

    return {"latent_df": latent_df, "df_new": df_new, "summary": summary}
