from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .config import EMB_MODEL_PATH, META_PATH, RND, STACK_PATH
from .features import build_feature_frame, embed_texts

# Load trained model and metadata at module import time for efficiency
with STACK_PATH.open("rb") as f:
    _STACK = pickle.load(f)

with META_PATH.open("rb") as f:
    _META = pickle.load(f)

_FEAT_NAMES: list[str] = _META["feature_names"]
_N_TABULAR: int = _META["n_tabular"]
_PCA_MEAN: np.ndarray = _META["pca_mean_"]
_PCA_COMP: np.ndarray = _META["pca_components_"]


def _pca_transform(embs: np.ndarray) -> np.ndarray:
    return (embs - _PCA_MEAN) @ _PCA_COMP.T


def _make_matrix(df_feat: pd.DataFrame, embs: np.ndarray) -> np.ndarray:
    X_tab = df_feat[_FEAT_NAMES[:_N_TABULAR]].astype(float).values
    X_emb = _pca_transform(embs)
    return np.hstack([X_tab, X_emb])


# Public prediction function


def predict(df_raw: pd.DataFrame) -> tuple[float, int, pd.DataFrame]:
    df_raw = df_raw.copy()
    df_raw["req"] = (
        df_raw.request_title.fillna("")
        + " "
        + df_raw.request_text_edit_aware.fillna("")
    ).str.lower()
    df_raw["request_time_utc"] = pd.to_datetime(
        df_raw["request_time_utc"], errors="coerce", utc=True
    )
    embs = embed_texts(df_raw["req"].tolist(), str(EMB_MODEL_PATH))
    df_feat = build_feature_frame(df_raw, embs)
    X = _make_matrix(df_feat, embs)
    proba = _STACK.predict_proba(X)[:, 1][0]  # scalar
    label = int(proba >= 0.5)
    return float(proba), label, df_feat.iloc[0:1]
