#!/usr/bin/env python3
# text_clustering.py

import os
import textwrap

import hdbscan
import numpy as np
import pandas as pd
import umap
from scipy.stats import zscore
from sentence_transformers import SentenceTransformer


def load_and_clean(csv_path: str, text_column: str):
    df = pd.read_csv(csv_path)
    df = df[df[text_column].notna()]
    df = df[df[text_column].str.strip().astype(bool)]
    # remove boilerplate intros
    intro_patterns = [
        r"(?i)^hey(,\s*guys)?:",
        r"(?i)^hi(,\s*everyone)?:",
        r"(?i)^hello(,\s*all)?:",
        r"(?i)\bi love this sub\b",
        r"(?i)\bfirst time (posting|here)\b",
        r"(?i)\bi\.?m in [A-Za-z ]+\b",
    ]
    for pat in intro_patterns:
        df[text_column] = df[text_column].str.replace(pat, "", regex=True)
    df = df.reset_index(drop=True)
    return df, df[text_column].tolist()


def embed_texts(
    texts: list, model_name: str, batch_size: int = 32, cache_path: str = None
) -> np.ndarray:
    if cache_path is None:
        cache_path = r"C:\Users\mzouicha\OneDrive - Amadeus Workplace\Desktop\STAGE\raop-pizza\embeddings_all-MiniLM-L6-v2.npy"
    if os.path.exists(cache_path):
        return np.load(cache_path)
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    embs = np.array(embs)
    np.save(cache_path, embs)
    return embs


def reduce_dimensionality(
    embs: np.ndarray,
    n_components: int = 50,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    metric: str = "cosine",
    seed: int = 42,
):
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    X_umap = reducer.fit_transform(embs)
    return X_umap, reducer


def cluster_points(
    X: np.ndarray,
    min_cluster_size: int = 20,
    min_samples: int = 15,
    metric: str = "euclidean",
):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X)
    return labels, clusterer


def compute_centroids(X: np.ndarray, labels: np.ndarray):
    unique = sorted(c for c in set(labels) if c >= 0)
    return {c: X[labels == c].mean(axis=0) for c in unique}


def compute_distances(X: np.ndarray, centroids: dict, scale: bool = True) -> np.ndarray:
    dmat = np.stack(
        [np.linalg.norm(X - centroids[c], axis=1) for c in sorted(centroids)], axis=1
    )
    return zscore(dmat, axis=0) if scale else dmat


def inspect_clusters(
    texts: list, labels: np.ndarray, examples: int = 3, width: int = 80
):
    for cid in sorted(set(labels)):
        if cid < 0:
            continue
        idxs = np.where(labels == cid)[0][:examples]
        print(f"\n=== Cluster {cid} (size={sum(labels==cid)}) ===")
        for i in idxs:
            print(" •", textwrap.shorten(texts[i], width, placeholder="…"))
