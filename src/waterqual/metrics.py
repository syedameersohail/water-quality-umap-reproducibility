"""Embedding-quality metrics shared across pipeline stages.

These are the exact definitions used to produce the manuscript's embedding
diagnostics (Task A ablation, Task B validation, Task C sweep, Task D
robustness). They are collected here so every stage computes them identically.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_recall(X_high: np.ndarray, X_low: np.ndarray, k: int = 10) -> float:
    """Mean k-nearest-neighbour recall between a high-D space and an embedding.

    For each point, the fraction of its ``k`` nearest neighbours in ``X_high``
    that remain among its ``k`` nearest neighbours in ``X_low`` (self excluded),
    averaged over all points.
    """
    nn_high = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X_high)
    nn_low = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X_low)
    idx_high = nn_high.kneighbors(X_high, return_distance=False)[:, 1:]
    idx_low = nn_low.kneighbors(X_low, return_distance=False)[:, 1:]
    recalls = [
        len(set(idx_high[i]) & set(idx_low[i])) / k
        for i in range(X_high.shape[0])
    ]
    return float(np.mean(recalls))


def median_neighbourhood_overlap(emb_a: np.ndarray, emb_b: np.ndarray, k: int = 5) -> float:
    """Median k-NN neighbourhood overlap between two aligned embeddings.

    Rows of ``emb_a`` and ``emb_b`` must correspond to the same points, in order.
    """
    nn_a = NearestNeighbors(n_neighbors=k + 1).fit(emb_a).kneighbors(emb_a, return_distance=False)[:, 1:]
    nn_b = NearestNeighbors(n_neighbors=k + 1).fit(emb_b).kneighbors(emb_b, return_distance=False)[:, 1:]
    overlaps = [len(set(a) & set(b)) / k for a, b in zip(nn_a, nn_b)]
    return float(np.median(overlaps))
