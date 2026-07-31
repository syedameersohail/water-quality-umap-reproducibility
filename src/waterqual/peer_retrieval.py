"""Peer-retrieval methods in the standardised 12-D chemistry space and the 2-D
UMAP portfolio view.

Three retrieval strategies (identical to the manuscript definitions):

* **12-D reference** — the ``k`` nearest sites in the standardised 12-dimensional
  feature space (Euclidean). This is the chemical-distance *reference* set, not a
  "ground truth".
* **UMAP direct** — the ``k`` nearest sites in the 2-D UMAP embedding.
* **Hybrid** — take the ``shortlist`` nearest sites in UMAP, then re-rank them by
  12-D chemical distance and keep the closest ``k``.

Distance ties are resolved deterministically by
``sklearn.neighbors.NearestNeighbors`` (stable kd-/ball-tree ordering).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class PeerSets:
    """Peer indices for one focal site, by method (self already excluded)."""
    ref_12d: np.ndarray
    umap_direct: np.ndarray
    hybrid: np.ndarray
    dist_12d_ref: np.ndarray          # 12-D distances of the 12-D reference peers
    dist_12d_umap: np.ndarray         # 12-D distances of the UMAP-direct peers
    dist_12d_hybrid: np.ndarray       # 12-D distances of the hybrid peers


class PeerRetriever:
    """Fit nearest-neighbour indices once, then query focal sites cheaply."""

    def __init__(self, X_scaled: np.ndarray, umap_2d: np.ndarray,
                 k: int = 5, shortlist: int = 20):
        assert X_scaled.shape[0] == umap_2d.shape[0], "row mismatch between X and UMAP"
        self.X = X_scaled
        self.umap = umap_2d
        self.k = k
        self.shortlist = shortlist
        # +1 because the focal site itself is always its own nearest neighbour.
        self._nn_12d = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X_scaled)
        self._nn_umap = NearestNeighbors(n_neighbors=shortlist + 1, metric="euclidean").fit(umap_2d)

    def peers_for(self, idx: int) -> PeerSets:
        k = self.k
        # 12-D reference peers
        d12, i12 = self._nn_12d.kneighbors(self.X[idx:idx + 1])
        ref_12d = i12[0][1:k + 1]
        dist_ref = d12[0][1:k + 1]

        # UMAP shortlist + direct peers
        _, iumap = self._nn_umap.kneighbors(self.umap[idx:idx + 1])
        umap_direct = iumap[0][1:k + 1]
        shortlist = iumap[0][1:self.shortlist + 1]

        # Hybrid: re-rank the UMAP shortlist by 12-D distance, keep closest k
        hybrid_d = np.linalg.norm(self.X[shortlist] - self.X[idx], axis=1)
        order = np.argsort(hybrid_d)[:k]
        hybrid = shortlist[order]

        dist_umap = np.linalg.norm(self.X[umap_direct] - self.X[idx], axis=1)
        dist_hybrid = hybrid_d[order]

        return PeerSets(
            ref_12d=ref_12d,
            umap_direct=umap_direct,
            hybrid=hybrid,
            dist_12d_ref=dist_ref,
            dist_12d_umap=dist_umap,
            dist_12d_hybrid=dist_hybrid,
        )

    @staticmethod
    def overlap_at_k(peers: np.ndarray, reference: np.ndarray) -> int:
        """Number of peers that also appear in the 12-D reference set."""
        return len(set(peers.tolist()) & set(reference.tolist()))
