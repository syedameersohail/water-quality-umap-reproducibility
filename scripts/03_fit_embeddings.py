#!/usr/bin/env python
"""
03_fit_embeddings.py  —  Fit the national embeddings on the 8,857 x 12 matrix (Task B, Part 1).

Fits three 2-D embeddings on the standardised feature matrix and saves aligned
site_id coordinate tables:
  * PCA   (linear baseline)
  * t-SNE (perplexity=30, init=pca)
  * UMAP  (the RETAINED configuration used for the displayed portfolio view and
           for peer retrieval: n_neighbors=50, min_dist=0.10, random_state=42)

Determinism: UMAP uses random_state=42 with n_jobs=1. Exact coordinates can still
differ across package versions / platforms; structural findings are robust (see
the Task C sweep and Task D robustness analyses).

Input : paths.matrix_scaled
Output: data/processed/umap_v2.csv  (+ pca_v2.csv, tsne_v2.csv, site_embeddings_v2.parquet,
        embedding_parameters.json)

Usage:
    python scripts/03_fit_embeddings.py [--config config/analysis_config.yaml]
"""
from __future__ import annotations

import argparse
import inspect
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from waterqual import load_config          # noqa: E402


def log(*a):
    print(*a)
    sys.stdout.flush()


def fit_embeddings(cfg) -> None:
    feature_order = cfg.feature_order
    scaled_path = cfg.require(
        "matrix_scaled",
        hint="Regenerate with scripts/02_build_matrix.py.",
    )
    df = pd.read_csv(scaled_path)
    assert list(df.columns[1:]) == feature_order, "feature column order mismatch"
    site_ids = df["site_id"].tolist()
    X = df[feature_order].values.astype(np.float64)
    log(f"[load] {scaled_path}  ({X.shape[0]} sites x {X.shape[1]} features)")

    seed = cfg.random_state
    params: dict = {}

    # -- PCA -------------------------------------------------------------------
    pcfg = cfg["other_embeddings"]["pca"]
    t0 = time.time()
    pca = PCA(n_components=int(pcfg["n_components"]), random_state=int(pcfg["random_state"]))
    pca_2d = pca.fit_transform(X)
    log(f"[pca]  {time.time() - t0:.1f}s  explained_var={pca.explained_variance_ratio_.sum():.4f}")
    params["PCA"] = {**pcfg, "explained_variance_ratio": pca.explained_variance_ratio_.tolist()}

    # -- t-SNE -----------------------------------------------------------------
    tcfg = cfg["other_embeddings"]["tsne"]
    tsne_kwargs = dict(
        n_components=int(tcfg["n_components"]), perplexity=int(tcfg["perplexity"]),
        learning_rate=tcfg["learning_rate"], init=tcfg["init"],
        random_state=int(tcfg["random_state"]),
    )
    # scikit-learn renamed n_iter -> max_iter; support both.
    sig = inspect.signature(TSNE.__init__).parameters
    if "max_iter" in sig:
        tsne_kwargs["max_iter"] = int(tcfg["max_iter"])
    elif "n_iter" in sig:
        tsne_kwargs["n_iter"] = int(tcfg["max_iter"])
    t0 = time.time()
    tsne_2d = TSNE(**tsne_kwargs).fit_transform(X)
    log(f"[tsne] {time.time() - t0:.1f}s")
    params["t-SNE"] = tcfg

    # -- UMAP (retained configuration) ----------------------------------------
    ucfg = cfg["umap"]
    t0 = time.time()
    umap_2d = UMAP(
        n_components=int(ucfg["n_components"]), n_neighbors=int(ucfg["n_neighbors"]),
        min_dist=float(ucfg["min_dist"]), metric=ucfg["metric"],
        random_state=int(ucfg["random_state"]), n_jobs=int(ucfg["n_jobs"]),
    ).fit_transform(X)
    log(f"[umap] {time.time() - t0:.1f}s")
    params["UMAP"] = ucfg

    # -- Save ------------------------------------------------------------------
    out_dir = cfg.ensure_dir("data_processed")
    pd.DataFrame({"site_id": site_ids, "umap_1": umap_2d[:, 0], "umap_2": umap_2d[:, 1]}).to_csv(
        cfg.path("umap_embedding"), index=False)
    pd.DataFrame({"site_id": site_ids, "pca_1": pca_2d[:, 0], "pca_2": pca_2d[:, 1]}).to_csv(
        out_dir / "pca_v2.csv", index=False)
    pd.DataFrame({"site_id": site_ids, "tsne_1": tsne_2d[:, 0], "tsne_2": tsne_2d[:, 1]}).to_csv(
        out_dir / "tsne_v2.csv", index=False)
    pd.DataFrame({
        "site_id": site_ids,
        "pca_1": pca_2d[:, 0], "pca_2": pca_2d[:, 1],
        "tsne_1": tsne_2d[:, 0], "tsne_2": tsne_2d[:, 1],
        "umap_1": umap_2d[:, 0], "umap_2": umap_2d[:, 1],
    }).to_parquet(out_dir / "site_embeddings_v2.parquet", index=False)

    params["environment"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit-learn": __import__("sklearn").__version__,
        "umap-learn": __import__("umap").__version__,
        "platform": platform.platform(),
    }
    with open(out_dir / "embedding_parameters.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    log(f"[write] {out_dir}: umap_v2.csv, pca_v2.csv, tsne_v2.csv, "
        f"site_embeddings_v2.parquet, embedding_parameters.json")
    log("[done] embeddings fitted")


def main():
    ap = argparse.ArgumentParser(description="Fit PCA/t-SNE/UMAP embeddings (Task B Part 1).")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    fit_embeddings(load_config(args.config))


if __name__ == "__main__":
    main()
