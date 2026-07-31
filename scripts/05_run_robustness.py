#!/usr/bin/env python
"""
05_run_robustness.py  —  Embedding robustness analyses (Task D).

Two analyses, both against a reference UMAP embedding generated in the SAME run
and environment (no cross-environment distance comparisons):

  1. Repeated 80% site subsampling — `n_replicates` fits on `fraction` of sites
     (default 100 x 0.80 -> 7,086 sites), each compared to the reference on the
     shared sites via local neighbourhood overlap (Overlap@k) and Procrustes.
  2. Leave-one-feature-out (LOFO) — 12 fits, each omitting one feature, compared
     to the reference via Procrustes (global) and Overlap@k (local).

Both stages are checkpointed and resumable; re-running skips completed replicates
/ features. This is a LONG-RUNNING analysis (100 + 12 UMAP fits). Use --smoke-test
to verify the pipeline on a tiny subset first.

Input : paths.matrix_scaled
Output: outputs/robustness/subsampling/*, outputs/robustness/lofo/*

Usage:
    python scripts/05_run_robustness.py --config config/analysis_config.yaml
    python scripts/05_run_robustness.py --config config/analysis_config.yaml --smoke-test
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations  # noqa: F401  (kept for parity/extension)
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from waterqual import load_config          # noqa: E402
from waterqual.config import ConfigError   # noqa: E402


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a)
    sys.stdout.flush()


def _backends():
    try:
        import umap
        from sklearn.manifold import trustworthiness  # noqa: F401
        from sklearn.neighbors import NearestNeighbors  # noqa: F401
        from scipy.spatial import procrustes  # noqa: F401
    except ImportError as e:
        raise ConfigError(f"Missing dependency ({e}); pip install -r requirements.txt") from e
    return umap, procrustes


def _overlap(emb_a, emb_b, k):
    from sklearn.neighbors import NearestNeighbors
    a = NearestNeighbors(n_neighbors=k + 1).fit(emb_a).kneighbors(emb_a, return_distance=False)[:, 1:]
    b = NearestNeighbors(n_neighbors=k + 1).fit(emb_b).kneighbors(emb_b, return_distance=False)[:, 1:]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)]))


def _umap_kwargs(cfg):
    u = cfg["umap"]
    return dict(n_neighbors=int(u["n_neighbors"]), min_dist=float(u["min_dist"]),
                n_components=int(u["n_components"]), metric=u["metric"], n_jobs=int(u["n_jobs"]))


def run(cfg, smoke: bool) -> None:
    umap, procrustes = _backends()
    feature_order = cfg.feature_order
    seed = cfg.random_state
    ukw = _umap_kwargs(cfg)

    df = pd.read_csv(cfg.require("matrix_scaled", hint="Run scripts/02_build_matrix.py."))
    if list(df.columns[1:]) != feature_order:
        raise ConfigError("feature column order mismatch")
    site_ids = df["site_id"].to_numpy()
    X = df[feature_order].values.astype(np.float64)
    n_sites = X.shape[0]
    log(f"input: {n_sites} sites x {X.shape[1]} features")

    log("fitting reference embedding (same environment) ...")
    reference = umap.UMAP(random_state=seed, **ukw).fit_transform(X)

    rob = cfg["robustness"]
    k = int(rob["subsampling"]["k_neighbours"])
    frac = float(rob["subsampling"]["fraction"])
    n_rep = int(rob["subsampling"]["n_replicates"])
    n_feat_drop = 12
    if smoke:
        n_rep, n_feat_drop = 2, 2

    # -- 1. Subsampling --------------------------------------------------------
    sub_dir = cfg.ensure_dir(rob["subsampling"]["output_dir"] if not smoke
                             else rob["subsampling"]["output_dir"] + "/_smoke")
    sub_ckpt = sub_dir / "subsampling_summary.csv"
    done = set()
    if sub_ckpt.exists():
        done = set(pd.read_csv(sub_ckpt)["replicate"].tolist())
    n_sampled = round(frac * n_sites)
    log(f"[subsampling] {n_rep} replicates x {n_sampled} sites ({len(done)} done)")

    ref_idx = {sid: i for i, sid in enumerate(site_ids)}
    for r in range(n_rep):
        if r in done:
            continue
        rng = np.random.RandomState(seed + r)
        sample = np.sort(rng.choice(n_sites, size=n_sampled, replace=False))
        emb = umap.UMAP(random_state=seed, **ukw).fit_transform(X[sample])
        ref_sub = reference[sample]
        _, _, disp = procrustes(ref_sub, emb)
        ov = _overlap(ref_sub, emb, k)
        _append(sub_ckpt, {"replicate": r, "n_sampled": n_sampled,
                           "overlap_k": ov, "procrustes_disparity": disp})
        log(f"  replicate {r}: overlap@{k}={ov:.4f} procrustes={disp:.5f}")

    # -- 2. Leave-one-feature-out ---------------------------------------------
    lofo_dir = cfg.ensure_dir(rob["lofo"]["output_dir"] if not smoke
                              else rob["lofo"]["output_dir"] + "/_smoke")
    lofo_ckpt = lofo_dir / "lofo_summary.csv"
    done_f = set()
    if lofo_ckpt.exists():
        done_f = set(pd.read_csv(lofo_ckpt)["dropped_code"].astype(str).tolist())
    labels = cfg["features"]["labels"]
    log(f"[lofo] dropping {n_feat_drop} features ({len(done_f)} done)")
    for j, code in enumerate(feature_order[:n_feat_drop]):
        if code in done_f:
            continue
        keep = [c for c in feature_order if c != code]
        Xk = df[keep].values.astype(np.float64)
        emb = umap.UMAP(random_state=seed, **ukw).fit_transform(Xk)
        _, _, disp = procrustes(reference, emb)
        ov = _overlap(reference, emb, k)
        _append(lofo_ckpt, {"dropped_code": code, "dropped_name": labels[code]["short"],
                            "overlap_k": ov, "procrustes_disparity": disp})
        log(f"  drop {code} ({labels[code]['short']}): overlap@{k}={ov:.4f} procrustes={disp:.5f}")

    log(f"[write] {sub_ckpt}\n[write] {lofo_ckpt}")
    log("[done] robustness")


def _append(path: Path, row: dict) -> None:
    df = pd.DataFrame([row])
    header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=header, index=False)


def main():
    ap = argparse.ArgumentParser(description="Embedding robustness analyses (Task D).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke-test", action="store_true", help="2 replicates + 2 LOFO fits only.")
    args = ap.parse_args()
    run(load_config(args.config), smoke=args.smoke_test)


if __name__ == "__main__":
    main()
