#!/usr/bin/env python
"""
04_benchmark_peers.py  —  Systematic peer benchmarking across 100 focal sites (Task B, Parts 2-3).

For a random sample of `n_focal_sites` (default 100, seed 42) focal sites, compare
three peer-retrieval methods against the 12-D chemical-distance reference:
  * direct UMAP peers,
  * hybrid (UMAP shortlist re-ranked by 12-D distance).

Reports, per method, Overlap@5 with the 12-D reference, the mean chemical-distance
ratio (selected-peer / reference distance), and WFD water-body-typology agreement
(both the "all-5" and "available" macro-averages).

Also runs the illustrative single-site example (Saffron Brook, MD-48883750).

Headline verification values (manuscript):
  Overlap@5           UMAP 0.86   hybrid 2.43
  distance ratio      UMAP 1.9027 hybrid 1.1716
  WFD agreement       ref 0.463   UMAP 0.411   hybrid 0.458   (available macro-average)

Inputs : paths.matrix_scaled, paths.matrix_imputed_unscaled, paths.umap_embedding,
         paths.wfd_typology_lookup
Outputs: outputs/tables/peer_validation_per_site.csv, peer_validation_summary.csv,
         outputs/tables/illustrative_peers.csv

Usage:
    python scripts/04_benchmark_peers.py [--config config/analysis_config.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from waterqual import load_config          # noqa: E402
from waterqual.peer_retrieval import PeerRetriever   # noqa: E402


def log(*a):
    print(*a)
    sys.stdout.flush()


def _load(cfg):
    feature_order = cfg.feature_order
    scaled = pd.read_csv(cfg.require("matrix_scaled"))
    imputed = pd.read_csv(cfg.require("matrix_imputed_unscaled"))
    umap = pd.read_csv(cfg.require("umap_embedding"))
    assert list(scaled["site_id"]) == list(imputed["site_id"]) == list(umap["site_id"]), \
        "site_id alignment mismatch across inputs"
    site_ids = scaled["site_id"].tolist()
    X = scaled[feature_order].values.astype(np.float64)
    X_raw = imputed[feature_order].values.astype(np.float64)
    umap_2d = umap[["umap_1", "umap_2"]].values.astype(np.float64)

    wfd = pd.read_csv(cfg.require("wfd_typology_lookup"))
    col = cfg["peer_retrieval"]["wfd_typology_column"]
    wfd_map = wfd.drop_duplicates("site_id").set_index("site_id")[col].to_dict()
    return site_ids, X, X_raw, umap_2d, wfd_map


def _typology_agreement(focal_typo, peer_ids, wfd_map, k):
    """Return (matches_over_k, matches_over_available, n_available)."""
    if focal_typo is None or (isinstance(focal_typo, float) and np.isnan(focal_typo)):
        return np.nan, np.nan, 0
    peer_typos = [wfd_map.get(pid) for pid in peer_ids]
    known = [t for t in peer_typos if t is not None and not (isinstance(t, float) and np.isnan(t))]
    matches = sum(1 for t in known if t == focal_typo)
    over_k = matches / k
    over_avail = (matches / len(known)) if known else np.nan
    return over_k, over_avail, len(known)


def benchmark(cfg) -> None:
    pr_cfg = cfg["peer_retrieval"]
    k = int(pr_cfg["k_peers"])
    shortlist = int(pr_cfg["umap_shortlist"])
    n_focal = int(pr_cfg["n_focal_sites"])
    focal_seed = int(pr_cfg["focal_seed"])

    site_ids, X, X_raw, umap_2d, wfd_map = _load(cfg)
    log(f"[load] {len(site_ids)} sites; WFD lookup covers {len(wfd_map)} sites")

    retriever = PeerRetriever(X, umap_2d, k=k, shortlist=shortlist)

    # -- Illustrative single-site example -------------------------------------
    illustrative_id = pr_cfg["illustrative_focal_site"]
    illus_rows = []
    if illustrative_id in site_ids:
        fidx = site_ids.index(illustrative_id)
        ps = retriever.peers_for(fidx)
        for method, idxs in [("12D_reference", ps.ref_12d),
                             ("UMAP_direct", ps.umap_direct),
                             ("hybrid", ps.hybrid)]:
            for rank, pidx in enumerate(idxs, start=1):
                illus_rows.append({
                    "method": method, "rank": rank, "site_id": site_ids[pidx],
                    "distance_12d": float(np.linalg.norm(X[pidx] - X[fidx])),
                    "wfd_typology": wfd_map.get(site_ids[pidx], ""),
                })
        log(f"[illustrative] {illustrative_id}: "
            f"Overlap@{k} UMAP={retriever.overlap_at_k(ps.umap_direct, ps.ref_12d)}, "
            f"hybrid={retriever.overlap_at_k(ps.hybrid, ps.ref_12d)}")
    else:
        log(f"[illustrative] {illustrative_id} not in eligible population; skipped")

    # -- Systematic 100-site benchmark ----------------------------------------
    rng = np.random.RandomState(focal_seed)
    focal_idx = rng.choice(len(site_ids), size=n_focal, replace=False)

    rows = []
    for idx in focal_idx:
        sid = site_ids[idx]
        ps = retriever.peers_for(idx)
        ref_d = float(ps.dist_12d_ref.mean())
        umap_d = float(ps.dist_12d_umap.mean())
        hybrid_d = float(ps.dist_12d_hybrid.mean())
        focal_typo = wfd_map.get(sid)

        w12_k, w12_a, _ = _typology_agreement(focal_typo, [site_ids[i] for i in ps.ref_12d], wfd_map, k)
        wu_k, wu_a, _ = _typology_agreement(focal_typo, [site_ids[i] for i in ps.umap_direct], wfd_map, k)
        wh_k, wh_a, _ = _typology_agreement(focal_typo, [site_ids[i] for i in ps.hybrid], wfd_map, k)

        rows.append({
            "site_id": sid,
            "overlap_umap": retriever.overlap_at_k(ps.umap_direct, ps.ref_12d),
            "overlap_hybrid": retriever.overlap_at_k(ps.hybrid, ps.ref_12d),
            "chem_dist_12d": ref_d,
            "chem_dist_umap": umap_d,
            "chem_dist_hybrid": hybrid_d,
            "dist_ratio_umap": umap_d / ref_d if ref_d > 0 else np.nan,
            "dist_ratio_hybrid": hybrid_d / ref_d if ref_d > 0 else np.nan,
            "wfd_agreement_12d_all5": w12_k, "wfd_agreement_12d_available": w12_a,
            "wfd_agreement_umap_all5": wu_k, "wfd_agreement_umap_available": wu_a,
            "wfd_agreement_hybrid_all5": wh_k, "wfd_agreement_hybrid_available": wh_a,
        })

    per_site = pd.DataFrame(rows)

    metrics = [c for c in per_site.columns if c != "site_id"]
    summary = pd.DataFrame([
        {
            "metric": m,
            "mean": float(per_site[m].dropna().mean()),
            "std": float(per_site[m].dropna().std()),
            "median": float(per_site[m].dropna().median()),
            "n_valid": int(per_site[m].dropna().shape[0]),
        }
        for m in metrics
    ])

    tables_dir = cfg.ensure_dir("outputs_tables")
    per_site.to_csv(tables_dir / "peer_validation_per_site.csv", index=False)
    summary.to_csv(tables_dir / "peer_validation_summary.csv", index=False)
    if illus_rows:
        pd.DataFrame(illus_rows).to_csv(tables_dir / "illustrative_peers.csv", index=False)

    log("\n[summary] key metrics across %d focal sites:" % n_focal)
    for m in ["overlap_umap", "overlap_hybrid", "dist_ratio_umap", "dist_ratio_hybrid",
              "wfd_agreement_12d_available", "wfd_agreement_umap_available",
              "wfd_agreement_hybrid_available"]:
        val = per_site[m].dropna().mean()
        log(f"  {m:<32s} mean={val:.4f}")
    log(f"\n[write] {tables_dir}: peer_validation_per_site.csv, peer_validation_summary.csv")
    log("[done] benchmarking")


def main():
    ap = argparse.ArgumentParser(description="Systematic peer benchmarking (Task B Parts 2-3).")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    benchmark(load_config(args.config))


if __name__ == "__main__":
    main()
