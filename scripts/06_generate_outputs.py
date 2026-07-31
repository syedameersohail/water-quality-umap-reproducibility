#!/usr/bin/env python
"""
06_generate_outputs.py  —  Interpretation, sensitivity, and figures (Task E).

Produces the manuscript's interpretation and sensitivity artifacts that do NOT
require new UMAP fits:

  * Feature <-> axis Spearman correlations and the 12x12 feature correlation
    matrix (Figure 5, supplementary correlation tables).
  * Axis interpretation via top/bottom 5% tail contrasts along each UMAP axis
    (Tables 8-9: nutrient-mineralisation vs hardwater-softwater / pH).
  * National UMAP embedding figure and axis-driver overlays (Figures 4, 6).
  * Censoring rates by determinand (Table 15)      [needs paths.observations].
  * National support-threshold sensitivity (Table 18): recomputes the feature
    panel and eligible-site set across the n_min x n_sites_min grid and confirms
    invariance                                       [needs paths.site_det_stats].

Sections skip gracefully (with a clear message) when an optional input is absent.

Inputs : paths.matrix_scaled, matrix_pre_imputation, umap_embedding,
         wfd_typology_lookup, (optional) observations, site_det_stats
Outputs: outputs/figures/*.png, outputs/tables/*.csv

Usage:
    python scripts/06_generate_outputs.py [--config config/analysis_config.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from waterqual import load_config          # noqa: E402
from waterqual.config import ConfigError   # noqa: E402


def log(*a):
    print(*a)
    sys.stdout.flush()


def _short_labels(cfg):
    return {c: cfg["features"]["labels"][c]["short"] for c in cfg.feature_order}


def axis_interpretation(cfg, tables_dir, figures_dir):
    feature_order = cfg.feature_order
    short = _short_labels(cfg)
    scaled = pd.read_csv(cfg.require("matrix_scaled"))
    umap = pd.read_csv(cfg.require("umap_embedding"))
    assert list(scaled["site_id"]) == list(umap["site_id"])
    X = scaled[feature_order].values.astype(float)
    u1, u2 = umap["umap_1"].values, umap["umap_2"].values

    # -- Feature <-> axis correlations ----------------------------------------
    rows = []
    for j, c in enumerate(feature_order):
        r1 = spearmanr(X[:, j], u1).statistic
        r2 = spearmanr(X[:, j], u2).statistic
        rows.append({"determinant_code": c, "feature": short[c],
                     "spearman_umap1": r1, "spearman_umap2": r2})
    axis_corr = pd.DataFrame(rows)
    axis_corr.to_csv(tables_dir / "feature_axis_correlations.csv", index=False)

    corr = pd.DataFrame(np.corrcoef(X.T), index=[short[c] for c in feature_order],
                        columns=[short[c] for c in feature_order])
    corr.to_csv(tables_dir / "feature_correlation_matrix.csv")

    # -- Tail contrasts (top/bottom 5% along each axis) -----------------------
    def tails(vals, axis_name):
        lo, hi = np.percentile(vals, 5), np.percentile(vals, 95)
        low_mask, high_mask = vals <= lo, vals >= hi
        out = []
        for j, c in enumerate(feature_order):
            out.append({"axis": axis_name, "determinant_code": c, "feature": short[c],
                        "low_tail_mean": float(X[low_mask, j].mean()),
                        "high_tail_mean": float(X[high_mask, j].mean()),
                        "contrast": float(X[high_mask, j].mean() - X[low_mask, j].mean())})
        return out
    tail_df = pd.DataFrame(tails(u1, "umap_1") + tails(u2, "umap_2"))
    tail_df.to_csv(tables_dir / "axis_tail_contrasts.csv", index=False)
    log("[axis] feature_axis_correlations.csv, feature_correlation_matrix.csv, axis_tail_contrasts.csv")

    # -- Figures ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 4: national embedding coloured by region (if lookup available)
    try:
        wfd = pd.read_csv(cfg.path("wfd_typology_lookup")).drop_duplicates("site_id")
        reg = wfd.set_index("site_id")["region"].to_dict()
        regions = [reg.get(s, "Unknown") for s in umap["site_id"]]
    except Exception:
        regions = None
    fig, ax = plt.subplots(figsize=(8, 8))
    if regions is not None:
        cats = sorted(set(regions))
        cmap = plt.get_cmap("tab10")
        for i, cat in enumerate(cats):
            m = np.array(regions) == cat
            ax.scatter(u1[m], u2[m], s=1.5, alpha=0.5, color=cmap(i % 10), label=cat, rasterized=True)
        ax.legend(markerscale=6, fontsize=7, loc="best")
    else:
        ax.scatter(u1, u2, s=1.0, alpha=0.4, color="steelblue", rasterized=True)
    ax.set_xlabel("UMAP 1 (nutrient-mineralisation)")
    ax.set_ylabel("UMAP 2 (hardwater-softwater / pH)")
    ax.set_title("National UMAP embedding (8,857 sites)")
    ax.set_aspect("equal")
    fig.savefig(figures_dir / "figure04_national_umap_embedding.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 6: axis-driver overlays for the strongest driver of each axis
    d1 = axis_corr.loc[axis_corr.spearman_umap1.abs().idxmax()]
    d2 = axis_corr.loc[axis_corr.spearman_umap2.abs().idxmax()]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, drv, vals in [(axes[0], d1, X[:, feature_order.index(d1.determinant_code)]),
                          (axes[1], d2, X[:, feature_order.index(d2.determinant_code)])]:
        sc = ax.scatter(u1, u2, c=vals, s=1.5, cmap="viridis", alpha=0.6, rasterized=True)
        ax.set_title(f"{drv.feature} (rho1={drv.spearman_umap1:.2f}, rho2={drv.spearman_umap2:.2f})")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(figures_dir / "figure06_axis_driver_overlays.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log("[figures] figure04_national_umap_embedding.png, figure06_axis_driver_overlays.png")


def censoring_rates(cfg, tables_dir):
    obs_path = cfg.path("observations")
    if not obs_path.exists():
        log("[censoring] observations parquet absent -> Table 15 skipped "
            "(see data/README.md).")
        return
    import polars as pl
    feature_order = cfg.feature_order
    short = _short_labels(cfg)
    material = cfg["sensitivity"]["river_material"]
    o = (pl.read_parquet(obs_path, columns=["sampleMaterialType", "result", "determinand.notation"])
         .filter(pl.col("sampleMaterialType") == material)
         .with_columns(pl.col("result").cast(pl.Utf8).str.starts_with("<").alias("cens")))
    g = o.group_by("determinand.notation").agg(
        pl.len().alias("total_obs"), pl.col("cens").sum().alias("censored_obs"))
    rows = []
    for c in feature_order:
        r = g.filter(pl.col("determinand.notation") == int(c))
        if len(r) == 0:
            continue
        tot, cen = int(r["total_obs"][0]), int(r["censored_obs"][0])
        rows.append({"determinant_code": c, "feature": short[c], "total_obs": tot,
                     "censored_obs": cen, "censored_pct": round(100 * cen / tot, 2)})
    pd.DataFrame(rows).to_csv(tables_dir / "table15_censoring_rates.csv", index=False)
    log(f"[censoring] table15_censoring_rates.csv ({len(rows)} determinands)")


def support_threshold_sensitivity(cfg, tables_dir):
    sds_path = cfg.path("site_det_stats")
    if not sds_path.exists():
        log("[support] site_det_stats absent -> Table 18 skipped (see data/README.md).")
        return
    import polars as pl
    pp = cfg["preprocessing"]
    cov_t = float(pp["coverage_threshold"])
    min_obs = int(pp["min_observed_features"])
    grid = cfg["sensitivity"]["support_grid"]
    sds = pl.read_parquet(sds_path)

    rows = []
    for n_min in grid["n_min"]:
        for n_sites_min in grid["n_sites_min"]:
            filt = sds.filter(pl.col("n") >= n_min)
            counts = filt.group_by(["determinant_code", "unit"]).agg(
                pl.col("site_id").n_unique().alias("ns"))
            good = counts.filter(pl.col("ns") >= n_sites_min).select(["determinant_code", "unit"])
            mat = (sds.join(good, on=["determinant_code", "unit"], how="inner")
                   .select(["site_id", "determinant_code", "median"])
                   .pivot(values="median", index="site_id", on="determinant_code",
                          aggregate_function="first"))
            feats = [c for c in mat.columns if c != "site_id"]
            keep = sorted([c for c in feats
                           if mat.select(pl.col(c).is_not_null().mean()).item() >= cov_t])
            n_obs = pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in keep])
            n_elig = mat.select(["site_id"] + keep).filter(n_obs >= min_obs).shape[0]
            rows.append({"n_min": n_min, "n_sites_min": n_sites_min,
                         "n_features": len(keep), "n_eligible_sites": n_elig,
                         "feature_set": ";".join(sorted(keep))})
    out = pd.DataFrame(rows)
    out.to_csv(tables_dir / "table18_support_threshold_sensitivity.csv", index=False)
    n_feat_unique = out["n_features"].nunique()
    n_site_unique = out["n_eligible_sites"].nunique()
    log(f"[support] table18_support_threshold_sensitivity.csv "
        f"({len(out)} combos; distinct feature counts={n_feat_unique}, "
        f"distinct site counts={n_site_unique})")


def main():
    ap = argparse.ArgumentParser(description="Interpretation, sensitivity, figures (Task E).")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tables_dir = cfg.ensure_dir("outputs_tables")
    figures_dir = cfg.ensure_dir("outputs_figures")
    axis_interpretation(cfg, tables_dir, figures_dir)
    censoring_rates(cfg, tables_dir)
    support_threshold_sensitivity(cfg, tables_dir)
    log("[done] outputs")


if __name__ == "__main__":
    main()
