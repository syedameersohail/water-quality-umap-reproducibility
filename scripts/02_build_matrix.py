#!/usr/bin/env python
"""
02_build_matrix.py  —  Build and validate the analytical feature matrix (Task A).

Pipeline (eligibility BEFORE imputation):
  1. National feature screening: keep determinands with >= n_min observations at
     >= n_sites_min sites.
  2. Pivot the full site x determinand median table to a site-level matrix.
  3. Coverage filter: keep features present at >= coverage_threshold of sites.
  4. Assert exactly `expected_n_features` (12) features retained.
  5. Count observed (non-null) features per site BEFORE imputation.
  6. Keep sites with >= min_observed_features (9 of 12) observed.
  7. Assert exactly `expected_n_sites` (8,857) eligible sites.
  8. Median-impute using the eligible population's column medians.
  9. StandardScaler on the eligible, imputed matrix.
 10. Validate: unique ids, shape, no NaN/Inf, no zero-variance features.

Input : paths.site_det_stats   (site_det_stats_2015_2024.parquet)
Output: data/processed/feature_matrix_8857x12_{scaled,pre_imputation,imputed_unscaled}.csv
        data/processed/v2_feature_metadata.csv, site_exclusion_audit.csv

Usage:
    python scripts/02_build_matrix.py [--config config/analysis_config.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from waterqual import load_config          # noqa: E402
from waterqual.config import ConfigError   # noqa: E402


def log(*a):
    print(*a)
    sys.stdout.flush()


def build_matrix(cfg) -> None:
    pp = cfg["preprocessing"]
    feature_order = cfg.feature_order
    n_min = int(pp["n_min"])
    n_sites_min = int(pp["n_sites_min"])
    coverage_threshold = float(pp["coverage_threshold"])
    min_observed = int(pp["min_observed_features"])
    expected_features = int(pp["expected_n_features"])
    expected_sites = int(pp["expected_n_sites"])

    sds_path = cfg.require(
        "site_det_stats",
        hint="Regenerate with scripts/01_prepare_data.py, or place the parquet in data/raw/.",
    )
    log(f"[load] {sds_path}")
    sds = pl.read_parquet(sds_path)
    log(f"  shape={sds.shape}, sites={sds['site_id'].n_unique()}, "
        f"determinands={sds['determinant_code'].n_unique()}")

    # -- Step 1: national feature screening -----------------------------------
    log(f"[screen] n_min={n_min}, n_sites_min={n_sites_min}")
    filtered = sds.filter(pl.col("n") >= n_min)
    site_counts = filtered.group_by(["determinant_code", "unit"]).agg(
        pl.col("site_id").n_unique().alias("n_sites_for_det")
    )
    good_codes = (
        site_counts.filter(pl.col("n_sites_for_det") >= n_sites_min)
        .select(["determinant_code", "unit"])
    )
    log(f"  determinands passing screening: {len(good_codes)}")

    # -- Step 2: pivot full table ---------------------------------------------
    site_mat = (
        sds.join(good_codes, on=["determinant_code", "unit"], how="inner")
        .select(["site_id", "determinant_code", "median"])
        .pivot(values="median", index="site_id", on="determinant_code",
               aggregate_function="first")
    )
    all_features = sorted([c for c in site_mat.columns if c != "site_id"])
    log(f"  pivot shape={site_mat.shape}, features in pivot={len(all_features)}")

    # -- Step 3: coverage filter ----------------------------------------------
    keep, coverage_info = [], {}
    for c in all_features:
        cov = site_mat.select(pl.col(c).is_not_null().mean()).item()
        coverage_info[c] = cov
        if cov >= coverage_threshold:
            keep.append(c)
    keep = sorted(keep)
    log(f"[coverage] retained {len(keep)} features at >= {coverage_threshold:.0%}")

    # -- Step 4: assert feature panel -----------------------------------------
    if sorted(keep) != sorted(feature_order):
        raise ConfigError(
            f"Retained features {keep} do not match the configured panel "
            f"{sorted(feature_order)}. Investigate before proceeding."
        )
    if len(keep) != expected_features:
        raise ConfigError(f"Expected {expected_features} features, got {len(keep)}")
    log(f"  PASS: {len(keep)} features match configured panel")

    site_mat_final = site_mat.select(["site_id"] + keep)

    # -- Step 5-6: completeness before imputation -----------------------------
    n_obs = pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in keep])
    site_obs = site_mat_final.select(
        "site_id",
        n_obs.alias("n_observed"),
        (len(keep) - n_obs).alias("n_missing"),
    )
    eligible = site_mat_final.filter(n_obs >= min_observed)
    n_eligible = eligible.shape[0]
    n_excluded = site_mat_final.shape[0] - n_eligible
    log(f"[eligibility] >= {min_observed} of {len(keep)} observed: "
        f"{n_eligible} eligible, {n_excluded} excluded")

    # -- Step 7: assert site count --------------------------------------------
    if n_eligible != expected_sites:
        raise ConfigError(
            f"Eligible site count = {n_eligible}, expected {expected_sites}. "
            f"Difference {n_eligible - expected_sites}. Diagnose before proceeding."
        )
    log(f"  PASS: {n_eligible} eligible sites match expected")

    # -- Step 8: median imputation (eligible population) ----------------------
    medians = eligible.select([pl.col(c).median().alias(c) for c in keep]).row(0)
    median_dict = dict(zip(keep, medians))
    imputed = eligible.with_columns(
        [pl.col(c).fill_null(m).alias(c) for c, m in zip(keep, medians)]
    )

    # -- Step 9: standardise ---------------------------------------------------
    X_unscaled = imputed.select(keep).to_numpy().astype(np.float64)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_unscaled)
    site_ids = imputed["site_id"].to_list()

    # -- Step 10: validation ---------------------------------------------------
    assert len(set(site_ids)) == len(site_ids), "duplicate site_ids"
    assert X_scaled.shape == (expected_sites, expected_features), X_scaled.shape
    assert not np.isnan(X_scaled).any(), "NaN in scaled matrix"
    assert not np.isinf(X_scaled).any(), "Inf in scaled matrix"
    assert np.all(np.var(X_scaled, axis=0) > 0), "zero-variance feature"
    log("  PASS: unique ids, correct shape, no NaN/Inf, no zero-variance")

    # -- Save ------------------------------------------------------------------
    out_dir = cfg.ensure_dir("data_processed")

    def _frame(mat):
        df = pd.DataFrame(mat, columns=keep)
        df.insert(0, "site_id", site_ids)
        return df

    _frame(X_scaled).to_csv(cfg.path("matrix_scaled"), index=False)
    _frame(X_unscaled).to_csv(cfg.path("matrix_imputed_unscaled"), index=False)

    pre = eligible.select(["site_id"] + keep).to_pandas()
    pre.to_csv(cfg.path("matrix_pre_imputation"), index=False)

    labels = cfg["features"]["labels"]
    pd.DataFrame([
        {
            "determinant_code": c,
            "determinant_name": labels[c]["name"],
            "unit": labels[c]["unit"],
            "coverage_in_pivot": coverage_info[c],
            "column_median": median_dict[c],
            "scaler_mean": float(scaler.mean_[keep.index(c)]),
            "scaler_std": float(scaler.scale_[keep.index(c)]),
        }
        for c in keep
    ]).to_csv(cfg.path("feature_metadata"), index=False)

    obs_df = site_obs.to_pandas()
    obs_df["eligible"] = obs_df["site_id"].isin(set(site_ids))
    obs_df["exclusion_reason"] = np.where(
        obs_df["eligible"], "eligible",
        np.where(obs_df["n_observed"] == 0, "zero_observed_features", "fewer_than_9_observed"),
    )
    obs_df.to_csv(cfg.path("site_exclusion_audit"), index=False)

    log(f"[write] {out_dir}")
    log("  feature_matrix_8857x12_scaled.csv")
    log("  feature_matrix_8857x12_imputed_unscaled.csv")
    log("  feature_matrix_8857x12_pre_imputation.csv")
    log("  v2_feature_metadata.csv")
    log("  site_exclusion_audit.csv")
    log(f"[done] matrix ({n_eligible} sites x {len(keep)} features)")


def main():
    ap = argparse.ArgumentParser(description="Build and validate the feature matrix (Task A).")
    ap.add_argument("--config", default=None, help="Path to analysis_config.yaml")
    args = ap.parse_args()
    build_matrix(load_config(args.config))


if __name__ == "__main__":
    main()
