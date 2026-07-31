#!/usr/bin/env python
"""
01_prepare_data.py  —  Validate inputs and (optionally) build the site-level stats table.

By default this checks which pipeline inputs are present and prints acquisition
guidance for anything missing (see data/README.md for the full instructions).

With --build-site-stats it reconstructs `site_det_stats_2015_2024.parquet` (the
site x determinand median table consumed by scripts/02_build_matrix.py) from the
raw observations parquet, by aggregating numeric results per
(site_id, determinand_code, unit) over the 2015-2024 study period.

The raw Environment Agency observations are NOT redistributed with this package.
Download them from the Water Quality Explorer (see data/README.md), place the
parquet at data/raw/, then run this script.

Usage:
    python scripts/01_prepare_data.py                     # validate + report
    python scripts/01_prepare_data.py --build-site-stats  # build site_det_stats from observations
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from waterqual import load_config          # noqa: E402


def log(*a):
    print(*a)
    sys.stdout.flush()


REQUIRED = {
    "site_det_stats": "Site x determinand median table. Build with --build-site-stats from observations.",
    "wfd_typology_lookup": "WFD Cycle 3 typology lookup (redistributed in data/reference/).",
    "feature_mapping": "Authoritative feature identity (redistributed in data/reference/).",
}
OPTIONAL = {
    "observations": "Raw EA observations parquet (large; needed for site-stats build & censoring Table 15).",
    "matrix_scaled": "Processed scaled matrix (regenerable via scripts/02_build_matrix.py).",
    "umap_embedding": "Retained UMAP coordinates (regenerable via scripts/03_fit_embeddings.py).",
}


def validate(cfg) -> int:
    log("=== Input validation ===")
    missing_required = 0
    for key, desc in REQUIRED.items():
        p = cfg.path(key)
        ok = p.exists()
        log(f"  [{'OK ' if ok else 'MISS'}] {key:<22s} {p.relative_to(cfg.root)}")
        if not ok:
            log(f"         -> {desc}")
            missing_required += 1
    log("--- optional ---")
    for key, desc in OPTIONAL.items():
        p = cfg.path(key)
        log(f"  [{'OK ' if p.exists() else '-- '}] {key:<22s} {p.relative_to(cfg.root)}")
        if not p.exists():
            log(f"         -> {desc}")
    if missing_required:
        log(f"\n{missing_required} required input(s) missing. See data/README.md.")
    else:
        log("\nAll required inputs present.")
    return missing_required


def build_site_stats(cfg) -> None:
    import polars as pl
    obs_path = cfg.require(
        "observations",
        hint="Download the EA observations parquet to data/raw/ (see data/README.md).",
    )
    out_path = cfg.path("site_det_stats")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"[build] reading {obs_path}")
    o = pl.read_parquet(obs_path, columns=[
        "samplingPoint.notation", "determinand.notation", "determinand.prefLabel",
        "result", "unit", "phenomenonTime",
    ])
    # Numeric result: strip a leading '<' (left-censored non-detects) then cast.
    o = o.with_columns(
        pl.col("result").cast(pl.Utf8).str.strip_prefix("<").cast(pl.Float64, strict=False).alias("value"),
        pl.col("determinand.notation").cast(pl.Utf8).alias("determinant_code"),
    ).drop_nulls("value")
    stats = (
        o.group_by(["samplingPoint.notation", "determinant_code",
                    "determinand.prefLabel", "unit"])
        .agg(
            pl.len().alias("n"),
            pl.col("value").mean().alias("mean"),
            pl.col("value").median().alias("median"),
            pl.col("value").std().alias("std"),
            pl.col("value").min().alias("min"),
            pl.col("value").max().alias("max"),
            pl.col("value").quantile(0.10).alias("q10"),
            pl.col("value").quantile(0.90).alias("q90"),
        )
        .rename({"samplingPoint.notation": "site_id", "determinand.prefLabel": "determinant"})
    )
    stats.write_parquet(out_path)
    log(f"[build] wrote {out_path}  ({stats.shape[0]} rows, "
        f"{stats['site_id'].n_unique()} sites, {stats['determinant_code'].n_unique()} determinands)")
    log("Note: this reconstructs the site-level stats table. scripts/02_build_matrix.py "
        "asserts the expected 12-feature / 8,857-site panel, which validates the build.")


def main():
    ap = argparse.ArgumentParser(description="Validate inputs / build site-level stats (Task pre-A).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--build-site-stats", action="store_true",
                    help="Reconstruct site_det_stats parquet from the observations parquet.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.build_site_stats:
        build_site_stats(cfg)
    else:
        validate(cfg)


if __name__ == "__main__":
    main()
