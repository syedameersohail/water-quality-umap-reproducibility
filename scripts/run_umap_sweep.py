#!/usr/bin/env python
"""
run_umap_sweep.py  —  UMAP hyperparameter sweep (Task C).

Full grid: 5 n_neighbors x 4 min_dist x 10 seeds = 200 seed-specific fits.
For each parameter combination the 45 pairwise Procrustes disparities among its
10 seed embeddings are also computed (20 combos x 45 = 900 pairs).

All scientific settings (grid, seeds, metric, k values) come from the central
configuration file. Nothing is hard-coded here.

Reliability features (per reproducibility requirements):
  * Output directory created automatically.
  * Each fit is checkpointed incrementally (append + fsync) — a crash loses at
    most the in-flight fit.
  * --resume skips fits already present in the checkpoint (the default; passing
    --resume is a harmless explicit request for it).
  * A per-fit run log records parameters, seed, runtime, completion status and
    any error message.
  * Deterministic, clearly named output files.
  * On completion the seed metrics and Procrustes pairs are aggregated into the
    manuscript summary table (sweep_results.csv) and heatmaps.
  * Clear failure if the input matrix or a dependency is missing.

Commands:
    python scripts/run_umap_sweep.py --config config/analysis_config.yaml --resume
    python scripts/run_umap_sweep.py --config config/analysis_config.yaml --smoke-test

Approximate cost: one fit of 8,857 x 12 takes ~15-40 s depending on hardware, so
the full 200-fit sweep is roughly 1-3 hours single-threaded. It is resumable, so
it can be run in stages.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from waterqual import load_config          # noqa: E402
from waterqual.config import ConfigError   # noqa: E402
from waterqual.metrics import knn_recall   # noqa: E402


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a)
    sys.stdout.flush()


# --------------------------------------------------------------------------- #
# Lazy heavy imports so --help and import checks stay fast and dependency
# problems produce a clear message at run time.
# --------------------------------------------------------------------------- #
def _import_backends():
    try:
        import umap  # noqa: F401
        from sklearn.manifold import trustworthiness  # noqa: F401
        from scipy.spatial import procrustes  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ConfigError(
            f"A required dependency is missing ({e}). "
            f"Install the environment with: pip install -r requirements.txt"
        ) from e
    return umap, trustworthiness, procrustes


SEED_FIELDS = ["n_neighbors", "min_dist", "seed", "trustworthiness_k10",
               "trustworthiness_k50", "knn_recall_k10", "runtime_s",
               "status", "timestamp"]
PROC_FIELDS = ["n_neighbors", "min_dist", "seed_a", "seed_b", "procrustes_disparity"]


def _append_row(path: Path, fields: list[str], row: dict) -> None:
    """Append one row to a CSV, writing a header if the file is new, and fsync."""
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})
        f.flush()
        os.fsync(f.fileno())


def _load_done_seeds(path: Path) -> set[tuple]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path)
    df = df[df.get("status", "ok") == "ok"] if "status" in df.columns else df
    return {(int(r.n_neighbors), float(r.min_dist), int(r.seed)) for r in df.itertuples()}


def _load_done_combos(path: Path, pairs_per_combo: int) -> set[tuple]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path)
    counts = df.groupby(["n_neighbors", "min_dist"]).size()
    return {(int(nn), float(md)) for (nn, md), c in counts.items() if c >= pairs_per_combo}


def run_sweep(cfg, resume: bool, smoke: bool) -> None:
    umap, trustworthiness, procrustes = _import_backends()

    feature_order = cfg.feature_order
    sweep = cfg["sweep"]
    if smoke:
        grid = sweep["smoke_test"]
        out_dir = cfg.resolve(sweep["output_dir"]) / "_smoke_test"
    else:
        grid = {k: sweep[k] for k in ("n_neighbors", "min_dist", "seeds")}
        out_dir = cfg.resolve(sweep["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    nn_list = list(grid["n_neighbors"])
    md_list = list(grid["min_dist"])
    seeds = list(grid["seeds"])
    n_total = len(nn_list) * len(md_list) * len(seeds)
    pairs_per_combo = len(list(combinations(seeds, 2)))
    tk = sweep["trustworthiness_k"]
    rk = int(sweep["knn_recall_k"])

    seed_ckpt = out_dir / "sweep_checkpoint.csv"
    proc_ckpt = out_dir / "sweep_procrustes_pairs.csv"
    summary_path = out_dir / "sweep_results.csv"
    seed_metrics_path = out_dir / "sweep_seed_metrics.csv"
    heatmap_path = out_dir / "sweep_heatmaps.png"

    log("=" * 68)
    log(f"UMAP SWEEP  mode={'SMOKE' if smoke else 'FULL'}  grid={len(nn_list)}x{len(md_list)}x{len(seeds)}={n_total}")
    log(f"output_dir={out_dir}")
    log("=" * 68)

    # -- Load + validate input ------------------------------------------------
    matrix_path = cfg.require(
        "matrix_scaled",
        hint="Regenerate with scripts/02_build_matrix.py, or restore data/processed/.",
    )
    df = pd.read_csv(matrix_path)
    if list(df.columns[1:]) != feature_order:
        raise ConfigError(f"Feature column order mismatch in {matrix_path}")
    X = df[feature_order].values.astype(np.float64)
    if np.isnan(X).any() or np.isinf(X).any():
        raise ConfigError("Input matrix contains NaN/Inf values.")
    log(f"input matrix validated: {X.shape[0]} sites x {X.shape[1]} features")

    done_seeds = _load_done_seeds(seed_ckpt) if resume else set()
    done_combos = _load_done_combos(proc_ckpt, pairs_per_combo) if resume else set()
    if resume and (done_seeds or done_combos):
        log(f"resume: {len(done_seeds)} seed-fits and {len(done_combos)} combos already complete")

    runtimes: list[float] = []
    count_done = len(done_seeds)

    for nn in nn_list:
        for md in md_list:
            combo_emb: dict[int, np.ndarray] = {}
            for seed in seeds:
                key = (int(nn), float(md), int(seed))
                if key in done_seeds:
                    continue
                t0 = time.time()
                try:
                    emb = umap.UMAP(
                        n_neighbors=nn, min_dist=md, n_components=int(sweep["n_components"]),
                        metric=sweep["metric"], random_state=seed, n_jobs=int(sweep["n_jobs"]),
                    ).fit_transform(X)
                    row = {
                        "n_neighbors": nn, "min_dist": md, "seed": seed,
                        "trustworthiness_k10": float(trustworthiness(X, emb, n_neighbors=tk[0])),
                        "trustworthiness_k50": float(trustworthiness(X, emb, n_neighbors=tk[1])),
                        "knn_recall_k10": knn_recall(X, emb, k=rk),
                        "runtime_s": round(time.time() - t0, 2),
                        "status": "ok",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                    _append_row(seed_ckpt, SEED_FIELDS, row)
                    combo_emb[seed] = emb
                    runtimes.append(time.time() - t0)
                    count_done += 1
                    eta = timedelta(seconds=int(np.mean(runtimes) * (n_total - count_done)))
                    log(f"[{count_done}/{n_total}] nn={nn} md={md} seed={seed} "
                        f"T@10={row['trustworthiness_k10']:.4f} R@10={row['knn_recall_k10']:.4f} "
                        f"{row['runtime_s']:.1f}s ETA {eta}")
                except Exception as e:  # record failure, keep going
                    _append_row(seed_ckpt, SEED_FIELDS, {
                        "n_neighbors": nn, "min_dist": md, "seed": seed,
                        "runtime_s": round(time.time() - t0, 2), "status": "error",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    })
                    log(f"FAILED nn={nn} md={md} seed={seed}: {e}")

            # Procrustes pairs for this combo (only when all seeds are available)
            if len(seeds) > 1 and (int(nn), float(md)) not in done_combos and len(combo_emb) == len(seeds):
                for sa, sb in combinations(seeds, 2):
                    _, _, disp = procrustes(combo_emb[sa], combo_emb[sb])
                    _append_row(proc_ckpt, PROC_FIELDS, {
                        "n_neighbors": nn, "min_dist": md,
                        "seed_a": sa, "seed_b": sb, "procrustes_disparity": disp,
                    })
                log(f"procrustes complete: nn={nn} md={md} ({pairs_per_combo} pairs)")

    _aggregate(seed_ckpt, proc_ckpt, summary_path, seed_metrics_path,
               heatmap_path, nn_list, md_list, smoke)
    log("[done] sweep")


def _aggregate(seed_ckpt, proc_ckpt, summary_path, seed_metrics_path,
               heatmap_path, nn_list, md_list, smoke) -> None:
    if not seed_ckpt.exists():
        log("no checkpoint to aggregate")
        return
    seed_df = pd.read_csv(seed_ckpt)
    seed_df = seed_df[seed_df.get("status", "ok") == "ok"]
    seed_df.to_csv(seed_metrics_path, index=False)
    proc_df = pd.read_csv(proc_ckpt) if proc_ckpt.exists() else pd.DataFrame()

    rows = []
    for nn in nn_list:
        for md in md_list:
            s = seed_df[(seed_df.n_neighbors == nn) & (np.isclose(seed_df.min_dist, md))]
            if s.empty:
                continue
            p = proc_df[(proc_df.n_neighbors == nn) & (np.isclose(proc_df.min_dist, md))] \
                if not proc_df.empty else pd.DataFrame()
            rows.append({
                "n_neighbors": nn, "min_dist": md, "n_seeds": len(s),
                "trust_k10_mean": s.trustworthiness_k10.mean(), "trust_k10_std": s.trustworthiness_k10.std(),
                "trust_k50_mean": s.trustworthiness_k50.mean(), "trust_k50_std": s.trustworthiness_k50.std(),
                "knn_recall_k10_mean": s.knn_recall_k10.mean(), "knn_recall_k10_std": s.knn_recall_k10.std(),
                "procrustes_mean": p.procrustes_disparity.mean() if not p.empty else np.nan,
                "procrustes_std": p.procrustes_disparity.std() if not p.empty else np.nan,
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False)
    log(f"summary -> {summary_path} ({len(summary)} combos)")

    if not smoke and len(nn_list) > 1 and len(md_list) > 1:
        _heatmaps(summary, heatmap_path, nn_list, md_list)


def _heatmaps(summary, heatmap_path, nn_list, md_list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("trust_k10_mean", "Trustworthiness@10"),
               ("trust_k50_mean", "Trustworthiness@50"),
               ("knn_recall_k10_mean", "kNN Recall@10"),
               ("procrustes_mean", "Pairwise Procrustes disparity")]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, (col, title) in zip(axes, metrics):
        piv = summary.pivot(index="n_neighbors", columns="min_dist", values=col) \
            .reindex(index=sorted(nn_list), columns=sorted(md_list))
        cmap = "YlOrRd" if col == "procrustes_mean" else "YlOrRd_r"
        im = ax.imshow(piv.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(md_list))); ax.set_xticklabels([f"{v:.2f}" for v in sorted(md_list)])
        ax.set_yticks(range(len(nn_list))); ax.set_yticklabels(sorted(nn_list))
        ax.set_xlabel("min_dist"); ax.set_ylabel("n_neighbors"); ax.set_title(title, fontsize=9)
        for i in range(len(nn_list)):
            for j in range(len(md_list)):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"heatmaps -> {heatmap_path}")


def main():
    ap = argparse.ArgumentParser(description="UMAP hyperparameter sweep (Task C).")
    ap.add_argument("--config", default=None, help="Path to analysis_config.yaml")
    ap.add_argument("--resume", action="store_true",
                    help="Skip fits already present in the checkpoint. This is the default; "
                         "the flag is accepted for explicitness and matches the documented command.")
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore any existing checkpoint and recompute every fit.")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Run a single fit (nn=50, md=0.10, seed=42) to verify the pipeline.")
    args = ap.parse_args()
    # Completed fits are skipped by default (idempotent re-runs). --fresh overrides.
    run_sweep(load_config(args.config), resume=not args.fresh, smoke=args.smoke_test)


if __name__ == "__main__":
    main()
