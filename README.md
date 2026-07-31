# Water-quality UMAP reproducibility package

Reproducibility package for:

> **Syed Sohail & Daqing Chen (2026).** Machine-learning-derived chemical signatures for network-scale river monitoring: A case study of England's routine water-quality archive.

It reproduces the study's pipeline end to end: from Environment Agency river
observations to the 12-feature / 8,857-site analytical matrix, the retained UMAP
embedding, chemical peer retrieval, systematic benchmarking, robustness analyses,
and the UMAP hyperparameter sweep.

## Study purpose

England's routine water-quality archive contains millions of observations across
thousands of river sites. This work derives a compact **chemical fingerprint** per
site (long-term median values for 12 determinands), embeds the national site
portfolio with UMAP for visual exploration, and — crucially — retrieves
chemically similar "peer" sites to support monitoring decisions.

### Portfolio visualisation ≠ exact peer retrieval

This distinction runs through the whole package:

- The **2-D UMAP embedding is a portfolio view** — a map for seeing national
  structure (a nutrient–mineralisation axis and a hardwater–softwater / pH axis).
  It is *not* the space in which similarity is measured.
- **Exact peer retrieval happens in the standardised 12-dimensional space.** The
  reference peers are the nearest sites by 12-D Euclidean distance.
- The **hybrid method** uses UMAP only to shortlist candidates, then re-ranks them
  by 12-D chemical distance. It recovers most of the chemical fidelity of direct
  12-D retrieval (distance ratio ≈ 1.17 vs 1.90 for direct-UMAP retrieval) while
  respecting the map.

Fingerprints describe **typical cross-sectional chemistry** (site medians over
2015–2024), **not** temporal change.

## Repository structure

```
water-quality-umap-reproducibility/
├── README.md                     ├── LICENSE  ├── CITATION.cff  ├── .gitignore
├── requirements.txt
├── config/analysis_config.yaml   # every scientific parameter, in one place
├── data/
│   ├── README.md                 # data acquisition & placement
│   ├── raw/                       # large EA parquets (download; gitignored)
│   ├── reference/                 # small lookups (WFD typology, feature mapping)
│   └── processed/                 # small analytical matrices + embedding (tracked)
├── src/waterqual/                # config loader + shared metrics + peer retrieval
├── scripts/                      # numbered pipeline stages + run_umap_sweep.py
├── outputs/
│   ├── figures/ tables/          # canonical manuscript figures & table sources
│   └── verification/             # expected_metrics.json
└── docs/manuscript_output_map.md # every result → the code that makes it
```

## Requirements

- **Python 3.10–3.11** (numpy is pinned `< 2`; the UMAP/numba stack requires it).
- Install dependencies:

```bash
pip install -r requirements.txt
```

All scientific settings live in `config/analysis_config.yaml`; scripts create the
output directories they need and use only relative paths.

## Data

The large raw observations are **not** shipped here (Environment Agency open data,
OGL v3.0). See [`data/README.md`](data/README.md) for download, filters, expected
filenames, columns, and placement. Small processed matrices, the retained UMAP
embedding, and the lookups **are** included, so the embedding, benchmarking,
robustness, and sweep stages run without the raw download.

Check what you have:

```bash
python scripts/01_prepare_data.py
```

## Pipeline execution order

Run from the repository root. Stages that only need the included processed data
are marked ✅; stages needing the raw download are marked ⬇.

```bash
# ⬇  0. Build the site-level stats table from raw observations (if starting from raw)
python scripts/01_prepare_data.py --build-site-stats

# ⬇  A. Build & validate the 12-feature / 8,857-site matrix   (needs site_det_stats)
python scripts/02_build_matrix.py --config config/analysis_config.yaml

# ✅ B1. Fit PCA / t-SNE / UMAP (retained configuration)
python scripts/03_fit_embeddings.py --config config/analysis_config.yaml

# ✅ B2-3. Systematic peer benchmarking across 100 focal sites
python scripts/04_benchmark_peers.py --config config/analysis_config.yaml

# ✅ C. UMAP hyperparameter sweep (200 fits) — resumable; see below
python scripts/run_umap_sweep.py --config config/analysis_config.yaml --resume

# ✅ D. Robustness: 80% subsampling (×100) + leave-one-feature-out (×12)
python scripts/05_run_robustness.py --config config/analysis_config.yaml

# ✅/⬇ E. Interpretation, figures, sensitivity (censoring needs raw observations)
python scripts/06_generate_outputs.py --config config/analysis_config.yaml
```

`--config` may be omitted; it defaults to `config/analysis_config.yaml`.

### Expected runtime / expensive stages

A single UMAP fit of 8,857 × 12 takes roughly **1.5–3 minutes** (first fit in a
process is slower due to numba JIT).

| Stage | Cost |
|-------|------|
| 02 build matrix | seconds (deterministic, no UMAP) |
| 03 fit embeddings | a few minutes (1 UMAP + t-SNE) |
| 04 benchmark | seconds (no new fits) |
| **C sweep** | **200 fits ≈ 1–3 hours** (resumable) |
| **D robustness** | **112 fits ≈ 2–5 hours** (resumable) |
| 06 outputs | seconds–minutes (support sweep recomputes 20 pivots) |

## The UMAP hyperparameter sweep (Task C)

Grid: `n_neighbors ∈ {15,30,50,100,150}` × `min_dist ∈ {0.01,0.10,0.25,0.50}` ×
`seed ∈ {0…9}` = **200 fits**. For each of the 20 parameter combinations the 45
pairwise Procrustes disparities among its 10 seeds are computed (**900 pairs**).

```bash
# Full sweep (resumable — safe to interrupt and re-run):
python scripts/run_umap_sweep.py --config config/analysis_config.yaml --resume

# Lightweight smoke test — one fit (nn=50, md=0.10, seed=42):
python scripts/run_umap_sweep.py --config config/analysis_config.yaml --smoke-test
```

**Outputs** (in `outputs/sweep/`): `sweep_checkpoint.csv` (per-fit log with
parameters, seed, runtime, status), `sweep_procrustes_pairs.csv`,
`sweep_seed_metrics.csv` (200 rows), `sweep_results.csv` (20-combo summary),
`sweep_heatmaps.png`.

**Checkpointing & resumption.** Every fit is appended to the checkpoint and
`fsync`-ed immediately, so an interruption loses at most the in-flight fit.
Completed fits are skipped on the next run (this is the default; `--resume` states
it explicitly). Use `--fresh` to recompute everything. Aggregation into the
summary tables runs automatically once the grid is complete. The full run needs no
special hardware — single-threaded (`n_jobs=1`, required for deterministic seeds),
~1–3 hours, a few MB of output.

## Regenerating manuscript figures & tables

- Canonical figures and machine-readable table sources are provided in
  `outputs/figures/` and `outputs/tables/`.
- `scripts/03_fit_embeddings.py` + `scripts/06_generate_outputs.py` regenerate the
  national embedding (Figure 4), axis-driver overlays (Figure 6), correlation
  artifacts (Figure 5), and the support-threshold sensitivity table (Table 18).
- `scripts/run_umap_sweep.py` regenerates the sweep heatmaps (Figure 11).
- `docs/manuscript_output_map.md` maps **every** reported figure/table/metric to
  its script, inputs, output file, and config keys.

## Key expected verification values

From `outputs/verification/expected_metrics.json`:

| Quantity | Value |
|---|---|
| Retained chemical features | **12** |
| Eligible sites | **8,857** |
| Benchmark focal sites | **100** |
| Overlap@5 — direct UMAP / hybrid | **0.86 / 2.43** |
| Chemical-distance ratio — direct UMAP / hybrid | **1.9027 / 1.1716** |
| WFD typology agreement — reference / UMAP / hybrid | **0.463 / 0.411 / 0.458** |
| Sweep seed-specific fits | **200** (900 Procrustes pairs) |
| Support-threshold grid | 20 combos → **identical** 12-feature / 8,857-site panel |

These were regenerated by this package's own scripts during assembly (the rebuilt
scaled matrix is byte-identical to the canonical one; Overlap@5, distance ratios,
and WFD agreement match). Structural counts are exact; UMAP-derived quantities can
vary slightly (≈ ±0.01) across package versions and platforms — this is expected
and is exactly what the sweep and robustness analyses characterise.

## Known limitations

- **UMAP is stochastic.** `random_state=42` and `n_jobs=1` make a single run
  reproducible, but exact coordinates differ across umap-learn / numba versions
  and platforms. Similarity is therefore measured in the 12-D space, and stability
  is quantified by the sweep (Procrustes) and robustness analyses.
- **Cross-sectional only.** Fingerprints are long-term medians; the package makes
  no temporal-trend claims.
- **Data currency.** The EA archive is periodically revised; a fresh download may
  differ slightly from the 2015–2024 snapshot used in the manuscript. The matrix
  builder asserts the 12-feature / 8,857-site panel and will flag any drift.
- **WFD typology coverage** is ~96% of focal sites; agreement is reported over
  sites with a known typology.
- The τ coverage sweep (Table 14) and drop-2 same-environment table (Table 16)
  are documented and parameterised in the config but are preserved as workflows
  rather than part of the core reproduced set (see `docs/manuscript_output_map.md`).

## Citation

See `CITATION.cff`. Please cite the manuscript above if you use this code or the
derived results.

## Licence

Code: MIT (`LICENSE`). Data: Environment Agency open data under the Open
Government Licence v3.0 — *Contains Environment Agency data © Environment Agency
and database right.*
