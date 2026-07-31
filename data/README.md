# Data

This directory holds the pipeline's inputs and processed artifacts. Large raw
data are **not redistributed** — this file explains how to obtain and place them.

```
data/
├── raw/         # large source data — download; NOT tracked in git
├── interim/     # intermediate build products
├── processed/   # small analytical matrices + embedding (tracked; regenerable)
└── reference/   # small lookups / metadata (tracked; redistributable)
```

## Data source

River water-quality observations are **Environment Agency open data**, released
under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

- **Water Quality Explorer / archive API:** https://environment.data.gov.uk/water-quality
- **WFD Cycle 3 river water body catchments (GeoPackage):**
  https://environment.data.gov.uk/dataset/cd84a955-fd0a-4f5d-9bcb-b869c8906f9e

Study scope used in the manuscript:

| Filter            | Value                                             |
|-------------------|---------------------------------------------------|
| Water type        | Freshwater rivers (`RIVER / RUNNING SURFACE WATER`) |
| Region            | The seven English River Basin Districts           |
| Period            | 2015–2024 (2014 deliberately excluded)            |
| Determinands      | See the 12-feature panel in `reference/authoritative_feature_mapping.csv` |

The download can be scripted against the WQE API using the basin identifiers in
`reference/clean_basins.csv` (`area_id` per region/sub-area).

## Expected files

### `data/raw/` (download; not tracked — see `.gitignore`)

| File | Description | Approx size |
|------|-------------|-------------|
| `observations_stitched_skip2014.parquet` | Row-per-observation EA river data, 2015–2024. Columns include `samplingPoint.notation`, `determinand.notation`, `determinand.prefLabel`, `result`, `unit`, `phenomenonTime`, `sampleMaterialType`, `samplingPoint.region`. | ~87 MB |
| `site_det_stats_2015_2024.parquet` | Site × determinand median table (aggregated from observations). Columns: `site_id`, `determinant_code`, `determinant`, `unit`, `n`, `median`, `mean`, `std`, `min`, `max`, `q10`, `q90`. Build it with `scripts/01_prepare_data.py --build-site-stats`. | ~9.5 MB |

### `data/reference/` (redistributed here; EA open data)

| File | Description |
|------|-------------|
| `site_wfd_typology_lookup.csv` | Per-site WFD Cycle 3 join: `site_id`, `lon`, `lat`, `region`, `water_body_typology`, class fields. |
| `authoritative_feature_mapping.csv` | Feature identity `determinant_code × unit × name` for the 12 retained features. |
| `clean_basins.csv` | EA basin/sub-area identifiers used to script the download. |

### `data/processed/` (tracked; small; regenerable)

Provided so the embedding, benchmarking, robustness, and sweep stages run without
the large raw download. All are regenerable from `data/raw/` via
`scripts/02_build_matrix.py` and `scripts/03_fit_embeddings.py`.

| File | Produced by | Description |
|------|-------------|-------------|
| `feature_matrix_8857x12_scaled.csv` | `02_build_matrix.py` | Standardised analytical matrix (8,857 × 12). |
| `feature_matrix_8857x12_pre_imputation.csv` | `02_build_matrix.py` | Eligible sites, before median imputation (with gaps). |
| `feature_matrix_8857x12_imputed_unscaled.csv` | `02_build_matrix.py` | Imputed but unscaled values (raw units, for chemistry profiles). |
| `v2_feature_metadata.csv` | `02_build_matrix.py` | Per-feature coverage, medians, scaler params. |
| `site_exclusion_audit.csv` | `02_build_matrix.py` | Per-site eligibility and exclusion reason. |
| `umap_v2.csv` | `03_fit_embeddings.py` | Retained UMAP coordinates (`site_id`, `umap_1`, `umap_2`). |
| `embedding_parameters.json` | `03_fit_embeddings.py` | Model parameters + environment record. |

## Where to put downloaded files

Place the two raw parquet files in `data/raw/`. The configured paths are in
`config/analysis_config.yaml` under `paths:` — adjust there if you store them
elsewhere. Then verify:

```bash
python scripts/01_prepare_data.py            # reports what is present / missing
```

## Column and unit notes

- Determinand identity is **`determinand.notation` (code) × `unit`**. Feature
  labels are taken from the source data, not hand-maintained. The 12 features and
  their units are listed in `reference/authoritative_feature_mapping.csv`.
- `result` values beginning with `<` are left-censored non-detects. The
  site-stats builder strips the `<` before aggregation; censoring rates per
  determinand are reported separately (manuscript Table 15).
- **Signatures describe typical cross-sectional chemistry** (long-term site
  medians), **not** temporal change.

## Licensing / redistribution

The observations and WFD data are EA open data under OGL v3.0 and may be
redistributed with attribution: *Contains Environment Agency data © Environment
Agency and database right.* The large raw parquets are excluded here only for
size, not licensing. The pipeline **code** is MIT-licensed (see `LICENSE`).

## Integrity

After building/placing inputs, you can record checksums for your own provenance:

```bash
python - <<'PY'
import hashlib, pathlib
for p in sorted(pathlib.Path("data").rglob("*")):
    if p.is_file() and p.suffix in {".parquet", ".csv"}:
        print(hashlib.sha256(p.read_bytes()).hexdigest()[:16], p)
PY
```
