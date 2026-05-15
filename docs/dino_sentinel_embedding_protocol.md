# DINO Sentinel-first embedding protocol

## Objective

This document records the auditable REV-P DINO Sentinel-first workflow. The DINO track is a review-only, self-supervised representation path for structural inspection of eligible Sentinel patch material. It is not a supervised flood susceptibility classifier, does not create labels or targets, and does not promote clusters into scientific classes.

## Methodological decisions

- `review_only=true`
- `supervised_training=false`
- `labels_created=false`
- `predictive_claims=false`
- `multimodal_hold=true`
- clusters are structural diagnostics only
- Sentinel-first is the active path
- multimodal stacks remain on hold until Recife balance/recovery is resolved

## Rationale

DINO is used as a frozen visual encoder because the project does not currently have validated observed-flood labels, supervised targets, or a cleared CRS/preflight gate. Self-supervised embeddings allow material-level comparison, neighbor review, outlier review, and structural clustering without claiming predictive performance.

DINOv2 with registers is the preferred backbone because register tokens are designed to improve representation stability in vision transformer feature spaces. DINOv2 without registers remains a control path. DINOv3 is recorded as a future comparison only if available and explicitly reviewed.

## Version summary

| Version | Script | Objective | Inputs | Local outputs | QA | Status |
| --- | --- | --- | --- | --- | --- | --- |
| v1fw | `scripts/dino/revp_v1fw_dino_embedding_extraction_scaffold.py` | Dry-run scaffold and execution schema for future embeddings | v1fu manifest, optional v1fv preflight, DINO config | `local_runs/dino_embeddings/v1fw/` | dry-run checks, no model/pixel read by default | implemented |
| v1fx | `scripts/dino/revp_v1fx_dino_smoke_embedding_execution.py` | Explicit smoke execution with real Sentinel pixel read and local embeddings | v1fu manifest, v1fv preflight, DINO config | `local_runs/dino_embeddings/v1fx/` | model attempts, metadata, failures, summary, QA | implemented |
| v1fy | `scripts/dino/revp_v1fy_dino_embedding_corpus_analysis.py` | Exploratory corpus analysis from local embeddings | v1fx local embedding manifest and metadata | `local_runs/dino_embeddings/v1fy/` | corpus, PCA, clustering, neighbors, region diagnostics | implemented |
| v1fz | `scripts/dino/revp_v1fz_dino_balanced_embedding_corpus.py` | Balanced Sentinel embedding subset by region | v1fu manifest, v1fv preflight, DINO config | `local_runs/dino_embeddings/v1fz/` | balanced selection, embeddings, PCA/clustering/neighbors/regions | implemented |
| v1ga | `scripts/dino/revp_v1ga_dino_embedding_structural_consistency_analysis.py` | Structural consistency analysis across regions, neighbors, clusters, and seeds | v1fz local manifest and embeddings | `local_runs/dino_embeddings/v1ga/` | consistency, centroid, cluster stability, outlier QA | implemented |

## Inputs and outputs

Primary versioned inputs:

- `manifests/dino_inputs/revp_v1fu_dino_sentinel_input_manifest/dino_sentinel_input_manifest_v1fu.csv`
- `configs/dino_embedding_extraction.example.yaml`
- scripts under `scripts/dino/`

Private/local inputs:

- `local_runs/dino_asset_preflight/v1fv/dino_local_asset_preflight_v1fv.csv`
- Sentinel raster references resolved inside the private `PROJETO` workspace

Local-only outputs:

- runtime CSV/JSON QA products under `local_runs/dino_embeddings/`
- embedding `.npz` files under `local_runs/dino_embeddings/*/embeddings/`

No `.npz`, `.npy`, rasters, GeoTIFFs, checkpoints, or `local_runs/` outputs are intended for Git.

## QA and audit checklist

- `.npz` embeddings are only under `local_runs/`
- heavy outputs are outside Git
- `pytest` passes
- local manifests are generated
- QA CSV files are generated
- summary JSON files are generated
- v1fz balanced corpus contains Curitiba, Petropolis/Petrópolis, and Recife
- no training loop, optimizer, labels, or targets are created
- no predictive metrics or supervised claims are emitted

## Exploratory analysis scope

The analysis layer supports:

- PCA coordinates and explained variance
- lightweight structural clustering
- nearest-neighbor and reciprocal-neighbor checks
- outlier diagnostics
- region centroids, dispersion, and intra/inter-region similarity
- cluster stability across seeds and K values

These outputs are structural diagnostics for review. They are not semantic classes, not flood labels, and not model performance evidence.

## Current limitations

- The balanced corpus is intentionally small and exploratory.
- CPU execution is acceptable for smoke and audit runs, but full runs may be slow.
- DINO embeddings depend on local availability or explicitly allowed model download.
- Multimodal assets remain excluded from the active path.
- Regional comparisons are descriptive and structural only.

## Valid next steps

- Expand Sentinel embeddings toward the full 128-patch corpus.
- Repeat v1ga consistency analysis on larger local-only runs.
- Review medoids, outliers, and reciprocal neighbors manually.
- Add visual QA panels only if kept local or explicitly approved for versioned documentation.
- Revisit multimodal stacks only after Recife recovery/balance blockers are resolved.
