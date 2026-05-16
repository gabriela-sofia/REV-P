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
| v1gb | `scripts/dino/revp_v1gb_dino_embedding_local_visual_structural_review.py` | Local visual structural review of embeddings, medoids, neighbors, and outliers | v1fz local manifest and embeddings | `local_runs/dino_embeddings/v1gb/` | visual panels, spatial consistency, multiscale checks, medoids, outlier taxonomy | implemented |
| v1gc | `scripts/dino/revp_v1gc_dino_embedding_geo_structural_diagnostics.py` | Geo-structural diagnostics linking local patch geometry with embedding neighborhoods | v1fz local manifest and embeddings | `local_runs/dino_embeddings/v1gc/` | geo-distance comparisons, graph topology, cross-region bridges, transition candidates | implemented |
| v1gd | `scripts/dino/revp_v1gd_dino_embedding_perturbation_robustness_diagnostics.py` | Perturbation robustness diagnostics for local Sentinel DINO embeddings | v1fz local manifest and embeddings | `local_runs/dino_embeddings/v1gd/` | controlled perturbations, drift, neighbor persistence, graph robustness, regional robustness | implemented |
| v1ge | `scripts/dino/revp_v1ge_dino_expanded_sentinel_embedding_corpus.py` | Expanded Sentinel embedding corpus run with resume and regional balancing | v1fu manifest, v1fv preflight, DINO config | `local_runs/dino_embeddings/v1ge/` | embedding consistency, hashes, failures by region, resume/skip audit | implemented |
| v1gf | `scripts/dino/revp_v1gf_dino_structural_evidence_index.py` | Integrated structural evidence index for manual review triage | local v1fz/v1ge and v1ga-v1gd outputs | `local_runs/dino_embeddings/v1gf/` | guardrails, review priority summary, no-label/no-target QA | implemented |
| v1gg | `scripts/dino/revp_v1gg_dino_human_review_package.py` | Local-only human review package for medoids, outliers, bridges, and representatives | v1gf structural evidence index and local visual manifests | `local_runs/dino_embeddings/v1gg/` | review manifest, batches, local README, human review guardrails | implemented |
| v1gh | `scripts/dino/revp_v1gh_dino_longitudinal_structural_diagnostics.py` | Longitudinal comparison of structural diagnostics across DINO phases | local v1fz-v1gg outputs | `local_runs/dino_embeddings/v1gh/` | neighbor, outlier, medoid, bridge, review-priority, and regional stability | implemented |
| v1gi | `scripts/dino/revp_v1gi_dino_structural_provenance_tracker.py` | Patch-to-embedding-to-diagnostic provenance tracking | local v1fz-v1gg outputs | `local_runs/dino_embeddings/v1gi/` | provenance index, diagnostic history, review traceability | implemented |
| v1gj | `scripts/dino/revp_v1gj_multimodal_readiness_audit.py` | Multimodal readiness audit without multimodal execution | v1fu/v1fv Sentinel manifests and local DINO outputs | `local_runs/dino_embeddings/v1gj/` | readiness table, blockers, asset inventory, multimodal-disabled guardrails | implemented |

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
- local visual review panels for nearest neighbors, medoids, edge cases, region exemplars, and outliers
- local spatial consistency diagnostics and multiscale structural sanity checks
- geo-structural diagnostics comparing embedding neighborhoods with patch centroids, local graph topology, and cross-region bridge candidates
- perturbation robustness diagnostics under small controlled image changes, without creating training augmentations
- expanded local corpus execution with resume/skip-existing support
- integrated structural evidence indexing for review triage
- local-only human review packaging for later manual inspection
- longitudinal stability checks across DINO diagnostic phases
- provenance tracing from patch to embedding to review package
- multimodal readiness audit while multimodal execution remains disabled

These outputs are structural diagnostics for review. They are not semantic classes, not flood labels, and not model performance evidence.

## Visual structural review

v1gb adds local-only visual panels to support later human review of structural embedding behavior. The panels are intended to make nearest-neighbor relationships, reciprocal pairs, medoids, edge cases, isolated embeddings, and regional exemplars auditable without turning clusters into labels.

The medoid and representative selections are structural conveniences only. They indicate positions in an embedding space for inspection, not real-world class membership, not flood occurrence, and not validation evidence. Visual QA is also limited by the current local corpus size, available Sentinel patch rendering, and the fact that image panels are derived from local runtime reads rather than versioned raw data.

## Geo-structural diagnostics

v1gc compares embedding relationships with local spatial geometry when centroid coordinates or raster metadata bounds are available. It produces distance-vs-similarity tables, regional overlap metrics, compactness summaries, graph components, hubs, bridge candidates, transition candidates, and topology continuity checks.

The graph layer is diagnostic only: nodes are patches, edges are embedding-nearest-neighbor relations, and cross-region bridges are candidates for human review. They are not classes, not labels, not validation targets, and not evidence of predictive performance. Topology metrics depend on the current corpus size, coordinate availability, and chosen top-k neighborhood; they should be treated as audit aids for selecting examples for later manual inspection.

## Perturbation robustness diagnostics

v1gd applies small reversible perturbations to local Sentinel renderings to measure whether DINO embedding relationships are structurally stable. The perturbations include light Gaussian noise, brightness scaling, contrast scaling, blur, crop jitter, and optional band dropout. They are used only for sensitivity audit and are not saved as a training set.

The robustness outputs compare original versus perturbed embeddings through cosine drift, nearest-neighbor persistence, cluster-assignment stability, medoid persistence, graph edge persistence, bridge persistence, hub stability, and regional drift summaries. These diagnostics support manual review of sensitivity and do not establish predictive reliability, class membership, or supervised performance.

## Expanded corpus and human review triage

v1ge expands Sentinel-first embedding execution beyond the initial small balanced corpus when local compute and model availability allow it. The run remains local-only and supports `--limit`, `--per-region-limit`, `--resume`, and `--skip-existing` so partial execution can be audited without overwriting previous local embeddings.

v1gf consolidates structural diagnostics into a single evidence index per patch. The `review_priority` field is a deterministic triage cue for human inspection only. It is not a label, not a class, not a target, and not evidence that a patch has any flood or susceptibility status.

v1gg packages local references for future human review of medoids, outliers, bridges, robust/unstable embeddings, reciprocal-neighbor examples, and region representatives. It does not copy raw rasters and does not version local images. Human notes are intentionally blank until manual inspection occurs.

## Longitudinal diagnostics and provenance

v1gh compares structural signals across the local DINO phases to check whether neighbor relationships, outlier flags, medoid roles, bridge roles, regional summaries, and review-priority triage remain traceable across versions. The output is an audit of diagnostic persistence, not a claim of temporal environmental change.

v1gi records patch-to-embedding-to-diagnostic provenance. It tracks which versions touched each patch, which diagnostics were produced, which QA files passed, which local visualizations exist, and whether each patch appears in medoid, bridge, outlier, or human-review package roles.

## Multimodal readiness hold

v1gj audits structural readiness for future multimodal work without activating multimodal execution. It records Sentinel availability, local preflight status, known blocker categories, asset inventory, and guardrails. Readiness does not equal execution: `multimodal_execution_enabled=false` and `multimodal_training_enabled=false` remain active constraints.

## Current limitations

- The balanced corpus is intentionally small and exploratory.
- Visual review panels are local QA aids and should not be interpreted as semantic cluster explanations.
- Geo-structural graph diagnostics are sensitive to local corpus size, coordinate provenance, and nearest-neighbor settings.
- Perturbation diagnostics are local sensitivity audits; they are not training augmentations and do not validate operational robustness.
- Expanded corpus execution is still constrained by local model availability, CPU/GPU speed, and private asset access.
- `review_priority` and human-review package entries are audit workflow aids only.
- Longitudinal stability is diagnostic persistence across local outputs, not environmental time-series inference.
- Multimodal readiness is a blocker audit and compatibility preparation layer, not fusion, stack generation, or multimodal training.
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
