# DINO Sentinel-first scientific evidence summary

## Scope

This document summarizes the review-only DINO Sentinel-first track in REV-P. It records what was technically demonstrated, which local evidence products were generated, and which scientific claims remain explicitly forbidden.

The DINO track uses a frozen self-supervised encoder for structural inspection of Sentinel patches. It does not create labels, targets, supervised classifiers, flood truth, susceptibility classes, or predictive performance claims.

## What was technically demonstrated

- A Sentinel-first DINO input manifest was created from consolidated repository manifests.
- Local asset preflight confirmed which Sentinel references can be resolved in the private workspace.
- DINOv2 with registers was loaded as a frozen encoder in local execution.
- Real Sentinel pixels were read only for explicit embedding or robustness execution stages.
- Local embeddings were generated under `local_runs/` only.
- A balanced regional corpus was produced for Curitiba, Petrópolis, and Recife.
- An expanded local corpus run produced 12 embeddings, 4 per region.
- Structural diagnostics were generated for neighbors, clusters, medoids, outliers, graph bridges, perturbation robustness, longitudinal stability, provenance, and human review triage.

## Local evidence produced

Local runtime evidence is stored under `local_runs/dino_embeddings/` and is intentionally not versioned:

- v1fw: dry-run extraction scaffold and output schema.
- v1fx: smoke embedding execution.
- v1fy: exploratory embedding corpus analysis.
- v1fz: balanced regional corpus and structural analysis.
- v1ga: structural consistency analysis.
- v1gb: local visual structural review.
- v1gc: geo-structural diagnostics.
- v1gd: perturbation robustness diagnostics.
- v1ge: expanded Sentinel embedding corpus.
- v1gf: structural evidence index.
- v1gg: human review package.
- v1gh: longitudinal structural diagnostics.
- v1gi: structural provenance tracker.
- v1gj: multimodal readiness audit with multimodal execution disabled.
- v1gk: reproducibility audit.

## What was not claimed

The DINO track does not claim:

- observed flood labels;
- binary targets;
- automatic labels;
- susceptibility classes;
- cluster-to-class interpretation;
- predictive performance;
- supervised model accuracy;
- vulnerability inference;
- flood occurrence inference;
- multimodal fusion readiness beyond blocker auditing.

`review_priority` is a human-review triage field only. It is not a scientific label, not a target, and not a proxy for vulnerability or flood status.

## Embedding status

Real embeddings were generated locally using DINOv2 with registers as a frozen encoder. The expanded local corpus contains 12 Sentinel embeddings with embedding dimension 768:

- Curitiba: 4
- Petrópolis: 4
- Recife: 4

Embedding arrays remain local-only in `local_runs/` and are not intended for Git.

## Structural analysis status

The current structural layer supports:

- nearest-neighbor analysis;
- reciprocal pair checks;
- PCA/manifold coordinates;
- lightweight clustering diagnostics;
- medoid and edge-case review;
- structural outlier diagnostics;
- geo-structural graph diagnostics;
- cross-region bridge candidates;
- perturbation robustness checks;
- longitudinal diagnostic persistence;
- provenance and review traceability.

All outputs are review-only diagnostics. They support manual inspection and method audit, not classification.

## Robustness status

v1gd tested controlled perturbations for local sensitivity audit:

- light Gaussian noise;
- brightness scaling;
- contrast scaling;
- light blur;
- crop jitter;
- controlled band dropout.

These perturbations are not training augmentations. They are not used to train or validate a supervised model.

## Human review status

v1gg generated a local human review package from structural diagnostics. It includes review items for bridge candidates, outliers, medoids, robust or unstable embeddings, and regional examples.

Human notes remain empty until manual review. The package does not copy raw rasters and does not version visual outputs.

## Multimodal status

Multimodal remains on hold.

- `multimodal_execution_enabled=false`
- `multimodal_training_enabled=false`
- `multimodal_hold=true`

v1gj is only a readiness audit. Readiness is not execution, not stack generation, not fusion, and not training.

## Current limitations

- The expanded corpus is still a local audit subset, not the full 128-patch run.
- DINO embeddings depend on local model availability and local asset access.
- Structural graph and perturbation diagnostics are sensitive to corpus size and top-k settings.
- Regional comparisons are descriptive and structural only.
- Multimodal fusion remains blocked by bindings, geometry, CRS, and recovery/readiness questions.

## Valid next steps

- Commit the versioned scripts and documentation after final audit.
- Run the full 128-patch Sentinel embedding corpus locally when compute time is acceptable.
- Regenerate v1gf-v1gk from the larger corpus.
- Conduct manual review of v1gg items before writing scientific interpretation.
- Keep multimodal in hold until blockers are cleared by explicit evidence.
