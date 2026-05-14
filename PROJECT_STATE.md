# Project State

Status date: 2026-05-14

## Canonical stage

REV-P is currently frozen as a review-only, audit-first dataset preparation pipeline.

## Current facts

- External validation package exists for review.
- Patch grounding exists for review.
- Training-readiness manifests exist for review.
- Sentinel-first DINO path is the valid next implementation path.
- Multimodal continuation is conditional on Recife stack recovery/balance.

## Current blockers

- No observed-flood binary ground truth.
- No validated susceptibility labels.
- No supervised model target.
- patch_bound_validated = 0/59.
- preflight_ready = 0/59.
- CRS gate remains blocked.
- Recife ext/bg naming issue remains unresolved for canonical TIF binding.
- Multimodal path remains on hold until Recife balance/recovery.

## Current allowed DINO scope

DINO may be used only as a frozen self-supervised visual encoder for:
- embedding extraction;
- nearest-neighbor retrieval;
- PCA/UMAP projection;
- clustering;
- outlier detection;
- visual/manual review support.

DINO must not be reported as a supervised flood susceptibility classifier at this stage.
