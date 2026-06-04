# Protocolo C v2ae - Multi-Region Registry Hardening

- canonical regions: `3`
- canonical events: `4`
- canonical event-patch packages: `172`
- consolidated blocker rows: `12`
- consolidated readiness rows: `730`
- reopen condition rows: `3`
- safe-use policy rows: `4`
- registry consistency QA: `CONSISTENT`
- Recife status: `REGION_HARDENED_CONTEXTUAL_COORDINATE_NON_OPERATIONAL`
- Petropolis status: `REGION_HARDENED_DOCUMENT_ONLY_NO_GEODATA`
- Curitiba status: `REGION_HARDENED_CONTEXT_ONLY_HOLD`
- selected next target: `EVENT_PATCH_PACKAGE_V2_QA_AUTOMATION`
- suggested next version: `v2af — Event-Patch Package V2 QA Automation`

v2ae consolidated the distributed state into hardened canonical multi-region registries. It modified no prior output, sought no new source, inferred no event/coordinate/date/crosswalk, promoted no context to occurrence, executed no overlay, and created no ground truth, ground reference or label.

## Canonical region status
Recife: `REGION_HARDENED_CONTEXTUAL_COORDINATE_NON_OPERATIONAL` (contextual coordinate layer, no occurrence coordinate). Petropolis: `REGION_HARDENED_DOCUMENT_ONLY_NO_GEODATA` (official document only, no public geodata). Curitiba: `REGION_HARDENED_CONTEXT_ONLY_HOLD` (event candidate and hydromet context, no occurrence layer).

## Canonical events and packages
4 canonical events and 172 canonical event-patch packages were consolidated, preserving every event_patch_candidate_id.

## Blockers
Global blockers: no_ground_reference, no_observed_geometry, no_occurrence_coordinates, no_overlay, no_training_label, patch_truth_forbidden. Region descriptor blockers: locality_only (Recife), document_only (Petropolis), context_only (Curitiba).

## Consolidated readiness
730 readiness rows across region, event and package scope; overlay, ground reference and training readiness are BLOCKED everywhere.

## Reopen conditions
Each region can only be reopened by a new qualifying public source (occurrence coordinates/geometry for Recife, public geodata/official crosswalk for Petropolis, official occurrence layer/event table for Curitiba). Region, name similarity, file order, inferred dates and inferred crosswalks are forbidden reopen bases.

## Safe and prohibited use
Safe: review-only, contextual support, evidence audit, DINO review support, package QA. Prohibited: ground truth, label, patch positive/negative, overlay truth, event validated by Sentinel, hydromet as occurrence, context layer as occurrence.

## Registry consistency QA
Result: `CONSISTENT` (10 checks, 0 failures).

## Why there is still no overlay
No overlay was executed and overlay readiness stays BLOCKED; hardening consolidates state but establishes no observed occurrence geometry.

## Why there is still no ground reference
No region or package has a linkable observed occurrence geometry; without it there is no basis for ground reference, and none was created.

## Why there is still no label
Labels require observed occurrence truth, which does not exist; creating one would be an unsupported overclaim.

## Next programming step
The score-based ranker selected `EVENT_PATCH_PACKAGE_V2_QA_AUTOMATION` (`v2af — Event-Patch Package V2 QA Automation`).
