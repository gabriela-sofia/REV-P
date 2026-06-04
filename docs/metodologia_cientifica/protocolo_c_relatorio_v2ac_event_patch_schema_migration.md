# Protocolo C v2ac - Event-Patch Schema Migration Implementation

- packages migrated to v2: `172`
- packages with explicit namespace: `171`
- packages with explicit DINO crosswalk: `171`
- packages without anchor crosswalk: `172`
- packages with unlinkable date: `171`
- packages with missing date: `1`
- packages valid against the contract: `171`
- namespace population rows: `172`
- crosswalk population rows: `172`
- temporal population rows: `172`
- blocker normalization rows: `1548`
- readiness rows: `2580`
- migration diff rows: `172`
- ground-reference blocker rows: `24`
- selected next target: `EVENT_PATCH_PACKAGE_V2_QA_HARNESS`
- suggested next version: `v2ad — Event-Patch Package V2 QA Harness`

v2ac migrated the event-patch packages to the hardened v2 schema additively. It modified no prior output, inferred no crosswalk, inferred no Sentinel date, applied no cross-namespace date, executed no overlay, and created no ground truth, ground reference or label.

## How many packages migrated
172 event-patch packages were migrated to the v2 schema, all preserving their original event_patch_candidate_id.

## Namespace and crosswalk
171 packages carry an explicit event-patch candidate namespace. 171 have an explicit DINO crosswalk via identical patch_id. No package has an anchor/REFPATCH or scaffolding crosswalk, because no explicit key exists; anchor/refpatch fields stay empty with an explicit blocker.

## Temporal status
171 packages have a Sentinel date recovered only in a parallel namespace, kept unlinkable with an empty sentinel_scene_date. 1 have no recoverable date. No date was inferred or applied by region.

## Schema validation
171 packages validate against the v2ab contract as schema-valid non-operational; the remainder are blocked (e.g. missing patch id).

## Persisting blockers
Every package keeps no_observed_geometry, no_occurrence_coordinates, no_overlay, no_ground_reference, no_training_label and patch_truth_forbidden; plus unlinkable_sentinel_date or no_sentinel_date and no_explicit_anchor_crosswalk.

## Why there is still no overlay
No overlay was executed and overlay status stays BLOCKED. The migration normalizes identity and temporal fields but establishes no observed occurrence geometry.

## Why there is still no ground reference
No package has a linkable observed occurrence geometry; without it there is no basis for ground reference, and none was created.

## Next programming step
The score-based ranker selected `EVENT_PATCH_PACKAGE_V2_QA_HARNESS` (`v2ad — Event-Patch Package V2 QA Harness`).
