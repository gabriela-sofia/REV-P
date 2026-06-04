# Protocolo C v1uz - Curitiba Context-Only Hold and Multi-Region Priority Re-Ranking

- Curitiba hold status: `CURITIBA_CONTEXT_ONLY_HOLD_NON_OPERATIONAL`
- non-occurrence guards: `6`
- Curitiba event-patch hold updates: `54`
- multi-region closures: `4`
- multi-region blocker rows: `44`
- multi-region readiness rows: `48`
- Recife closure: `RECIFE_CONTEXTUAL_COORDINATE_LAYER_CONSOLIDATED_NON_OPERATIONAL`
- Petropolis closure: `PETROPOLIS_DOCUMENT_ONLY_NO_GEODATA`
- Curitiba closure: `CURITIBA_CONTEXT_ONLY_HOLD_NON_OPERATIONAL`
- selected next programming target: `SENTINEL_DATE_RECOVERY_FOR_EVENT_PATCH_PACKAGES`
- suggested next version: `v2aa — Sentinel Date Recovery for Event-Patch Packages`

v1uz consolidated Curitiba as a context-only hold and re-ranked the next real programming target. It did not execute overlay, geocoding, centroid use, label creation, ground truth, ground reference, operational validation, DINO execution, model training, event inference, coordinate inference or raw data versioning.

## Curitiba final status
Curitiba is `CURITIBA_CONTEXT_ONLY_HOLD_NON_OPERATIONAL`. An official event candidate (`CUR_2022_01_15`) and hydromet support exist, and public/contextual layers (administrative, drainage, context) exist, but there is no observed occurrence layer, no possible-occurrence layer, no base for controlled feature download, no overlay preflight and no ground reference. Hydromet support is temporal hazard context only and is not an observed occurrence.

## Why Curitiba entered context-only hold
The v1uy deepening probed public geodata endpoints and classified layers as administrative, drainage or unknown context. No queryable occurrence table, endpoint or layer was found, so no controlled feature download was recommended. Context layers, alerts and hydromet series cannot be promoted to observed occurrence, ground reference or label.

## Recife final status
Recife is `RECIFE_CONTEXTUAL_COORDINATE_LAYER_CONSOLIDATED_NON_OPERATIONAL`. The strongest evidence is a contextual coordinate layer; there is no occurrence coordinate and no observed geometry, so overlay and ground reference remain blocked.

## Petropolis final status
Petropolis is `PETROPOLIS_DOCUMENT_ONLY_NO_GEODATA`. Only official documents are available with no public geodata and no observed geometry.

## Multi-region blockers
Every region shares `no_observed_geometry`, `no_occurrence_coordinates`, `no_sentinel_date`, `no_overlay`, `no_ground_reference` and `no_training_label`. Region-specific blockers are `locality_only` (Recife), `document_only` (Petropolis) and `context_only`/`hydromet_only` (Curitiba). `patch_truth_forbidden` applies to all.

## Multi-region readiness
Event registry, official source and temporal support are present across regions. Occurrence coordinate support, observed geometry support, overlay readiness, ground reference readiness and training readiness are absent or blocked for every region.

## Next programming target
The score-based ranker selected `SENTINEL_DATE_RECOVERY_FOR_EVENT_PATCH_PACKAGES`. Sentinel scene dates are missing for the large majority of event-patch candidates, and recovering them is metadata-only with low overclaim risk and high blocker reduction.

## Suggested next version
`v2aa — Sentinel Date Recovery for Event-Patch Packages` (planning only; implementation not started).

## Why there is still no ground reference, overlay or label
No region has an observed occurrence geometry tied to its event. Without observed occurrence geometry there is no basis for overlay, no basis for ground reference, and no basis for a training label. Creating any of them now would be an unsupported overclaim.
