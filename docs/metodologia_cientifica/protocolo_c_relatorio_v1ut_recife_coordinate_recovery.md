# Protocolo C v1ut - Recife Coordinate Recovery from Public CKAN

- event_id: `REC_2022_05_24_30`
- coordinate assets located: `24`
- rows with coordinates reported by v1uk: `438366`
- rows in Recife plausible range after reparse: `438363`
- review-only coordinate candidates: `0`
- max status: `RECIFE_PUBLIC_COORDINATE_CANDIDATE_FOR_REVIEW`
- ground_truth_operational: `false`
- can_create_ground_reference: `false`
- can_create_training_label: `false`
- can_reopen_protocol_b: `false`
- dino_usage: `SUPPORT_ONLY`
- no_overlay_executed: `true`
- no_coordinates_invented: `true`
- geocoding_executed: `false`
- centroid_used: `false`

v1ut recovers only coordinates explicitly present in public CKAN assets already downloaded locally. Raw coordinate values remain local-only; versionable outputs store counts, hashes, classifications and blockers.

Next recommended action: `v1uu - Recife Contextual Coordinate Layer Consolidation` because public coordinates exist, but they are contextual or not joined to event-window hazard rows.

## Result
REC_2022 does not become ground reference, ground truth, patch positive, patch negative, observed flood label, flood detected or operationally validated in v1ut.
