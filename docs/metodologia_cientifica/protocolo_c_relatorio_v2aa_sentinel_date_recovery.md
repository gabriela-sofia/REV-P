# Protocolo C v2aa - Sentinel Date Recovery for Event-Patch Packages

- registries scanned: `123` (parseable for dates: `23`)
- filename/scene-id dates extracted: `4`
- sidecar explicit-date resolutions: `15`
- patches consolidated: `8` (recovered: `7`, missing/blocked: `1`)
- usable (HIGH/MEDIUM) dates: `7`
- event-patch candidates with placed temporal distance: `0` of `172`
- readiness update rows: `1032`
- ground-reference blocker rows: `21`
- selected next target: `EVENT_PATCH_PACKAGE_SCHEMA_HARDENING`
- suggested next version: `v2ab — Event-Patch Package Schema Hardening`

v2aa recovered Sentinel scene dates only from existing filenames, scene ids and explicit sidecar date fields. It never downloaded data, queried the web, inferred a date, used an approximate date as real, used created_at/modified_at or file mtimes, executed overlay, or created ground truth, ground reference or labels.

## How many registries were scanned
123 versionable registries (datasets/ and configs/) were scanned for patch and date metadata; 23 contained both a patch id and a date or filename field and were parsed. local_only was never scanned.

## Which date sources worked
Real Sentinel scene dates were recovered from canonical scene-id values (pattern `YYYYMMDDT...`) and from explicit `scene_date` fields in the anchor Sentinel patch registries. The event-patch candidate patch ids (CUR/PET/REC numeric namespace) carry no Sentinel scene date in any versionable registry, so they remain without a recoverable date.

## How many patches recovered a date / stayed missing
7 patches recovered a Sentinel date; 1 remained missing or blocked. 7 dates are usable (HIGH/MEDIUM confidence) for review-only temporal linkage.

## How many event-patch candidates improved
0 of 172 event-patch candidates obtained a placed temporal distance. The remainder stay temporally blocked because their patch ids have no recoverable Sentinel date locally.

## Which regions improved
Regions with any reduction: PET, REC. The recovered dates belong to the anchor reference-patch namespace; the bulk candidate namespace keeps no_sentinel_date dominant.

## Was no_sentinel_date reduced
Partially and only where real dates exist. The blocker remains dominant for the event-patch candidate namespace, so it is recorded as still-blocked in the readiness and blocker matrices.

## Why there is still no overlay
No overlay was executed and overlay readiness stays BLOCKED. A recovered acquisition date does not establish observed occurrence geometry, which overlay would require.

## Why there is still no ground reference
A Sentinel acquisition date is temporal metadata, not an observed occurrence. Without observed occurrence geometry there is no basis for ground reference, and none was created.

## Next programming step
The score-based ranker selected `EVENT_PATCH_PACKAGE_SCHEMA_HARDENING` (`v2ab — Event-Patch Package Schema Hardening`).
