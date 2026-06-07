# v2ap - readiness do link evento-patch

patch_truth_allowed=false para todos. patch_level_reference_candidate=true exige
geometria de evento + geometria de patch + crosswalk Sentinel explicito.

| candidate_id | event_geometry | patch_geometry | explicit_crosswalk | status |
| --- | --- | --- | --- | --- |
| PET_2022_02_15 | true | true | false | NEEDS_EXPLICIT_SENTINEL_CROSSWALK |
| REC_2022_05_24_30 | false | true | true | EVENT_REFERENCE_ONLY_NEEDS_EVENT_GEOMETRY |
| REC_2024_06_14_16 | false | true | false | EVENT_REFERENCE_ONLY_NEEDS_EVENT_GEOMETRY |
| CTB_2023_10_28_30 | false | true | false | EVENT_REFERENCE_ONLY_NEEDS_EVENT_GEOMETRY |
| CTB_2022_01_15_16 | false | true | false | EVENT_REFERENCE_ONLY_NEEDS_EVENT_GEOMETRY |
| CTB_2024_02_18_20 | false | true | false | EVENT_REFERENCE_ONLY_NEEDS_EVENT_GEOMETRY |
| REC_2023_02_05_06 | false | true | false | EVENT_REFERENCE_ONLY_NEEDS_EVENT_GEOMETRY |
| PET_2022_03_20_21 | true | true | false | NEEDS_EXPLICIT_SENTINEL_CROSSWALK |
| PET_2024_03_21_28 | false | true | false | EVENT_REFERENCE_ONLY_NEEDS_EVENT_GEOMETRY |
