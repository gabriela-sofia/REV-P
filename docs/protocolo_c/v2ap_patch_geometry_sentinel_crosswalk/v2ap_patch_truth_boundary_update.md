# v2ap - atualizacao do patch truth boundary

patch_truth_allowed=false nesta etapa. Mesmo com geometria+crosswalk explicitos,
o maximo e READY_FOR_EXTERNAL_VALIDATION, nunca ground truth operacional.

| candidate_id | event_geometry | patch_geometry | crosswalk | patch_truth_allowed | patch_reference_status |
| --- | --- | --- | --- | --- | --- |
| PET_2022_02_15 | READY | READY | MISSING | false | PARTIAL_PATCH_READINESS_NEEDS_GEOMETRY_OR_CROSSWALK |
| REC_2022_05_24_30 | MISSING | READY | EXPLICIT | false | PARTIAL_PATCH_READINESS_NEEDS_GEOMETRY_OR_CROSSWALK |
| REC_2024_06_14_16 | MISSING | READY | MISSING | false | EVENT_REFERENCE_ONLY |
| CTB_2023_10_28_30 | MISSING | READY | MISSING | false | EVENT_REFERENCE_ONLY |
| CTB_2022_01_15_16 | MISSING | READY | MISSING | false | EVENT_REFERENCE_ONLY |
| CTB_2024_02_18_20 | MISSING | READY | MISSING | false | EVENT_REFERENCE_ONLY |
| REC_2023_02_05_06 | MISSING | READY | MISSING | false | EVENT_REFERENCE_ONLY |
| PET_2022_03_20_21 | READY | READY | MISSING | false | PARTIAL_PATCH_READINESS_NEEDS_GEOMETRY_OR_CROSSWALK |
| PET_2024_03_21_28 | MISSING | READY | MISSING | false | EVENT_REFERENCE_ONLY |
