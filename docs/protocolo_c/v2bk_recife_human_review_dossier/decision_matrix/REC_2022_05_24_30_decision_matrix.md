# Recife candidate decision matrix - REC_2022_05_24_30

| axis | status | evidence | blocker | next action | promotion |
| --- | --- | --- | --- | --- | --- |
| TEMPORALITY | TEMPORALITY_SUPPORTED_FOR_HUMAN_REVIEW | ANA Capibaribe stage (dated) + APAC May accumulation | Parsed local rainfall series absent | HUMAN_REVIEW_TEMPORALITY | false |
| LOCAL_PRECIPITATION | PENDING_LOCAL_SERIES | None usable (A301 precip empty; proxies regional) | Cemaden/APAC local series missing | REQUEST_CEMADEN_APAC_LOCAL_SERIES | false |
| HYDROLOGICAL_CONTEXT | PRESENT_CONTEXT_ONLY | ANA river stage at Sao Lourenco da Mata (39187800, RMR) | River stage is not precipitation and not flood extent | KEEP_AS_CONTEXT_ONLY | false |
| CHARTER_SPATIAL_PRODUCT | PASS | Charter 758 Recife landslide-scars map | None for spatial anchor | REVIEW_PRODUCT | false |
| GEOMETRY_ACCESS | MAP_PRESENT_PENDING_VECTOR_CRS | Full-resolution raster map present | No machine-readable CRS, no vector | REQUEST_CHARTER_VECTOR_CRS_FROM_CENAD | false |
| HAZARD_TYPING | LANDSLIDE_SCARS_PENDING_CONFIRMATION | Product feature LANDSLIDE_SCARS | Official legend/class confirmation pending | CONFIRM_FEATURE_NOT_FLOOD_EXTENT | false |
| HUMAN_REVIEW | PENDING | Dossier assembled | Awaiting human decision | EXECUTE_HUMAN_REVIEW | false |
| FINAL_TRUTH | BLOCKED | None | Final ground truth prohibited | NONE_FINAL_GROUND_TRUTH_PROHIBITED | false |
