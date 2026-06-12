# Region packet - Recife

- Reference status: `PROTOCOL_VALIDATED_CANDIDATE_REFERENCE`
- Phenomenon scope: LANDSLIDE_SCARS_WITH_FLOOD_EVENT_CONTEXT
- Evidence basis: Charter 758 raster landslide-scars product
- Evidence score: 0.76 | uncertainty: MODERATE
- Allowed use: PROTOCOL_C_REFERENCE_REVIEW|ARTICLE_EVIDENCE|PUBLIC_DELIVERY_TABLE
- Forbidden use: SUPERVISED_LABEL|NEGATIVE_LABEL|TRAINING_TARGET|FLOOD_EXTENT_TRUTH

## Gates C0-C7
| gate | status | evidence |
| --- | --- | --- |
| C0_PROVENANCE | PASS_PUBLIC_PROVENANCE_RECORDED | Charter 758 + APAC/ANA public sources |
| C1_TEMPORALITY | PASS_PUBLIC_TEMPORAL_EVIDENCE | APAC monthly + ANA stage |
| C2_VALID_SERIES_OR_STATION | PARTIAL_PASS_HYDROLOGICAL_CONTEXT_LOCAL_RAINFALL_GAP | ANA station; A301 gap |
| C3_SPATIAL_ANCHOR | PASS_OFFICIAL_CARTOGRAPHIC_PRODUCT | Charter Recife raster |
| C4_CANDIDATE_GEOMETRY | PASS_RASTER_CARTOGRAPHIC_EVIDENCE_FOR_REFERENCE | Charter raster |
| C5_PROTOCOL_VALIDATION | AUTO_ADJUDICATED_BY_PROTOCOL | Coherent evidence |
| C6_CANDIDATE_REFERENCE | PROTOCOL_VALIDATED_CANDIDATE_REFERENCE | Aggregated public evidence |
| C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH | NOT_CREATED_BLOCKED_FOR_TRAINING | None |

## Guardrails
operational_label=0; negative=0; training=0; C7 NOT_CREATED_BLOCKED_FOR_TRAINING.
regional_proxy != local station; sentinel_preview != truth; dino != truth; patch_boundary != event geometry.
