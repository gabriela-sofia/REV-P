# Region packet - Curitiba

- Reference status: `PROTOCOL_VALIDATED_TEMPORAL_REFERENCE`
- Phenomenon scope: URBAN_FLOOD_EVENT_TEMPORAL_CONTEXT
- Evidence basis: A807 LOCAL strong precipitation + Sentinel preview/patch (visual review)
- Evidence score: 0.7 | uncertainty: MODERATE
- Allowed use: PROTOCOL_C_REFERENCE_REVIEW|ARTICLE_EVIDENCE|PUBLIC_DELIVERY_TABLE
- Forbidden use: SUPERVISED_LABEL|NEGATIVE_LABEL|TRAINING_TARGET|FLOOD_EXTENT_TRUTH

## Gates C0-C7
| gate | status | evidence |
| --- | --- | --- |
| C0_PROVENANCE | PASS_PUBLIC_PROVENANCE_RECORDED | INMET A807 + Sentinel public sources |
| C1_TEMPORALITY | PASS_PUBLIC_TEMPORAL_EVIDENCE | A807 LOCAL strong precipitation |
| C2_VALID_SERIES_OR_STATION | PASS_LOCAL_STATION_SERIES | A807 LOCAL station, 0 missing |
| C3_SPATIAL_ANCHOR | PENDING_NO_OFFICIAL_CARTOGRAPHIC_PRODUCT | No Charter-like product |
| C4_CANDIDATE_GEOMETRY | VISUAL_REVIEW_CONTEXT_NOT_GEOMETRY | Sentinel preview + patch link; no acquisition date |
| C5_PROTOCOL_VALIDATION | AUTO_ADJUDICATED_BY_PROTOCOL | Strong temporal + visual review |
| C6_CANDIDATE_REFERENCE | PROTOCOL_VALIDATED_TEMPORAL_REFERENCE | A807 local temporal evidence |
| C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH | NOT_CREATED_BLOCKED_FOR_TRAINING | None |

## Guardrails
operational_label=0; negative=0; training=0; C7 NOT_CREATED_BLOCKED_FOR_TRAINING.
regional_proxy != local station; sentinel_preview != truth; dino != truth; patch_boundary != event geometry.
