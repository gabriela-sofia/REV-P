# Region packet - Petropolis

- Reference status: `PROTOCOL_VALIDATED_CONTEXTUAL_REFERENCE`
- Phenomenon scope: LANDSLIDE_FLOOD_REGIONAL_TEMPORAL_CONTEXT
- Evidence basis: A610 REGIONAL_PROXY temporal context
- Evidence score: 0.55 | uncertainty: HIGH
- Allowed use: PROTOCOL_C_REFERENCE_REVIEW|ARTICLE_EVIDENCE|PUBLIC_DELIVERY_TABLE
- Forbidden use: SUPERVISED_LABEL|NEGATIVE_LABEL|TRAINING_TARGET|FLOOD_EXTENT_TRUTH

## Gates C0-C7
| gate | status | evidence |
| --- | --- | --- |
| C0_PROVENANCE | PASS_PUBLIC_PROVENANCE_RECORDED | INMET A610 public source |
| C1_TEMPORALITY | PASS_REGIONAL_TEMPORAL_EVIDENCE | A610 regional precipitation ready |
| C2_VALID_SERIES_OR_STATION | PARTIAL_PASS_REGIONAL_PROXY_NOT_LOCAL | A610 regional proxy, not local |
| C3_SPATIAL_ANCHOR | PENDING_NO_LOCAL_CARTOGRAPHIC_ANCHOR | No local spatial anchor |
| C4_CANDIDATE_GEOMETRY | PENDING_NO_GEOMETRY_EVIDENCE | No raster/vector product |
| C5_PROTOCOL_VALIDATION | AUTO_ADJUDICATED_BY_PROTOCOL | Regional temporal context |
| C6_CANDIDATE_REFERENCE | PROTOCOL_VALIDATED_CONTEXTUAL_REFERENCE | Regional temporal context |
| C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH | NOT_CREATED_BLOCKED_FOR_TRAINING | None |

## Guardrails
operational_label=0; negative=0; training=0; C7 NOT_CREATED_BLOCKED_FOR_TRAINING.
regional_proxy != local station; sentinel_preview != truth; dino != truth; patch_boundary != event geometry.
