# Validated candidate reference packet - REC_2022_05_24_30

## Identificacao
Candidate: `REC_2022_05_24_30` | Package: `ARP_v2az_0005` | Event-patch: `FACT_v2at_0005` | Janela: 2022-05-24 a 2022-06-02.

## Evidencia Charter
Charter 758 (CENAD, ativacao 2022-05-30); produto raster
2022-06-02 - LANDSLIDE_SCARS (landslide scars). Vetor/CRS nao disponiveis (limitacao tecnica).

## Evidencia APAC/ANA/INMET
APAC mensal (contexto), ANA Capibaribe cota (contexto hidrologico), INMET A301 PRECIP_FULL_GAP
(lacuna instrumental), proxies regionais auditados (A357=PRECIP_FULL_GAP; A328=PRECIP_PARTIAL; A320=PRECIP_AVAILABLE).

## Gates C0-C7 atualizados
| gate | previous | adjudicated | confidence |
| --- | --- | --- | --- |
| C0_PROVENANCE | PASS_FOR_REVIEW | PASS_PUBLIC_PROVENANCE_RECORDED | HIGH |
| C1_TEMPORALITY | TEMPORALITY_SUPPORTED_FOR_HUMAN_REVIEW | PASS_PUBLIC_TEMPORAL_EVIDENCE | MODERATE |
| C2_VALID_SERIES_OR_STATION | PARTIAL_FOR_HUMAN_REVIEW | PARTIAL_PASS_HYDROLOGICAL_CONTEXT_LOCAL_RAINFALL_GAP | MODERATE |
| C3_SPATIAL_ANCHOR | PASS | PASS_OFFICIAL_CARTOGRAPHIC_PRODUCT | HIGH |
| C4_CANDIDATE_GEOMETRY | MAP_PRESENT_PENDING_VECTOR_CRS | PASS_RASTER_CARTOGRAPHIC_EVIDENCE_FOR_REFERENCE | HIGH |
| C5_PROTOCOL_VALIDATION |  | AUTO_ADJUDICATED_BY_PROTOCOL | MODERATE |
| C6_CANDIDATE_REFERENCE | CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW | PROTOCOL_VALIDATED_CANDIDATE_REFERENCE | MODERATE |
| C7_OPERATIONAL_LABEL_OR_FINAL_SUPERVISED_TRUTH |  | NOT_CREATED_BLOCKED_FOR_TRAINING | HIGH |

## Status final protocolar
`PROTOCOL_VALIDATED_CANDIDATE_REFERENCE` | phenomenon: LANDSLIDE_SCARS_WITH_FLOOD_EVENT_CONTEXT | score 0.76.

## Allowed use
PROTOCOL_C_REFERENCE_REVIEW|ARTICLE_EVIDENCE|PUBLIC_DELIVERY_TABLE

## Forbidden use
SUPERVISED_LABEL|NEGATIVE_LABEL|TRAINING_TARGET|FLOOD_EXTENT_TRUTH

## Limitacoes
vector/CRS not available (vector overlay); local rainfall series gap (A301 empty, Cemaden pending).

## Proximos passos
Solicitar vetor/CRS (overlay) e serie local de chuva (Cemaden/APAC) - melhorias, nao blockers.
Reaplicar protocolo a Curitiba/Petropolis conforme learning matrix.
