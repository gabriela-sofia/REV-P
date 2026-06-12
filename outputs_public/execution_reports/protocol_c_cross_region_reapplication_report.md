# Protocol C - cross-region reapplication report

## 1. Resumo executivo
A politica refinada do Protocolo C (calibrada em Recife) foi reaplicada a Curitiba e
Petropolis com adjudicacao automatica fail-closed. Recife permanece candidate reference;
Curitiba evolui para referencia temporal; Petropolis para referencia contextual regional.

## 2. Politica refinada do Protocolo C
Licenca/confirmacao externa nao sao blockers; revisao humana manual separada e substituida
por adjudicacao automatica; raster sustenta candidate reference; vetor/CRS e limitacao
tecnica; Sentinel preview e DINO sao apoio de revisao (nao truth); proxy regional nao vira
estacao local; ausencia instrumental e lacuna, nao negativo.

## 3. Recife como calibrador
`PROTOCOL_VALIDATED_CANDIDATE_REFERENCE` (score 0.76): produto cartografico
oficial raster + contexto temporal/hidrologico publico.

## 4. Reavaliacao de Curitiba
`PROTOCOL_VALIDATED_TEMPORAL_REFERENCE` (score 0.7): A807 LOCAL com
precipitacao forte (3/3 seeds) e preview/patch para revisao visual; sem acquisition_date
Sentinel, nao vira candidate reference espacial completa.

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

## 5. Reavaliacao de Petropolis
`PROTOCOL_VALIDATED_CONTEXTUAL_REFERENCE` (score 0.55): A610
REGIONAL_PROXY com evidencia temporal regional; sem ancora espacial local, permanece contexto.

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

## 6. Matriz comparativa das tres regioes
| region | reference status | score | uncertainty |
| --- | --- | --- | --- |
| Recife | PROTOCOL_VALIDATED_CANDIDATE_REFERENCE | 0.76 | MODERATE |
| Curitiba | PROTOCOL_VALIDATED_TEMPORAL_REFERENCE | 0.7 | MODERATE |
| Petropolis | PROTOCOL_VALIDATED_CONTEXTUAL_REFERENCE | 0.55 | HIGH |

## 7. O que cada regiao permite afirmar
- Recife: existe produto cartografico oficial de deslizamento (referencia candidata).
- Curitiba: existe forte evidencia temporal local datada (referencia temporal).
- Petropolis: existe contexto temporal regional (referencia contextual).

## 8. O que cada regiao NAO permite afirmar
- Nenhuma e label supervisionado, negativo ou alvo de treino.
- Curitiba nao tem ancora cartografica nem data Sentinel; Petropolis usa proxy regional.
- Preview Sentinel nao e truth; DINO nao e truth; patch boundary nao e geometria de evento.

## 9. Por que ainda nao existe label operacional
Falta ancora/geometria vetorial validada e/ou serie local consolidada; C7 permanece
NOT_CREATED_BLOCKED_FOR_TRAINING em todas as regioes.

## 10. Como isso fortalece o TCC
Tres regioes com referencias protocolares graduadas (candidate/temporal/contextual),
auditaveis, com proveniencia publica e limitacoes explicitas, sem overclaim.

## 11. Proximos passos
- Recife: solicitar vetor/CRS (overlay) e serie local de chuva.
- Curitiba: recuperar acquisition_date Sentinel para subir de tier.
- Petropolis: buscar ancora/cartografia local especifica.
