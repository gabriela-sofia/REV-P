# Protocol C - Recife validated candidate reference report

## 1. Resumo executivo
Recife (`REC_2022_05_24_30`) foi adjudicado automaticamente pelo Protocolo C e promovido a
`PROTOCOL_VALIDATED_CANDIDATE_REFERENCE` com base em evidencia publica real ja coletada e auditada.
Score de evidencia: 0.76 | incerteza: MODERATE.

## 2. Por que licenca/confirmacao externa foi reclassificada
Charter 758, APAC e ANA sao fontes publicas oficiais para uso academico/metodologico.
Licenca, redistribuicao e confirmacao externa passam a ser metadados de proveniencia, nao
blockers. Ver v2bl_non_blocking_limitations_reclassification.csv.

## 3. Por que revisao humana manual separada nao e mais obrigatoria
Com dado disponivel, rastreado, coerente em tempo/local/fenomeno e classificado, o protocolo
valida automaticamente (AUTO_ADJUDICATED_BY_PROTOCOL). Nao se cria label nem treino.

## 4. Evidencias reais usadas
- Charter 758 raster (2022-06-02, LANDSLIDE_SCARS).
- APAC mensal maio/2022 (contexto temporal).
- ANA Capibaribe / Sao Lourenco da Mata (contexto hidrologico).
- INMET A301 PRECIP_FULL_GAP (lacuna instrumental documentada); proxies regionais auditados.

## 5. Gates antigos vs gates novos
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

## 6. Promocao de Recife
`CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW` -> `PROTOCOL_VALIDATED_CANDIDATE_REFERENCE` (promotion_type
PROTOCOL_LEVEL_REFERENCE; promotion_allowed=true).

## 7. O que isso permite afirmar
- Existe produto cartografico oficial de deslizamento em Recife no evento (referencia candidata).
- Ha contexto temporal e hidrologico publico datado.
- Recife e referencia candidata validada pelo protocolo para revisao/artigo/entrega publica.

## 8. O que isso NAO permite afirmar
- Nao e flood extent; nao e geometria vetorial; nao e precipitacao local instrumental.
- Nao e label supervisionado, negativo nem alvo de treino.

## 9. Por que ainda nao e label operacional
Falta vetor/CRS para overlay e serie local de chuva; C7 permanece NOT_CREATED_BLOCKED_FOR_TRAINING.

## 10. Como reaplicar para Curitiba e Petropolis
Ver v2bl_reapplication_learning_matrix.csv: asset date (Curitiba), proxy-vs-contexto
(Petropolis), instrument gap, raster-as-reference e remocao de blockers de licenca.

## 11. Guardrails finais
operational_label=0; negative=0; training=0; raster!=vector; landslide scars!=flood extent;
ANA cota!=precipitacao; APAC PDF!=serie horaria; A301 vazia=instrument gap; C7 BLOCKED.
