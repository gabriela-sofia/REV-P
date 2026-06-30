# SUSC-17A - Reference Evidence Protocol (review-only)

Status: review-only. `trainable=false`; `ground_truth=false`; `score_v6_changed=false`; `score_v7_created=false`; `eligible_for_score_v7_future_count=0`; `readiness_status=REFERENCE_PROTOCOL_READY_REVIEW_ONLY`.

O SUSC-17A formaliza o protocolo de evidencia observacional review-only. Ele separa event_record, source_footprint, derived_patch_link e score_evaluation_reference, trata footprint como evidencia candidata e nao como ground truth, nao transforma ausencia documental em negativo, nao altera o score v6 oficial, nao cria score v7, nao cria treino, modelo, label supervisionado ou ground truth.

## 1. O que o 17A resolve

O 17A transforma o stub criado no SUSC-16D em um protocolo formal e auditavel de evidencia observacional. Ele cataloga, classifica e qualifica eventos, footprints, fontes, incerteza e elegibilidade de uso, definindo o contrato que um futuro mini-benchmark (17B) poderia consumir, sem ainda criar benchmark, score v7, treino, label ou ground truth.

## 2. Por que separar event_record / source_footprint / derived_patch_link / score_evaluation_reference

Esses quatro objetos tem regimes de confianca diferentes. O `event_record` e a identidade do evento; o `source_footprint` e a geometria oficial/tecnica; o `derived_patch_link` e a ligacao footprint->patch; o `score_evaluation_reference` e o link efetivamente usavel para avaliar o score v6. Mistura-los esconderia que uma geometria existe mas nao tem link de patch, ou que um link existe mas sem data do evento. O registry mantem os quatro niveis explicitos no campo `reference_level`.

## 3. Por que footprint nao e ground truth

Os footprints sao geometrias oficiais candidatas (parse local, `moderate_candidate`), sem data explicita, sem validacao QA forte e sem confirmacao independente patch-a-patch. Eles indicam onde houve inundacao reportada, mas nao provam o rotulo de cada patch. Por isso `ground_truth=false` e `not_ground_truth_reason` em todas as linhas.

## 4. Por que ausencia documental nao e negativo

A ausencia de footprint ou de documento sobre um patch significa apenas que nao ha evidencia registrada, nao que o evento nao ocorreu. Tratar ausencia como negativo verdadeiro criaria rotulos falsos. O protocolo marca `no_negative_control_reason` em todas as linhas e nao cria nenhum controle negativo.

## 5. Classes em avaliacao forte, moderada ou apenas contexto

- Forte (valida patch-level, elegivel a calibracao): `official_observed_event_polygon`, `official_observed_event_point`, `technical_remote_sensing_flood_footprint`.
- Moderada (nunca forte automatica, exige `uncertainty_m` e QA): `official_address_resolved`.
- Apenas contexto (nao valida patch-level): `street_neighborhood_context_only`, `risk_area_not_event`, `alert_only`, `administrative_disaster_record`, `documentary_context`.
- Excluidas de avaliacao/calibracao: `rejected_non_event`, `insufficient_reference`.

## 6. Campos preenchidos como ausencia controlada e por que

- `event_date`, `pre_event_window`, `post_event_window`, `temporal_resolution`: `not_available` porque os footprints elegiveis do SUSC-16A nao tem data explicita extraida.
- `uncertainty_m`: `not_available` porque nenhuma incerteza metrica foi quantificada nas fontes locais.
- `patch_link_id`, `patch_id`, `link_quality`: `not_available` para footprints sem link de patch resolvido.
- `geometry_type`/`evidence_class` insuficiente para 6 footprints cujo geometry parse falhou (`insufficient_reference`).

Nenhum desses campos foi inventado.

## 7. O que ainda bloqueia o 17B

["no_event_dates_for_temporal_pre_post_windows", "too_few_distinct_strong_footprint_sources", "strong_references_concentrated_in_one_region"]

Em numeros: 65 referencias fortes patch-linked, mas vindas de apenas 2 footprints distintos, em regiao unica, com 0 datadas. Sem janelas temporais e sem diversidade, um mini-benchmark de evento seria fragil.

## 8. Numeros do protocolo

- Registros de referencia: 75.
- Eventos unicos: 12.
- Footprints unicos: 12.
- Patch links: 65.
- Por classe: {"insufficient_reference": 6, "official_observed_event_point": 3, "official_observed_event_polygon": 66}.
- Por tier: {"TIER_B_MODERATE_GEOMETRY_PATCH_LINKED": 65, "TIER_C_COARSE_GEOMETRY_UNLINKED": 4, "TIER_E_INSUFFICIENT": 6}.
- Elegiveis a avaliacao: 69.
- Elegiveis a avaliacao forte: 65.
- Elegiveis a calibracao: 65.
- Elegiveis a score v7 futuro: 0.

## 9. Proximo marco recomendado

SUSC-17C Sentinel-1/SAR Canary (review-only) para produzir footprints tecnicos datados; SUSC-17B permanece bloqueado ate haver janelas temporais e diversidade de footprints fortes
