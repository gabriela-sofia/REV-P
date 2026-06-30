# SUSC-16D - desenho controlado de calibracao candidata review-only

Status: review-only. `trainable=false`; `ground_truth=false`; `official_score_changed=false`; `score_v7_created=false`; `eligible_for_score_v7_future=0`; `readiness_status=BLOCKED_REVIEW_ONLY`.

O SUSC-16D desenha calibracao candidata review-only a partir do diagnostico SUSC-16C. Ele separa susceptibility_signal de evidence_confidence, trata o documentary_component como confianca documental e nao como prova de baixa suscetibilidade fisica, e nao altera o score v6 oficial, nao cria score v7, nao cria treino, modelo, ground truth ou negativo verdadeiro.

## 1. Entrada herdada do SUSC-16C

O SUSC-16D consome o diagnostico do SUSC-16C sem reabrir o escopo. Numeros reproduzidos do 16C:

- Links footprint-patch: 65 (esperado 65).
- Patches observacionais unicos: 62 (esperado 62).
- Casos low/medium com footprint: 57 (esperado 57).
- `documentary_component` foi o componente mais baixo em 65/65 casos.
- `urban_flash_flood_underrepresented`: 60.
- `rainfall_trigger_underweighted`: 5.
- Melhor sensibilidade review-only: `increase_spectral_water_weight` com `event_hit_rate_top_30_simulated`=0.366667.

## 2. Separacao conceitual obrigatoria

O 16D separa dois conceitos que o score v6 mantinha misturados:

- `susceptibility_signal`: fisico, hidrologico, urbano, espectral e de chuva (16 features).
- `evidence_confidence`: documental, fonte, footprint e incerteza (5 sinais).

O `documentary_component` e tratado como confianca documental. A ausencia de documentacao NAO e prova de baixa suscetibilidade fisica e NAO e negativo verdadeiro.

## 3. Matriz de candidatos por link/patch

A matriz `susc_16d_calibration_candidate_matrix.csv` tem 65 linhas (uma por link footprint-patch elegivel do 16C). Cada linha registra `score_v6`, `score_v6_class`, `documentary_component`, `urban_prop`, features divergentes, `failure_mode` e a familia de ajuste candidata. Todas as linhas: `has_observed_footprint=true`, `ground_truth=false`, `trainable=false`, `score_v6_unchanged=true`, `score_v7_created=false`, `eligible_for_score_v7_future=false`, com `not_ground_truth_reason` preenchido.

Distribuicao por familia de ajuste primaria: {"increase_rainfall_trigger_weight": 5, "increase_urban_flash_flood_weight": 60}.

## 4. Hipoteses de calibracao candidatas

As hipoteses estao em `susc_16d_calibration_hypotheses.csv`, com justificativa, features, modos de falha cobertos, risco metodologico e status review-only:

1. `decouple_documentary_component_from_susceptibility` - separar confianca documental do sinal fisico.
2. `increase_urban_flash_flood_weight` - exposicao urbana subrepresentada (60/65).
3. `increase_rainfall_trigger_weight` - gatilho de chuva/runoff subponderado (5/65).
4. `increase_spectral_water_weight` - melhor sensibilidade review-only do 16C.
5. `guard_low_documentary_high_physical_signal` - guarda contra colapso do score por ausencia documental.

Nenhuma hipotese e aplicada ao score oficial. `eligible_for_score_v7_future=false` para todas.

## 5. Mapeamento modo de falha -> hipotese de peso

`susc_16d_failure_mode_to_weight_hypothesis.csv` liga cada modo de falha do 16C a uma familia de ajuste candidata e marca a separacao de conceito (susceptibility_signal vs evidence_confidence).

## 6. Politica de direcao de features

`susc_16d_feature_direction_policy.json` classifica cada feature como `susceptibility_signal` ou `evidence_confidence`, registra a direcao esperada e a estabilidade observada no 16C, e marca `can_change_official_score_v6=false` para todas.

## 7. Fundacao minima para SUSC-17A (Reference Evidence Protocol)

Foi criada a base minima do Reference Evidence Protocol como stub fail-closed:

- Schema: `schemas/suscetibilidade/susc_17a_reference_evidence_protocol_schema_v1.json`.
- Registro stub headers-only (0 registros): `susc_17a_reference_evidence_protocol_registry_stub.csv`.
- Politica de classes: `susc_17a_reference_evidence_protocol_class_policy.json`.

Somente `official_observed_event_polygon`, `official_observed_event_point` e `technical_remote_sensing_flood_footprint` podem sustentar avaliacao forte. Nenhuma classe vira ground truth ou negativo verdadeiro. Nenhum registro foi inventado.

## 8. O que nao pode ser afirmado

Nao se pode afirmar ground truth, negativo verdadeiro, prontidao para treino, score v7 pronto ou causalidade. Footprints permanecem evidencia observacional candidata; controles permanecem 'no documented footprint', nao negativos verdadeiros.

## 9. Proximo marco recomendado

`SUSC-17A Reference Evidence Protocol`: popular o protocolo de referencia com evidencia observacional auditada (sem ground truth automatico), ou `SUSC-17B Event-Based Mini-Benchmark` se houver referencias fortes suficientes para um mini-benchmark review-only.
