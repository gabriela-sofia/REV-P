# SUSC-17C22 - Aceite formal automatizado do QA sensorial e feature matrix candidata

## Objetivo
Operacionalizar a 'revisao humana' como gates objetivos aplicados pelo agente: consolidar as features sensoriais reais (pre-evento 17C20, deltas pre/pos 17C21), aplicar politica de aceite, montar matriz candidata review-only por patch e bloquear qualquer promocao cientifica indevida.

## Aceite de features pre-evento
- Features pre-evento avaliadas: 40.
- Aceitas como accepted_review_only_sensor_feature: 40.
- Criterios objetivos: feature real, hash conferido, replay byte-identico, cena pre-evento, fallback declarado.

## Aceite de deltas observacionais
- Deltas avaliados: 40.
- Aceitos como accepted_observational_change_review_only: 40.
- Delta nunca e feature pre-evento, label, ground truth ou score.

## Matrizes candidatas por patch
- Matriz sensorial candidata (pre-evento): 5 linhas.
- Matriz de deltas observacionais: 5 linhas.
- Deltas ficam em bloco separado; nunca misturados com features pre-evento.

## Fallback
- Fallback usado: 5 patches; aceito para review-only: 5; aceito para ciencia agora: 0.
- Fallback mantem source_role=materialization_fallback, can_be_used_for_ground_reference=false e requires_external_policy_review=true.

## Guardrails
- Features promovidas a treino: 0; deltas usados como label: 0; pos-evento como pre-evento: 0.
- Ground Reference: 0; ground truth: 0; label: 0; score v7: inexistente; score v6 intacto; 17B bloqueado.

## minimum_success_achieved: True

## Proximo marco recomendado
SUSC-17C23 Revisao de politica da fonte fallback e requisitos formais para promover feature sensorial review-only a uso cientifico
