# REV-P v2ci - inventario TP2-ready de evidencia observacional candidata

Este marco e um inventario TP2-ready, nao um fechamento de TP2. Ele organiza
evidencia observacional candidata ja presente no repositorio e registra bloqueios
metodologicos de forma auditavel.

## Escopo

- Modo review-only.
- Sem download externo e sem internet.
- Sem inferencia de geometria ausente, CRS, data de evento ou hash.
- Sem criacao de labels, negativos formais ou treino.
- Sem afirmacao de intersecao espacial observada.

## Resultado do inventario

Total de candidatos inventariados: 38.
Total de pares candidato-patch: 38.

## Status TP2

- `TP2_BLOCKED`: 15
- `TP2_CANDIDATE_ONLY`: 23

## Travas metodologicas

- `formal_labels_available`: PASS (Nenhum label formal criado.)
- `formal_negatives_available`: PASS (Nenhum negativo formal criado.)
- `training_ready`: PASS (Treino permanece bloqueado.)
- `ground_truth_operational`: PASS (Sem ground truth operacional patch-level.)
- `supervised_model_allowed`: PASS (Classificador supervisionado nao permitido.)
- `prediction_claim_allowed`: PASS (Sem reivindicacao preditiva.)
- `intersection_claim_allowed`: PASS (Nenhuma intersecao observada foi afirmada neste marco.)
- `ground_truth_ready_status_absent`: PASS (Status de promocao operacional nao utilizado.)

## Interpretacao conservadora

Evidencia textual, evidencia visual, geometria candidata, geometria observada
validada e ground truth operacional sao categorias distintas. Um item so poderia
entrar como `TP2_READY_FOR_REPLAY` se houvesse geometria observada vetorial, CRS
conhecido, proveniencia e hash. Quando qualquer desses elementos falta, o item
permanece bloqueado ou candidato apenas.
