# v1tp — Automated Reviewer A/B Adjudication

## Objetivo

Dois perfis independentes de revisão automatizada por caso. Reviewer A conservador (exige fonte observacional independente, penaliza baixa precisão, bloqueia overclaim). Reviewer B integrador (valoriza consistência cruzada e contexto DINO/patch, ainda bloqueia overclaim).

## Resultado
Decisões: 18 (A=9, B=9). Validadas review-only: 9.

## Limitação

Nenhum revisor cria label, target, ground truth operacional, C3 automático, C4 ou negativo formal. DINO/hidromet são contexto.
