# v1pw — DINO Review-Only Queue Expansion

## Objetivo

Gerar fila DINO expandida a partir de v1pv, priorizando Protocol C patches e referências Sentinel TIF. Não executa embedding. Não cria label.

## Scene date

A elegibilidade para fila DINO NÃO requer scene_date confirmada. A extração de embedding é uma representação visual review-only independente de adjudicação temporal.

## Guardrails

can_create_label, can_train_model e target_created sempre false.

## Resultado

Itens na fila: 100. Protocol C priority: 0. Sentinel TIF priority: 100.
