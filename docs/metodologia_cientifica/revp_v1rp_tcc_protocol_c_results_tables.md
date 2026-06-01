# v1rp — TCC Results Tables for Protocol C

## Objetivo

Gerar tabelas TCC-ready a partir dos resumos P0/P1/P2: estado C1-C4, fontes externas faltantes, fluxo de revisao, e papel review-only do DINO.

## Tabelas

c_level_status, external_sources, review_workflow, dino_role.

## Invariante DINO

DINO e review-only: dino_validates_event=false, dino_can_create_label=false. Serve apenas para priorizar revisao humana, nunca como prova de evento.

## Guardrails

Nenhum c_level e label operacional. Nenhuma tabela cria target ou ground truth.
