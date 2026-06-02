# Protocolo C v1os — Fixture Contamination Audit

## Objetivo

v1os audita datasets do Protocolo C em busca de linhas que correspondem a padrões de test fixtures/dados sintéticos. Não remove nem altera dados — apenas registra.

## Heurísticas

- `patch_id` matching `^REC_\d{5}$` (ex: REC_00001, REC_00002): padrão de ID sequencial curto usado em test fixtures, nunca em dados reais de Recife.
- `resolution_id` ou `selected_source_id` matching `^[A-Z]\d{1,4}$` (ex: R1, R2): IDs mínimos criados por testes, não por pipeline real.
- `candidate_id` matching `^C\d{1,3}$` sem `source_id`: padrão de fixture mínimo.

## Ação tomada

Scripts v1oo e v1op chamam `is_fixture_row()` e rejeitam estas linhas com `blocked_reason=FIXTURE_OR_SYNTHETIC_INPUT_BLOCKED`. Nenhum fixture chega a `PRODUCT_DATE_CONFIRMED` no pipeline real.

## Resultado

Total de linhas suspeitas: 0. Alta severidade: 0.
