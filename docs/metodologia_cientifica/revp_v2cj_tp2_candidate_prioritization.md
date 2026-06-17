# REV-P v2cj - priorizacao conservadora de candidatos TP2

Este marco transforma o inventario `v2ci` em uma lista ordenada para revisao
humana. A coluna `review_priority_score` mede prioridade de revisao, nao verdade,
nao label e nao desempenho preditivo.

Entradas:

- `outputs_public/tables/revp_tp2_candidate_inventory_v2ci.csv`
- `outputs_public/tables/revp_tp2_patch_pair_candidates_v2ci.csv`

Saidas:

- `outputs_public/tables/revp_tp2_candidate_priority_v2cj.csv`
- `outputs_public/execution_reports/revp_tp2_candidate_priority_report_v2cj.md`

Mesmo `HIGH_REVIEW_PRIORITY` nao fecha TP2. A etapa apenas indica candidatos que
merecem pacote de digitalizacao ou revisao manual antes de qualquer replay.

