# revp_temporal_acquisition_gap_plan_mv1

## 1. Escopo
Plano publico e auditavel de lacunas de aquisicao temporal para a Vertente A.

## 2. Relacao com o manifesto temporal normalizado
- Entrada principal: `outputs_public/tables/revp_normalized_temporal_patch_coverage_mv1.csv`

## 3. Por que ainda nao se calcula drift
- Nenhum patch possui 3 ou mais datas limpas no manifesto normalizado.

## 4. Arquivos de entrada
- `outputs_public/tables/revp_normalized_temporal_asset_manifest_mv1.csv`
- `outputs_public/tables/revp_normalized_temporal_patch_coverage_mv1.csv`
- `outputs_public/metrics/revp_normalized_temporal_asset_manifest_mv1.json`
- `outputs_public/execution_reports/revp_normalized_temporal_asset_manifest_mv1.md`

## 5. Regra de lacuna temporal
- `missing_dates_to_3 = max(0, 3 - datas_limpas_atuais)`.

## 6. Patches com 0, 1, 2 e 3+ datas
- 0 datas: 328
- 1 data: 128
- 2 datas: 0
- 3+ datas: 0

## 7. Slots minimos de aquisicao necessarios
- Total de slots faltantes: 1240
- Para 1 patch elegivel: 2
- Para 20 patches elegiveis: 40
- Para 30 patches elegiveis: 60

## 8. Prioridades de aquisicao
- P1_NEEDS_2_DATES: 128
- P3_REPAIR_METADATA_FIRST: 120
- P4_MANUAL_REVIEW: 208

## 9. Requisitos tecnicos por nova aquisicao
- Data real nao especificada ainda.
- Mesma geometria de patch, mesmo contrato de pre-processamento e DINOv2 frozen em etapa posterior.
- Sem criacao de rotulos e sem assumir evento.

## 10. Estimativa minima para viabilizar piloto de drift
- `step_a_acquisition_plan_status`: `STEP_A_ACQUISITION_PLAN_REQUIRES_MULTI_DATE_BACKFILL`

## 11. Blockers restantes
- BLOCKED_BY_METADATA_REPAIR: 110
- BLOCKED_BY_MISSING_PATCH_LINKAGE: 2
- BLOCKED_BY_UNKNOWN_REGION: 8
- MANUAL_REVIEW_REQUIRED: 208
- NEEDS_2_MORE_DATES: 128

## 12. Proximos passos permitidos dentro da Vertente A
- Reparar metadados determinísticos restantes e planejar aquisicoes reais sem download automatico.

## 13. Limitacoes
- Esta etapa nao inventa datas, nao baixa assets, nao consulta API e nao calcula drift.

## 14. Guardrails preservados
- Branch auditada: `analysis/temporal-asset-readiness-mv1`
- sem download de assets
- sem consulta STAC API ou GEE
- sem gerar URLs de download
- sem calcular embedding drift
- sem gerar embeddings novos
- sem treino
- sem rotulo
- sem positivos ou negativos formais
- sem classe-zero
- sem GT operacional
- sem tabela multimodal de atributos
- sem alterar manifestos originais
