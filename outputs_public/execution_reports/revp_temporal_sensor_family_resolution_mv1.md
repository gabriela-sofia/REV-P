# revp_temporal_sensor_family_resolution_mv1

## 1. Escopo
Camada publica e auditavel de resolucao deterministica de familia de sensor para slots de backfill temporal.

## 2. Relacao com o fail-closed do request manifest
- Status anterior: `STEP_A_BACKFILL_REQUESTS_REQUIRE_METADATA_REPAIR_FIRST`.
- Esta etapa explica se o bloqueio por selecao manual de sensor pode ser reparado sem busca externa.

## 3. Por que DINOv2 nao e sensor de aquisicao
- DINOv2 aparece aqui como embedding derivado. Ele nao identifica, por si so, a familia fisica de aquisicao do asset fonte.
- Por isso, embedding derivado sem sensor source preservado permanece bloqueado para backfill temporal.

## 4. Arquivos de entrada
- `outputs_public/tables/revp_temporal_acquisition_gap_plan_mv1.csv`
- `outputs_public/tables/revp_temporal_acquisition_requirements_mv1.csv`
- `outputs_public/tables/revp_temporal_backfill_request_manifest_mv1.csv`
- `outputs_public/tables/revp_normalized_temporal_asset_manifest_mv1.csv`
- `outputs_public/tables/revp_normalized_temporal_patch_coverage_mv1.csv`
- `outputs_public/metrics/revp_temporal_backfill_request_manifest_mv1.json`
- `outputs_public/metrics/revp_normalized_temporal_asset_manifest_mv1.json`

## 5. Regras de resolucao
- Asset limpo `sentinel_2_optical` resolve para `sentinel_2_optical_preferred` quando nao ha conflito.
- Asset limpo `sentinel_1_sar` vira `sentinel_1_sar_optional_support`, sem substituir Sentinel-2.
- Evidencia DINOv2 derivada isolada nao resolve sensor de aquisicao.

## 6. Regras de bloqueio
- Contexto ausente de patch, regiao ou vinculo preservavel bloqueia.
- Asset unknown bloqueia.
- Familias de aquisicao conflitantes bloqueiam.
- Metadata blocker anterior bloqueia acionabilidade posterior.

## 7. Slots resolvidos
- Sentinel-2: 0
- Sentinel-1 suporte: 0

## 8. Slots bloqueados
- DINOv2 derivado sem sensor source: 256
- Unknown: 711
- Ambiguidade: 0

## 9. Impacto potencial sobre requests acionaveis
- Slots potencialmente acionaveis apos resolucao: 0
- Patches potencialmente acionaveis apos resolucao: 0
- `step_a_sensor_resolution_status`: `STEP_A_SENSOR_RESOLUTION_BLOCKED`

## 10. Batches que poderiam ser reconstruidos
- 1 patch: False, aquisicoes: 0
- 20 patches: False, aquisicoes: 0
- 30 patches: False, aquisicoes: 0

## 11. Proximos passos permitidos dentro da Vertente A
- Revisar manualmente a origem dos embeddings derivados e recuperar sensor source apenas com evidencia explicita.
- Reemitir manifesto dry-run acionavel somente se uma etapa posterior produzir requisitos resolvidos para Sentinel-2 sem blockers.

## 12. Limitacoes
- Nenhuma data nova foi criada.
- Nenhum backend externo foi consultado.
- Nenhum manifesto anterior foi alterado.

## 13. Guardrails preservados
- Branch auditada: `analysis/temporal-asset-readiness-mv1`
- sem download de assets
- sem consulta STAC API ou GEE
- sem gerar URLs de download
- sem calcular embedding drift
- sem gerar embeddings novos
- sem treino
- sem rotulo
- sem criar evidencia de ausencia supervisionada
- sem classe-zero
- sem GT operacional
- sem tabela multimodal de atributos
- sem alterar manifestos originais
