# revp_temporal_backfill_request_manifest_mv1

## 1. Escopo
Manifesto publico e auditavel de requisicoes futuras de backfill temporal para a Vertente A.

## 2. Entradas usadas
- `outputs_public/tables/revp_temporal_acquisition_gap_plan_mv1.csv`
- `outputs_public/tables/revp_temporal_acquisition_requirements_mv1.csv`
- `outputs_public/metrics/revp_temporal_acquisition_gap_plan_mv1.json`

## 3. Regra de conversao slot -> request
- Cada slot faltante do arquivo de requisitos vira uma linha de requisicao dry-run.
- A requisicao permanece bloqueada quando metadados, regiao, vinculo de patch ou familia de sensor ainda impedem busca futura.

## 4. Totais de requests
- Requests totais: 1240
- Requests acionaveis: 0
- Requests bloqueados antes da busca: 1240
- Patches marcados como acionaveis na entrada anterior: 128
- Requisitos marcados como acionaveis na entrada anterior: 256
- Recheck aplicado: `FAIL_CLOSED_REQUIRES_SUPPORTED_SENSOR_FAMILY_AND_COMPLETE_METADATA`

## 5. Contagem por status
- BLOCKED_BY_MANUAL_REVIEW: 624
- BLOCKED_BY_MISSING_PATCH_LINKAGE: 6
- BLOCKED_BY_UNKNOWN_REGION: 283
- BLOCKED_BY_UNSUPPORTED_SENSOR_SELECTION: 327

## 6. Contagem por regiao
- Curitiba: 437
- Example: 3
- Olinda/PE: 3
- Petropolis: 489
- Recife: 281
- multiple_or_none: 3
- unknown: 24

## 7. Contagem por familia de sensor
- requires_manual_sensor_selection: 1240

## 8. Batches minimos de dry-run
- `BATCH_MIN_1_PATCH_TEMPORAL_DRIFT_PILOT`: 0 patches, 0 novas aquisicoes planejadas em dry-run
- `BATCH_MIN_20_PATCHES_TEMPORAL_DRIFT_PILOT`: 0 patches, 0 novas aquisicoes planejadas em dry-run
- `BATCH_MIN_30_PATCHES_TEMPORAL_DRIFT_READY`: 0 patches, 0 novas aquisicoes planejadas em dry-run

## 9. Status da etapa
- `step_a_backfill_request_status`: `STEP_A_BACKFILL_REQUESTS_REQUIRE_METADATA_REPAIR_FIRST`

## 10. Limitacoes
- Nenhuma data futura foi inventada.
- Nenhuma busca STAC, API, GEE, download ou execucao externa foi realizada.
- Nenhum embedding, drift, treino, rotulo ou evidencia de ausencia foi criado.

## 11. Proximo passo permitido
- Apenas revisao tecnica do manifesto e, futuramente, execucao controlada por operador externo com contrato de metadados completo.

## 12. Guardrails preservados
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
