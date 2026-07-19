# revp_normalized_temporal_asset_manifest_mv1

## 1. Escopo
Manifesto temporal normalizado derivado para a Vertente A, em modo shadow.

## 2. Relacao com as tres etapas anteriores
- Usa readiness audit, backfill queue e repair candidates como entradas.

## 3. Por que o manifesto e derivado/shadow
- O manifesto aplica candidatos somente na camada derivada e nao altera manifestos originais.

## 4. Arquivos de entrada
- `outputs_public/tables/revp_temporal_asset_readiness_mv1.csv`
- `outputs_public/metrics/revp_temporal_asset_readiness_mv1.json`
- `outputs_public/tables/revp_temporal_asset_backfill_queue_mv1.csv`
- `outputs_public/metrics/revp_temporal_asset_backfill_queue_mv1.json`
- `outputs_public/tables/revp_temporal_metadata_repair_candidates_mv1.csv`
- `outputs_public/metrics/revp_temporal_metadata_repair_candidates_mv1.json`

## 5. Regras de aplicacao de candidatos
- Apenas `is_applicable=true`, `is_deterministic=true` e confianca alta/media permitida.

## 6. Regras de rejeicao
- Rejeitados, ambiguos, contexto fraco e valores unknown/missing nao sao aplicados.

## 7. Definicao de asset temporal limpo
- Exige patch, regiao, tipo temporal, data, vinculo patch/asset e cloud cover numerico para Sentinel-2 optico.

## 8. Cobertura temporal por patch
- HAS_ASSETS_BUT_NO_CLEAN_DATE: 328
- PARTIAL_NEEDS_2_DATES: 128

## 9. Contagem de patches com 1, 2 e 3+ datas
- 1 data: 128
- 2 datas: 0
- 3+ datas: 0

## 10. Status da Vertente A
- `step_a_normalized_manifest_status`: `STEP_A_NORMALIZED_MANIFEST_REQUIRES_ADDITIONAL_ACQUISITIONS`

## 11. Blockers restantes
- NOT_TEMPORAL_ASSET: 248
- BLOCKED_MISSING_ASSET_TYPE: 238
- BLOCKED_AMBIGUOUS_REPAIR: 173
- BLOCKED_MISSING_ACQUISITION_DATE: 128
- BLOCKED_MISSING_REGION: 128
- BLOCKED_MISSING_PATCH_ID: 2

## 12. Proximos passos permitidos dentro da Vertente A
- Curar aquisicoes adicionais e reparar metadados determinísticos restantes.

## 13. Limitacoes
- Esta etapa nao calcula drift temporal e nao mede acuracia.

## 14. Guardrails preservados
- Branch auditada: `analysis/temporal-asset-readiness-mv1`
- manifesto derivado em modo shadow
- sem alterar manifestos originais
- sem download automatico
- sem STAC ou API
- sem calculo de drift
- sem treino
- sem rotulo
- sem positivos ou negativos formais
- sem classe-zero
- sem GT operacional
- sem tabela multimodal de atributos
