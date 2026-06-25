# Relatorio de execucao - evolucao real de metadados DATA-06/07/08/09

## 1. Branch / worktree
- branch: dados/evolucao-real-metadados-data-06-09
- worktree: REV-P-evolucao-real-metadados-data-06-09
- modo: replay-only (strict=True)

## 2. Inputs reais encontrados
- input local real encontrado: False

## 3. Inputs faltantes
- DATA-06: BLOCKED_NO_REAL_TEMPORAL_WINDOW
- DATA-07: BLOCKED_NO_REAL_SENSOR_LINEAGE
- DATA-08: BLOCKED_NO_CONFIG

## 4. Filas de aquisicao geradas
- geradas: True
- outputs_public/mv2_data_06_09_real_acquisition_queue/

## 5. Targets prontos para metadata-only
- 0

## 6. Providers disponiveis / bloqueados
- configurados: GEE, CDSE_STAC, CDSE_ODATA, TRACEABILITY
- bloqueados: GEE, CDSE_STAC, CDSE_ODATA, TRACEABILITY

## 7. Metadata execution
- NO_CALL

## 8. Lineage consensus
- NO_CALL

## 9. MV2-16
- READY_FOR_MV2_16_DRY_RUN

## 10. Dia 10
- BLOCKED (bloqueado ate raster local-only + SCL QA existirem)

## 11. Chamadas / downloads / rasters / crops
- 0 / 0 / 0 / 0

## 12. Proxima acao humana objetiva
- Preencher inputs_local/data_06_temporal_windows/ e data_07_sensor_lineage/ com janela temporal e lineage Sentinel-2 reais (fonte rastreavel), criar configs/api_config.local.json local (network+metadata on, raster/canary off) e re-rodar com --allow-live-metadata.
