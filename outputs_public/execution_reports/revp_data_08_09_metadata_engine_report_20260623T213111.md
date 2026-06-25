# Relatorio de execucao - motor metadata-only Sentinel DATA-08/09

## 1. Branch / worktree
- branch: dados/motor-metadados-sentinel-data-08-09
- worktree: REV-P-motor-metadados-sentinel-data-08-09
- modo: replay-only (strict=True)

## 2. Targets elegiveis
- elegiveis (promovido + Sentinel-2): 0

## 3. Motivos de bloqueio
- DATA-06: BLOCKED_NO_FILLED_TEMPLATE
- DATA-07: UNKNOWN_BLOCKED
- DATA-08: BLOCKED_NO_CONFIG

## 4. Providers configurados
- GEE, CDSE_STAC, CDSE_ODATA, TRACEABILITY

## 5. Providers bloqueados
- GEE, CDSE_STAC, CDSE_ODATA, TRACEABILITY

## 6. Chamadas / downloads / rasters / crops
- live_calls: 0
- downloads: 0
- rasters: 0
- crops: 0

## 7. Consenso de lineage
- status: NO_CALL
- contagens: {"STRONG": 0, "MEDIUM_REVIEW": 0, "WEAK_BLOCKED": 0, "CONFLICT": 0, "NO_MATCH": 0, "NO_CALL": 0}
- lineage confirmado (STRONG): 0

## 8. MV2-16
- readiness: READY_FOR_MV2_16_DRY_RUN

## 9. Dia 10
- BLOCKED

## 10. Execucao de metadados
- NO_CALL

## 11. Proximos inputs humanos
- DATA-06: preencher janela temporal com fonte rastreavel e rodar a promocao.
- DATA-07: confirmar lineage Sentinel-2 elegivel com referencia de fonte.
- DATA-08: criar configs/api_config.local.json local (nao versionado) habilitando
  apenas allow_network + allow_metadata_calls; exportar variaveis de ambiente do
  provedor (sem versionar segredo).
- Somente com os tres acima o motor sai de replay-only para metadata-only real.
