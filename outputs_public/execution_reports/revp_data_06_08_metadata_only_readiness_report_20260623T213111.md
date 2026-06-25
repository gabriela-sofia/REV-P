# Relatorio de execucao - desbloqueio metadata-only DATA-06/07/08

## 1. Branch / worktree
- branch: dados/desbloqueio-metadata-only-data-06-08
- worktree: REV-P-dados-metadata-only-data-06-08
- base: marco/pre-unificacao-gates-mv1 (HEAD 1c5744b)

## 2. Estado DATA-06
- BLOCKED_NO_FILLED_TEMPLATE
- targets no template de promocao: 10
- janela com fonte rastreavel: nenhuma preenchida (template humano pendente)

## 3. Estado DATA-07
- UNKNOWN_BLOCKED
- targets: 10
- Sentinel-2 elegivel: 0

## 4. Estado DATA-08
- BLOCKED_NO_CONFIG
- config local presente: False

## 5. Templates criados
- outputs_public/mv2_data_temporal_window_human_pack/ (DATA-06)
- outputs_public/mv2_data_sensor_lineage_human_pack/ (DATA-07)
- outputs_public/mv2_data_metadata_only_probe/mv2_data_08_*.md (DATA-08)
- configs/api_config.metadata_only.example.json

## 6. Proximos inputs humanos
- DATA-06: preencher janela temporal com fonte rastreavel e rodar a promocao.
- DATA-07: preencher sensor_family + sensor_source_ref e rodar a promocao.
- DATA-08: criar configs/api_config.local.json local (nao versionado) para sair de dry-run.

## 7. Chamadas / downloads / rasters / crops
- 0 / 0 / 0 / 0

## 8. Dia 10
- BLOCKED

## 9. Criterio para sair de dry-run
- DATA-06 PROMOTED_METADATA_READY + DATA-07 SENTINEL_2_ELIGIBLE +
  DATA-08 READY_METADATA_ONLY (config local habilitada metadata-only).

## 10. Proximo comando recomendado
- python scripts/mv2_data_06_08_metadata_only_readiness_orchestrator.py
