# Aquisicao real DATA-06/07 - relatorio consolidado

## Branch / worktree
- branch: dados/aquisicao-real-data-06-07
- worktree: REV-P-aquisicao-real-data-06-07
- base: dados/evolucao-real-metadados-data-06-09 (HEAD b0c98a4)

## Targets investigados
- 10 targets: 5 Recife (REC_00019/00183/00204/00205/00227) e 5 Petropolis (PET_00016/00104/00119/00140/00240).

## Fontes consultadas
- Internas auditaveis (committed):
  - datasets/v2at_event_patch_package_registry.csv (vinculo patch->evento + janela + sentinel_sensor_family)
  - datasets/event_sentinel_temporal_window_registry.csv (janelas por evento)
  - manifests/dino_inputs/revp_v1fu_dino_sentinel_input_manifest/dino_sentinel_input_manifest_v1fu.csv (source_asset_type SENTINEL_TIF_ASSET)
  - datasets/v2bd_patch_asset_lineage_registry.csv (asset_sensor, asset_type)
- Oficiais publicas (web, leves: URL/titulo/data):
  - CEMADEN/MCTI - inundacoes e deslizamentos em Recife (PE), maio de 2022
  - Copernicus EMS GloFAS - Floods and landslides in Rio de Janeiro State, Brazil, fev-mar 2022 (S2 pos-evento 17/02/2022)

## Janelas temporais candidatas (DATA-06)
- Recife (REC_2022_05_24_30): 2022-05-24 a 2022-05-30 - STRONG (interno auditavel + CEMADEN, corroborado por INMET alerta vermelho 28-29/05/2022).
- Petropolis (PET_2022_02_15): 2022-02-15 - STRONG (interno auditavel + Copernicus EMS, S2 pos-evento 17/02/2022).
- 10/10 STRONG -> input local DATA-06 criado (nao versionado).

## Sensor lineage candidato (DATA-07)
- Recife (5): familia Sentinel-2 documentada por v2at (SENTINEL2_MSI, obs 2022-05-24) + v1fu (SENTINEL_TIF_ASSET).
  spectral_eligible=true, mas product_id/scene_id S2 explicito AUSENTE -> MEDIUM / NEEDS_REVIEW.
- Petropolis (5): v1fu marca SENTINEL_TIF_ASSET, mas v2at registra sensor UNKNOWN.
  Sem declaracao Sentinel-2 explicita -> UNKNOWN_BLOCKED (nao inferido por nome de asset).
- 0 lineage completo forte -> input local DATA-07 NAO criado.

## Inputs locais criados
- inputs_local/data_06_temporal_windows/data_06_temporal_windows_real_candidate.csv (git-ignored, 10 linhas).
- DATA-07: nenhum (evidencia incompleta, aguarda product_id S2 explicito).

## Por que cada target foi promovido ou bloqueado
- DATA-06 promovido (10/10): janela de evento real com vinculo patch->evento interno auditavel + fonte oficial publica.
- DATA-07 bloqueado: familia S2 documentada apenas para Recife; product_id/scene_id S2 explicito ausente em todos; Petropolis sem declaracao S2.

## Status consolidado
- DATA-06: PROMOTED_METADATA_READY
- DATA-07: BLOCKED_NO_REAL_SENSOR_LINEAGE
- DATA-08: BLOCKED_NO_CONFIG
- metadata_execution: NO_CALL
- lineage_consensus: NO_CALL
- MV2-16: READY_FOR_MV2_16_DRY_RUN
- Dia 10: BLOCKED

## Chamadas / downloads / rasters / crops
- 0 / 0 / 0 / 0

## Proxima acao humana objetiva
1. DATA-07: confirmar o product_id/scene_id Sentinel-2 explicito de cada asset (historico/export GEE COPERNICUS/S2_SR_HARMONIZED ou manifest de export original) e registrar como sensor_source_ref; para Petropolis, confirmar se os SENTINEL_TIF_ASSET sao Sentinel-2 ou Sentinel-1.
2. Com DATA-07 elegivel, criar inputs_local/data_07_sensor_lineage/ e re-rodar o orquestrador.
3. DATA-08: criar configs/api_config.local.json local (network+metadata on, raster/canary off) para habilitar metadata-only.
4. So entao, com autorizacao explicita, rodar o orquestrador com --allow-live-metadata.
