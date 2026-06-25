# Fila de aquisicao real DATA-06/07/08/09

## Estado
- targets DATA-06 (janela temporal): 10
- targets DATA-07 (sensor lineage): 10
- targets DATA-06 com bbox local disponivel: 0

## DATA-06 - Janela temporal
Fontes aceitas: CEMADEN, Defesa Civil, Copernicus EMS/CEMS, SGB/CPRM, ANA, boletim municipal oficial, artigo/publicacao cientifica, registro interno auditavel com referencia.
Acao: preencher `inputs_local/data_06_temporal_windows/` com janela inicio/fim e
fonte oficial rastreavel. Nunca inventar data; nunca resolver por bbox.

## DATA-07 - Sensor lineage
Fontes aceitas: historico/export GEE, manifest de asset original, script de export, metadata oficial Sentinel, registro interno auditavel ligando asset_ref a source_asset_ref.
Acao: preencher `inputs_local/data_07_sensor_lineage/` ligando `asset_ref` a
`source_asset_ref` Sentinel-2. Nunca inferir sensor por nome/visual; nunca marcar
Sentinel-2 sem fonte.

## DATA-08 - Config metadata-only
Ver `mv2_data_08_metadata_config_checklist.md`. Criar `configs/api_config.local.json`
local (nunca versionar) habilitando apenas network + metadata calls.

## Garantias
- nenhuma resolucao automatica por bbox ou filename;
- DINO/PNG/NPZ nao sao promovidos a raster;
- chamadas/downloads/rasters/crops: 0/0/0/0.
