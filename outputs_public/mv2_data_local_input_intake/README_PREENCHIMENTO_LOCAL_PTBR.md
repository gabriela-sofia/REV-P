# Preenchimento local DATA-06/07/08

Os CSVs de exemplo desta pasta sao ficticios e publicos. Nao copie dados reais
para `outputs_public`.

## Onde colocar inputs reais
- DATA-06: `inputs_local/data_06_temporal_windows/`
- DATA-07: `inputs_local/data_07_sensor_lineage/`
- DATA-08: `inputs_local/data_08_metadata_config/` ou `configs/api_config.local.json`

`inputs_local/`, `.env`, `configs/api_config.local.json`, secrets, credenciais e
tokens permanecem gitignored. O intake publico grava somente status, contagens,
hashes, caminhos redigidos e flags nao sensiveis.
