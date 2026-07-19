# MV2-DATA-03 — Runbook de configuração (CONFIG_MISSING)

Nenhuma config local privada foi encontrada, então **nenhuma chamada de rede** foi
executada (fail-closed). Para habilitar o probe de metadados e o canary privado:

## 1. Criar config local privada (git-ignored)

Crie um destes arquivos (NÃO versionar, NÃO colocar segredo):

- `local_only/mv2_data_api_raster_harness/api_config.local.json`
- `data_local/mv2_data_api_raster_harness/api_config.local.json`
- `private_outputs/mv2_data_api_raster_harness/api_config.local.json`

Conteúdo (apenas flags + nome do diretório privado):

```json
{
  "gee_enabled": false,
  "cdse_stac_enabled": true,
  "cdse_odata_enabled": true,
  "allow_network": true,
  "allow_metadata_calls": true,
  "allow_raster_download": false,
  "allow_canary_download": false,
  "max_download_mb": 50,
  "private_output_dir": "local_only/mv2_data_live_api_canary"
}
```

## 2. Credenciais SÓ por variável de ambiente (nunca em arquivo)

- `CDSE_TOKEN` ou `CDSE_USERNAME` + `CDSE_PASSWORD` (CDSE STAC/OData)
- `GEE_PROJECT` + `GOOGLE_APPLICATION_CREDENTIALS` (Earth Engine)

## 3. Permissões opcionais por env var

- `REV_P_ALLOW_NETWORK=1`, `REV_P_ALLOW_METADATA_CALLS=1`
- `REV_P_ALLOW_RASTER_DOWNLOAD=1`
- `REV_P_ALLOW_RASTER_CANARY=YES` (confirmação explícita do canary)

Mesmo com tudo habilitado: no máximo **1** raster canary privado, só para a âncora
oficial, raster só em diretório privado, e **o Dia 10 do corpus continua bloqueado**.
