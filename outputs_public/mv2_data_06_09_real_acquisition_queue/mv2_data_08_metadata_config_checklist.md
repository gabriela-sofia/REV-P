# DATA-08 - Checklist de aquisicao de config metadata-only

Crie LOCALMENTE (nunca versione) `configs/api_config.local.json`:

```json
{
  "allow_network": true,
  "allow_metadata_calls": true,
  "allow_raster_download": false,
  "allow_canary_download": false,
  "providers": {
    "GEE": {"enabled": false, "project_id_env": "REV_P_GEE_PROJECT_ID"},
    "CDSE_STAC": {"enabled": false},
    "CDSE_ODATA": {"enabled": false}
  }
}
```

## Regras
- [ ] `allow_network=true` e `allow_metadata_calls=true`.
- [ ] `allow_raster_download=false` e `allow_canary_download=false` (frente metadata-only).
- [ ] Segredos somente em variaveis de ambiente; nunca no arquivo versionado.
- [ ] Nunca versionar `configs/api_config.local.json`, `.env`, token ou credencial.

Enquanto este arquivo nao existir localmente, DATA-08 fica `BLOCKED_NO_CONFIG`
e o motor permanece em replay-only.
