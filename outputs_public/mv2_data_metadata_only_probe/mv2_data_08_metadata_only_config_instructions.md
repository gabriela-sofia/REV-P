# DATA-08 - Instrucoes de config metadata-only

## Estado padrao (dry-run)
Sem `configs/api_config.local.json`, o preflight retorna `BLOCKED_NO_CONFIG` e
nenhuma chamada e feita.

## Exemplo seguro versionado
`configs/api_config.metadata_only.example.json` mantem todos os flags em `false`.

## Para sair de dry-run (acao manual humana)
Crie LOCALMENTE (nunca versione) `configs/api_config.local.json` com:

```json
{
  "allow_network": true,
  "allow_metadata_calls": true,
  "allow_raster_download": false,
  "allow_canary_download": false
}
```

E exporte as variaveis de ambiente necessarias (sem versionar segredo), por exemplo:

```
REV_P_GEE_PROJECT_ID=<seu_project_id>
```

## Nunca criar/versionar
- `configs/api_config.local.json`
- `.env`
- qualquer token ou credencial

Mesmo com `allow_metadata_calls=true`, `allow_raster_download` e
`allow_canary_download` permanecem `false`: a frente e metadata-only.
