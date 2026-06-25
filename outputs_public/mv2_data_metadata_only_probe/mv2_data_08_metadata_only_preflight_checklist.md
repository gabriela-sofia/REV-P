# DATA-08 - Checklist de preflight metadata-only

- [ ] DATA-06 promovido: janela temporal com fonte rastreavel (`PROMOTED_METADATA_READY`).
- [ ] DATA-07 promovido: sensor `SENTINEL_2` elegivel com `sensor_source_ref`.
- [ ] `configs/api_config.local.json` criado localmente (nao versionado).
- [ ] `allow_network=true` e `allow_metadata_calls=true`.
- [ ] `allow_raster_download=false` e `allow_canary_download=false`.
- [ ] Variaveis de ambiente de provedor exportadas (sem segredo no repo).
- [ ] Nenhum `.env`/token/credencial versionado.

Enquanto qualquer item acima estiver pendente, o preflight permanece
`BLOCKED_NO_CONFIG`/`BLOCKED_BY_FLAGS` e o Dia 10 segue `BLOCKED`.
