# Replay metadata-only Sentinel-2

Esta pasta guarda fixtures publicas e leves usadas para exercitar o motor de
metadados sem tocar a rede. Por padrao o motor roda offline (`--replay-only`):
ele le estas fixtures vazias e nao executa nenhuma chamada real.

## Conteudo
- `fixtures/gee_empty_result.json`
- `fixtures/stac_empty_result.json`
- `fixtures/odata_empty_result.json`
- `fixtures/traceability_empty_result.json`

Todas sao ficticias e vazias (`features: []` / `value: []` / `items: []`).

## Regras de publicacao
- fixtures publicas sao sempre ficticias ou vazias;
- se no futuro houver resposta real, publicar somente campos leves e redigidos
  (`redact_sensitive_fields`);
- nunca publicar token, URL assinada, path local, raster ou payload pesado;
- `odata_s3path` e mantido em branco por politica.

## Como o replay e usado
1. `load_replay_fixture(provider)` carrega a fixture vazia;
2. `normalize_*_response(raw, target)` converte para o contrato canonico;
3. fixture vazia produz `NO_MATCH` (chamada simulada, nenhum item);
4. `hash_raw_response` registra um hash estavel da resposta bruta.
