# DATA-07 - Instrucoes de recuperacao de sensor family

Este pacote e de entrada humana rastreavel. O sensor nunca e inferido pelo nome
visual do arquivo ou do asset.

## Como preencher
1. Abra `mv2_data_07_sensor_lineage_human_template.csv`.
2. Para cada `patch_id`/`asset_id`, identifique a origem real do asset e preencha
   `source_asset_ref` (referencia da cena/produto de origem) e `source_asset_type`.
3. Preencha `sensor_family` com um dos valores permitidos.
4. Preencha `sensor_source_ref` com a evidencia rastreavel que comprova a familia
   do sensor (metadado de produto, manifesto, registro auditavel). Sem isso, o
   registro permanece bloqueado.
5. `spectral_eligible` so pode ser `true` para `SENTINEL_2` com `sensor_source_ref`.
   `SENTINEL_1` recebe `support_only=true`. As demais familias bloqueiam o baseline.
6. Atualize `review_status` apenas apos conferencia.

## Valores permitidos de sensor_family
- SENTINEL_2
- SENTINEL_1
- DINO_DERIVED
- PNG_RENDER
- NPZ_EMBEDDING
- UNKNOWN
- CONFLICT

## Regras inviolaveis
- So `SENTINEL_2` pode ser `spectral_eligible=true`.
- `SENTINEL_1` e `support_only=true` (suporte SAR, nao baseline optico S2).
- `DINO_DERIVED`, `PNG_RENDER`, `NPZ_EMBEDDING`, `UNKNOWN` e `CONFLICT` bloqueiam
  o baseline espectral.
- Sensor nunca e inferido por nome visual.
- `sensor_source_ref` e obrigatorio para promover.
- Sem preenchimento, o estagio permanece `UNKNOWN_BLOCKED`.
