# Pacote MapBiomas/GEE - SUSC-19B

Extracao das features territoriais faltantes para os 300 patches, somente revisao.

## O que o script faz

1. Carrega os 300 patches como `FeatureCollection` (asset editavel `PATCHES_ASSET`).
2. Seleciona a banda `classification_2022` da colecao MapBiomas.
3. Calcula a proporcao de classes por patch com `frequencyHistogram`.
4. Deriva `mapbiomas_class_majority`, `water_prop`, `exposed_soil_prop`,
   `impervious_proxy` e a distribuicao de classes.
5. Exporta um CSV leve (sem raster) com uma linha por patch.

## Parametros editaveis

- `COLLECTION_ASSET`: colecao MapBiomas (padrao Colecao 9).
- `YEAR`: ano de referencia (padrao 2022).
- `SCALE`: resolucao em metros (padrao 30).
- `PATCHES_ASSET`: sua `FeatureCollection` com `patch_id` e geometria dos 300 patches.
- Listas de classes (agua, solo exposto, urbano, vegetacao) conforme a legenda oficial.

## Como preparar os patches

Use `gee_export_manifest_19b.csv` (patch_id + bbox) para construir a
`FeatureCollection` no Earth Engine ou para gerar um asset via upload.

## Restricoes

- Nao contem credenciais; a autenticacao e feita pelo usuario no Earth Engine.
- Nao baixa raster pesado; exporta apenas CSV leve.
- Resultado e somente revisao: nao e ground truth, nao habilita treino e nao cria score_v7.
