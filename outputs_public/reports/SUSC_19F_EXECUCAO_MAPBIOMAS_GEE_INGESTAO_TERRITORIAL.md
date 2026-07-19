# SUSC-19F - Execucao MapBiomas/GEE e ingestao territorial

## Estado herdado do 19B/19E

O 19B identificou a lacuna territorial (MapBiomas/solo exposto/agua/impermeabilizacao)
nos 300 patches e preparou o pacote MapBiomas/GEE. O 19E consolidou a comunicacao
review-only. Esta etapa executa o pacote e ingere as features territoriais reais,
sem criar score novo e sem alterar o score_v6.

## Ambiente GEE

Status do ambiente: `gee_autenticado_pronto`. Verificacoes de Python, biblioteca earthengine,
CLI, `ee.Initialize()` (sem impressao de credenciais), conectividade, permissao de
compute e pasta gravavel em `auditoria_ambiente_gee_19f.csv`.

## Execucao MapBiomas

Colecao `projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1`, ano 2022, escala 30 m. Para
cada patch calcula-se o histograma de classes (`frequencyHistogram`) e derivam-se a
classe majoritaria, `water_prop`, `exposed_soil_prop` e `impervious_proxy`. Nao se baixa
raster pesado; exporta-se apenas CSV leve.

## Export / recuperacao

Rotas de recuperacao em `recuperacao_export_mapbiomas_19f.csv`. CSV leve recuperado:
`true`. Patches preenchidos:
300 de 300.

## Ingestao territorial

Quando ha CSV real, `matriz_territorial_mapbiomas_19f.csv` recebe as features
territoriais reais e `matriz_multimodal_19f_atualizada.csv` atualiza a cobertura de
completude. urban_prop e vegetation_prop do 19A sao preservados. Nenhum valor e
inventado: sem CSV real, a feature fica vazia e o status e `aguardando_export`.

## Cobertura antes/depois

| Metrica | 19A | 19B | 19F |
| --- | --- | --- | --- |
| cobertura_territorial | 0.3333 | 0.3333 | 1.0000 |
| cobertura_total | 0.5633 | 0.5633 | 0.6744 |
| patches_preenchidos | 0 | 0 | 300 |
| features_destravadas | 0 | 0 | 4 |

| Familia | Cobertura | Status | Principal lacuna |
| --- | --- | --- | --- |
| fisica_topografica | 1.0000 | completa | nenhuma |
| espectral_umidade | 1.0000 | completa | nenhuma |
| chuva_hidrometeorologica | 1.0000 | completa | nenhuma |
| territorial | 1.0000 | completa | nenhuma |
| documental | 0.0200 | rara | documental so em Recife |
| observacional | 0.0267 | rara | evidencia so em Recife e SAR de Curitiba |

## Lacunas restantes

Nenhuma lacuna territorial: as 6 features estao completas.

## Por que nao e ground truth

Uso/cobertura do solo e feature territorial escalavel review-only; MapBiomas nao e
verdade de campo de enchente e nunca vira ground truth.

## Por que nao e treino

eligible_for_training=false; a feature territorial nao habilita treino supervisionado
e nao cria rotulo nem target.

## Por que nao e score_v7

O score_v7 permanece `SCORE_V7_NAO_AUTORIZADO`. Preencher territorial nao valida
score, nao cria benchmark e nao altera o score_v6, que segue baseline oficial.

## Proximo marco recomendado

SUSC-19G (recalibracao candidata review-only apos territorial).
