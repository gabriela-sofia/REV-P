# SUSC-19A - Matriz multimodal escalavel por patch

## Estado herdado do 18H

O 18H consolidou a cadeia 17C ate 18G: Recife como referencia forte review-only,
Curitiba como segunda regiao tecnica SAR sem geometria oficial e Petropolis
bloqueado por fenomeno misto. Estado do 17B: `17B_APROXIMACAO_COM_SEGUNDA_REGIAO_TECNICA` (17B nao criado).

## Motivacao da matriz multimodal

Deixar de depender do footprint como base central e organizar uma matriz unica,
com uma linha por patch, que sustente a generalizacao do REV-P. A matriz integra
features fisicas, territoriais, espectrais e de chuva reais, alem de score_v6 e
evidencia documental/observacional review-only.

## Fontes de features

As features fisicas, territoriais basicas, espectrais e de chuva vem de
`datasets/suscetibilidade/susc_features_by_patch_v1.csv`. O score_v6 vem de `datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv` por join. A evidencia
documental/observacional vem do 18H e dos overlays tecnicos SAR do 18G. O
patch_stats SAR (pos-evento) e bloqueado como feature pre-evento.

## Universo de patches

Total de patches consolidados: 300 (Recife
100, Curitiba
100, Petropolis
100). Cada patch e uma linha
unica, com geometria de patch e bbox.

## Cobertura por regiao

| Regiao | Patches | Fisico | Territorial | Espectral | Chuva | Documental | Observacional |
| --- | --- | --- | --- | --- | --- | --- | --- |
| recife | 100 | 1.0000 | 0.3333 | 1.0000 | 1.0000 | 0.0500 | 0.0500 |
| curitiba | 100 | 1.0000 | 0.3333 | 1.0000 | 1.0000 | 0.0000 | 0.0200 |
| petropolis | 100 | 1.0000 | 0.3333 | 1.0000 | 1.0000 | 0.0100 | 0.0100 |

## Cobertura por familia

| Familia | Presentes/Esperadas | Cobertura media | Principal lacuna |
| --- | --- | --- | --- |
| fisica_topografica | 7/7 | 1.0000 | nenhuma |
| territorial | 2/6 | 0.3333 | MapBiomas_class_majority;exposed_soil_prop;water_prop;impervious_proxy |
| espectral | 4/4 | 1.0000 | AWEI e NDMI nao materializados (opcionais) |
| chuva | 4/4 | 1.0000 | nenhuma |
| documental | 1/1 | 0.0200 | documentacao so em Recife e contexto misto em Petropolis |
| observacional | 1/1 | 0.0267 | evidencia observacional so em Recife e SAR de Curitiba |

## Lacunas

A maior lacuna e territorial: faltam MapBiomas, exposed_soil_prop, water_prop e
impervious_proxy em todas as regioes. A evidencia documental e observacional
existe apenas em poucos patches (Recife e Curitiba SAR). A missingness e explicita
em `missingness_por_patch.csv` e nao e escondida.

## Como Recife, Curitiba e Petropolis entram

- Recife: features completas mais evidencia observacional forte review-only.
- Curitiba: features completas mais overlay tecnico SAR em 2 patches; geometria
  oficial ausente.
- Petropolis: features completas, porem evidencia bloqueada por fenomeno misto.

## Por que SAR e footprints sao canarios

O footprint e o SAR indicam onde olhar (canario), mas nao sao a base estrutural
nem geometria oficial. O patch_stats SAR e pos-evento e nunca vira feature pre-evento.

## Por que a matriz e a base estrutural

Ela e escalavel (uma linha por patch), auditavel (fonte e temporalidade por
feature) e honesta quanto a missingness. E o eixo que sustenta a generalizacao.

## Por que nao e ground truth, nem treino, nem score_v7

Toda a matriz e review-only: ground_truth falso, treino desabilitado e score_v7
nao permitido. O `coverage_score` mede completude da matriz, nunca suscetibilidade.
O score_v6 permanece intacto e nenhum benchmark 17B foi criado.

## Proximo marco recomendado

**SUSC-19B - Auditoria de cobertura multimodal**: aprofundar a missingness por
regiao e feature e priorizar o preenchimento das lacunas territoriais.
