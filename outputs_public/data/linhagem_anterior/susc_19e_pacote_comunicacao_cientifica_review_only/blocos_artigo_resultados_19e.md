# Blocos para artigo - Resultados (review-only)

## Consolidacao multimodal
Foram consolidados 300 patches (100 por regiao). As familias
fisica, espectral e de chuva tem cobertura completa; a familia territorial tem
cobertura parcial (33.3%) por lacuna de MapBiomas, solo
exposto, agua e impermeabilizacao.

## Amostra observacional
7 patches com evidencia observacional review-only (Recife
5, Curitiba 2); background nao rotulado de
293. Petropolis fica fora, como fenomeno misto bloqueado.

## score_v6 observacional
O score_v6 medio dos observados e 0.596 contra 0.521
do background. Ainda assim, 0/7 observados estao
no top-30 global e 3/7 no top-30 regional. O
hit-rate e exploratorio review-only e nao e benchmark.

## Aderencia e divergencia
Familias que sustentam a aderencia: fisica_topografica, urbana_territorial, espectral_umidade. Familias que derrubam os observados:
hidrologica, chuva_hidrometeorologica. Ha coerencia urbana e topografica e divergencia hidrologica e de chuva
antecedente.

## Divergencia relevante
O patch curitiba_01101 tem score_v6 baixo apesar da evidencia
observacional review-only, sendo a divergencia mais forte; permanece baixo nos
cenarios de sensibilidade nao oficiais.

## Bloqueio do score_v7
O score_v7 permanece bloqueado por amostra minima, missingness territorial e ausencia
de benchmark. O score_v6 permanece intacto e nenhum score_v7 e criado.
