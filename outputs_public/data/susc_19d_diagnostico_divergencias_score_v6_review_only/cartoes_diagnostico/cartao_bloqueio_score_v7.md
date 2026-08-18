# Cartao de diagnostico - Bloqueio do score_v7

## Estado
status_score_v7 = `SCORE_V7_NAO_AUTORIZADO`.

## Motivos do bloqueio
- amostra minima (7 observados; 0 no top-30 global);
- missingness territorial herdado do 19B (faltam MapBiomas_class_majority, MapBiomas_class_distribution, exposed_soil_prop, water_prop, impervious_proxy);
- ausencia de benchmark 17B;
- hipoteses sao review-only e nenhuma vira score oficial.

## O que muda com o diagnostico
Nada no score_v6 (intacto) e nada no score_v7 (nao criado). O diagnostico apenas
organiza aderencias, divergencias, falhas e hipoteses testaveis no futuro.

## Por que nao e ground truth
Evidencia observacional review-only sem geometria de ocorrencia confirmada por patch; SAR e pos-evento; nao e verdade de campo.

## Por que nao e treino
eligible_for_training=false; nenhuma linha habilita treino; a amostra e minima.

## Por que nao cria score_v7
score_v7 bloqueado por amostra, missingness territorial e ausencia de benchmark; as hipoteses sao review-only e nenhuma vira score oficial.

## Por que nao cria 17B
Sem geometria oficial e sem eventos suficientes; nenhum benchmark 17B e criado.
