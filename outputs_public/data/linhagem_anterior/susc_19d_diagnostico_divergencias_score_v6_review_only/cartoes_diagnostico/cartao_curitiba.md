# Cartao de diagnostico - Curitiba

## Evidencia
Dois overlays tecnicos SAR review-only (curitiba_01050, curitiba_01101). SAR nao e
geometria oficial e nunca vira feature pre-evento.

## score_v6
Um patch com score alto (curitiba_01050) e um com score baixo (curitiba_01101):

| patch | score_v6 | classe | percentil global | documental |
| --- | --- | --- | --- | --- |
| curitiba_01050 | 0.669632 | high | 0.7600 | documental_ausente |
| curitiba_01101 | 0.357789 | low | 0.2000 | documental_ausente |

## Aderencias
curitiba_01050 mantem coerencia urbana/espectral.

## Divergencias
Familia hidrologica e chuva divergem; documental ausente (evidence_support_index=0.0)
em ambos.

## Hipotese review-only
Componente hidrologico diferenciado para alagamento urbano e reducao da penalizacao
por baixa documentacao (HIP_05, HIP_02); nenhuma vira score oficial.

## Por que nao e ground truth
Evidencia observacional review-only sem geometria de ocorrencia confirmada por patch; SAR e pos-evento; nao e verdade de campo.

## Por que nao e treino
eligible_for_training=false; nenhuma linha habilita treino; a amostra e minima.

## Por que nao cria score_v7
score_v7 bloqueado por amostra, missingness territorial e ausencia de benchmark; as hipoteses sao review-only e nenhuma vira score oficial.

## Por que nao cria 17B
Sem geometria oficial e sem eventos suficientes; nenhum benchmark 17B e criado.
