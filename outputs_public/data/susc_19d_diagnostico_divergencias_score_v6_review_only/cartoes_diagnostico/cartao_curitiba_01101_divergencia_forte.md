# Cartao de diagnostico - curitiba_01101 (divergencia forte)

## Evidencia
Overlay tecnico SAR review-only. SAR e pos-evento e nao e geometria oficial.

## score_v6
score_v6=0.357789 (classe low); percentil global
0.2000 e regional 0.0800 - o menor da amostra
observacional. Componente topografia_hidrologia e o mais baixo entre os observados e
o documental e 0.0.

## Aderencias
Aderencia parcial em urbana_territorial.

## Divergencias
Divergencia forte em fisica_topografica; o score_v6 classifica o patch
como baixo apesar da evidencia observacional review-only. Permanece baixo em todos os
cenarios de sensibilidade nao oficiais.

## Hipotese review-only
Caso emblematico para separar documental de suscetibilidade e revisar o componente
hidrologico (HIP_01, HIP_02, HIP_05); nenhuma vira score oficial.

## Por que nao e ground truth
Evidencia observacional review-only sem geometria de ocorrencia confirmada por patch; SAR e pos-evento; nao e verdade de campo.

## Por que nao e treino
eligible_for_training=false; nenhuma linha habilita treino; a amostra e minima.

## Por que nao cria score_v7
score_v7 bloqueado por amostra, missingness territorial e ausencia de benchmark; as hipoteses sao review-only e nenhuma vira score oficial.

## Por que nao cria 17B
Sem geometria oficial e sem eventos suficientes; nenhum benchmark 17B e criado.
