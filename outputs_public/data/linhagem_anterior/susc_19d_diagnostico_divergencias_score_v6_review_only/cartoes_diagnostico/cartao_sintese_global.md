# Cartao de diagnostico - Sintese global

## Evidencia
7 patches observacionais review-only (Recife
5, Curitiba 2); background nao
rotulado de 293 (nunca negativo).

## score_v6
Score medio observado 0.5958 contra background
0.5208; observados no top-30 global:
0.

## Aderencias
Familias que sustentam a aderencia: fisica_topografica, urbana_territorial, espectral_umidade.

## Divergencias
Familias que derrubam os observados: hidrologica, chuva_hidrometeorologica. O score_v6 pesa topografia_hidrologia
(0.4) e chuva (0.25); a hidrologia divergente e a chuva antecedente menor limitam o
ranking mesmo com topografia, urbano e espectral coerentes.

## Hipotese review-only
Sete hipoteses review-only (separar documental, chuva como gatilho, hidrologia urbana,
preencher territorial, ampliar amostra); nenhuma vira score oficial.

## Por que nao e ground truth
Evidencia observacional review-only sem geometria de ocorrencia confirmada por patch; SAR e pos-evento; nao e verdade de campo.

## Por que nao e treino
eligible_for_training=false; nenhuma linha habilita treino; a amostra e minima.

## Por que nao cria score_v7
score_v7 bloqueado por amostra, missingness territorial e ausencia de benchmark; as hipoteses sao review-only e nenhuma vira score oficial.

## Por que nao cria 17B
Sem geometria oficial e sem eventos suficientes; nenhum benchmark 17B e criado.
