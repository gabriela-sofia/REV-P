# Cartao observacional - Recife

## Amostra
5 patches observacionais review-only (background da regiao:
95).

## Evidencia
Canarios review-only (documental).

## score_v6
Score medio observado 0.6287 contra background
0.5697. Resultado top-k: 2/5_em_top30_regional.

## Features coerentes
Direcoes consistentes: 9 (divergentes:
7). Coerencia principal em elevacao, declividade,
HAND, urbanizacao e indices espectrais.

## Divergencias
Distancia a agua, TWI, fluxo e chuva antecedente divergem da direcao esperada.

## Limitacoes
Amostra pequena; sem geometria oficial; potencia amostral baixa.

## Por que nao e ground truth / treino / score_v7 / 17B
Evidencia review-only sem geometria de ocorrencia; nao e verdade de campo, nao
habilita treino, nao autoriza score_v7 e nao cria benchmark 17B.
