# Cartao observacional - Curitiba

## Amostra
2 patches observacionais review-only (background da regiao:
98).

## Evidencia
Overlays tecnicos SAR review-only (curitiba_01050, curitiba_01101).

## score_v6
Score medio observado 0.5137 contra background
0.5151. Resultado top-k: 1/2_em_top30_regional.

## Features coerentes
Direcoes consistentes: 8 (divergentes:
8). Coerencia principal em elevacao, declividade,
HAND, urbanizacao e indices espectrais.

## Divergencias
Distancia a agua, TWI, fluxo e chuva antecedente divergem da direcao esperada.

## Limitacoes
Amostra pequena; sem geometria oficial; potencia amostral baixa.

## Por que nao e ground truth / treino / score_v7 / 17B
Evidencia review-only sem geometria de ocorrencia; nao e verdade de campo, nao
habilita treino, nao autoriza score_v7 e nao cria benchmark 17B.
