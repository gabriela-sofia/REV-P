# v1qx — DINO Embedding vs Real Physical-Variable Correlation (review-only)

## Objetivo

Primeiro passo concreto rumo a um framework multimodal: testar se o espaço de embedding DINO (auxiliar) é consistente com variáveis físicas reais e independentes (elevação, declividade — causais), para os 17 patches de Petrópolis onde ambas existem.

## Limite metodológico

Isto é um relatório de correlação exploratória, nunca uma feature de modelo. Nenhuma coluna aqui é usada como entrada de classificador. n pequeno (exploratório, não robusto estatisticamente).

## Resultado

r(similaridade_embedding, |Δelevação|) = 0.0109; r(similaridade_embedding, |Δdeclividade|) = -0.4654; r(PCA1, elevação) = 0.0043; r(PCA1, declividade) = -0.7272.
