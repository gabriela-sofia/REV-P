# Protocolo C - resolucao final de negativos e controles v1kk-v1kr

Esta etapa executa rotas validas sem inventar ground truth: expande controles conservadores, tenta adquirir patches multimodais, constroi tabela numerica, endurece gates e separa C4 operacional de experimento com controles.

Controles conservadores, mesmo quando fortes, nao sao negativos formais. A etapa mantem DINO congelado, nao salva pesos e nao cria label operacional.
