# v1qz — DINO x SEDEC Recife Join (descriptive only, not a model)

## Objetivo

Verificar honestamente quantos dos 163 registros SEDEC reais (primary real-vs-real do pipeline_final_v5.py) caem dentro dos 37 patches Sentinel reais que já têm embedding DINO, e reportar uma comparação puramente descritiva — nunca inferencial — da estrutura do embedding entre label=1 e label=0.

## Por que não é um modelo

Apenas 81/163 registros caem dentro de algum patch com Sentinel real (n=81 com embedding válido; 23 patches únicos). Com 22 negativos, qualquer p-valor, AUC ou coeficiente seria ruído, não evidência — muito abaixo do que o próprio pipeline_final_v5.py já sinalizou como frágil em n=22 negativos. Nenhum número aqui deve ser citado como achado.

## Caminho para crescer isso de verdade

Só aumentando a cobertura de patches Sentinel reais de Recife (hoje 37 de ~100+ do dataset_final.csv) via exportação GEE adicional é que esse n cresce de forma honesta — não há atalho estatístico para contornar a falta de dado.
