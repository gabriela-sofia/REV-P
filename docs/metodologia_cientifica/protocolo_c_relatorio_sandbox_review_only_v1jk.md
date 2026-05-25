# Relatorio v1jk - Sandbox review-only

## Resultado

A v1jk executa uma analise exploratoria local sobre os 9 anchors oficiais. Ela consolida features DINO, espectrais e de terreno, calcula rankings de revisao e registra um sandbox one-class quando a biblioteca necessaria esta disponivel.

## Features

As features usadas sao:

- similaridade e distancia DINO pre/pos;
- NDWI, NDBI e deltas de bandas;
- DEM, slope e aspect;
- status S1 e QA Sentinel-2.

A tabela e local porque depende de rasters em `local_runs`. O registry publico resume apenas o boundary do sandbox.

## Interpretacao

O ranking mostra quais anchors merecem revisao estrutural prioritaria. A distribuicao de distancias compara diagnosticos pareados ja armazenados. A PCA ajuda a enxergar agrupamentos exploratorios.

Nenhum desses elementos e uma classe. Nenhum deles e label.

## Sandbox

O sandbox one-class, quando roda, fica com status `INVALID_FOR_SCIENTIFIC_CLAIM`. Ele e valido apenas para verificar se o pipeline tecnico aceita uma matriz de features pequena.

Ele nao salva pesos e nao altera o gate cientifico.

## Limite

O treino supervisionado permanece bloqueado por falta de:

- negativos formais;
- protocolo de ausencia;
- governanca de labels;
- split por evento/localidade;
- auditoria de vazamento;
- validacao independente.

Portanto, v1jk e uma etapa de engenharia e revisao, nao uma etapa de modelagem cientifica.
