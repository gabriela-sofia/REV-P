# Pacote GEE Sentinel-1 - SUSC-18E Curitiba

## Escopo

Este pacote usa a AOI tecnica derivada dos 43 poligonos CUR para consultar
Sentinel-1 GRD (`VV` e `VH`) em janelas pre/pos do evento `CUR_2022_01_15`.

## O que este pacote nao faz

- Nao cria geometria oficial de ocorrencia.
- Nao cria ground truth.
- Nao cria treino.
- Nao cria score_v7.
- Nao cria benchmark 17B.
- Nao substitui a resposta oficial solicitada no 18D.

## Resultado esperado

Exportar o raster tecnico e o manifesto para um diretorio local privado, por exemplo:

`local_runs/suscetibilidade/18e_sar_curitiba/resultados_gee`

Apos a execucao externa, registrar metadados, CRS, datas, orbit pass e caminho local
antes de qualquer ingestao. Um footprint tecnico so pode ser usado como evidencia
candidata somente revisao.
