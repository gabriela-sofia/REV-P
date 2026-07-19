# SUSC-18E - Footprint tecnico SAR de Curitiba

## Resultado

- Status 18E: `18E_PACOTE_GEE_SENTINEL1_PRONTO`
- Status 17B: `17B_BLOQUEADO_POR_GEOMETRIA_OFICIAL`
- Raster SAR local suficiente: `false`
- Footprint produzido: `false`
- Pacote GEE criado: `true`
- Patches CUR usados para AOI: `43`
- Vinculos fortes: `0`

## Interpretacao

Os 43 patches CUR foram usados para preparar uma AOI tecnica de busca SAR. Isso
gera uma fila executavel e um pacote GEE/ASF/Copernicus, mas nao resolve a
geometria oficial de ocorrencia. No estado atual, nao ha par raster Sentinel-1
local pre/pos suficiente, portanto nenhum footprint foi produzido.

## Diferenca metodologica

Patches candidatos preparados sao alvos espaciais para processamento e revisao.
Vinculos fortes exigem geometria real de ocorrencia ou sobreposicao aceita por
regra especifica. Como o 18E nao recebeu footprint tecnico revisado e o 18D segue
sem resposta oficial, `strong_patch_links=0`.

## Proxima acao pesada

Executar o pacote GEE ou obter par Sentinel-1 GRD por ASF/Copernicus, salvar os
resultados privados em `local_runs/suscetibilidade/18e_sar_curitiba/resultados_gee` e revisar o footprint tecnico
antes de qualquer ingestao leve.
