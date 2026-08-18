# SUSC-17C8 - Extracao real controlada de features para patches candidatos

Este pacote executa uma tentativa controlada sobre as camadas adaptaveis do 17C7: `physical_static`, `urban_territorial` e `rainfall_trigger`.

Resultado: nenhum valor foi marcado como real, porque nao ha artefato local candidato-especifico verificavel para DEM/HAND/hidrologia, MapBiomas/territorial/espectral ou CHIRPS/runoff. A matriz oficial existente demonstra colunas e pipelines para patches oficiais, mas nao autoriza copiar, interpolar ou inferir valores para os patches candidatos.

## Contagens

- Patches candidatos: 5
- Linhas de feature tentadas: 90
- Linhas reais extraidas: 0
- Linhas bloqueadas: 90

## Garantias

- Review-only: sim.
- Valor sintetico tratado como real: nao.
- Proveniencia manifestada: sim.
- Uso indevido de dado pos-evento: nao.
- Patch oficial ou patch-link oficial criado: nao.
- Raw raster commitado: nao.
- Score v6 alterado: nao.
- Score v7 criado: nao.
- Treino, modelo, label ou ground truth criado: nao.

## Bloqueio 17B

O 17B permanece bloqueado porque os patches continuam candidatos, nao oficiais, sem politica aceita de promocao e sem valores reais candidato-especificos com proveniencia suficiente.
