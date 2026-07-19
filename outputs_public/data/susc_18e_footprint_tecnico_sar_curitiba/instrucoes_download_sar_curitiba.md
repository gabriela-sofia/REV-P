# Instrucoes de busca SAR - Curitiba SUSC-18E

## Onde colocar resultados locais

Use diretorio privado:

`local_runs/suscetibilidade/18e_sar_curitiba/resultados_gee`

## Fontes aceitas

- Google Earth Engine, com o script em `outputs_public/data/susc_18e_footprint_tecnico_sar_curitiba/pacote_gee/gee_sentinel1_curitiba_2022_01_15.js`.
- ASF, buscando Sentinel-1 GRD com `VV` e `VH`.
- Copernicus Dataspace, buscando Sentinel-1 GRD com `VV` e `VH`.

## Regras

1. A AOI deve ser a bbox tecnica dos 43 patches CUR: `-49.40354686,-25.59974008,-49.08763059,-25.29008811`.
2. O par pre/pos deve cobrir o evento `CUR_2022_01_15`.
3. Registrar CRS, data, orbit pass, polarizacoes e fonte.
4. Raster bruto permanece em caminho local privado.
5. Sem footprint revisado, manter `footprint_produzido=false`.
6. Footprint tecnico nao substitui geometria oficial de ocorrencia do 18D.
