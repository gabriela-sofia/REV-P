# SUSC-18F - Ingestao e validacao SAR de Curitiba

## Estado herdado do 18E2

- Earth Engine autenticado: `true`
- GEE consultado: `true`
- Cenas Sentinel-1: `pre=3`, `pos=3`
- Status herdado: `18E2_TAREFAS_GEE_INICIADAS_AGUARDANDO_CONCLUSAO`

## Tasks atualizadas

- Flood mask: `SUCCEEDED`
- Flood vector: `SUCCEEDED`
- Patch stats: `SUCCEEDED`

## Exports recuperados

- Patch_stats local: `true` (43 linhas)
- Footprint vetorial local: `false`
- Flood mask local: `false`

## Validacao tecnica

- Status 18F: `18F_REFERENCIA_TECNICA_SAR_CURITIBA_PARCIAL_POR_PATCH_STATS`
- Vínculos técnicos criados: `43`
- Vínculos técnicos fortes somente revisao: `0`

## Gate 17B

- Status 17B: `17B_APROXIMACAO_COM_EVIDENCIA_TECNICA_CURITIBA`
- Nenhum benchmark 17B foi criado.

## Guardrails

Sem ground truth, sem treino, sem score_v7, score_v6 intacto e footprint tecnico
nao substitui geometria oficial de ocorrencia.

## Proxima acao pesada

Recuperar do Drive o raster `flood_mask_curitiba_2022_01_15.tif`, se autorizado,
ou usar o vetor/patch_stats ja recuperados para revisao tecnica 18G sem promover
verdade de referencia.
