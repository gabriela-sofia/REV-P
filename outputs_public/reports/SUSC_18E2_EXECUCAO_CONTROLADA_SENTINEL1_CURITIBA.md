# SUSC-18E2 - Execucao controlada Sentinel-1 Curitiba

## Estado herdado do 18E

- Pacote GEE Sentinel-1 criado: `true`
- Raster SAR local suficiente no 18E: `false`
- Footprint produzido no 18E: `false`
- 17B: `17B_BLOQUEADO_POR_GEOMETRIA_OFICIAL`

## Rotas testadas

- Rota A - Python Earth Engine API: `tentada`
- Rota B - Earth Engine CLI: `tentada`
- Rota C - pacote JS GEE: `validada`

## Autenticacao e consulta

- Earth Engine autenticado: `true`
- GEE consultado: `true`
- Cenas Sentinel-1 pre-evento: `3`
- Cenas Sentinel-1 pos-evento: `3`

## Resultados

- Export iniciado: `true`
- Task IDs: `YNY6EV5ZXU25NT737DDJ7VCJ, KF4J7MQQDB64TDEM7S5CNLPZ, PCSCLVHXO2BGXK6P63XCFF4K`
- Flood mask local: `false`
- Flood vector local: `false`
- Patch stats local: `true`
- Arquivos locais: `local_runs/suscetibilidade/18e_sar_curitiba/resultados_gee`

## Bloqueio tecnico

- Tipo: `not_available`
- Detalhe: `nenhum bloqueio terminal registrado`
- Acao corretiva: `aguardar conclusao dos exports se houver tarefas pendentes`

## Handoff 18F

- Pronto para 18F: `true`
- Motivo: `not_available`

## Proxima acao pesada

Corrigir o bloqueio tecnico, reexecutar `python scripts\suscetibilidade\run_susc_18e2_gee_sentinel1_curitiba.py`
e, se houver tarefas iniciadas, aguardar a conclusao do export antes de ingerir resultado local.
