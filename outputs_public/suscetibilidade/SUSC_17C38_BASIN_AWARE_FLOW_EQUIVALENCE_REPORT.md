# SUSC-17C38 - Flow Accumulation Basin-Aware e Revalidacao de Equivalencia

## Objetivo
Recalcular flow accumulation (e TWI/TPI/HAND) num DOMINIO HIDROLOGICO UNICO E AMPLO (nao janela por patch) cobrindo os 100 patches oficiais + buffer, e reavaliar equivalencia com as features oficiais. Testar se o truncamento espacial causou a incompatibilidade do 17C37.

## Dominio e grid
- Dominio basin-aware: buffer 5.0 km; grid unico; flow accumulation NAO por janela de patch.
- DEM: 4 tentativas de tile Copernicus GLO-30; grid criado: True.
- Processamento: 3.73s; max flow accumulation: 171.0.
- Patches oficiais recomputados: 100/100.

## Equivalencia por feature (17C37 -> 17C38)
  - hand_mean: calibratable -> equivalent | spearman=0.9127 | scale_ratio=0.9704 | improved=false
  - twi_mean: calibratable -> calibratable | spearman=0.8805 | scale_ratio=0.1757 | improved=true
  - tpi_250m_mean: calibratable -> calibratable | spearman=0.8425 | scale_ratio=1.3625 | improved=false
  - flow_acc_log_mean: incompatible -> incompatible | spearman=-0.2299 | scale_ratio=0.6728 | improved=true

## flow_acc (foco do marco)
- flow_acc 17C37: incompatible -> 17C38: incompatible; melhorou: True.

## Decisao
- method_equivalence_accepted=False; method_calibration_possible=False.
- equivalentes=1; calibraveis=2; incompativeis=1.
- can_compute_score_v6_final_replay=False; result_class=B_flow_acc_still_incompatible.

## Guardrails
- Equivalencia so em patches oficiais (canarios nao decidem); flow_acc em dominio unico (nao janela por patch); matriz oficial e score v6 intactos; sem v7/GT/treino; sem normalizacao canary-only; raster pesado so em local_runs; controle nao vira negativo verdadeiro.

## minimum_success_achieved: True

## Proximo marco recomendado
SUSC-17C39 Buscar metodo/DEM/algoritmo original do score v6 (twi escala ~116) ou adotar analise por features brutas; flow_acc segue incompativel mesmo basin-aware. 17B fail-closed.
