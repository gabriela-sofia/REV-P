# SUSC-17C37 - Validacao de Equivalencia Metodologica Topography/Hydrology

## Objetivo
Validar se o pipeline hidrologico local do 17C36 e equivalente/calibravel/incompativel com as features oficiais (susc_features_by_patch_v1.csv), recomputando patches OFICIAIS amostrados com o MESMO pipeline e comparando aos valores oficiais.

## Amostra e recomputo
- Populacao oficial: 100 patches recife.
- Amostra oficial: 24 (extremos por feature + proximos aos canarios + estratificado por classe).
- Recomputados (pipeline 17C36, DEM Copernicus GLO-30): 24.
- Comparacoes patch x feature: 96.

## Metricas de equivalencia por feature
  - hand_mean: status=calibratable; spearman=0.9287; rel_err_med=0.2487; scale_ratio_med=0.7824 (oficial 1.9457..42.25; recomputado 0.8552..28.7888)
  - twi_mean: status=calibratable; spearman=0.8165; rel_err_med=0.8271; scale_ratio_med=0.1729 (oficial 6.6409..1033.1852; recomputado 6.4742..9.1382)
  - tpi_250m_mean: status=calibratable; spearman=0.9209; rel_err_med=0.6314; scale_ratio_med=1.2474 (oficial -1.8725..1.842; recomputado -1.178..2.637)
  - flow_acc_log_mean: status=incompatible; spearman=-0.5148; rel_err_med=0.4967; scale_ratio_med=1.1487 (oficial 0.7253..4.942; recomputado 0.9771..1.4633)

## Decisao
- method_equivalence_accepted=False; method_calibration_possible=False.
- Equivalentes: 0; calibraveis: 3; incompativeis: 1.
- can_compute_score_v6_final_replay=False; can_compute_calibrated_component_replay=True.
- result_class: C_method_incompatible.

## Interpretacao (honesta)
O pipeline 17C36 gera features reais mas o metodo/DEM/unidade originais do score v6 nao estao documentados; a comparacao em patches oficiais mede diretamente a equivalencia. O score final v6 comparavel so e liberado se as 4 features criticas forem equivalent. Caso contrario, o score final permanece bloqueado (calibracao futura ou busca do metodo original).

## Guardrails
- susc_features_by_patch_v1.csv e score v6 oficial intactos; sem v7/GT/treino; equivalencia avaliada SO em patches oficiais (canarios nao calibram); ocorrencia nao e feature; sem normalizacao canary-only; raster pesado so em local_runs; controle nao vira negativo verdadeiro.

## minimum_success_achieved: True

## Proximo marco recomendado
SUSC-17C38 Politica de calibracao review-only das features topograficas calibraveis (ex.: TWI/flow_acc por relacao monotonica) OU busca do metodo/DEM original do score v6; so entao replay v6 comparavel. Manter score v6 oficial intacto e 17B fail-closed.
