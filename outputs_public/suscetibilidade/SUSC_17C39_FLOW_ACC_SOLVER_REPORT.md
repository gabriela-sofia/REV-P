# SUSC-17C39 - Flow Accumulation Solver: Method Recovery, Variants e Calibracao Transferivel

## Objetivo
Resolver operacionalmente o flow_acc_log_mean via forense + variantes hidrologicas + calibracao supervisionada nos 100 patches OFICIAIS + crossval + aplicacao aos canarios.

## Diagnostico
- flow_acc oficial correlaciona com TWI=0.806 e HAND=-0.5978 (spearman).

## Variantes diretas (top 5 por |spearman|)
  - V17_log1p_accumulation_plus_slope_weight: spearman=-0.5781 status=incompatible improved_vs_17c38=true
  - V09_inverse_flow_acc_log1p: spearman=0.2811 status=incompatible improved_vs_17c38=true
  - V03_d8_raw_cell_count_basin_5km: spearman=-0.2798 status=incompatible improved_vs_17c38=true
  - V06_d8_log1p_with_drainage_burning: spearman=-0.2733 status=incompatible improved_vs_17c38=true
  - V10_downstream_distance_proxy: spearman=-0.2508 status=incompatible improved_vs_17c38=true
- Variantes registradas: 14; resultados patch x variante: 1398.
- Melhor variante direta: V17 (spearman=-0.5781, status=incompatible).

## Calibracao transferivel (crossval 5-fold, so oficiais)
- Modelos: 4; folds: 20.
- Melhor calibrador: C03 (spearman=0.5194, pearson=0.4918, rank=0.5194, passa=False).

## Solucao selecionada
- Tipo: replacement_contract_review_only | result_class: D_replacement_contract.
- method_recovered=False; variant_equivalent=False; calibrated_surrogate_validated=False; replacement_contract=True.
- Aplicada a 11 canarios (calibrated_flow_acc_log_mean_review_only, not_original_method=true).

## Readiness
- can_compute_calibrated_score_v6_replay=False; score final v6 NAO computado neste marco.

## Guardrails
- Calibracao SO nos 100 patches oficiais (canarios nunca treinam); surrogate validado por crossval antes de aplicar; matriz oficial e score v6 intactos; sem v7/GT/treino; metodo calibrado nunca mascarado como original; raster pesado so em local_runs.

## minimum_success_achieved: True

## Proximo marco recomendado
SUSC-17C40 Migrar para score/componentes review-only transparentes (flow_acc como replacement contract documentado) ou buscar metodo/DEM original. 17B fail-closed.
