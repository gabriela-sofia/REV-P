# SUSC-17C36 - Pipeline Hidrologico Local para TWI/TPI/Flow Accumulation/HAND Full

## Objetivo
Pipeline hidrologico local auditavel (DEM Copernicus GLO-30 + numpy) para reproduzir twi_mean, tpi_250m_mean, flow_acc_log_mean e hand_mean/min/max dos canarios, completando o subindice topography_hydrology do replay review-only do score v6.

## Metodo
- Inventario do metodo original: 6 features; recuperacao majoritaria feature_names_only (DEM/algoritmo/unidade originais nao documentados no repo).
- DEM: 3 tentativas; artefato local: 1 (Copernicus GLO-30, ~30 m, raw em local_runs, NAO commitado).
- Grid hidrologico: fill-sinks (priority-flood), D8 flow direction+accumulation, slope, TPI 250m (integral image), TWI=ln(a/tan(beta)), HAND (downstream ate celula de drenagem).

## Features topograficas (reconstruidas)
- TPI: 11; flow_acc: 11 (full 11); TWI: 11 (full 11); HAND: 11 (full 11/proxy 0).
- Topography matrix: 11 linhas; completas: 11; paridade media: 1.0.

## Resultado (honesto)
- result_class: A_topography_reconstructed_score_final_blocked_on_method_equivalence.
- As features topograficas/hidrologicas FORAM reproduzidas em escala local auditavel a partir de DEM real. Porem o metodo NAO e comprovadamente identico ao score v6 original (metodo original nao documentado; twi original ~116 tem escala atipica) -> method_equivalence_status=method_reconstructed_not_proven_equivalent.
- Score v6 full replay computavel: False; score final review-only computado: 0. O score final NAO foi computado (guardrail: nao computar se metodo nao for equivalente).

## Guardrails
- Score v6 oficial intacto; sem v7; sem GT/treino; ocorrencia nao e feature; proxy != full flaggeado; OSM != hidrografia oficial; raster pesado so em local_runs (nunca outputs_public); controle nao vira negativo verdadeiro.

## minimum_success_achieved: True

## Proximo marco recomendado
SUSC-17C37 Validacao de equivalencia metodologica (recomputar as features topograficas dos PATCHES OFICIAIS com o MESMO pipeline e comparar com os valores originais); se equivalente, computar o score v6 replay final review-only; manter score v6 oficial intacto e 17B fail-closed.
