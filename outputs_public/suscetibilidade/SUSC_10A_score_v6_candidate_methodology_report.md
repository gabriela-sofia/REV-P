# SUSC-10A — Score v6 Candidato (determinístico, review-only)

> O SUSC-10A cria um score v6 candidato determinístico e review-only. Ele não é modelo supervisionado, não usa ground truth, não valida ocorrência real de enchente e não autoriza afirmações de evento observado por patch.

## Construção
- Features usadas (aprovadas pelos gates): **19**.
- Excluídas do score principal (sinalizadas SUSC-06B, diagnóstico): ['curvature_laplacian_mean', 'flow_acc_log_p75', 'rain_3d_7d_ratio', 'water_occurrence_patch'].
- Normalização determinística: winsorize 1/99 → robust min-max → orientação por direção esperada.
- Subíndices e pesos documentados: topography_hydrology_index=0.4, rainfall_trigger_index=0.25, urban_spectral_index=0.2, vegetation_mitigation_index=-0.1, evidence_support_index=0.05.
- Score final reescalado para [0,1]; classes por tercis globais (low/medium/high).

## Resultados
- n=300; score médio=0.5226; classes: {'medium': 99, 'low': 100, 'high': 101}.
- Por região: ver `SUSC_10A_score_v6_candidate_by_region.csv`.
- Top 20 patches: `SUSC_10A_score_v6_candidate_top_patches.csv`.
- Contribuições por feature: `SUSC_10A_score_v6_candidate_feature_contributions.csv`.

## O que NÃO é
- Não usa label heurístico, score v5 ou proxy como feature/target.
- Não treina modelo; não persiste modelo; não cria ground truth.
- Não é validação de ocorrência por patch.

## Limitações
Pesos são escolha metodológica (não aprendidos). `evidence_support_index` usa aderência espacial review-only (07B), não ground truth. As 4 features sinalizadas ficam em diagnóstico.

> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.
