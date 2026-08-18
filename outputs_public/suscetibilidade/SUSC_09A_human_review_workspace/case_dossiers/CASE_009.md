# CASE_009 — petropolis_00467 (petropolis)

> AVISO: aderência espacial review-only. NÃO é ground truth de enchente por patch. Mesmo aprovação humana não cria ground truth automaticamente.

- **Região:** petropolis
- **Patch:** petropolis_00467
- **Relação espacial:** near_patch_buffer_candidate
- **evidence_id:** REGION_LEVEL_petropolis
- **coordinate_id:** COORD_0009
- **Fonte:** official_coordinate_recovery_hardened_registry.csv  (datasets/official_coordinate_recovery_hardened_registry.csv)
- **Geometria:** point_set
- **Data/período:** unknown
- **Score/proxy:** 0.6880287043189368
- **Features físicas:** slope_mean=15.2236; elevation_mean=931.3329; hand_mean=45.4975; tpi_250m_mean=-5.4458
- **Features hidrológicas:** distance_to_water_mean=4360.0813; twi_mean=11.4739; flow_acc_log_mean=1.819; water_occurrence_patch=0.0
- **Features espectrais:** ndbi_mean=-0.106; mndwi_mean=-0.5349; ndvi_mean=0.6035
- **Interpretação:** aderência espacial review-only (near_patch_buffer_candidate).
- **Limitações:** Aderencia espacial review-only; NAO e ocorrencia confirmada por patch. 
- **Conflito conhecido:** missing_or_weak_temporal_link
- **machine_pre_review:** `candidate_for_tcc_with_caution`
- **Mapa:** `maps_svg/CASE_009.svg`

## Perguntas para o revisor humano

1) fonte oficial/tecnica/derivada? 2) coordenada e evento/setor de risco/estacao/poligono candidato/contexto? 3) data compativel? 4) geometria dentro/proxima do patch? 5) proxy alto? 6) features fisicas explicam? 7) conflito entre fontes? 8) uso forte/cautela/contexto? 9) deve ficar bloqueado? 10) respeita review-only?

## Campos a preencher (no form)

source_verified, geometry_verified, temporal_match_verified, patch_relation_verified, conflict_checked, approved_for_tcc_example, approved_for_score_v6_context, approved_for_ground_truth(=false), final_case_strength, review_notes

> Governança: can_be_ground_truth=false · allowed_for_training=false · review_only=true
