# CASE_002 — recife_00276 (recife)

> AVISO: aderência espacial review-only. NÃO é ground truth de enchente por patch. Mesmo aprovação humana não cria ground truth automaticamente.

- **Região:** recife
- **Patch:** recife_00276
- **Relação espacial:** near_patch_buffer_candidate
- **evidence_id:** REGION_LEVEL_recife
- **coordinate_id:** COORD_0003
- **Fonte:** recife_defesa_civil_risk_locations.geojson  (datasets/external_sources/recife_minimal_tp/event_polygon_REC_2022_05_24_30/raw/recife_defesa_civil_risk_locations.geojson)
- **Geometria:** point_set
- **Data/período:** unknown
- **Score/proxy:** 0.3720974184782609
- **Features físicas:** slope_mean=6.6368; elevation_mean=46.8888; hand_mean=24.3838; tpi_250m_mean=0.4066
- **Features hidrológicas:** distance_to_water_mean=3705.5576; twi_mean=9.808; flow_acc_log_mean=0.8001; water_occurrence_patch=0.0
- **Features espectrais:** ndbi_mean=0.132; mndwi_mean=-0.4454; ndvi_mean=0.3049
- **Interpretação:** aderência espacial review-only (near_patch_buffer_candidate).
- **Limitações:** Aderencia espacial review-only; NAO e ocorrencia confirmada por patch. Ponto de risco != footprint de evento. 
- **Conflito conhecido:** geometry_source_disagreement
- **machine_pre_review:** `blocked_source_conflict`
- **Mapa:** `maps_svg/CASE_002.svg`

## Perguntas para o revisor humano

1) fonte oficial/tecnica/derivada? 2) coordenada e evento/setor de risco/estacao/poligono candidato/contexto? 3) data compativel? 4) geometria dentro/proxima do patch? 5) proxy alto? 6) features fisicas explicam? 7) conflito entre fontes? 8) uso forte/cautela/contexto? 9) deve ficar bloqueado? 10) respeita review-only?

## Campos a preencher (no form)

source_verified, geometry_verified, temporal_match_verified, patch_relation_verified, conflict_checked, approved_for_tcc_example, approved_for_score_v6_context, approved_for_ground_truth(=false), final_case_strength, review_notes

> Governança: can_be_ground_truth=false · allowed_for_training=false · review_only=true
