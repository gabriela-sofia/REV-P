# CASE_006 — recife_00299 (recife)

> AVISO: aderência espacial review-only. NÃO é ground truth de enchente por patch. Mesmo aprovação humana não cria ground truth automaticamente.

- **Região:** recife
- **Patch:** recife_00299
- **Relação espacial:** bbox_overlap
- **evidence_id:** REGION_LEVEL_recife
- **coordinate_id:** COORD_0005
- **Fonte:** recife_digitization_aoi_context.geojson  (datasets/gis_workbench/recife_minimal_tp/layers/recife_digitization_aoi_context.geojson)
- **Geometria:** polygon
- **Data/período:** unknown
- **Score/proxy:** 0.3617568387681159
- **Features físicas:** slope_mean=5.9776; elevation_mean=63.2415; hand_mean=36.2013; tpi_250m_mean=0.536
- **Features hidrológicas:** distance_to_water_mean=4008.3594; twi_mean=13.4138; flow_acc_log_mean=0.8184; water_occurrence_patch=0.0
- **Features espectrais:** ndbi_mean=0.162; mndwi_mean=-0.4292; ndvi_mean=0.2531
- **Interpretação:** aderência espacial review-only (bbox_overlap).
- **Limitações:** Aderencia espacial review-only; NAO e ocorrencia confirmada por patch. Geometria e contexto/AOI, nao footprint de evento. 
- **Conflito conhecido:** geometry_is_context_not_event
- **machine_pre_review:** `context_only`
- **Mapa:** `maps_svg/CASE_006.svg`

## Perguntas para o revisor humano

1) fonte oficial/tecnica/derivada? 2) coordenada e evento/setor de risco/estacao/poligono candidato/contexto? 3) data compativel? 4) geometria dentro/proxima do patch? 5) proxy alto? 6) features fisicas explicam? 7) conflito entre fontes? 8) uso forte/cautela/contexto? 9) deve ficar bloqueado? 10) respeita review-only?

## Campos a preencher (no form)

source_verified, geometry_verified, temporal_match_verified, patch_relation_verified, conflict_checked, approved_for_tcc_example, approved_for_score_v6_context, approved_for_ground_truth(=false), final_case_strength, review_notes

> Governança: can_be_ground_truth=false · allowed_for_training=false · review_only=true
