# SUSC-08 — Casebook de Revisão Espaço-Temporal

> O SUSC-08 não cria ground truth de enchente por patch. Aderência espacial é review-only, NÃO ocorrência confirmada. Todos os casos exigem revisão humana.

Total de casos: **13** (9 espaciais reais + contexto/lacuna).

## Resumo por força preliminar

| força | nº |
|-------|----|
| strong_candidate | 0 |
| moderate_candidate | 3 |
| weak_contextual | 9 |
| blocked_conflict | 0 |
| insufficient | 1 |

## CASE_001 — recife_00019 (recife) `[weak_contextual]`

- **Relação espacial:** bbox_overlap · **fonte:** patch_boundary_REC_00019_from_lineage.geojson
- **Geometria:** polygon · **coord_id:** COORD_0001 · **data:** unknown
- **Proxy suscetibilidade:** 0.5605429800724637
- **Físicas:** slope_mean=0.8168; elevation_mean=5.5737; hand_mean=3.0008; tpi_250m_mean=-0.0177
- **Hidrológicas:** distance_to_water_mean=2226.8878; twi_mean=116.4579; flow_acc_log_mean=0.9056; water_occurrence_patch=0.0001
- **Espectrais:** ndbi_mean=-0.2493; mndwi_mean=-0.4194; ndvi_mean=0.6242
- **Confiança:** coord=traceable_review_only · temporal=low · espacial=low
- **Vínculo evidência↔patch:** regional_manual · **conflitos:** geometry_is_patch_self_boundary
- **Uso recomendado:** context_only · **requires_human_review:** true
- **Limitações:** Aderencia espacial review-only; NAO e ocorrencia confirmada por patch. Fonte e a propria fronteira do patch (quase circular). 
- **Governança:** ground_truth=false · training=false · review_only=true

## CASE_002 — recife_00276 (recife) `[moderate_candidate]`

- **Relação espacial:** near_patch_buffer_candidate · **fonte:** recife_defesa_civil_risk_locations.geojson
- **Geometria:** point_set · **coord_id:** COORD_0003 · **data:** unknown
- **Proxy suscetibilidade:** 0.3720974184782609
- **Físicas:** slope_mean=6.6368; elevation_mean=46.8888; hand_mean=24.3838; tpi_250m_mean=0.4066
- **Hidrológicas:** distance_to_water_mean=3705.5576; twi_mean=9.808; flow_acc_log_mean=0.8001; water_occurrence_patch=0.0
- **Espectrais:** ndbi_mean=0.132; mndwi_mean=-0.4454; ndvi_mean=0.3049
- **Confiança:** coord=traceable_review_only · temporal=low · espacial=medium
- **Vínculo evidência↔patch:** regional_manual · **conflitos:** geometry_source_disagreement
- **Uso recomendado:** tcc_example_with_caution · **requires_human_review:** true
- **Limitações:** Aderencia espacial review-only; NAO e ocorrencia confirmada por patch. Ponto de risco != footprint de evento. 
- **Governança:** ground_truth=false · training=false · review_only=true

## CASE_003 — recife_00276 (recife) `[moderate_candidate]`

- **Relação espacial:** near_patch_buffer_candidate · **fonte:** recife_defesa_civil_risk_areas_geojson.geojson
- **Geometria:** point_set · **coord_id:** COORD_0004 · **data:** unknown
- **Proxy suscetibilidade:** 0.3720974184782609
- **Físicas:** slope_mean=6.6368; elevation_mean=46.8888; hand_mean=24.3838; tpi_250m_mean=0.4066
- **Hidrológicas:** distance_to_water_mean=3705.5576; twi_mean=9.808; flow_acc_log_mean=0.8001; water_occurrence_patch=0.0
- **Espectrais:** ndbi_mean=0.132; mndwi_mean=-0.4454; ndvi_mean=0.3049
- **Confiança:** coord=traceable_review_only · temporal=low · espacial=medium
- **Vínculo evidência↔patch:** regional_manual · **conflitos:** geometry_source_disagreement
- **Uso recomendado:** tcc_example_with_caution · **requires_human_review:** true
- **Limitações:** Aderencia espacial review-only; NAO e ocorrencia confirmada por patch. Ponto de risco != footprint de evento. 
- **Governança:** ground_truth=false · training=false · review_only=true

## CASE_004 — recife_00229 (recife) `[weak_contextual]`

- **Relação espacial:** bbox_overlap · **fonte:** recife_digitization_aoi_context.geojson
- **Geometria:** polygon · **coord_id:** COORD_0005 · **data:** unknown
- **Proxy suscetibilidade:** 0.6861477807971015
- **Físicas:** slope_mean=4.2009; elevation_mean=15.9321; hand_mean=10.1962; tpi_250m_mean=-0.1978
- **Hidrológicas:** distance_to_water_mean=1945.0542; twi_mean=53.8115; flow_acc_log_mean=1.9022; water_occurrence_patch=0.0004
- **Espectrais:** ndbi_mean=-0.0618; mndwi_mean=-0.4429; ndvi_mean=0.4768
- **Confiança:** coord=traceable_review_only · temporal=low · espacial=low
- **Vínculo evidência↔patch:** regional_manual · **conflitos:** geometry_is_context_not_event
- **Uso recomendado:** context_only · **requires_human_review:** true
- **Limitações:** Aderencia espacial review-only; NAO e ocorrencia confirmada por patch. Geometria e contexto/AOI, nao footprint de evento. 
- **Governança:** ground_truth=false · training=false · review_only=true

## CASE_005 — recife_00276 (recife) `[weak_contextual]`

- **Relação espacial:** bbox_overlap · **fonte:** recife_digitization_aoi_context.geojson
- **Geometria:** polygon · **coord_id:** COORD_0005 · **data:** unknown
- **Proxy suscetibilidade:** 0.3720974184782609
- **Físicas:** slope_mean=6.6368; elevation_mean=46.8888; hand_mean=24.3838; tpi_250m_mean=0.4066
- **Hidrológicas:** distance_to_water_mean=3705.5576; twi_mean=9.808; flow_acc_log_mean=0.8001; water_occurrence_patch=0.0
- **Espectrais:** ndbi_mean=0.132; mndwi_mean=-0.4454; ndvi_mean=0.3049
- **Confiança:** coord=traceable_review_only · temporal=low · espacial=low
- **Vínculo evidência↔patch:** regional_manual · **conflitos:** geometry_is_context_not_event
- **Uso recomendado:** context_only · **requires_human_review:** true
- **Limitações:** Aderencia espacial review-only; NAO e ocorrencia confirmada por patch. Geometria e contexto/AOI, nao footprint de evento. 
- **Governança:** ground_truth=false · training=false · review_only=true

## CASE_006 — recife_00299 (recife) `[weak_contextual]`

- **Relação espacial:** bbox_overlap · **fonte:** recife_digitization_aoi_context.geojson
- **Geometria:** polygon · **coord_id:** COORD_0005 · **data:** unknown
- **Proxy suscetibilidade:** 0.3617568387681159
- **Físicas:** slope_mean=5.9776; elevation_mean=63.2415; hand_mean=36.2013; tpi_250m_mean=0.536
- **Hidrológicas:** distance_to_water_mean=4008.3594; twi_mean=13.4138; flow_acc_log_mean=0.8184; water_occurrence_patch=0.0
- **Espectrais:** ndbi_mean=0.162; mndwi_mean=-0.4292; ndvi_mean=0.2531
- **Confiança:** coord=traceable_review_only · temporal=low · espacial=low
- **Vínculo evidência↔patch:** regional_manual · **conflitos:** geometry_is_context_not_event
- **Uso recomendado:** context_only · **requires_human_review:** true
- **Limitações:** Aderencia espacial review-only; NAO e ocorrencia confirmada por patch. Geometria e contexto/AOI, nao footprint de evento. 
- **Governança:** ground_truth=false · training=false · review_only=true

## CASE_007 — recife_00322 (recife) `[weak_contextual]`

- **Relação espacial:** bbox_overlap · **fonte:** recife_digitization_aoi_context.geojson
- **Geometria:** polygon · **coord_id:** COORD_0005 · **data:** unknown
- **Proxy suscetibilidade:** 0.3993700634057971
- **Físicas:** slope_mean=4.5792; elevation_mean=44.5086; hand_mean=30.4205; tpi_250m_mean=0.4796
- **Hidrológicas:** distance_to_water_mean=3524.3636; twi_mean=20.1976; flow_acc_log_mean=0.879; water_occurrence_patch=0.0
- **Espectrais:** ndbi_mean=0.0728; mndwi_mean=-0.4324; ndvi_mean=0.3426
- **Confiança:** coord=traceable_review_only · temporal=low · espacial=low
- **Vínculo evidência↔patch:** regional_manual · **conflitos:** geometry_is_context_not_event
- **Uso recomendado:** context_only · **requires_human_review:** true
- **Limitações:** Aderencia espacial review-only; NAO e ocorrencia confirmada por patch. Geometria e contexto/AOI, nao footprint de evento. 
- **Governança:** ground_truth=false · training=false · review_only=true

## CASE_008 — recife_00276 (recife) `[weak_contextual]`

- **Relação espacial:** near_patch_buffer_candidate · **fonte:** recife_risk_areas_context.geojson
- **Geometria:** point_set · **coord_id:** COORD_0006 · **data:** unknown
- **Proxy suscetibilidade:** 0.3720974184782609
- **Físicas:** slope_mean=6.6368; elevation_mean=46.8888; hand_mean=24.3838; tpi_250m_mean=0.4066
- **Hidrológicas:** distance_to_water_mean=3705.5576; twi_mean=9.808; flow_acc_log_mean=0.8001; water_occurrence_patch=0.0
- **Espectrais:** ndbi_mean=0.132; mndwi_mean=-0.4454; ndvi_mean=0.3049
- **Confiança:** coord=traceable_review_only · temporal=low · espacial=low
- **Vínculo evidência↔patch:** regional_manual · **conflitos:** geometry_is_context_not_event
- **Uso recomendado:** context_only · **requires_human_review:** true
- **Limitações:** Aderencia espacial review-only; NAO e ocorrencia confirmada por patch. Ponto de risco != footprint de evento. Geometria e contexto/AOI, nao footprint de evento. 
- **Governança:** ground_truth=false · training=false · review_only=true

## CASE_009 — petropolis_00467 (petropolis) `[moderate_candidate]`

- **Relação espacial:** near_patch_buffer_candidate · **fonte:** official_coordinate_recovery_hardened_registry.csv
- **Geometria:** point_set · **coord_id:** COORD_0009 · **data:** unknown
- **Proxy suscetibilidade:** 0.6880287043189368
- **Físicas:** slope_mean=15.2236; elevation_mean=931.3329; hand_mean=45.4975; tpi_250m_mean=-5.4458
- **Hidrológicas:** distance_to_water_mean=4360.0813; twi_mean=11.4739; flow_acc_log_mean=1.819; water_occurrence_patch=0.0
- **Espectrais:** ndbi_mean=-0.106; mndwi_mean=-0.5349; ndvi_mean=0.6035
- **Confiança:** coord=traceable_review_only · temporal=low · espacial=medium
- **Vínculo evidência↔patch:** regional_manual · **conflitos:** missing_or_weak_temporal_link
- **Uso recomendado:** tcc_example_with_caution · **requires_human_review:** true
- **Limitações:** Aderencia espacial review-only; NAO e ocorrencia confirmada por patch. 
- **Governança:** ground_truth=false · training=false · review_only=true

## CASE_010 — REGION_LEVEL_petropolis (petropolis) `[weak_contextual]`

- **Relação espacial:** same_region_period · **fonte:** SUSC-07A regional/documentary
- **Geometria:**  · **coord_id:**  · **data:** dated
- **Proxy suscetibilidade:** 
- **Confiança:** coord=none · temporal=medium · espacial=region_only
- **Vínculo evidência↔patch:** regional · **conflitos:** none
- **Uso recomendado:** context_only · **requires_human_review:** true
- **Limitações:** Associacao regional datada (Petropolis 2022); contexto, nao overlap de patch.
- **Governança:** ground_truth=false · training=false · review_only=true

## CASE_011 — REGION_LEVEL_petropolis (petropolis) `[weak_contextual]`

- **Relação espacial:** same_region_period · **fonte:** SUSC-07A regional/documentary
- **Geometria:**  · **coord_id:**  · **data:** dated
- **Proxy suscetibilidade:** 
- **Confiança:** coord=none · temporal=medium · espacial=region_only
- **Vínculo evidência↔patch:** regional · **conflitos:** none
- **Uso recomendado:** context_only · **requires_human_review:** true
- **Limitações:** Associacao regional datada (Petropolis 2022); contexto, nao overlap de patch.
- **Governança:** ground_truth=false · training=false · review_only=true

## CASE_012 — REGION_LEVEL_petropolis (petropolis) `[weak_contextual]`

- **Relação espacial:** same_region_period · **fonte:** SUSC-07A regional/documentary
- **Geometria:**  · **coord_id:**  · **data:** dated
- **Proxy suscetibilidade:** 
- **Confiança:** coord=none · temporal=medium · espacial=region_only
- **Vínculo evidência↔patch:** regional · **conflitos:** none
- **Uso recomendado:** context_only · **requires_human_review:** true
- **Limitações:** Associacao regional datada (Petropolis 2022); contexto, nao overlap de patch.
- **Governança:** ground_truth=false · training=false · review_only=true

## CASE_013 — REGION_LEVEL_curitiba (curitiba) `[insufficient]`

- **Relação espacial:** insufficient_for_patch_link · **fonte:** no_real_coordinate_extracted
- **Geometria:**  · **coord_id:**  · **data:** unknown
- **Proxy suscetibilidade:** 
- **Confiança:** coord=none · temporal=low · espacial=none
- **Vínculo evidência↔patch:** none · **conflitos:** none
- **Uso recomendado:** acquire_geometry_first · **requires_human_review:** true
- **Limitações:** Curitiba sem coordenada real extraida; lacuna de aquisicao (GeoCuritiba/IPPUC).
- **Governança:** ground_truth=false · training=false · review_only=true
