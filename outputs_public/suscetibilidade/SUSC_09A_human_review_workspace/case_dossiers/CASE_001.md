# CASE_001 — recife_00019 (recife)

> AVISO: aderência espacial review-only. NÃO é ground truth de enchente por patch. Mesmo aprovação humana não cria ground truth automaticamente.

- **Região:** recife
- **Patch:** recife_00019
- **Relação espacial:** bbox_overlap
- **evidence_id:** REGION_LEVEL_recife
- **coordinate_id:** COORD_0001
- **Fonte:** patch_boundary_REC_00019_from_lineage.geojson  (datasets/external_sources/recife_minimal_tp/derived/patch_boundary_REC_00019_from_lineage.geojson)
- **Geometria:** polygon
- **Data/período:** unknown
- **Score/proxy:** 0.5605429800724637
- **Features físicas:** slope_mean=0.8168; elevation_mean=5.5737; hand_mean=3.0008; tpi_250m_mean=-0.0177
- **Features hidrológicas:** distance_to_water_mean=2226.8878; twi_mean=116.4579; flow_acc_log_mean=0.9056; water_occurrence_patch=0.0001
- **Features espectrais:** ndbi_mean=-0.2493; mndwi_mean=-0.4194; ndvi_mean=0.6242
- **Interpretação:** aderência espacial review-only (bbox_overlap).
- **Limitações:** Aderencia espacial review-only; NAO e ocorrencia confirmada por patch. Fonte e a propria fronteira do patch (quase circular). 
- **Conflito conhecido:** geometry_is_patch_self_boundary
- **machine_pre_review:** `context_only`
- **Mapa:** `maps_svg/CASE_001.svg`

## Perguntas para o revisor humano

1) fonte oficial/tecnica/derivada? 2) coordenada e evento/setor de risco/estacao/poligono candidato/contexto? 3) data compativel? 4) geometria dentro/proxima do patch? 5) proxy alto? 6) features fisicas explicam? 7) conflito entre fontes? 8) uso forte/cautela/contexto? 9) deve ficar bloqueado? 10) respeita review-only?

## Campos a preencher (no form)

source_verified, geometry_verified, temporal_match_verified, patch_relation_verified, conflict_checked, approved_for_tcc_example, approved_for_score_v6_context, approved_for_ground_truth(=false), final_case_strength, review_notes

> Governança: can_be_ground_truth=false · allowed_for_training=false · review_only=true
