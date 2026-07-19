# SUSC-17C5 Geometry-to-Patch Linkage Resolver

## Estado inicial herdado do 17C

- Branch: `marco/reavaliacao-pos-mapbiomas-sensibilidade-territorial`
- HEAD: `f94c0c6`
- Area staged: 0 arquivo(s)
- Candidatas 17C: 63
- Eventos 17C datados: 26
- Linhas 17C com geometria forte candidata: 11
- `score_v6` alterado: False
- `score_v7` criado: False

## Fontes e registries usados

- `outputs_public/data/susc_17c_strong_reference_acquisition_canary/susc_17c_source_target_pack.csv`: 63 (17c_target_pack)
- `outputs_public/data/susc_17c_strong_reference_acquisition_canary/susc_17c_strong_reference_acquisition_summary.json`: not_csv (17c_summary)
- `outputs_public/data/susc_17c_strong_reference_acquisition_canary/reference_canary_qa_queue.csv`: 5 (17c_qa_queue)
- `outputs_public/suscetibilidade/susc_17c4_candidate_geometries.geojson`: 1 (charter_17c4_geometry)
- `outputs_public/suscetibilidade/susc_17c4_extracted_reference_candidates.csv`: 1 (charter_17c4_refs)
- `outputs_public/suscetibilidade/susc_17c6_candidate_patch_grid.csv`: 5 (candidate_patch_grid_17c6)
- `outputs_public/suscetibilidade/susc_17c6_candidate_patch_grid.geojson`: 5 (candidate_patch_geojson_17c6)
- `datasets/suscetibilidade/susc_features_by_patch_v1.csv`: 300 (official_patch_features)
- `outputs_public/tables/revp_patch_event_spatial_binding_v2ec.csv`: 53 (spatial_binding_v2ec)
- `datasets/event_patch_linkage_registry.csv`: 9 (event_patch_linkage_registry)
- `datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv`: 300 (score_v6_official)
- `.`: not_applicable (git_initial_state)

## Geometrias

- Geometrias normalizadas: 63
- Geometrias resolvidas com objeto real: 1
- Geometrias unresolved/bloqueadas: 62
- Blockers registrados: 62

## Patch-links

- Patch-links gerados: 67
- Strong link candidates: 5
- Strong link candidates em patch oficial: 0
- Entradas em QA: 5
- Eligible for evaluation: 5
- Eligible for training: 0
- Ground truth: 0
- score_v7_allowed=true: 0

### Links por classe

| item | count |
|---|---|
| exact_polygon_overlap | 5 |
| same_region_only | 49 |
| unresolved_geometry | 13 |

## Decisao 17B

17B continua **bloqueado**. A sprint criou links auditaveis para QA humana, mas todos permanecem review-only; nao ha QA accepted automatico, nao ha patch-link oficial aceito, nao ha ground truth, nao ha treino e nao ha score_v7.

O desbloqueio e parcial apenas para o proximo passo operacional: `17D Human QA` pode revisar os links candidatos em `reference_patch_link_qa_queue.csv`.

## Proximos passos para SUSC-17D Human QA

1. Revisar visualmente cada item de `reference_patch_link_qa_queue.csv`.
2. Confirmar se o patch canario 17C6 e aceitavel como patch de revisao, sem mistura com patch oficial.
3. Para PET, adquirir coordenadas/CRS reais dos pontos oficiais antes de qualquer link forte.
4. Para qualquer `same_region_only`, manter bloqueado ate existir geometria real.
5. Manter `ground_truth=false`, `eligible_for_training=false` e `score_v7_allowed=false`.
