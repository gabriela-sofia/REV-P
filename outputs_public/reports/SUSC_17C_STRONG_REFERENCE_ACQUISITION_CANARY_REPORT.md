# SUSC-17C Strong Reference Acquisition Canary

## Estado inicial

- Branch: `marco/reavaliacao-pos-mapbiomas-sensibilidade-territorial`
- HEAD: `5270e87`
- Area staged: 0 arquivo(s)
- Worktree fora dos outputs 17C desta sprint: 293 caminho(s)
- `score_v6` alterado: False
- `score_v7` criado: False

## Fontes e registries encontrados

- `outputs_public/tables/revp_observed_event_registry_v2dz.csv`: 53 linhas (observed_event_registry)
- `outputs_public/tables/revp_patch_event_spatial_binding_v2ec.csv`: 53 linhas (patch_event_spatial_binding)
- `outputs_public/tables/revp_patch_event_temporal_alignment_v2eb.csv`: 53 linhas (patch_event_temporal_alignment)
- `datasets/observed_event_reference_candidate_registry.csv`: 9 linhas (observed_event_reference_candidates)
- `datasets/official_observed_event_vector_registry.csv`: 6 linhas (official_observed_event_vector_registry)
- `datasets/consolidated_observed_event_vector_candidate_registry.csv`: 18 linhas (consolidated_vector_candidate_registry)
- `datasets/event_patch_linkage_registry.csv`: 9 linhas (event_patch_linkage_registry)
- `datasets/c4_s1_completion_queue.csv`: 8 linhas (c4_s1_completion_queue)
- `datasets/ground_reference_evidence_source_registry.csv`: 8 linhas (evidence_source_registry)
- `outputs_public/suscetibilidade/susc_17c3_official_source_acquisition_targets.csv`: 9 linhas (susc_17c3_source_targets)
- `outputs_public/suscetibilidade/susc_17c31_sar_metadata_feasibility.csv`: 1 linhas (susc_17c31_sar_metadata)
- `outputs_public/suscetibilidade/susc_17c31_technical_footprint_candidate_registry.csv`: 1 linhas (susc_17c31_technical_footprint)
- `outputs_public/suscetibilidade/susc_17c31_ground_reference_readiness_evaluation.csv`: 42 linhas (susc_17c31_ground_reference_readiness)
- `outputs_public/suscetibilidade/susc_17c33_event_anchored_canary_patch_registry.csv`: 11 linhas (susc_17c33_canary_patches)
- `datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv`: 300 linhas (score_v6_official)

## Contagens de candidatos

Total de linhas candidatas: 63
Eventos unicos: 52

### Por cidade

| item | count |
|---|---|
| Curitiba | 11 |
| Petropolis | 30 |
| Recife | 9 |
| UNKNOWN | 13 |

### Por regiao

| item | count |
|---|---|
| CUR | 11 |
| PET | 30 |
| REC | 9 |
| UNKNOWN | 13 |

### Por fonte/classe

| item | count |
|---|---|
| administrative_disaster_record | 14 |
| documentary_context | 3 |
| insufficient | 33 |
| official_address_resolved | 2 |
| official_observed_event_point | 10 |
| technical_remote_sensing_flood_footprint | 1 |

### Por forca documental

| item | count |
|---|---|
| documentary_only | 3 |
| insufficient | 34 |
| official_context_pending_geometry | 16 |
| strong_candidate_pending_qa | 10 |

## Resolucao temporal, geometria e SAR

- Eventos unicos com data resolvida e janelas pre/post: 26
- Linhas com geometria forte candidata: 11
- Linhas com SAR factivel por metadado/manifest: 1
- Linhas elegiveis para avaliacao review-only: 10
- Linhas elegiveis para calibracao: 0
- Linhas elegiveis para treino: 0
- Linhas ground truth: 0
- Linhas com score_v7_allowed=true: 0

## Eventos priorizados para canario

- 1. `REC_2022_05_24_30` / REC / technical_remote_sensing_flood_footprint / score 120 / data resolvida; classe fonte forte; geometria ou bbox candidata; SAR metadata factivel; patch-link possivel mas nao aceito
- 2. `EVENT_PET2022_CPRM_ANEXOVII_24022022` / PET / official_observed_event_point / score 117 / data resolvida; classe fonte forte; geometria ou bbox candidata; patch-link possivel mas nao aceito
- 3. `EVENT_PET2022_CPRM_ANEXOIII_20022022` / PET / official_observed_event_point / score 104 / data resolvida; classe fonte forte; geometria ou bbox candidata; patch-link possivel mas nao aceito
- 4. `EVENT_PET2022_CPRM_ANEXOII_19022022` / PET / official_observed_event_point / score 104 / data resolvida; classe fonte forte; geometria ou bbox candidata; patch-link possivel mas nao aceito
- 5. `EVENT_PET2022_CPRM_ANEXOIV_22022022` / PET / official_observed_event_point / score 104 / data resolvida; classe fonte forte; geometria ou bbox candidata; patch-link possivel mas nao aceito

## Criterios minimos

- minimo 3 eventos datados: True (26)
- minimo 2 regioes priorizadas: True (PET, REC)
- minimo 1 footprint tecnico/oficial por regiao prioritaria: False
- minimo 20 patch-links fortes: False (linkage forte=1; intersecoes confirmadas=0)

## Blockers reais

- patch-links fortes insuficientes: 1 linkage interno forte e 0 intersecoes espaciais confirmadas
- footprint tecnico/oficial por regiao prioritaria ainda incompleto
- QA humana 17D pendente; nenhuma linha accepted
- score_v7, treino supervisionado e ground truth continuam proibidos

## Decisao 17B

17B permanece **bloqueado**. A sprint destrava uma fila auditavel de aquisicao forte e alguns candidatos review-only, mas ainda nao ha 20 patch-links fortes confirmados, nao ha footprint tecnico/oficial completo por regiao prioritaria, e nenhuma linha foi aceita por QA humana.

## Proximos passos 17D Human QA

1. Revisar manualmente os 3-5 eventos de `reference_canary_qa_queue.csv`.
2. Confirmar CRS, geometria observada e intersecao patch-evento antes de qualquer aceite.
3. Para Recife, transformar a viabilidade SAR em artefato tecnico sob politica explicita, sem publicar raster pesado.
4. Para Petropolis/Curitiba, completar metadado Sentinel-1 pre/post ou registrar bloqueio por ausencia.
5. Manter `ground_truth=false`, `eligible_for_training=false` e `score_v7_allowed=false` ate nova decisao metodologica.
