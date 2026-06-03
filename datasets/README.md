# Datasets auditÃ¡veis do REV-P

## O que este diretÃ³rio documenta

Este diretÃ³rio registra os datasets e corpora produzidos ou utilizados pelo REV-P como
evidÃªncia cientÃ­fica auditÃ¡vel. Ele nÃ£o contÃ©m dados brutos â€” contÃ©m registros
estruturados que descrevem o que existe, onde estÃ¡, como foi produzido e quais sÃ£o as
suas limitaÃ§Ãµes.

## Quatro categorias de material

**Dataset pÃºblico:** manifest ou registro commitado neste repositÃ³rio. AcessÃ­vel a
qualquer leitura, sem dependÃªncia de ambiente local.

**Registro auditÃ¡vel:** tabela que descreve um corpus local sem replicar os arquivos
pesados. Prova que o corpus existe e como foi construÃ­do, sem exigir que o repositÃ³rio
hospede os rasters.

**Dado local:** arquivo que existe apenas no workspace privado (rasters Sentinel,
embeddings `.npz`, shapefiles brutos). Referenciado pelos manifests pÃºblicos, mas nÃ£o
versionado.

**Artefato pesado:** dado que nÃ£o pode ou nÃ£o deve ser versionado por tamanho, por
conteÃºdo sensÃ­vel ou por ser reproduzÃ­vel a partir dos scripts e manifests pÃºblicos.

## Por que o GitHub publica rastreabilidade, nÃ£o rasters

Os GeoTIFFs Sentinel originais tÃªm entre 10 MB e 200 MB por arquivo. O corpus de 128
patches totaliza mÃºltiplos gigabytes. Versionar esses arquivos incharia o repositÃ³rio
sem benefÃ­cio cientÃ­fico: os patches sÃ£o gerados a partir de imagens Sentinel-2 Level-2A
de acesso pÃºblico, e a metodologia de derivaÃ§Ã£o estÃ¡ documentada nos manifests.

O que prova a legitimidade cientÃ­fica do corpus nÃ£o Ã© a presenÃ§a dos rasters â€” Ã© a
rastreabilidade da cadeia: qual imagem Sentinel originou cada patch, qual preflight foi
executado, qual QA foi aprovado antes da extraÃ§Ã£o de embeddings.

## Arquivos neste diretÃ³rio

| Arquivo | ConteÃºdo |
|---|---|
| `dataset_registry.csv` | Registro geral de datasets e corpora do projeto |
| `patch_corpus_registry.csv` | Registro dos corpora de patches Sentinel por estÃ¡gio |
| `patch_corpus_taxonomy_registry.csv` | Taxonomia v1iw que distingue corpus territorial consolidado (59 patches) do manifesto Sentinel candidato (128 assets) |
| `external_evidence_registry.csv` | Registro das evidÃªncias GIS externas por regiÃ£o |
| `contextual_reference_layer_registry.csv` | Camada de referÃªncia contextual validada: status de evidÃªncia e claims permitidos/proibidos por patch |
| `ground_reference_evidence_source_registry.csv` | InventÃ¡rio de fontes de referÃªncia categorizado pelo Protocolo C: tipo, grau de observaÃ§Ã£o, allowed_use, forbidden_use |
| `schemas/dataset_registry_schema.csv` | Schema de campos de dataset_registry.csv |
| `schemas/patch_corpus_schema.csv` | Schema de campos de patch_corpus_registry.csv |
| `schemas/patch_corpus_taxonomy_schema.csv` | Schema de campos de patch_corpus_taxonomy_registry.csv |
| `schemas/external_evidence_schema.csv` | Schema de campos de external_evidence_registry.csv |
| `schemas/contextual_reference_layer_schema.csv` | Schema de campos de contextual_reference_layer_registry.csv |
| `schemas/ground_reference_evidence_source_schema.csv` | Schema de campos de ground_reference_evidence_source_registry.csv |
| `flood_event_candidate_registry.csv` | Registro de eventos de inundaÃ§Ã£o candidatos por regiÃ£o â€” status de confirmaÃ§Ã£o, elegibilidade e bloqueadores (etapa de aquisiÃ§Ã£o Protocolo C) |
| `patch_event_reference_link_registry.csv` | VÃ­nculos patch-evento-fonte com alinhamento temporal/espacial, candidatura e claims permitidos/proibidos (etapa de aquisiÃ§Ã£o Protocolo C) |
| `schemas/flood_event_candidate_schema.csv` | Schema de campos de flood_event_candidate_registry.csv |
| `schemas/patch_event_reference_link_schema.csv` | Schema de campos de patch_event_reference_link_registry.csv |
| `ground_reference_gap_matrix.csv` | Matriz de lacunas de evidÃªncia por regiÃ£o: gates abertos, evidÃªncia faltante, risco metodolÃ³gico e prÃ³ximos passos permitidos/proibidos (etapa de fechamento Protocolo C) |
| `review_gate_reference_registry.csv` | Registry de revisÃµes humanas ou placeholders: decisÃ£o, materiais revisados, consistency checks, allowed_claim e forbidden_claim por revisÃ£o (etapa de fechamento Protocolo C) |
| `reference_promotion_decision_registry.csv` | Registry de decisÃµes formais de promoÃ§Ã£o: gates satisfeitos/falhados, final_reference_status, protocol_b_reassessment_allowed (etapa de fechamento Protocolo C) |
| `schemas/ground_reference_gap_matrix_schema.csv` | Schema de campos de ground_reference_gap_matrix.csv |
| `schemas/human_reference_review_schema.csv` | Schema de campos de review_gate_reference_registry.csv |
| `schemas/reference_promotion_decision_schema.csv` | Schema de campos de reference_promotion_decision_registry.csv |
| `observational_evidence_acquisition_plan.csv` | Plano de aquisiÃ§Ã£o de evidÃªncias observacionais por regiÃ£o (v1hl): fontes-alvo, prioridades, forÃ§a metodolÃ³gica, gates relacionados e acesso esperado |
| `schemas/observational_evidence_acquisition_plan_schema.csv` | Schema de campos de observational_evidence_acquisition_plan.csv |
| `regional_ground_reference_readiness.csv` | ProntidÃ£o regional para ground reference (v1hl): status de gates por regiÃ£o, evidÃªncia mais forte, lacunas crÃ­ticas, risco metodolÃ³gico e allowed/forbidden claims |
| `schemas/regional_ground_reference_readiness_schema.csv` | Schema de campos de regional_ground_reference_readiness.csv |
| `evidence_acquisition_tracker.csv` | Tracker de aquisiÃ§Ã£o (v1hm): estado atual de cada fonte-alvo por regiÃ£o â€” acquisition_status, license_status, current_blocker, next_action e forbidden_use |
| `schemas/evidence_acquisition_tracker_schema.csv` | Schema de campos de evidence_acquisition_tracker.csv |
| `evidence_source_intake_registry.csv` | Intake registry (v1hm): fontes acessadas ou em processo â€” event_link_status, intake_decision, blocked_reason, allowed_use e forbidden_use por entrada |
| `schemas/evidence_source_intake_schema.csv` | Schema de campos de evidence_source_intake_registry.csv |
| `evidence_license_provenance_registry.csv` | Registry de licenÃ§a e proveniÃªncia (v1hm): license_status, redistribution_status, raw_data_publication_allowed, local_only_required e use_for_operational_ground_truth_allowed por fonte |
| `schemas/evidence_license_provenance_schema.csv` | Schema de campos de evidence_license_provenance_registry.csv |
| `event_evidence_dossier_registry.csv` | DossiÃªs de evidÃªncia por evento candidato (v1ho): status do dossiÃª, lacunas de evidÃªncia mÃ­nima, decisÃ£o de continuidade e guardrails de promoÃ§Ã£o |
| `schemas/event_evidence_dossier_schema.csv` | Schema de campos de event_evidence_dossier_registry.csv |
| `event_evidence_requirements_registry.csv` | Requisitos mÃ­nimos de evidÃªncia por evento candidato (v1ho): tipo de requisito, gate alvo, status atual, blocking_if_missing e forbidden_if_missing |
| `schemas/event_evidence_requirements_schema.csv` | Schema de campos de event_evidence_requirements_registry.csv |
| `event_dossier_decision_registry.csv` | DecisÃµes de continuidade por dossiÃª (v1ho): decision_status, prÃ³ximos passos permitidos/proibidos, can_reassess_protocol_b=false e can_start_multimodal=false |
| `schemas/event_dossier_decision_schema.csv` | Schema de campos de event_dossier_decision_registry.csv |
| `event_candidate_screening_registry.csv` | Eventos candidatos por regiÃ£o (v1hn): status, prioridade de busca, gates potencialmente endereÃ§Ã¡veis e guardrails de promoÃ§Ã£o e ground truth |
| `schemas/event_candidate_screening_schema.csv` | Schema de campos de event_candidate_screening_registry.csv |
| `event_source_search_backlog.csv` | Backlog de fontes a pesquisar por evento candidato (v1hn): fonte, famÃ­lia, modo de acesso, status da busca e gates que a fonte poderia suportar |
| `schemas/event_source_search_backlog_schema.csv` | Schema de campos de event_source_search_backlog.csv |
| `event_patch_screening_scope.csv` | Escopo de triagem por patch (v1hn): quais patches do corpus estÃ£o no perÃ­metro de busca de cada evento candidato, com guardrails de sobreposiÃ§Ã£o espacial e promoÃ§Ã£o |
| `schemas/event_patch_screening_scope_schema.csv` | Schema de campos de event_patch_screening_scope.csv |
| `regional_external_search_plan.csv` | Planos de busca externa por regiÃ£o (v1hp): fonte-alvo, gate, modo, prioridade, status e forbidden_use bloqueando ground truth e labels |
| `schemas/regional_external_search_plan_schema.csv` | Schema de campos de regional_external_search_plan.csv |
| `source_request_package_registry.csv` | Pacotes de solicitaÃ§Ã£o formal a instituiÃ§Ãµes (v1hp): instituiÃ§Ã£o, tipo de solicitaÃ§Ã£o, evidÃªncia solicitada, status e cannot_establish_ground_truth_alone=true |
| `schemas/source_request_package_schema.csv` | Schema de campos de source_request_package_registry.csv |
| `gate_search_question_registry.csv` | Perguntas de busca por gate e regiÃ£o (v1hp): current_answer_status, blocking_if_unanswered e forbidden_if_unanswered |
| `schemas/gate_search_question_schema.csv` | Schema de campos de gate_search_question_registry.csv |
| `regional_request_priority_matrix.csv` | Matriz de prioridade regional de solicitaÃ§Ã£o (v1hp): evento prioritÃ¡rio por regiÃ£o, razÃ£o, prÃ³ximo gate a fechar, protocol_b_status=BLOCKED e multimodal_status=HOLD |
| `schemas/regional_request_priority_matrix_schema.csv` | Schema de campos de regional_request_priority_matrix.csv |
| `observed_event_reference_candidate_registry.csv` | 9 eventos observados candidatos (v1hq): G1/G2/G3 fechados documentalmente, G4 em triagem espacial, operational_ground_truth_status=NOT_ESTABLISHED, protocol_b_status=BLOCKED, can_be_used_as_training_label=false para todos |
| `schemas/observed_event_reference_candidate_schema.csv` | Schema de campos de observed_event_reference_candidate_registry.csv |
| `observed_event_reference_gap_registry.csv` | Lacunas metodolÃ³gicas por evento observado candidato (v1hq): o que falta para avanÃ§ar Ã  ligaÃ§Ã£o patch-evento e ground reference |
| `schemas/observed_event_reference_gap_schema.csv` | Schema de campos de observed_event_reference_gap_registry.csv |
| `observed_event_reference_decision_registry.csv` | DecisÃµes metodolÃ³gicas por evento observado candidato (v1hq): can_promote_to_ground_reference=false, can_generate_training_label=false, can_reopen_protocol_b=false para todos |
| `schemas/observed_event_reference_decision_schema.csv` | Schema de campos de observed_event_reference_decision_registry.csv |
| `manual_external_evidence_needed_registry.csv` | InventÃ¡rio de dados externos que precisam ser trazidos manualmente por regiÃ£o (v1hq): categoria, provedor, modo de aquisiÃ§Ã£o, cannot_establish_ground_truth_alone=true para todos |
| `schemas/manual_external_evidence_needed_schema.csv` | Schema de campos de manual_external_evidence_needed_registry.csv |
| `event_patch_linking_preflight_registry.csv` | Preflight de prÃ©-ligaÃ§Ã£o eventoâ€“patch (v1hr): escopo regional, status de overlay, bloqueios e guardrails â€” promotion_allowed=false, can_create_training_label=false, protocol_b_status=BLOCKED para todos |
| `schemas/event_patch_linking_preflight_schema.csv` | Schema de campos de event_patch_linking_preflight_registry.csv |
| `manual_geocoding_target_registry.csv` | Alvos de geocodificaÃ§Ã£o manual (v1hr): 22 localidades por evento â€” geocoding_status=NOT_GEOCODED ou NEEDS_MANUAL_REVIEW, requires_official_confirmation=true, cannot_establish_ground_truth_alone=true para todos |
| `schemas/manual_geocoding_target_schema.csv` | Schema de campos de manual_geocoding_target_registry.csv |
| `event_sentinel_temporal_window_registry.csv` | Janelas temporais Sentinel por evento (v1hr): perÃ­odos prÃ©/evento/pÃ³s metadata-only â€” acquisition_status=NOT_ACQUIRED, cannot_establish_ground_truth_alone=true para todos |
| `schemas/event_sentinel_temporal_window_schema.csv` | Schema de campos de event_sentinel_temporal_window_registry.csv |
| `patch_linking_dependency_registry.csv` | DependÃªncias metodolÃ³gicas para patch-linking real (v1hr): o que deve ser resolvido antes de overlay, human review e ground reference â€” current_status=OPEN, required_before_ground_reference=true para todas |
| `schemas/patch_linking_dependency_schema.csv` | Schema de campos de patch_linking_dependency_registry.csv |
| `observed_source_acquisition_manifest.csv` | Manifest pÃºblico de aquisiÃ§Ã£o v1hs: metadados de fontes pÃºblicas candidatas â€” acquisition_status, hash SHA-256 (quando adquirido), license_status, cannot_establish_ground_truth_alone=true, can_generate_training_label=false, protocol_b_status=BLOCKED, multimodal_status=HOLD para todas |
| `schemas/observed_source_acquisition_manifest_schema.csv` | Schema de campos de observed_source_acquisition_manifest.csv |
| `observed_source_acquisition_gap_registry.csv` | Lacunas de aquisiÃ§Ã£o por regiÃ£o (v1hs): gap_type, required_action, priority_level, blocks_ground_reference â€” 11 solicitaÃ§Ãµes formais e 12 buscas manuais pendentes |
| `schemas/observed_source_acquisition_gap_schema.csv` | Schema de campos de observed_source_acquisition_gap_registry.csv |
| `acquired_source_review_registry.csv` | RevisÃ£o inicial metadata-only de fontes adquiridas (v1hs): can_support_patch_linking=false, can_support_ground_reference=false, can_generate_training_label=false para todas |
| `schemas/acquired_source_review_schema.csv` | Schema de campos de acquired_source_review_registry.csv |
| `programmatic_source_review_registry.csv` | RevisÃ£o assistida de fontes por evidÃªncia textual (v1hu): forÃ§a de evidÃªncia, candidatos G1â€“G4, requires_manual_review=true, can_close_gate_automatically=false para todas |
| `schemas/assisted_source_review_schema.csv` | Schema de campos de programmatic_source_review_registry.csv |
| `programmatic_gate_support_registry.csv` | Suporte candidato por gate e fonte (v1hu): 152 linhas (38Ã—4 gates), can_close_gate_automatically=false, can_generate_training_label=false para todas |
| `schemas/assisted_gate_support_schema.csv` | Schema de campos de programmatic_gate_support_registry.csv |
| `assisted_source_evidence_gap_registry.csv` | Lacunas da revisÃ£o assistida (v1hu): REVIEW_GATE_REQUIRED e GEOMETRY_NOT_AVAILABLE para todas, blocks_ground_reference=true para lacunas crÃ­ticas |
| `schemas/assisted_source_evidence_gap_schema.csv` | Schema de campos de assisted_source_evidence_gap_registry.csv |
| `event_gate_decision_matrix.csv` | Matriz de decisÃ£o por gate por evento (v1hv): 36 linhas (9 eventos Ã— 4 gates), event_gate_status, gate_can_advance, requires_reviewer_confirmation=true, can_close_gate_automatically=false para todas |
| `schemas/event_gate_decision_matrix_schema.csv` | Schema de campos de event_gate_decision_matrix.csv |
| `event_ground_reference_readiness_registry.csv` | ProntidÃ£o para ground reference por evento (v1hv): 9 linhas, overall_readiness, ground_reference_candidate, can_promote_to_ground_reference=false, can_create_training_label=false para todas |
| `schemas/event_ground_reference_readiness_schema.csv` | Schema de campos de event_ground_reference_readiness_registry.csv |
| `event_next_action_registry.csv` | PrÃ³ximas aÃ§Ãµes por evento (v1hv): 30 aÃ§Ãµes (â‰¥3 por evento), can_automate=false para todas |
| `schemas/event_next_action_schema.csv` | Schema de campos de event_next_action_registry.csv |
| `source_event_validation_registry.csv` | ValidaÃ§Ã£o assistida por fonte (v1hw): 38 linhas, source_validation_status, event_confirmation_status, phenomenon_status, temporal_alignment_status, can_generate_training_label=false, can_support_ground_reference=false para todas |
| `schemas/source_event_validation_schema.csv` | Schema de campos de source_event_validation_registry.csv |
| `event_assisted_validation_decision_registry.csv` | DecisÃ£o assistida por evento (v1hw): 9 linhas, validation_decision, event_confirmation_status_final, can_promote_to_ground_reference=false, can_create_training_label=false para todas |
| `schemas/event_assisted_validation_decision_schema.csv` | Schema de campos de event_assisted_validation_decision_registry.csv |
| `event_patch_compatibility_precheck_registry.csv` | Precheck de compatibilidade patchâ€“evento (v1hw): 9 linhas, precheck_status, can_execute_overlay_now=false para todas |
| `schemas/event_patch_compatibility_precheck_schema.csv` | Schema de campos de event_patch_compatibility_precheck_registry.csv |
| `event_priority_for_geocoding_registry.csv` | Prioridade para geocodificaÃ§Ã£o controlada (v1hw): 9 linhas, 2 selecionados (PET_2022_02_15 e PET_2024_03_21_28), can_execute_overlay_after_geocoding=false para todas |
| `schemas/event_priority_for_geocoding_schema.csv` | Schema de campos de event_priority_for_geocoding_registry.csv |
| `targeted_missing_source_search_registry.csv` | 22 alvos de busca dirigida de lacunas (v1ht): 7 Recife, 8 PetrÃ³polis, 7 Curitiba â€” estratÃ©gia, modo, prioridade, fallback_action, cannot_establish_ground_truth_alone=true para todas |
| `schemas/targeted_missing_source_search_schema.csv` | Schema de campos de targeted_missing_source_search_registry.csv |
| `formal_source_request_target_registry.csv` | 12 solicitaÃ§Ãµes formais (v1ht + v1hx): 11 identificadas em v1ht + FR_CTB_005 (Simepar) adicionada em v1hx; instituiÃ§Ã£o, base legal, template, blocks_ground_reference, cannot_establish_ground_truth_alone=true para todas |
| `schemas/formal_source_request_target_schema.csv` | Schema de campos de formal_source_request_target_registry.csv |
| `formal_request_package_registry.csv` | 12 pacotes de pedido formal preenchidos (v1hx): um por solicitaÃ§Ã£o formal; package_status=DRAFT; paths para docs/templates/protocolo_c_solicitacoes_preenchidas/; cannot_establish_ground_truth_alone=true para todos |
| `schemas/formal_request_package_schema.csv` | Schema de campos de formal_request_package_registry.csv |
| `acquired_portal_deep_review_registry.csv` | 8 portais PORTAL_HOMEPAGE_ACQUIRED revisados em v1hy: todos PORTAL_HOMEPAGE_GENERIC / CONFIRM_CONTEXTUAL_ONLY; 1729 links extraÃ­dos, 0 event-specific; can_support_patch_linking=false; cannot_establish_ground_truth_alone=true para todos |
| `schemas/acquired_portal_deep_review_schema.csv` | Schema de 20 campos para revisÃ£o profunda de portais |
| `pre_geocoding_closure_registry.csv` | 9 eventos com registro de fechamento prÃ©-geocodificaÃ§Ã£o (v1hy): 3 selecionados (PET_2022_02_15, PET_2024_03_21_28, REC_2022_05_24_30); todos os guardrails operacionais false/BLOCKED/HOLD/SUPPORT_ONLY |
| `schemas/pre_geocoding_closure_schema.csv` | Schema de 22 campos para fechamento prÃ©-geocodificaÃ§Ã£o |
| `authoritative_spatial_source_inventory.csv` | InventÃ¡rio de fontes espaciais autoritativas (v1ia): 15 entradas por evento/localidade â€” classe, autoridade, acesso, resolves_blocker_id; `can_create_training_label=false`, `can_support_ground_reference_future=false` para todas |
| `schemas/authoritative_spatial_source_inventory_schema.csv` | Schema de 29 campos para inventÃ¡rio de fontes espaciais autoritativas |
| `controlled_geocoding_execution_preflight_registry.csv` | Preflight de execuÃ§Ã£o de geocodificaÃ§Ã£o controlada (v1ia): 13 linhas â€” 2 READY_FOR_TRIAGE_ONLY (PET_2024), 5 WAITING_OFFICIAL_SOURCE (REC), 6 WAITING_PHENOMENON_SEPARATION (PET_2022); `can_execute_overlay_after_geocoding=false`, `can_create_training_label_after_geocoding=false` para todos |
| `schemas/controlled_geocoding_execution_preflight_schema.csv` | Schema de 28 campos para preflight de execuÃ§Ã£o de geocodificaÃ§Ã£o controlada |
| `authoritative_spatial_gap_registry.csv` | Lacunas de fonte autoritativa (v1ia): 6 entradas â€” 1 CRITICAL (separaÃ§Ã£o fenÃ´meno PET_2022, bloqueia geocodificaÃ§Ã£o), 4 HIGH, 1 MEDIUM; `blocks_training_label=true` invariante para todas |
| `schemas/authoritative_spatial_gap_schema.csv` | Schema de 22 campos para lacunas de fonte autoritativa |
| `observational_reference_promotion_registry.csv` | PromoÃ§Ã£o graduada por evento candidato (v1ib): 9 linhas â€” 1 LEVEL_5 (PET_2022_02_15), 1 LEVEL_6 (PET_2024_03_21_28), 2 HOLD, 3 contextuais, 2 bloqueadas; `can_be_called_ground_truth_operational=false`, `can_create_training_label=false`, `can_reopen_protocol_b=false` invariantes |
| `protocolo_c_event_evidence_level_matrix.csv` | Matriz G1â€“G7 de evidÃªncia por evento (v1ib): 9 linhas; resume estado de cada gate por evento â€” confirmaÃ§Ã£o, fonte, temporal, localidade, fenÃ´meno, suporte espacial, prontidÃ£o geocodificaÃ§Ã£o |
| `schemas/observational_reference_promotion_schema.csv` | Schema de 26 campos para promoÃ§Ã£o graduada de referÃªncia observacional candidata (v1ib) |
| `event_locality_phenomenon_separation_registry.csv` | SeparaÃ§Ã£o fenomenolÃ³gica por localidade (v1ic): 8 linhas para PET_2022_02_15; `phenomenon_class`, `phenomenon_confidence`, `blocks_controlled_geocoding=true` para todas; ChÃ¡cara Flora e Caxambu MASS_MOVEMENT_CONFIRMED (HIGH); `can_create_training_label=false`, `multimodal_status=HOLD` invariantes |
| `event_phenomenon_separation_decision_registry.csv` | DecisÃ£o de separaÃ§Ã£o fenomenolÃ³gica (v1ic): 1 linha para PET_2022_02_15; `phenomenon_separation_status=PARTIAL_SEPARATION`; `can_advance_to_controlled_geocoding_future=false`; `required_next_action=obter PKG_FR_PET_001`; `forbidden_claim` explÃ­cito |
| `schemas/event_locality_phenomenon_separation_schema.csv` | Schema de 29 campos para separaÃ§Ã£o fenomenolÃ³gica por localidade (v1ic) |
| `schemas/event_phenomenon_separation_decision_schema.csv` | Schema de 21 campos para decisÃ£o de separaÃ§Ã£o fenomenolÃ³gica (v1ic) |
| `observed_reference_source_package_registry.csv` | Pacotes de referÃªncia cartogrÃ¡fica candidatos (v1id): 1 linha; PKG_FR_PET_001 (DRM-RJ completo) registrado como REQUIRED_NOT_INGESTED; `operational_ground_truth_status=BLOCKED` invariante |
| `schemas/observed_reference_source_package_schema.csv` | Schema de 21 campos para registry de pacotes de referÃªncia cartogrÃ¡fica (v1id) |
| `ground_reference_evidence_registry.csv` | Auditoria de ground reference (v1ie): 10 linhas; candidatos SGB/CPRM+FBDS auditados com pyshp; suscetibilidade e feiÃ§Ãµes de deslizamento histÃ³ricas sem vÃ­nculo temporal ao evento 2022-02-15; Gate 6 FAIL para todos; `operational_ground_truth_status=BLOCKED`, `ml_label_status=BLOCKED_UNTIL_SPLIT_AND_LEAKAGE_PROTOCOL` invariantes |
| `schemas/ground_reference_evidence_registry_schema.csv` | Schema de 24 campos para auditoria de evidÃªncia de ground reference (v1ie) |
| `official_observed_event_vector_registry.csv` | AquisiÃ§Ã£o e auditoria de vetores oficiais observados (v1if): 6 linhas; ZIP SGB/CPRM baixado (20.9MB) â€” 11 PDFs extraÃ­dos, 0 vetores; todos BLOCKED pelos 11 gates; `ml_label_status=BLOCKED_UNTIL_SPLIT_AND_LEAKAGE_PROTOCOL` invariante; 4 instituiÃ§Ãµes pendentes de solicitaÃ§Ã£o formal |
| `schemas/official_observed_event_vector_registry_schema.csv` | Schema de 32 campos para registry de vetores observados oficiais (v1if) |
=======
| `official_ground_truth_request_registry.csv` | Rastreamento de solicitaÃ§Ãµes institucionais de ground truth (v1ig): 5 linhas â€” REQ_SGB_PET_001, REQ_DRM_PET_001, REQ_DC_PET_001, REQ_INPE_PET_001, REQ_SEDEC_001; todas NOT_YET_SUBMITTED; `acceptance_status=PENDING_RESPONSE`, `ground_truth_relevance=UNKNOWN_PENDING_RESPONSE` para todas; atualizaÃ§Ã£o manual pelo pesquisador apÃ³s cada interaÃ§Ã£o institucional |
| `schemas/official_ground_truth_request_registry_schema.csv` | Schema de 24 campos para registry de rastreamento de solicitaÃ§Ãµes institucionais (v1ig) |
| `official_ground_truth_response_acceptance_criteria.csv` | 13 critÃ©rios de aceitaÃ§Ã£o de respostas institucionais (v1ig): cobertura: origem, canal, geometria, CRS, data, fenÃ´meno, separaÃ§Ã£o hidrolÃ³gico/geolÃ³gico, dicionÃ¡rio, metadados, licenÃ§a, escala, auditabilidade, ausÃªncia de ambiguidade risco/suscetibilidade; 6 critÃ©rios REJECTED, 7 HOLD_PENDING_CLARIFICATION |
| `schemas/official_ground_truth_response_acceptance_criteria_schema.csv` | Schema de 7 campos para critÃ©rios de aceitaÃ§Ã£o de respostas institucionais (v1ig) |
| `official_open_event_vector_discovery_registry.csv` | Descoberta e validaÃ§Ã£o de vetores observados em bases pÃºblicas abertas (v1ih): 18 candidatos locais auditados por 10 gates â€” 0 ground truth confirmados; `operational_ground_truth_status=BLOCKED`, `can_create_training_label=false` para todos |
| `schemas/official_open_event_vector_discovery_registry_schema.csv` | Schema de 30 campos para registry de descoberta de vetores observados em bases abertas (v1ih) |
| `targeted_official_repository_event_vector_registry.csv` | MineraÃ§Ã£o dirigida em repositÃ³rios oficiais (v1ii-R1): 12 recursos auditados em 6 repositÃ³rios â€” 0 ground truth confirmados; lacuna de disponibilidade pÃºblica documentada; `operational_ground_truth_status=BLOCKED`, `can_create_training_label=false` para todos |
| `schemas/targeted_official_repository_event_vector_registry_schema.csv` | Schema de 27 campos para registry de mineraÃ§Ã£o em repositÃ³rios oficiais (v1ii-R1) |
 6acdc62 (Audita vetores de eventos em repositÃ³rios oficiais)

## Protocolo C e camada de referÃªncia

A camada de referÃªncia contextual foi refinada pelo Protocolo C, que organiza a distinÃ§Ã£o entre evidÃªncia contextual, proxy auditÃ¡vel, candidato de referÃªncia e validaÃ§Ã£o operacional. Ground truth operacional continua bloqueado no estado atual.

O `contextual_reference_layer_registry.csv` registra o status de evidÃªncia e os claims permitidos/proibidos por patch.

O `ground_reference_evidence_source_registry.csv` Ã© o inventÃ¡rio de fontes de referÃªncia: classifica cada fonte por family (ex.: HYDROGEOMORPHOLOGICAL_CONTEXT, OPERATIONAL_FLOOD_PRODUCT), grau de observaÃ§Ã£o (CONTEXTUAL, OPERATIONAL_ALGORITHMIC, EXPERT_INTERPRETED), e registra o allowed_use e forbidden_use de cada fonte. Fontes nÃ£o adquiridas localmente sÃ£o marcadas como NOT_ACQUIRED ou METHODOLOGICAL_REFERENCE_ONLY e nÃ£o podem ser usadas como referÃªncia aplicada a patches.

Veja [`docs/metodologia_cientifica/protocolo_c_construcao_referencia_operacional.md`](../docs/metodologia_cientifica/protocolo_c_construcao_referencia_operacional.md) para a formulaÃ§Ã£o completa do Protocolo C.

A etapa de fechamento de evidÃªncias adiciona mais trÃªs registros metadata-only: `ground_reference_gap_matrix.csv` mapeia os gates de promoÃ§Ã£o abertos por regiÃ£o com evidÃªncia faltante e risco metodolÃ³gico; `review_gate_reference_registry.csv` organiza revisÃµes humanas executadas ou placeholders com decisÃ£o e claims; e `reference_promotion_decision_registry.csv` registra decisÃµes formais de promoÃ§Ã£o com `protocol_b_reassessment_allowed=false` em todas as linhas atuais. O Protocolo C agora inclui fechamento de evidÃªncias, revisÃ£o humana e decisÃ£o de promoÃ§Ã£o â€” formando trilha auditÃ¡vel para eventual ground reference. Ground truth operacional permanece nÃ£o estabelecido. Veja [`protocolo_c_fechamento_evidencias_ground_reference.md`](../docs/metodologia_cientifica/protocolo_c_fechamento_evidencias_ground_reference.md) e [`protocolo_c_revisao_humana_referencia.md`](../docs/metodologia_cientifica/protocolo_c_revisao_humana_referencia.md).

A etapa de aquisiÃ§Ã£o adiciona dois registros metadata-only: `flood_event_candidate_registry.csv` organiza eventos candidatos por regiÃ£o (com status de confirmaÃ§Ã£o e elegibilidade), e `patch_event_reference_link_registry.csv` registra os vÃ­nculos entre patches, eventos e fontes com alinhamentos e bloqueadores explÃ­citos. No estado atual, nenhum evento tem `eligible_for_reference_search=true` e nenhum vÃ­nculo tem `promotion_allowed=true`. Veja [`docs/metodologia_cientifica/protocolo_c_aquisicao_ground_reference.md`](../docs/metodologia_cientifica/protocolo_c_aquisicao_ground_reference.md) para a justificativa metodolÃ³gica desta etapa.

A camada de plano de aquisiÃ§Ã£o (v1hl) transforma metodologia em roteiro real: `observational_evidence_acquisition_plan.csv` organiza fontes-alvo por regiÃ£o, classifica pela forÃ§a metodolÃ³gica, documenta prioridades de aquisiÃ§Ã£o e mapeia quais gates cada fonte pode fechar. `regional_ground_reference_readiness.csv` registra a prontidÃ£o regional para cada gate, identifica a evidÃªncia mais forte jÃ¡ disponÃ­vel, descreve as lacunas crÃ­ticas, e documenta allowed/forbidden claims por regiÃ£o. Essa camada continua metadata-only e nÃ£o treina, prediz ou declara ground truth.

A camada de aquisiÃ§Ã£o operacional (v1hm) coloca o plano em prÃ¡tica: `evidence_acquisition_tracker.csv` rastreia o estado atual de cada fonte-alvo, com acquisition_status, license_status, current_blocker e forbidden_use. `evidence_source_intake_registry.csv` registra fontes acessadas ou em processo com decisÃ£o de intake (ACCEPT_METADATA_ONLY, BLOCK_USE, REQUEST_MORE_INFORMATION). `evidence_license_provenance_registry.csv` documenta licenÃ§a, redistribuiÃ§Ã£o e proveniÃªncia para cada fonte, garantindo que raw_data_publication_allowed=false quando redistribuiÃ§Ã£o nÃ£o for explicitamente pÃºblica e que use_for_operational_ground_truth_allowed=FALSE em todas as linhas atuais. O GitHub continua contendo apenas metadados pÃºblicos seguros â€” dados brutos permanecem local-only.

A camada de dossiÃªs de evidÃªncia (v1ho) especifica o que precisa ser encontrado por evento candidato: `event_evidence_dossier_registry.csv` tem um dossiÃª por evento com status (DOSSIER_OPEN para EVENT_SEARCH_TARGET, DOSSIER_PARTIAL para PENDING_SOURCE_REVIEW), lacunas de evidÃªncia mÃ­nima e `can_support_ground_reference_candidate=false` em todas as linhas. `event_evidence_requirements_registry.csv` registra os requisitos crÃ­ticos por gate (EVENT_CONFIRMATION, TEMPORAL_EVIDENCE, SPATIAL_EVIDENCE, REVIEW_GATE, PROMOTION_DECISION) com `current_status=MISSING` ou `PARTIAL` e `blocking_if_missing=true` para requisitos crÃ­ticos. `event_dossier_decision_registry.csv` tem uma decisÃ£o por dossiÃª com `can_reassess_protocol_b=false` e `can_start_multimodal=false` em todas as linhas. Ground truth operacional, Protocolo B e multimodal permanecem bloqueados/hold.

A camada de busca externa e solicitaÃ§Ã£o regional (v1hp) transforma os dossiÃªs em aÃ§Ã£o concreta: `regional_external_search_plan.csv` organiza onze planos de busca por regiÃ£o (Recife, PetrÃ³polis, Curitiba), especificando fonte-alvo, gate, modo de busca (PUBLIC_PORTAL_REVIEW ou FORMAL_REQUEST), prioridade e status atual. `source_request_package_registry.csv` detalha sete pacotes de solicitaÃ§Ã£o a instituiÃ§Ãµes (COMPDEC Recife, CPRM, Defesa Civil PetrÃ³polis, Defesa Civil Curitiba, GeoCuritiba) com `cannot_establish_ground_truth_alone=true` em todas as linhas e dado bruto local-only quando recebido. `gate_search_question_registry.csv` registra 17 perguntas de busca mapeadas a gates G1â€“G7, com `blocking_if_unanswered=true` para perguntas de G1, G3, G4 e G7. `regional_request_priority_matrix.csv` consolida a prioridade de solicitaÃ§Ã£o por regiÃ£o com forbidden_claim bloqueando declaraÃ§Ã£o de ground truth, flood label, training label, flood detection, flood prediction e supervised training em todas as entradas. Protocolo B permanece BLOCKED e multimodal permanece HOLD em toda a camada.

A camada de referÃªncias observacionais candidatas (v1hq) inicia a primeira prova documental de eventos observados: `observed_event_reference_candidate_registry.csv` registra 9 eventos (3 por regiÃ£o) com G1/G2/G3 fechados documentalmente e G4 em triagem espacial â€” prova de existÃªncia do evento, fonte rastreÃ¡vel e temporalidade. `observed_event_reference_gap_registry.csv` cataloga as lacunas por evento (patch overlay nÃ£o executado, revisÃ£o humana nÃ£o feita, licenÃ§a pendente, geometria ausente). `observed_event_reference_decision_registry.csv` registra uma decisÃ£o por evento com `can_promote_to_ground_reference=false` e `can_generate_training_label=false` em todas as linhas. `manual_external_evidence_needed_registry.csv` cataloga os dados externos que precisam ser trazidos manualmente por regiÃ£o, com `cannot_establish_ground_truth_alone=true` em todas as entradas. Ground truth operacional nÃ£o estÃ¡ estabelecido. G4 nesta etapa Ã© apenas triagem espacial, nÃ£o overlay patch-level. Protocolo B permanece BLOCKED. Multimodal permanece HOLD. DINO permanece SUPPORT_ONLY.

A camada de prÃ©-ligaÃ§Ã£o eventoâ€“patch (v1hr) prepara as condiÃ§Ãµes para patch-linking sem executÃ¡-lo: `event_patch_linking_preflight_registry.csv` organiza o escopo regional de triagem para cada um dos 9 eventos com `promotion_allowed=false`, `can_create_training_label=false`, `protocol_b_status=BLOCKED` em todas as linhas. `manual_geocoding_target_registry.csv` lista 22 localidades a geocodificar manualmente (7 Recife, 10 PetrÃ³polis, 5 Curitiba) com `geocoding_status=NOT_GEOCODED` ou `NEEDS_MANUAL_REVIEW`, sem coordenadas criadas. `event_sentinel_temporal_window_registry.csv` define janelas temporais metadata-only de prÃ©/evento/pÃ³s-evento para busca futura de assets Sentinel, com `acquisition_status=NOT_ACQUIRED` em todas. `patch_linking_dependency_registry.csv` documenta 48 dependÃªncias (SOURCE_GEOMETRY, MANUAL_GEOCODING, LICENSE_PROVENANCE, SENTINEL_TEMPORAL_SEARCH, REVIEW_GATE, mais PHENOMENON_SEPARATION para PetrÃ³polis) com `current_status=OPEN` e `required_before_ground_reference=true` para todas. Nenhum overlay foi executado. Nenhuma geocodificaÃ§Ã£o foi realizada. Nenhuma coordenada foi criada. Nenhum dado bruto foi baixado.

A triagem de eventos candidatos (v1hn) organiza eventos candidatos reais por regiÃ£o em trÃªs registros metadata-only: `event_candidate_screening_registry.csv` lista cinco eventos candidatos (Recife 2021, Recife 2022, PetrÃ³polis fev/2022, Curitiba 2022, Curitiba 2023) com status (EVENT_SEARCH_TARGET ou PENDING_SOURCE_REVIEW), prioridade de busca e gates potencialmente endereÃ§Ã¡veis. `event_source_search_backlog.csv` organiza as fontes a pesquisar por evento candidato, com referÃªncia cruzada ao tracker (v1hm). `event_patch_screening_scope.csv` registra quais patches do corpus DINO estÃ£o no perÃ­metro de busca de cada evento candidato â€” com `spatial_overlap_assessed=false` e `promotion_allowed=false` em todas as linhas. Nenhum dado foi baixado, nenhum gate foi fechado e DINOv2 permanece review-only.

A revisÃ£o assistida de fontes observacionais (v1hu) analisa as 38 fontes registradas no manifest de aquisiÃ§Ã£o e, quando o arquivo local estÃ¡ disponÃ­vel em `local_only/`, extrai indÃ­cios documentais de evento, data, localidade, fenÃ´meno e impacto. `programmatic_source_review_registry.csv` registra 38 linhas com forÃ§a de evidÃªncia, candidatos a gates G1â€“G4 e guardrails obrigatÃ³rios (`requires_manual_review=true`, `can_generate_training_label=false`, `protocol_b_status=BLOCKED`, `can_support_patch_linking=false` para todas). `programmatic_gate_support_registry.csv` lista 152 pares (38 fontes Ã— 4 gates) com `can_close_gate_automatically=false` em todas as linhas. `assisted_source_evidence_gap_registry.csv` documenta 156 lacunas com `REVIEW_GATE_REQUIRED` e `GEOMETRY_NOT_AVAILABLE` para todas as fontes. Resultados: 2 fontes com STRONG_DOCUMENTARY_SUPPORT (NHESS e Copernicus IoD sobre PetrÃ³polis), 2 com WEAK, 34 METADATA_ONLY. Os PDFs locais (DRM-RJ e DiÃ¡rio Oficial) nÃ£o puderam ser lidos por ausÃªncia de biblioteca PDF. Nenhum gate foi fechado automaticamente. RevisÃ£o humana obrigatÃ³ria antes de qualquer avanÃ§o.

A matriz programática de decisÃ£o por evento (v1hv) agrega as evidÃªncias de v1hu ao nÃ­vel do evento observado: `event_gate_decision_matrix.csv` consolida 36 linhas (9 eventos Ã— 4 gates) com `event_gate_status`, `gate_can_advance`, `requires_reviewer_confirmation=true` e `can_close_gate_automatically=false` para todas. `event_ground_reference_readiness_registry.csv` registra 9 linhas de prontidÃ£o com `overall_readiness`, `ground_reference_candidate`, `can_promote_to_ground_reference=false`, `can_create_training_label=false` e `can_reopen_protocol_b=false` para todas. `event_next_action_registry.csv` lista 30 prÃ³ximas aÃ§Ãµes (â‰¥3 por evento) com `can_automate=false` em todas. Resultados: PET_2022_02_15 Ã© o Ãºnico evento com READY_FOR_REVIEW e ground_reference_candidate=true; PET_2024_03_21_28 tem PARTIAL_READINESS; CTB_2024_02_18_20 tem BLOCKED (acesso); demais 6 eventos INSUFFICIENT. Ground truth operacional nÃ£o estÃ¡ estabelecido. Protocolo B permanece BLOCKED. Multimodal permanece HOLD.

A aquisiÃ§Ã£o dirigida e fechamento de lacunas (v1ht) endurece o script de aquisiÃ§Ã£o v1hs com `--target-gaps` e 17 domÃ­nios no allowlist; cria 22 alvos de busca dirigida (7 Recife, 8 PetrÃ³polis, 7 Curitiba) em `targeted_missing_source_search_registry.csv`; identifica 11 solicitaÃ§Ãµes formais em `formal_source_request_target_registry.csv` com templates prontos em `docs/templates/`; e documenta o estado atual: 3 alvos BROWSER_MANUAL_REQUIRED, 7 NOT_FOUND_REQUIRES_FORMAL, 3 REVIEW_GATE_REQUIRED, 9 WEB_ACCESSIBLE pendentes. Nenhum novo arquivo foi adquirido nesta execuÃ§Ã£o de catalogaÃ§Ã£o; a aquisiÃ§Ã£o real legada de v1hs permanece com 6 arquivos em local_only/. Nenhum gate foi fechado. Ground truth operacional nÃ£o estÃ¡ estabelecido. Protocolo B permanece BLOCKED. Multimodal permanece HOLD.

A resoluÃ§Ã£o assistida de lacunas (v1hx) executa `--allow-web` nos 9 alvos WEB_ACCESSIBLE: 8 homepages de portal adquiridas em `local_only/` (WEB_ACQUIRED_LOCAL_ONLY) e 1 bloqueada (simepar.br, WEB_BLOCKED_BY_ACCESS); `targeted_missing_source_search_registry.csv` atualizado com WEB_* statuses reais; `observed_source_acquisition_manifest.csv` expandido de 38 para 46 linhas (ACQ_v1hx_001â€“008); 9 gaps atualizados para PORTAL_ACQUIRED ou OPEN com nota WEB_BLOCKED. Backend PDF ALL_MISSING (pypdf/PyPDF2/pdfminer/fitz/pdfplumber â€” aÃ§Ã£o pendente: `pip install pypdf`). FR_CTB_005 (Simepar) adicionado ao `formal_source_request_target_registry.csv`; 12 pacotes preenchidos criados em `docs/templates/protocolo_c_solicitacoes_preenchidas/` e registrados em `formal_request_package_registry.csv`. Re-execuÃ§Ã£o de v1hw com 46 fontes produziu 3 decisÃµes DOWNGRADE_TO_CONTEXTUAL_ONLY (REC e CTB) â€” portais tÃªm vocabulÃ¡rio genÃ©rico sem confirmaÃ§Ã£o especÃ­fica de evento. A revisÃ£o profunda prÃ©-geocodificaÃ§Ã£o (v1hy) instala pypdf localmente (v6.12.0), extrai 1729 links de 8 HTMLs de portal â€” nenhum passa o filtro combinado de domÃ­nio + keyword de evento. `acquired_portal_deep_review_registry.csv` confirma todos os portais como PORTAL_HOMEPAGE_GENERIC. `pre_geocoding_closure_registry.csv` registra 9 eventos com todos os guardrails explicitamente false/BLOCKED/HOLD/SUPPORT_ONLY e seleciona 3 eventos para geocodificaÃ§Ã£o controlada futura. Nenhum gate foi fechado. Ground truth operacional nÃ£o estÃ¡ estabelecido. Protocolo B permanece BLOCKED. Multimodal permanece HOLD.

O pacote operacional de geocodificaÃ§Ã£o controlada (v1hz) constrÃ³i o escopo espacial auditÃ¡vel: `controlled_geocoding_event_scope_registry.csv` registra 9 eventos (3 selecionados) com `geocoding_execution_allowed_now=false`, `overlay_execution_allowed_now=false`, `can_create_training_label=false`, `protocol_b_status=BLOCKED` para todos. `controlled_geocoding_target_registry.csv` registra 13 alvos de localidade (5 Recife + 6 PetrÃ³polis 2022 + 2 PetrÃ³polis 2024) â€” 6 bloqueados por PHENOMENON_SEPARATION_REQUIRED (PET_2022_02_15, CRITICAL), 7 prontos para futura geocodificaÃ§Ã£o controlada; `coordinate_value_public=""` e `geometry_file_public=""` em todos os 13 (nenhuma coordenada ou geometria criada). `spatial_ground_reference_blocker_registry.csv` documenta 13 bloqueios (1 CRITICAL, 4 HIGH, 2 MEDIUM, 6 LOW) com `blocks_training_label=true` invariante. `future_controlled_geocoding_queue.csv` lista 5 itens de fila com `can_execute_now=false`, `can_create_training_label_after_execution=false` para todos. Nenhuma geocodificaÃ§Ã£o foi executada. Nenhuma coordenada foi criada. Nenhuma geometria foi inferida. Ground truth operacional nÃ£o estÃ¡ estabelecido. Protocolo B permanece BLOCKED. Multimodal permanece HOLD.

O inventÃ¡rio de fontes espaciais autoritativas (v1ia) identifica e classifica as fontes necessÃ¡rias para os 13 alvos: `authoritative_spatial_source_inventory.csv` cataloga 15 fontes reais (nenhuma inventada), classificadas por classe, autoridade, status de acesso e capacidade de resolver bloqueios da v1hz; 5 fontes com FORMAL_REQUEST_REQUIRED. `controlled_geocoding_execution_preflight_registry.csv` avalia prontidÃ£o de cada alvo â€” 2 READY_FOR_TRIAGE_ONLY (ValparaÃ­so e Floresta, PET_2024), 5 WAITING_OFFICIAL_SOURCE (Recife, aguardam COMPDEC), 6 WAITING_PHENOMENON_SEPARATION (PET_2022, bloqueio CRITICAL ativo). `authoritative_spatial_gap_registry.csv` documenta 6 lacunas com 1 CRITICAL (separaÃ§Ã£o de fenÃ´meno PET_2022 bloqueia geocodificaÃ§Ã£o), 4 HIGH e 1 MEDIUM. Nenhuma geocodificaÃ§Ã£o foi executada. Nenhuma coordenada foi criada. Ground truth operacional nÃ£o estÃ¡ estabelecido. Protocolo B permanece BLOCKED. Multimodal permanece HOLD.

A consolidaÃ§Ã£o de referÃªncias observacionais candidatas (v1ib) transforma o Protocolo C de camada de bloqueio em camada de promoÃ§Ã£o positiva: classifica 9 eventos por nÃ­veis de evidÃªncia acumulada (LEVEL_0â€“LEVEL_6). `observational_reference_promotion_registry.csv` registra 9 linhas com nÃ­vel, decisÃ£o de promoÃ§Ã£o e guardrails invariantes (`can_be_called_ground_truth_operational=false`, `can_create_training_label=false`, `can_reopen_protocol_b=false`, `multimodal_status=HOLD`, `dino_usage_status=SUPPORT_ONLY` para todos). `protocolo_c_event_evidence_level_matrix.csv` documenta o estado de cada gate G1â€“G7 por evento. Resultados: PET_2022_02_15 promovido a STRONG candidata (LEVEL_5_SPATIAL_TRIAGE_READY) â€” 3 fontes SOURCE_CONFIRMED, 6 localidades, fenÃ´meno MIXED documentado, bloqueado para geocodificaÃ§Ã£o por separaÃ§Ã£o de fenÃ´meno; PET_2024_03_21_28 promovido a secundÃ¡ria (LEVEL_6_READY_FOR_CONTROLLED_GEOCODING); REC_2022_05_24_30 em LEVEL_4 (aguarda COMPDEC); 5 eventos em LEVEL_0 ou LEVEL_1 por ausÃªncia de fonte confirmada. Nenhum evento Ã© ground truth operacional. Nenhum label criado. Protocolo B permanece BLOCKED.

A separaÃ§Ã£o assistida de fenÃ´meno por localidade (v1ic) executa a classificaÃ§Ã£o fenomenolÃ³gica das 8 localidades de PET_2022_02_15  com base em evidÃªncia textual das fontes jÃ¡ adquiridas (DRM-RJ PDF 57p. e NHESS HTML). `event_locality_phenomenon_separation_registry.csv` registra 8 linhas com `phenomenon_class`, `phenomenon_confidence`, `source_ids`, `hydrological_terms_found`, `mass_movement_terms_found` e `blocks_controlled_geocoding=true` para todas. `event_phenomenon_separation_decision_registry.csv` registra a decisÃ£o consolidada: PARTIAL_SEPARATION â€” 0 HYDROLOGICAL_CONFIRMED, 3 MASS_MOVEMENT_CONFIRMED (ChÃ¡cara Flora e Caxambu com HIGH, Morin com LOW), 5 MIXED_CONFIRMED. O achado hidrolÃ³gico mais importante Ã© o transbordamento do Rio Quitandinha documentado pelo DRM-RJ, mas a localidade Quitandinha permanece MIXED por sobreposiÃ§Ã£o com lista de deslizamentos. Nenhuma localidade atingiu HYDROLOGICAL_CONFIRMED â€” o bloqueio de geocodificaÃ§Ã£o controlada se mantÃ©m. `can_advance_to_controlled_geocoding_future=false`, `can_create_training_label=false`, `multimodal_status=HOLD`, `dino_usage_status=SUPPORT_ONLY` para todas as linhas.

A validaÃ§Ã£o assistida fonteâ€“evento (v1hw) valida individualmente as 38 fontes contra os 9 eventos candidatos: `source_event_validation_registry.csv` registra 38 linhas com classificaÃ§Ã£o de fenÃ´meno, alinhamento temporal, evidÃªncia espacial, impacto, licenÃ§a e `can_generate_training_label=false`, `can_support_ground_reference=false` para todas. `event_assisted_validation_decision_registry.csv` registra 9 decisÃµes por evento com `validation_decision`, confirmaÃ§Ã£o final, fenÃ´meno e `can_promote_to_ground_reference=false`, `can_create_training_label=false` em todas. `event_patch_compatibility_precheck_registry.csv` registra 9 prÃ©-verificaÃ§Ãµes de compatibilidade com `can_execute_overlay_now=false` em todas. `event_priority_for_geocoding_registry.csv` lista 9 eventos com prioridade de geocodificaÃ§Ã£o â€” 2 selecionados (PET_2022_02_15 com REQUEST_PHENOMENON_SEPARATION e PET_2024_03_21_28 com KEEP_AS_PRIORITY) â€” `can_execute_overlay_after_geocoding=false` para todas. Nenhum overlay foi executado. Nenhuma geocodificaÃ§Ã£o foi realizada. Nenhuma coordenada foi criada. Ground truth operacional nÃ£o estÃ¡ estabelecido. Protocolo B permanece BLOCKED. Multimodal permanece HOLD.

A descoberta e validaÃ§Ã£o de vetores em bases abertas (v1ih) audita 18 candidatos locais identificados em bases oficiais pÃºblicas por 10 gates de validaÃ§Ã£o (fonte oficial, geometria, CRS, data, compatibilidade temporal, fenÃ´meno, observaÃ§Ã£o direta, separabilidade, escala espacial, ground truth). Resultado: 0 candidatos passam todos os gates. O gate mais restritivo Ã© Gate 05 (data de evento compatÃ­vel) com 39% de aprovaÃ§Ã£o; Gate 09 (escala patch-level) com 0%. `official_open_event_vector_discovery_registry.csv` documenta todos os 18 candidatos com as classificaÃ§Ãµes de bloqueio e allowed/forbidden claims.

A mineraÃ§Ã£o dirigida em repositÃ³rios oficiais (v1ii-R1) estende a busca de v1ih a 6 repositÃ³rios pÃºblicos com APIs e catÃ¡logos indexados (RIGeo/SGB, CKAN Recife, CKAN Pernambuco/APAC, Dados Abertos RJ/DRM-RJ, GeoCuritiba/IPPUC, dados.gov.br/S2ID/Atlas). 12 recursos foram auditados â€” 4 EVENT_CONFIRMATION_ONLY, 3 CARTOGRAPHIC_LEAD_ONLY, 2 BLOCKED_NO_DATE, 2 SCAN_FAILED_CONTROLLED, 1 RISK_SUSCEPTIBILITY_ONLY â€” com 0 ground truth confirmados. `targeted_official_repository_event_vector_registry.csv` documenta todos os recursos com classificaÃ§Ã£o, gate breakdown e guardrails. A lacuna de disponibilidade pÃºblica de vetor observado datado em escala patch-level Ã© confirmada como resultado cientÃ­fico do Protocolo C.

## O que nÃ£o estÃ¡ aqui

- GeoTIFFs, rasters, GeoJSONs brutos, shapefiles, geodatabases
- Embeddings `.npz` ou `.npy`
- Outputs de execuÃ§Ã£o local (`local_runs/`)
- Dados de validaÃ§Ã£o externa pesados (PE3D/MDE, SGB/RIGeo, GeoCuritiba)
- Qualquer arquivo que contenha paths absolutos de mÃ¡quina local

Os registros descrevem esses materiais. Os materiais ficam locais.

## NavegaÃ§Ã£o relacionada

- [`docs/metodologia_cientifica/research_datasets_and_artifacts.md`](../docs/metodologia_cientifica/research_datasets_and_artifacts.md) â€” narrativa metodolÃ³gica completa
- [`docs/metodologia_cientifica/patch_lineage_and_grounding.md`](../docs/metodologia_cientifica/patch_lineage_and_grounding.md) â€” linhagem dos patches
- [`manifests/`](../manifests/) â€” manifests CSV/JSON por estÃ¡gio do pipeline
