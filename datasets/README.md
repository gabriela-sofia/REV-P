# Datasets auditáveis do REV-P

## O que este diretório documenta

Este diretório registra os datasets e corpora produzidos ou utilizados pelo REV-P como
evidência científica auditável. Ele não contém dados brutos — contém registros
estruturados que descrevem o que existe, onde está, como foi produzido e quais são as
suas limitações.

## Quatro categorias de material

**Dataset público:** manifest ou registro commitado neste repositório. Acessível a
qualquer leitura, sem dependência de ambiente local.

**Registro auditável:** tabela que descreve um corpus local sem replicar os arquivos
pesados. Prova que o corpus existe e como foi construído, sem exigir que o repositório
hospede os rasters.

**Dado local:** arquivo que existe apenas no workspace privado (rasters Sentinel,
embeddings `.npz`, shapefiles brutos). Referenciado pelos manifests públicos, mas não
versionado.

**Artefato pesado:** dado que não pode ou não deve ser versionado por tamanho, por
conteúdo sensível ou por ser reproduzível a partir dos scripts e manifests públicos.

## Por que o GitHub publica rastreabilidade, não rasters

Os GeoTIFFs Sentinel originais têm entre 10 MB e 200 MB por arquivo. O corpus de 128
patches totaliza múltiplos gigabytes. Versionar esses arquivos incharia o repositório
sem benefício científico: os patches são gerados a partir de imagens Sentinel-2 Level-2A
de acesso público, e a metodologia de derivação está documentada nos manifests.

O que prova a legitimidade científica do corpus não é a presença dos rasters — é a
rastreabilidade da cadeia: qual imagem Sentinel originou cada patch, qual preflight foi
executado, qual QA foi aprovado antes da extração de embeddings.

## Arquivos neste diretório

| Arquivo | Conteúdo |
|---|---|
| `dataset_registry.csv` | Registro geral de datasets e corpora do projeto |
| `patch_corpus_registry.csv` | Registro dos corpora de patches Sentinel por estágio |
| `patch_corpus_taxonomy_registry.csv` | Taxonomia v1iw que distingue corpus territorial consolidado (59 patches) do manifesto Sentinel candidato (128 assets) |
| `external_evidence_registry.csv` | Registro das evidências GIS externas por região |
| `contextual_reference_layer_registry.csv` | Camada de referência contextual validada: status de evidência e claims permitidos/proibidos por patch |
| `ground_reference_evidence_source_registry.csv` | Inventário de fontes de referência categorizado pelo Protocolo C: tipo, grau de observação, allowed_use, forbidden_use |
| `schemas/dataset_registry_schema.csv` | Schema de campos de dataset_registry.csv |
| `schemas/patch_corpus_schema.csv` | Schema de campos de patch_corpus_registry.csv |
| `schemas/patch_corpus_taxonomy_schema.csv` | Schema de campos de patch_corpus_taxonomy_registry.csv |
| `schemas/external_evidence_schema.csv` | Schema de campos de external_evidence_registry.csv |
| `schemas/contextual_reference_layer_schema.csv` | Schema de campos de contextual_reference_layer_registry.csv |
| `schemas/ground_reference_evidence_source_schema.csv` | Schema de campos de ground_reference_evidence_source_registry.csv |
| `flood_event_candidate_registry.csv` | Registro de eventos de inundação candidatos por região — status de confirmação, elegibilidade e bloqueadores (etapa de aquisição Protocolo C) |
| `patch_event_reference_link_registry.csv` | Vínculos patch-evento-fonte com alinhamento temporal/espacial, candidatura e claims permitidos/proibidos (etapa de aquisição Protocolo C) |
| `schemas/flood_event_candidate_schema.csv` | Schema de campos de flood_event_candidate_registry.csv |
| `schemas/patch_event_reference_link_schema.csv` | Schema de campos de patch_event_reference_link_registry.csv |
| `ground_reference_gap_matrix.csv` | Matriz de lacunas de evidência por região: gates abertos, evidência faltante, risco metodológico e próximos passos permitidos/proibidos (etapa de fechamento Protocolo C) |
| `review_gate_reference_registry.csv` | Registry de revisões supervisoras ou placeholders: decisão, materiais revisados, consistency checks, allowed_claim e forbidden_claim por revisão (etapa de fechamento Protocolo C) |
| `reference_promotion_decision_registry.csv` | Registry de decisões formais de promoção: gates satisfeitos/falhados, final_reference_status, protocol_b_reassessment_allowed (etapa de fechamento Protocolo C) |
| `schemas/ground_reference_gap_matrix_schema.csv` | Schema de campos de ground_reference_gap_matrix.csv |
| `schemas/review_gate_reference_schema.csv` | Schema de campos de review_gate_reference_registry.csv |
| `schemas/reference_promotion_decision_schema.csv` | Schema de campos de reference_promotion_decision_registry.csv |
| `observational_evidence_acquisition_plan.csv` | Plano de aquisição de evidências observacionais por região (v1hl): fontes-alvo, prioridades, força metodológica, gates relacionados e acesso esperado |
| `schemas/observational_evidence_acquisition_plan_schema.csv` | Schema de campos de observational_evidence_acquisition_plan.csv |
| `regional_ground_reference_readiness.csv` | Prontidão regional para ground reference (v1hl): status de gates por região, evidência mais forte, lacunas críticas, risco metodológico e allowed/forbidden claims |
| `schemas/regional_ground_reference_readiness_schema.csv` | Schema de campos de regional_ground_reference_readiness.csv |
| `evidence_acquisition_tracker.csv` | Tracker de aquisição (v1hm): estado atual de cada fonte-alvo por região — acquisition_status, license_status, current_blocker, next_action e forbidden_use |
| `schemas/evidence_acquisition_tracker_schema.csv` | Schema de campos de evidence_acquisition_tracker.csv |
| `evidence_source_intake_registry.csv` | Intake registry (v1hm): fontes acessadas ou em processo — event_link_status, intake_decision, blocked_reason, allowed_use e forbidden_use por entrada |
| `schemas/evidence_source_intake_schema.csv` | Schema de campos de evidence_source_intake_registry.csv |
| `evidence_license_provenance_registry.csv` | Registry de licença e proveniência (v1hm): license_status, redistribution_status, raw_data_publication_allowed, local_only_required e use_for_operational_ground_truth_allowed por fonte |
| `schemas/evidence_license_provenance_schema.csv` | Schema de campos de evidence_license_provenance_registry.csv |
| `event_evidence_dossier_registry.csv` | Dossiês de evidência por evento candidato (v1ho): status do dossiê, lacunas de evidência mínima, decisão de continuidade e guardrails de promoção |
| `schemas/event_evidence_dossier_schema.csv` | Schema de campos de event_evidence_dossier_registry.csv |
| `event_evidence_requirements_registry.csv` | Requisitos mínimos de evidência por evento candidato (v1ho): tipo de requisito, gate alvo, status atual, blocking_if_missing e forbidden_if_missing |
| `schemas/event_evidence_requirements_schema.csv` | Schema de campos de event_evidence_requirements_registry.csv |
| `event_dossier_decision_registry.csv` | Decisões de continuidade por dossiê (v1ho): decision_status, próximos passos permitidos/proibidos, can_reassess_protocol_b=false e can_start_multimodal=false |
| `schemas/event_dossier_decision_schema.csv` | Schema de campos de event_dossier_decision_registry.csv |
| `event_candidate_screening_registry.csv` | Eventos candidatos por região (v1hn): status, prioridade de busca, gates potencialmente endereçáveis e guardrails de promoção e ground truth |
| `schemas/event_candidate_screening_schema.csv` | Schema de campos de event_candidate_screening_registry.csv |
| `event_source_search_backlog.csv` | Backlog de fontes a pesquisar por evento candidato (v1hn): fonte, família, modo de acesso, status da busca e gates que a fonte poderia suportar |
| `schemas/event_source_search_backlog_schema.csv` | Schema de campos de event_source_search_backlog.csv |
| `event_patch_screening_scope.csv` | Escopo de triagem por patch (v1hn): quais patches do corpus estão no perímetro de busca de cada evento candidato, com guardrails de sobreposição espacial e promoção |
| `schemas/event_patch_screening_scope_schema.csv` | Schema de campos de event_patch_screening_scope.csv |
| `regional_external_search_plan.csv` | Planos de busca externa por região (v1hp): fonte-alvo, gate, modo, prioridade, status e forbidden_use bloqueando ground truth e labels |
| `schemas/regional_external_search_plan_schema.csv` | Schema de campos de regional_external_search_plan.csv |
| `source_request_package_registry.csv` | Pacotes de solicitação formal a instituições (v1hp): instituição, tipo de solicitação, evidência solicitada, status e cannot_establish_ground_truth_alone=true |
| `schemas/source_request_package_schema.csv` | Schema de campos de source_request_package_registry.csv |
| `gate_search_question_registry.csv` | Perguntas de busca por gate e região (v1hp): current_answer_status, blocking_if_unanswered e forbidden_if_unanswered |
| `schemas/gate_search_question_schema.csv` | Schema de campos de gate_search_question_registry.csv |
| `regional_request_priority_matrix.csv` | Matriz de prioridade regional de solicitação (v1hp): evento prioritário por região, razão, próximo gate a fechar, protocol_b_status=BLOCKED e multimodal_status=HOLD |
| `schemas/regional_request_priority_matrix_schema.csv` | Schema de campos de regional_request_priority_matrix.csv |
| `observed_event_reference_candidate_registry.csv` | 9 eventos observados candidatos (v1hq): G1/G2/G3 fechados documentalmente, G4 em triagem espacial, operational_ground_truth_status=NOT_ESTABLISHED, protocol_b_status=BLOCKED, can_be_used_as_training_label=false para todos |
| `schemas/observed_event_reference_candidate_schema.csv` | Schema de campos de observed_event_reference_candidate_registry.csv |
| `observed_event_reference_gap_registry.csv` | Lacunas metodológicas por evento observado candidato (v1hq): o que falta para avançar à ligação patch-evento e ground reference |
| `schemas/observed_event_reference_gap_schema.csv` | Schema de campos de observed_event_reference_gap_registry.csv |
| `observed_event_reference_decision_registry.csv` | Decisões metodológicas por evento observado candidato (v1hq): can_promote_to_ground_reference=false, can_generate_training_label=false, can_reopen_protocol_b=false para todos |
| `schemas/observed_event_reference_decision_schema.csv` | Schema de campos de observed_event_reference_decision_registry.csv |
| `manual_external_evidence_needed_registry.csv` | Inventário de dados externos que precisam ser trazidos manualmente por região (v1hq): categoria, provedor, modo de aquisição, cannot_establish_ground_truth_alone=true para todos |
| `schemas/manual_external_evidence_needed_schema.csv` | Schema de campos de manual_external_evidence_needed_registry.csv |
| `event_patch_linking_preflight_registry.csv` | Preflight de pré-ligação evento–patch (v1hr): escopo regional, status de overlay, bloqueios e guardrails — promotion_allowed=false, can_create_training_label=false, protocol_b_status=BLOCKED para todos |
| `schemas/event_patch_linking_preflight_schema.csv` | Schema de campos de event_patch_linking_preflight_registry.csv |
| `manual_geocoding_target_registry.csv` | Alvos de geocodificação manual (v1hr): 22 localidades por evento — geocoding_status=NOT_GEOCODED ou NEEDS_MANUAL_REVIEW, requires_official_confirmation=true, cannot_establish_ground_truth_alone=true para todos |
| `schemas/manual_geocoding_target_schema.csv` | Schema de campos de manual_geocoding_target_registry.csv |
| `event_sentinel_temporal_window_registry.csv` | Janelas temporais Sentinel por evento (v1hr): períodos pré/evento/pós metadata-only — acquisition_status=NOT_ACQUIRED, cannot_establish_ground_truth_alone=true para todos |
| `schemas/event_sentinel_temporal_window_schema.csv` | Schema de campos de event_sentinel_temporal_window_registry.csv |
| `patch_linking_dependency_registry.csv` | Dependências metodológicas para patch-linking real (v1hr): o que deve ser resolvido antes de overlay, review gate e ground reference — current_status=OPEN, required_before_ground_reference=true para todas |
| `schemas/patch_linking_dependency_schema.csv` | Schema de campos de patch_linking_dependency_registry.csv |
| `observed_source_acquisition_manifest.csv` | Manifest público de aquisição v1hs: metadados de fontes públicas candidatas — acquisition_status, hash SHA-256 (quando adquirido), license_status, cannot_establish_ground_truth_alone=true, can_generate_training_label=false, protocol_b_status=BLOCKED, multimodal_status=HOLD para todas |
| `schemas/observed_source_acquisition_manifest_schema.csv` | Schema de campos de observed_source_acquisition_manifest.csv |
| `observed_source_acquisition_gap_registry.csv` | Lacunas de aquisição por região (v1hs): gap_type, required_action, priority_level, blocks_ground_reference — 11 solicitações formais e 12 buscas manuais pendentes |
| `schemas/observed_source_acquisition_gap_schema.csv` | Schema de campos de observed_source_acquisition_gap_registry.csv |
| `acquired_source_review_registry.csv` | Revisão inicial metadata-only de fontes adquiridas (v1hs): can_support_patch_linking=false, can_support_ground_reference=false, can_generate_training_label=false para todas |
| `schemas/acquired_source_review_schema.csv` | Schema de campos de acquired_source_review_registry.csv |
| `programmatic_source_review_registry.csv` | revisão programática de fontes por evidência textual (v1hu): força de evidência, candidatos G1–G4, requires_manual_review=true, can_close_gate_automatically=false para todas |
| `schemas/programmatic_source_review_schema.csv` | Schema de campos de programmatic_source_review_registry.csv |
| `programmatic_gate_support_registry.csv` | Suporte candidato por gate e fonte (v1hu): 152 linhas (38×4 gates), can_close_gate_automatically=false, can_generate_training_label=false para todas |
| `schemas/programmatic_gate_support_schema.csv` | Schema de campos de programmatic_gate_support_registry.csv |
| `programmatic_source_evidence_gap_registry.csv` | Lacunas da revisão programática (v1hu): REVIEW_GATE_REQUIRED e GEOMETRY_NOT_AVAILABLE para todas, blocks_ground_reference=true para lacunas críticas |
| `schemas/programmatic_source_evidence_gap_schema.csv` | Schema de campos de programmatic_source_evidence_gap_registry.csv |
| `event_gate_decision_matrix.csv` | Matriz de decisão por gate por evento (v1hv): 36 linhas (9 eventos × 4 gates), event_gate_status, gate_can_advance, requires_reviewer_confirmation=true, can_close_gate_automatically=false para todas |
| `schemas/event_gate_decision_matrix_schema.csv` | Schema de campos de event_gate_decision_matrix.csv |
| `event_ground_reference_readiness_registry.csv` | Prontidão para ground reference por evento (v1hv): 9 linhas, overall_readiness, ground_reference_candidate, can_promote_to_ground_reference=false, can_create_training_label=false para todas |
| `schemas/event_ground_reference_readiness_schema.csv` | Schema de campos de event_ground_reference_readiness_registry.csv |
| `event_next_action_registry.csv` | Próximas ações por evento (v1hv): 30 ações (â‰¥3 por evento), can_automate=false para todas |
| `schemas/event_next_action_schema.csv` | Schema de campos de event_next_action_registry.csv |
| `source_event_validation_registry.csv` | validação programática por fonte (v1hw): 38 linhas, source_validation_status, event_confirmation_status, phenomenon_status, temporal_alignment_status, can_generate_training_label=false, can_support_ground_reference=false para todas |
| `schemas/source_event_validation_schema.csv` | Schema de campos de source_event_validation_registry.csv |
| `event_programmatic_validation_decision_registry.csv` | decisão programática por evento (v1hw): 9 linhas, validation_decision, event_confirmation_status_final, can_promote_to_ground_reference=false, can_create_training_label=false para todas |
| `schemas/event_programmatic_validation_decision_schema.csv` | Schema de campos de event_programmatic_validation_decision_registry.csv |
| `event_patch_compatibility_precheck_registry.csv` | Precheck de compatibilidade patch–evento (v1hw): 9 linhas, precheck_status, can_execute_overlay_now=false para todas |
| `schemas/event_patch_compatibility_precheck_schema.csv` | Schema de campos de event_patch_compatibility_precheck_registry.csv |
| `event_priority_for_geocoding_registry.csv` | Prioridade para geocodificação controlada (v1hw): 9 linhas, 2 selecionados (PET_2022_02_15 e PET_2024_03_21_28), can_execute_overlay_after_geocoding=false para todas |
| `schemas/event_priority_for_geocoding_schema.csv` | Schema de campos de event_priority_for_geocoding_registry.csv |
| `targeted_missing_source_search_registry.csv` | 22 alvos de busca dirigida de lacunas (v1ht): 7 Recife, 8 Petrópolis, 7 Curitiba — estratégia, modo, prioridade, fallback_action, cannot_establish_ground_truth_alone=true para todas |
| `schemas/targeted_missing_source_search_schema.csv` | Schema de campos de targeted_missing_source_search_registry.csv |
| `formal_source_request_target_registry.csv` | 12 solicitações formais (v1ht + v1hx): 11 identificadas em v1ht + FR_CTB_005 (Simepar) adicionada em v1hx; instituição, base legal, template, blocks_ground_reference, cannot_establish_ground_truth_alone=true para todas |
| `schemas/formal_source_request_target_schema.csv` | Schema de campos de formal_source_request_target_registry.csv |
| `formal_request_package_registry.csv` | 12 pacotes de pedido formal preenchidos (v1hx): um por solicitação formal; package_status=DRAFT; paths para docs/templates/protocolo_c_solicitacoes_preenchidas/; cannot_establish_ground_truth_alone=true para todos |
| `schemas/formal_request_package_schema.csv` | Schema de campos de formal_request_package_registry.csv |
| `acquired_portal_deep_review_registry.csv` | 8 portais PORTAL_HOMEPAGE_ACQUIRED revisados em v1hy: todos PORTAL_HOMEPAGE_GENERIC / CONFIRM_CONTEXTUAL_ONLY; 1729 links extraídos, 0 event-specific; can_support_patch_linking=false; cannot_establish_ground_truth_alone=true para todos |
| `schemas/acquired_portal_deep_review_schema.csv` | Schema de 20 campos para revisão profunda de portais |
| `pre_geocoding_closure_registry.csv` | 9 eventos com registro de fechamento pré-geocodificação (v1hy): 3 selecionados (PET_2022_02_15, PET_2024_03_21_28, REC_2022_05_24_30); todos os guardrails operacionais false/BLOCKED/HOLD/SUPPORT_ONLY |
| `schemas/pre_geocoding_closure_schema.csv` | Schema de 22 campos para fechamento pré-geocodificação |
| `authoritative_spatial_source_inventory.csv` | Inventário de fontes espaciais autoritativas (v1ia): 15 entradas por evento/localidade — classe, autoridade, acesso, resolves_blocker_id; `can_create_training_label=false`, `can_support_ground_reference_future=false` para todas |
| `schemas/authoritative_spatial_source_inventory_schema.csv` | Schema de 29 campos para inventário de fontes espaciais autoritativas |
| `controlled_geocoding_execution_preflight_registry.csv` | Preflight de execução de geocodificação controlada (v1ia): 13 linhas — 2 READY_FOR_TRIAGE_ONLY (PET_2024), 5 WAITING_OFFICIAL_SOURCE (REC), 6 WAITING_PHENOMENON_SEPARATION (PET_2022); `can_execute_overlay_after_geocoding=false`, `can_create_training_label_after_geocoding=false` para todos |
| `schemas/controlled_geocoding_execution_preflight_schema.csv` | Schema de 28 campos para preflight de execução de geocodificação controlada |
| `authoritative_spatial_gap_registry.csv` | Lacunas de fonte autoritativa (v1ia): 6 entradas — 1 CRITICAL (separação fenômeno PET_2022, bloqueia geocodificação), 4 HIGH, 1 MEDIUM; `blocks_training_label=true` invariante para todas |
| `schemas/authoritative_spatial_gap_schema.csv` | Schema de 22 campos para lacunas de fonte autoritativa |
| `observational_reference_promotion_registry.csv` | Promoção graduada por evento candidato (v1ib): 9 linhas — 1 LEVEL_5 (PET_2022_02_15), 1 LEVEL_6 (PET_2024_03_21_28), 2 HOLD, 3 contextuais, 2 bloqueadas; `can_be_called_ground_truth_operational=false`, `can_create_training_label=false`, `can_reopen_protocol_b=false` invariantes |
| `protocolo_c_event_evidence_level_matrix.csv` | Matriz G1–G7 de evidência por evento (v1ib): 9 linhas; resume estado de cada gate por evento — confirmação, fonte, temporal, localidade, fenômeno, suporte espacial, prontidão geocodificação |
| `schemas/observational_reference_promotion_schema.csv` | Schema de 26 campos para promoção graduada de referência observacional candidata (v1ib) |
| `event_locality_phenomenon_separation_registry.csv` | Separação fenomenológica por localidade (v1ic): 8 linhas para PET_2022_02_15; `phenomenon_class`, `phenomenon_confidence`, `blocks_controlled_geocoding=true` para todas; Chácara Flora e Caxambu MASS_MOVEMENT_CONFIRMED (HIGH); `can_create_training_label=false`, `multimodal_status=HOLD` invariantes |
| `event_phenomenon_separation_decision_registry.csv` | Decisão de separação fenomenológica (v1ic): 1 linha para PET_2022_02_15; `phenomenon_separation_status=PARTIAL_SEPARATION`; `can_advance_to_controlled_geocoding_future=false`; `required_next_action=obter PKG_FR_PET_001`; `forbidden_claim` explícito |
| `schemas/event_locality_phenomenon_separation_schema.csv` | Schema de 29 campos para separação fenomenológica por localidade (v1ic) |
| `schemas/event_phenomenon_separation_decision_schema.csv` | Schema de 21 campos para decisão de separação fenomenológica (v1ic) |
| `observed_reference_source_package_registry.csv` | Pacotes de referência cartográfica candidatos (v1id): 1 linha; PKG_FR_PET_001 (DRM-RJ completo) registrado como REQUIRED_NOT_INGESTED; `operational_ground_truth_status=BLOCKED` invariante |
| `schemas/observed_reference_source_package_schema.csv` | Schema de 21 campos para registry de pacotes de referência cartográfica (v1id) |
| `ground_reference_evidence_registry.csv` | Auditoria de ground reference (v1ie): 10 linhas; candidatos SGB/CPRM+FBDS auditados com pyshp; suscetibilidade e feições de deslizamento históricas sem vínculo temporal ao evento 2022-02-15; Gate 6 FAIL para todos; `operational_ground_truth_status=BLOCKED`, `ml_label_status=BLOCKED_UNTIL_SPLIT_AND_LEAKAGE_PROTOCOL` invariantes |
| `schemas/ground_reference_evidence_registry_schema.csv` | Schema de 24 campos para auditoria de evidência de ground reference (v1ie) |
| `official_observed_event_vector_registry.csv` | Aquisição e auditoria de vetores oficiais observados (v1if): 6 linhas; ZIP SGB/CPRM baixado (20.9MB) — 11 PDFs extraídos, 0 vetores; todos BLOCKED pelos 11 gates; `ml_label_status=BLOCKED_UNTIL_SPLIT_AND_LEAKAGE_PROTOCOL` invariante; 4 instituições pendentes de solicitação formal |
| `schemas/official_observed_event_vector_registry_schema.csv` | Schema de 32 campos para registry de vetores observados oficiais (v1if) |
=======
| `official_ground_truth_request_registry.csv` | Rastreamento de solicitações institucionais de ground truth (v1ig): 5 linhas — REQ_SGB_PET_001, REQ_DRM_PET_001, REQ_DC_PET_001, REQ_INPE_PET_001, REQ_SEDEC_001; todas NOT_YET_SUBMITTED; `acceptance_status=PENDING_RESPONSE`, `ground_truth_relevance=UNKNOWN_PENDING_RESPONSE` para todas; atualização manual pelo pesquisador após cada interação institucional |
| `schemas/official_ground_truth_request_registry_schema.csv` | Schema de 24 campos para registry de rastreamento de solicitações institucionais (v1ig) |
| `official_ground_truth_response_acceptance_criteria.csv` | 13 critérios de aceitação de respostas institucionais (v1ig): cobertura: origem, canal, geometria, CRS, data, fenômeno, separação hidrológico/geológico, dicionário, metadados, licença, escala, auditabilidade, ausência de ambiguidade risco/suscetibilidade; 6 critérios REJECTED, 7 HOLD_PENDING_CLARIFICATION |
| `schemas/official_ground_truth_response_acceptance_criteria_schema.csv` | Schema de 7 campos para critérios de aceitação de respostas institucionais (v1ig) |
| `official_open_event_vector_discovery_registry.csv` | Descoberta e validação de vetores observados em bases públicas abertas (v1ih): 18 candidatos locais auditados por 10 gates — 0 ground truth confirmados; `operational_ground_truth_status=BLOCKED`, `can_create_training_label=false` para todos |
| `schemas/official_open_event_vector_discovery_registry_schema.csv` | Schema de 30 campos para registry de descoberta de vetores observados em bases abertas (v1ih) |
| `targeted_official_repository_event_vector_registry.csv` | Mineração dirigida em repositórios oficiais (v1ii-R1): 12 recursos auditados em 6 repositórios — 0 ground truth confirmados; lacuna de disponibilidade pública documentada; `operational_ground_truth_status=BLOCKED`, `can_create_training_label=false` para todos |
| `schemas/targeted_official_repository_event_vector_registry_schema.csv` | Schema de 27 campos para registry de mineração em repositórios oficiais (v1ii-R1) |
 6acdc62 (Audita vetores de eventos em repositórios oficiais)

## Protocolo C e camada de referência

A camada de referência contextual foi refinada pelo Protocolo C, que organiza a distinção entre evidência contextual, proxy auditável, candidato de referência e validação operacional. Ground truth operacional continua bloqueado no estado atual.

O `contextual_reference_layer_registry.csv` registra o status de evidência e os claims permitidos/proibidos por patch.

O `ground_reference_evidence_source_registry.csv` é o inventário de fontes de referência: classifica cada fonte por family (ex.: HYDROGEOMORPHOLOGICAL_CONTEXT, OPERATIONAL_FLOOD_PRODUCT), grau de observação (CONTEXTUAL, OPERATIONAL_ALGORITHMIC, EXPERT_INTERPRETED), e registra o allowed_use e forbidden_use de cada fonte. Fontes não adquiridas localmente são marcadas como NOT_ACQUIRED ou METHODOLOGICAL_REFERENCE_ONLY e não podem ser usadas como referência aplicada a patches.

Veja [`docs/metodologia_cientifica/protocolo_c_construcao_referencia_operacional.md`](../docs/metodologia_cientifica/protocolo_c_construcao_referencia_operacional.md) para a formulação completa do Protocolo C.

A etapa de fechamento de evidências adiciona mais três registros metadata-only: `ground_reference_gap_matrix.csv` mapeia os gates de promoção abertos por região com evidência faltante e risco metodológico; `review_gate_reference_registry.csv` organiza revisões supervisoras executadas ou placeholders com decisão e claims; e `reference_promotion_decision_registry.csv` registra decisões formais de promoção com `protocol_b_reassessment_allowed=false` em todas as linhas atuais. O Protocolo C agora inclui fechamento de evidências, revisão supervisora e decisão de promoção — formando trilha auditável para eventual ground reference. Ground truth operacional permanece não estabelecido. Veja [`protocolo_c_fechamento_evidencias_ground_reference.md`](../docs/metodologia_cientifica/protocolo_c_fechamento_evidencias_ground_reference.md) e [`protocolo_c_revisao_supervisora_referencia.md`](../docs/metodologia_cientifica/protocolo_c_revisao_supervisora_referencia.md).

A etapa de aquisição adiciona dois registros metadata-only: `flood_event_candidate_registry.csv` organiza eventos candidatos por região (com status de confirmação e elegibilidade), e `patch_event_reference_link_registry.csv` registra os vínculos entre patches, eventos e fontes com alinhamentos e bloqueadores explícitos. No estado atual, nenhum evento tem `eligible_for_reference_search=true` e nenhum vínculo tem `promotion_allowed=true`. Veja [`docs/metodologia_cientifica/protocolo_c_aquisicao_ground_reference.md`](../docs/metodologia_cientifica/protocolo_c_aquisicao_ground_reference.md) para a justificativa metodológica desta etapa.

A camada de plano de aquisição (v1hl) transforma metodologia em roteiro real: `observational_evidence_acquisition_plan.csv` organiza fontes-alvo por região, classifica pela força metodológica, documenta prioridades de aquisição e mapeia quais gates cada fonte pode fechar. `regional_ground_reference_readiness.csv` registra a prontidão regional para cada gate, identifica a evidência mais forte já disponível, descreve as lacunas críticas, e documenta allowed/forbidden claims por região. Essa camada continua metadata-only e não treina, prediz ou declara ground truth.

A camada de aquisição operacional (v1hm) coloca o plano em prática: `evidence_acquisition_tracker.csv` rastreia o estado atual de cada fonte-alvo, com acquisition_status, license_status, current_blocker e forbidden_use. `evidence_source_intake_registry.csv` registra fontes acessadas ou em processo com decisão de intake (ACCEPT_METADATA_ONLY, BLOCK_USE, REQUEST_MORE_INFORMATION). `evidence_license_provenance_registry.csv` documenta licença, redistribuição e proveniência para cada fonte, garantindo que raw_data_publication_allowed=false quando redistribuição não for explicitamente pública e que use_for_operational_ground_truth_allowed=FALSE em todas as linhas atuais. O GitHub continua contendo apenas metadados públicos seguros — dados brutos permanecem local-only.

A camada de dossiês de evidência (v1ho) especifica o que precisa ser encontrado por evento candidato: `event_evidence_dossier_registry.csv` tem um dossiê por evento com status (DOSSIER_OPEN para EVENT_SEARCH_TARGET, DOSSIER_PARTIAL para PENDING_SOURCE_REVIEW), lacunas de evidência mínima e `can_support_ground_reference_candidate=false` em todas as linhas. `event_evidence_requirements_registry.csv` registra os requisitos críticos por gate (EVENT_CONFIRMATION, TEMPORAL_EVIDENCE, SPATIAL_EVIDENCE, REVIEW_GATE, PROMOTION_DECISION) com `current_status=MISSING` ou `PARTIAL` e `blocking_if_missing=true` para requisitos críticos. `event_dossier_decision_registry.csv` tem uma decisão por dossiê com `can_reassess_protocol_b=false` e `can_start_multimodal=false` em todas as linhas. Ground truth operacional, Protocolo B e multimodal permanecem bloqueados/hold.

A camada de busca externa e solicitação regional (v1hp) transforma os dossiês em ação concreta: `regional_external_search_plan.csv` organiza onze planos de busca por região (Recife, Petrópolis, Curitiba), especificando fonte-alvo, gate, modo de busca (PUBLIC_PORTAL_REVIEW ou FORMAL_REQUEST), prioridade e status atual. `source_request_package_registry.csv` detalha sete pacotes de solicitação a instituições (COMPDEC Recife, CPRM, Defesa Civil Petrópolis, Defesa Civil Curitiba, GeoCuritiba) com `cannot_establish_ground_truth_alone=true` em todas as linhas e dado bruto local-only quando recebido. `gate_search_question_registry.csv` registra 17 perguntas de busca mapeadas a gates G1–G7, com `blocking_if_unanswered=true` para perguntas de G1, G3, G4 e G7. `regional_request_priority_matrix.csv` consolida a prioridade de solicitação por região com forbidden_claim bloqueando declaração de ground truth, flood label, training label, flood detection, flood prediction e supervised training em todas as entradas. Protocolo B permanece BLOCKED e multimodal permanece HOLD em toda a camada.

A camada de referências observacionais candidatas (v1hq) inicia a primeira prova documental de eventos observados: `observed_event_reference_candidate_registry.csv` registra 9 eventos (3 por região) com G1/G2/G3 fechados documentalmente e G4 em triagem espacial — prova de existência do evento, fonte rastreável e temporalidade. `observed_event_reference_gap_registry.csv` cataloga as lacunas por evento (patch overlay não executado, revisão supervisora não feita, licença pendente, geometria ausente). `observed_event_reference_decision_registry.csv` registra uma decisão por evento com `can_promote_to_ground_reference=false` e `can_generate_training_label=false` em todas as linhas. `manual_external_evidence_needed_registry.csv` cataloga os dados externos que precisam ser trazidos manualmente por região, com `cannot_establish_ground_truth_alone=true` em todas as entradas. Ground truth operacional não está estabelecido. G4 nesta etapa é apenas triagem espacial, não overlay patch-level. Protocolo B permanece BLOCKED. Multimodal permanece HOLD. DINO permanece SUPPORT_ONLY.

A camada de pré-ligação evento–patch (v1hr) prepara as condições para patch-linking sem executá-lo: `event_patch_linking_preflight_registry.csv` organiza o escopo regional de triagem para cada um dos 9 eventos com `promotion_allowed=false`, `can_create_training_label=false`, `protocol_b_status=BLOCKED` em todas as linhas. `manual_geocoding_target_registry.csv` lista 22 localidades a geocodificar manualmente (7 Recife, 10 Petrópolis, 5 Curitiba) com `geocoding_status=NOT_GEOCODED` ou `NEEDS_MANUAL_REVIEW`, sem coordenadas criadas. `event_sentinel_temporal_window_registry.csv` define janelas temporais metadata-only de pré/evento/pós-evento para busca futura de assets Sentinel, com `acquisition_status=NOT_ACQUIRED` em todas. `patch_linking_dependency_registry.csv` documenta 48 dependências (SOURCE_GEOMETRY, MANUAL_GEOCODING, LICENSE_PROVENANCE, SENTINEL_TEMPORAL_SEARCH, REVIEW_GATE, mais PHENOMENON_SEPARATION para Petrópolis) com `current_status=OPEN` e `required_before_ground_reference=true` para todas. Nenhum overlay foi executado. Nenhuma geocodificação foi realizada. Nenhuma coordenada foi criada. Nenhum dado bruto foi baixado.

A triagem de eventos candidatos (v1hn) organiza eventos candidatos reais por região em três registros metadata-only: `event_candidate_screening_registry.csv` lista cinco eventos candidatos (Recife 2021, Recife 2022, Petrópolis fev/2022, Curitiba 2022, Curitiba 2023) com status (EVENT_SEARCH_TARGET ou PENDING_SOURCE_REVIEW), prioridade de busca e gates potencialmente endereçáveis. `event_source_search_backlog.csv` organiza as fontes a pesquisar por evento candidato, com referência cruzada ao tracker (v1hm). `event_patch_screening_scope.csv` registra quais patches do corpus DINO estão no perímetro de busca de cada evento candidato — com `spatial_overlap_assessed=false` e `promotion_allowed=false` em todas as linhas. Nenhum dado foi baixado, nenhum gate foi fechado e DINOv2 permanece review-only.

A revisão programática de fontes observacionais (v1hu) analisa as 38 fontes registradas no manifest de aquisição e, quando o arquivo local está disponível em `local_only/`, extrai indícios documentais de evento, data, localidade, fenômeno e impacto. `programmatic_source_review_registry.csv` registra 38 linhas com força de evidência, candidatos a gates G1–G4 e guardrails obrigatórios (`requires_manual_review=true`, `can_generate_training_label=false`, `protocol_b_status=BLOCKED`, `can_support_patch_linking=false` para todas). `programmatic_gate_support_registry.csv` lista 152 pares (38 fontes × 4 gates) com `can_close_gate_automatically=false` em todas as linhas. `programmatic_source_evidence_gap_registry.csv` documenta 156 lacunas com `REVIEW_GATE_REQUIRED` e `GEOMETRY_NOT_AVAILABLE` para todas as fontes. Resultados: 2 fontes com STRONG_DOCUMENTARY_SUPPORT (NHESS e Copernicus IoD sobre Petrópolis), 2 com WEAK, 34 METADATA_ONLY. Os PDFs locais (DRM-RJ e Diário Oficial) não puderam ser lidos por ausência de biblioteca PDF. Nenhum gate foi fechado automaticamente. revisão supervisora obrigatória antes de qualquer avanço.

A matriz programática de decisão por evento (v1hv) agrega as evidências de v1hu ao nível do evento observado: `event_gate_decision_matrix.csv` consolida 36 linhas (9 eventos × 4 gates) com `event_gate_status`, `gate_can_advance`, `requires_reviewer_confirmation=true` e `can_close_gate_automatically=false` para todas. `event_ground_reference_readiness_registry.csv` registra 9 linhas de prontidão com `overall_readiness`, `ground_reference_candidate`, `can_promote_to_ground_reference=false`, `can_create_training_label=false` e `can_reopen_protocol_b=false` para todas. `event_next_action_registry.csv` lista 30 próximas ações (â‰¥3 por evento) com `can_automate=false` em todas. Resultados: PET_2022_02_15 é o único evento com READY_FOR_REVIEW e ground_reference_candidate=true; PET_2024_03_21_28 tem PARTIAL_READINESS; CTB_2024_02_18_20 tem BLOCKED (acesso); demais 6 eventos INSUFFICIENT. Ground truth operacional não está estabelecido. Protocolo B permanece BLOCKED. Multimodal permanece HOLD.

A aquisição dirigida e fechamento de lacunas (v1ht) endurece o script de aquisição v1hs com `--target-gaps` e 17 domínios no allowlist; cria 22 alvos de busca dirigida (7 Recife, 8 Petrópolis, 7 Curitiba) em `targeted_missing_source_search_registry.csv`; identifica 11 solicitações formais em `formal_source_request_target_registry.csv` com templates prontos em `docs/templates/`; e documenta o estado atual: 3 alvos BROWSER_MANUAL_REQUIRED, 7 NOT_FOUND_REQUIRES_FORMAL, 3 REVIEW_GATE_REQUIRED, 9 WEB_ACCESSIBLE pendentes. Nenhum novo arquivo foi adquirido nesta execução de catalogação; a aquisição real legada de v1hs permanece com 6 arquivos em local_only/. Nenhum gate foi fechado. Ground truth operacional não está estabelecido. Protocolo B permanece BLOCKED. Multimodal permanece HOLD.

A resolução programática de lacunas (v1hx) executa `--allow-web` nos 9 alvos WEB_ACCESSIBLE: 8 homepages de portal adquiridas em `local_only/` (WEB_ACQUIRED_LOCAL_ONLY) e 1 bloqueada (simepar.br, WEB_BLOCKED_BY_ACCESS); `targeted_missing_source_search_registry.csv` atualizado com WEB_* statuses reais; `observed_source_acquisition_manifest.csv` expandido de 38 para 46 linhas (ACQ_v1hx_001–008); 9 gaps atualizados para PORTAL_ACQUIRED ou OPEN com nota WEB_BLOCKED. Backend PDF ALL_MISSING (pypdf/PyPDF2/pdfminer/fitz/pdfplumber — ação pendente: `pip install pypdf`). FR_CTB_005 (Simepar) adicionado ao `formal_source_request_target_registry.csv`; 12 pacotes preenchidos criados em `docs/templates/protocolo_c_solicitacoes_preenchidas/` e registrados em `formal_request_package_registry.csv`. Re-execução de v1hw com 46 fontes produziu 3 decisões DOWNGRADE_TO_CONTEXTUAL_ONLY (REC e CTB) — portais têm vocabulário genérico sem confirmação específica de evento. A revisão profunda pré-geocodificação (v1hy) instala pypdf localmente (v6.12.0), extrai 1729 links de 8 HTMLs de portal — nenhum passa o filtro combinado de domínio + keyword de evento. `acquired_portal_deep_review_registry.csv` confirma todos os portais como PORTAL_HOMEPAGE_GENERIC. `pre_geocoding_closure_registry.csv` registra 9 eventos com todos os guardrails explicitamente false/BLOCKED/HOLD/SUPPORT_ONLY e seleciona 3 eventos para geocodificação controlada futura. Nenhum gate foi fechado. Ground truth operacional não está estabelecido. Protocolo B permanece BLOCKED. Multimodal permanece HOLD.

O pacote operacional de geocodificação controlada (v1hz) constrói o escopo espacial auditável: `controlled_geocoding_event_scope_registry.csv` registra 9 eventos (3 selecionados) com `geocoding_execution_allowed_now=false`, `overlay_execution_allowed_now=false`, `can_create_training_label=false`, `protocol_b_status=BLOCKED` para todos. `controlled_geocoding_target_registry.csv` registra 13 alvos de localidade (5 Recife + 6 Petrópolis 2022 + 2 Petrópolis 2024) — 6 bloqueados por PHENOMENON_SEPARATION_REQUIRED (PET_2022_02_15, CRITICAL), 7 prontos para futura geocodificação controlada; `coordinate_value_public=""` e `geometry_file_public=""` em todos os 13 (nenhuma coordenada ou geometria criada). `spatial_ground_reference_blocker_registry.csv` documenta 13 bloqueios (1 CRITICAL, 4 HIGH, 2 MEDIUM, 6 LOW) com `blocks_training_label=true` invariante. `future_controlled_geocoding_queue.csv` lista 5 itens de fila com `can_execute_now=false`, `can_create_training_label_after_execution=false` para todos. Nenhuma geocodificação foi executada. Nenhuma coordenada foi criada. Nenhuma geometria foi inferida. Ground truth operacional não está estabelecido. Protocolo B permanece BLOCKED. Multimodal permanece HOLD.

O inventário de fontes espaciais autoritativas (v1ia) identifica e classifica as fontes necessárias para os 13 alvos: `authoritative_spatial_source_inventory.csv` cataloga 15 fontes reais (nenhuma inventada), classificadas por classe, autoridade, status de acesso e capacidade de resolver bloqueios da v1hz; 5 fontes com FORMAL_REQUEST_REQUIRED. `controlled_geocoding_execution_preflight_registry.csv` avalia prontidão de cada alvo — 2 READY_FOR_TRIAGE_ONLY (Valparaíso e Floresta, PET_2024), 5 WAITING_OFFICIAL_SOURCE (Recife, aguardam COMPDEC), 6 WAITING_PHENOMENON_SEPARATION (PET_2022, bloqueio CRITICAL ativo). `authoritative_spatial_gap_registry.csv` documenta 6 lacunas com 1 CRITICAL (separação de fenômeno PET_2022 bloqueia geocodificação), 4 HIGH e 1 MEDIUM. Nenhuma geocodificação foi executada. Nenhuma coordenada foi criada. Ground truth operacional não está estabelecido. Protocolo B permanece BLOCKED. Multimodal permanece HOLD.

A consolidação de referências observacionais candidatas (v1ib) transforma o Protocolo C de camada de bloqueio em camada de promoção positiva: classifica 9 eventos por níveis de evidência acumulada (LEVEL_0–LEVEL_6). `observational_reference_promotion_registry.csv` registra 9 linhas com nível, decisão de promoção e guardrails invariantes (`can_be_called_ground_truth_operational=false`, `can_create_training_label=false`, `can_reopen_protocol_b=false`, `multimodal_status=HOLD`, `dino_usage_status=SUPPORT_ONLY` para todos). `protocolo_c_event_evidence_level_matrix.csv` documenta o estado de cada gate G1–G7 por evento. Resultados: PET_2022_02_15 promovido a STRONG candidata (LEVEL_5_SPATIAL_TRIAGE_READY) — 3 fontes SOURCE_CONFIRMED, 6 localidades, fenômeno MIXED documentado, bloqueado para geocodificação por separação de fenômeno; PET_2024_03_21_28 promovido a secundária (LEVEL_6_READY_FOR_CONTROLLED_GEOCODING); REC_2022_05_24_30 em LEVEL_4 (aguarda COMPDEC); 5 eventos em LEVEL_0 ou LEVEL_1 por ausência de fonte confirmada. Nenhum evento é ground truth operacional. Nenhum label criado. Protocolo B permanece BLOCKED.

A separação metodológica de fenômeno por localidade (v1ic) executa a classificação fenomenológica das 8 localidades de PET_2022_02_15  com base em evidência textual das fontes já adquiridas (DRM-RJ PDF 57p. e NHESS HTML). `event_locality_phenomenon_separation_registry.csv` registra 8 linhas com `phenomenon_class`, `phenomenon_confidence`, `source_ids`, `hydrological_terms_found`, `mass_movement_terms_found` e `blocks_controlled_geocoding=true` para todas. `event_phenomenon_separation_decision_registry.csv` registra a decisão consolidada: PARTIAL_SEPARATION — 0 HYDROLOGICAL_CONFIRMED, 3 MASS_MOVEMENT_CONFIRMED (Chácara Flora e Caxambu com HIGH, Morin com LOW), 5 MIXED_CONFIRMED. O achado hidrológico mais importante é o transbordamento do Rio Quitandinha documentado pelo DRM-RJ, mas a localidade Quitandinha permanece MIXED por sobreposição com lista de deslizamentos. Nenhuma localidade atingiu HYDROLOGICAL_CONFIRMED — o bloqueio de geocodificação controlada se mantém. `can_advance_to_controlled_geocoding_future=false`, `can_create_training_label=false`, `multimodal_status=HOLD`, `dino_usage_status=SUPPORT_ONLY` para todas as linhas.

A validação programática fonte–evento (v1hw) valida individualmente as 38 fontes contra os 9 eventos candidatos: `source_event_validation_registry.csv` registra 38 linhas com classificação de fenômeno, alinhamento temporal, evidência espacial, impacto, licença e `can_generate_training_label=false`, `can_support_ground_reference=false` para todas. `event_programmatic_validation_decision_registry.csv` registra 9 decisões por evento com `validation_decision`, confirmação final, fenômeno e `can_promote_to_ground_reference=false`, `can_create_training_label=false` em todas. `event_patch_compatibility_precheck_registry.csv` registra 9 pré-verificações de compatibilidade com `can_execute_overlay_now=false` em todas. `event_priority_for_geocoding_registry.csv` lista 9 eventos com prioridade de geocodificação — 2 selecionados (PET_2022_02_15 com REQUEST_PHENOMENON_SEPARATION e PET_2024_03_21_28 com KEEP_AS_PRIORITY) — `can_execute_overlay_after_geocoding=false` para todas. Nenhum overlay foi executado. Nenhuma geocodificação foi realizada. Nenhuma coordenada foi criada. Ground truth operacional não está estabelecido. Protocolo B permanece BLOCKED. Multimodal permanece HOLD.

A descoberta e validação de vetores em bases abertas (v1ih) audita 18 candidatos locais identificados em bases oficiais públicas por 10 gates de validação (fonte oficial, geometria, CRS, data, compatibilidade temporal, fenômeno, observação direta, separabilidade, escala espacial, ground truth). Resultado: 0 candidatos passam todos os gates. O gate mais restritivo é Gate 05 (data de evento compatível) com 39% de aprovação; Gate 09 (escala patch-level) com 0%. `official_open_event_vector_discovery_registry.csv` documenta todos os 18 candidatos com as classificações de bloqueio e allowed/forbidden claims.

A mineração dirigida em repositórios oficiais (v1ii-R1) estende a busca de v1ih a 6 repositórios públicos com APIs e catálogos indexados (RIGeo/SGB, CKAN Recife, CKAN Pernambuco/APAC, Dados Abertos RJ/DRM-RJ, GeoCuritiba/IPPUC, dados.gov.br/S2ID/Atlas). 12 recursos foram auditados — 4 EVENT_CONFIRMATION_ONLY, 3 CARTOGRAPHIC_LEAD_ONLY, 2 BLOCKED_NO_DATE, 2 SCAN_FAILED_CONTROLLED, 1 RISK_SUSCEPTIBILITY_ONLY — com 0 ground truth confirmados. `targeted_official_repository_event_vector_registry.csv` documenta todos os recursos com classificação, gate breakdown e guardrails. A lacuna de disponibilidade pública de vetor observado datado em escala patch-level é confirmada como resultado científico do Protocolo C.

## O que não está aqui

- GeoTIFFs, rasters, GeoJSONs brutos, shapefiles, geodatabases
- Embeddings `.npz` ou `.npy`
- Outputs de execução local (`local_runs/`)
- Dados de validação externa pesados (PE3D/MDE, SGB/RIGeo, GeoCuritiba)
- Qualquer arquivo que contenha paths absolutos de máquina local

Os registros descrevem esses materiais. Os materiais ficam locais.

## Navegação relacionada

- [`docs/metodologia_cientifica/research_datasets_and_artifacts.md`](../docs/metodologia_cientifica/research_datasets_and_artifacts.md) — narrativa metodológica completa
- [`docs/metodologia_cientifica/patch_lineage_and_grounding.md`](../docs/metodologia_cientifica/patch_lineage_and_grounding.md) — linhagem dos patches
- [`manifests/`](../manifests/) — manifests CSV/JSON por estágio do pipeline
