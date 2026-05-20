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
| `external_evidence_registry.csv` | Registro das evidências GIS externas por região |
| `contextual_reference_layer_registry.csv` | Camada de referência contextual validada: status de evidência e claims permitidos/proibidos por patch |
| `ground_reference_evidence_source_registry.csv` | Inventário de fontes de referência categorizado pelo Protocolo C: tipo, grau de observação, allowed_use, forbidden_use |
| `schemas/dataset_registry_schema.csv` | Schema de campos de dataset_registry.csv |
| `schemas/patch_corpus_schema.csv` | Schema de campos de patch_corpus_registry.csv |
| `schemas/external_evidence_schema.csv` | Schema de campos de external_evidence_registry.csv |
| `schemas/contextual_reference_layer_schema.csv` | Schema de campos de contextual_reference_layer_registry.csv |
| `schemas/ground_reference_evidence_source_schema.csv` | Schema de campos de ground_reference_evidence_source_registry.csv |
| `flood_event_candidate_registry.csv` | Registro de eventos de inundação candidatos por região — status de confirmação, elegibilidade e bloqueadores (etapa de aquisição Protocolo C) |
| `patch_event_reference_link_registry.csv` | Vínculos patch-evento-fonte com alinhamento temporal/espacial, candidatura e claims permitidos/proibidos (etapa de aquisição Protocolo C) |
| `schemas/flood_event_candidate_schema.csv` | Schema de campos de flood_event_candidate_registry.csv |
| `schemas/patch_event_reference_link_schema.csv` | Schema de campos de patch_event_reference_link_registry.csv |
| `ground_reference_gap_matrix.csv` | Matriz de lacunas de evidência por região: gates abertos, evidência faltante, risco metodológico e próximos passos permitidos/proibidos (etapa de fechamento Protocolo C) |
| `human_reference_review_registry.csv` | Registry de revisões humanas ou placeholders: decisão, materiais revisados, consistency checks, allowed_claim e forbidden_claim por revisão (etapa de fechamento Protocolo C) |
| `reference_promotion_decision_registry.csv` | Registry de decisões formais de promoção: gates satisfeitos/falhados, final_reference_status, protocol_b_reassessment_allowed (etapa de fechamento Protocolo C) |
| `schemas/ground_reference_gap_matrix_schema.csv` | Schema de campos de ground_reference_gap_matrix.csv |
| `schemas/human_reference_review_schema.csv` | Schema de campos de human_reference_review_registry.csv |
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
| `patch_linking_dependency_registry.csv` | Dependências metodológicas para patch-linking real (v1hr): o que deve ser resolvido antes de overlay, human review e ground reference — current_status=OPEN, required_before_ground_reference=true para todas |
| `schemas/patch_linking_dependency_schema.csv` | Schema de campos de patch_linking_dependency_registry.csv |

## Protocolo C e camada de referência

A camada de referência contextual foi refinada pelo Protocolo C, que organiza a distinção entre evidência contextual, proxy auditável, candidato de referência e validação operacional. Ground truth operacional continua bloqueado no estado atual.

O `contextual_reference_layer_registry.csv` registra o status de evidência e os claims permitidos/proibidos por patch.

O `ground_reference_evidence_source_registry.csv` é o inventário de fontes de referência: classifica cada fonte por family (ex.: HYDROGEOMORPHOLOGICAL_CONTEXT, OPERATIONAL_FLOOD_PRODUCT), grau de observação (CONTEXTUAL, OPERATIONAL_ALGORITHMIC, EXPERT_INTERPRETED), e registra o allowed_use e forbidden_use de cada fonte. Fontes não adquiridas localmente são marcadas como NOT_ACQUIRED ou METHODOLOGICAL_REFERENCE_ONLY e não podem ser usadas como referência aplicada a patches.

Veja [`docs/metodologia_cientifica/protocolo_c_construcao_referencia_operacional.md`](../docs/metodologia_cientifica/protocolo_c_construcao_referencia_operacional.md) para a formulação completa do Protocolo C.

A etapa de fechamento de evidências adiciona mais três registros metadata-only: `ground_reference_gap_matrix.csv` mapeia os gates de promoção abertos por região com evidência faltante e risco metodológico; `human_reference_review_registry.csv` organiza revisões humanas executadas ou placeholders com decisão e claims; e `reference_promotion_decision_registry.csv` registra decisões formais de promoção com `protocol_b_reassessment_allowed=false` em todas as linhas atuais. O Protocolo C agora inclui fechamento de evidências, revisão humana e decisão de promoção — formando trilha auditável para eventual ground reference. Ground truth operacional permanece não estabelecido. Veja [`protocolo_c_fechamento_evidencias_ground_reference.md`](../docs/metodologia_cientifica/protocolo_c_fechamento_evidencias_ground_reference.md) e [`protocolo_c_revisao_humana_referencia.md`](../docs/metodologia_cientifica/protocolo_c_revisao_humana_referencia.md).

A etapa de aquisição adiciona dois registros metadata-only: `flood_event_candidate_registry.csv` organiza eventos candidatos por região (com status de confirmação e elegibilidade), e `patch_event_reference_link_registry.csv` registra os vínculos entre patches, eventos e fontes com alinhamentos e bloqueadores explícitos. No estado atual, nenhum evento tem `eligible_for_reference_search=true` e nenhum vínculo tem `promotion_allowed=true`. Veja [`docs/metodologia_cientifica/protocolo_c_aquisicao_ground_reference.md`](../docs/metodologia_cientifica/protocolo_c_aquisicao_ground_reference.md) para a justificativa metodológica desta etapa.

A camada de plano de aquisição (v1hl) transforma metodologia em roteiro real: `observational_evidence_acquisition_plan.csv` organiza fontes-alvo por região, classifica pela força metodológica, documenta prioridades de aquisição e mapeia quais gates cada fonte pode fechar. `regional_ground_reference_readiness.csv` registra a prontidão regional para cada gate, identifica a evidência mais forte já disponível, descreve as lacunas críticas, e documenta allowed/forbidden claims por região. Essa camada continua metadata-only e não treina, prediz ou declara ground truth.

A camada de aquisição operacional (v1hm) coloca o plano em prática: `evidence_acquisition_tracker.csv` rastreia o estado atual de cada fonte-alvo, com acquisition_status, license_status, current_blocker e forbidden_use. `evidence_source_intake_registry.csv` registra fontes acessadas ou em processo com decisão de intake (ACCEPT_METADATA_ONLY, BLOCK_USE, REQUEST_MORE_INFORMATION). `evidence_license_provenance_registry.csv` documenta licença, redistribuição e proveniência para cada fonte, garantindo que raw_data_publication_allowed=false quando redistribuição não for explicitamente pública e que use_for_operational_ground_truth_allowed=FALSE em todas as linhas atuais. O GitHub continua contendo apenas metadados públicos seguros — dados brutos permanecem local-only.

A camada de dossiês de evidência (v1ho) especifica o que precisa ser encontrado por evento candidato: `event_evidence_dossier_registry.csv` tem um dossiê por evento com status (DOSSIER_OPEN para EVENT_SEARCH_TARGET, DOSSIER_PARTIAL para PENDING_SOURCE_REVIEW), lacunas de evidência mínima e `can_support_ground_reference_candidate=false` em todas as linhas. `event_evidence_requirements_registry.csv` registra os requisitos críticos por gate (EVENT_CONFIRMATION, TEMPORAL_EVIDENCE, SPATIAL_EVIDENCE, HUMAN_REVIEW, PROMOTION_DECISION) com `current_status=MISSING` ou `PARTIAL` e `blocking_if_missing=true` para requisitos críticos. `event_dossier_decision_registry.csv` tem uma decisão por dossiê com `can_reassess_protocol_b=false` e `can_start_multimodal=false` em todas as linhas. Ground truth operacional, Protocolo B e multimodal permanecem bloqueados/hold.

A camada de busca externa e solicitação regional (v1hp) transforma os dossiês em ação concreta: `regional_external_search_plan.csv` organiza onze planos de busca por região (Recife, Petrópolis, Curitiba), especificando fonte-alvo, gate, modo de busca (PUBLIC_PORTAL_REVIEW ou FORMAL_REQUEST), prioridade e status atual. `source_request_package_registry.csv` detalha sete pacotes de solicitação a instituições (COMPDEC Recife, CPRM, Defesa Civil Petrópolis, Defesa Civil Curitiba, GeoCuritiba) com `cannot_establish_ground_truth_alone=true` em todas as linhas e dado bruto local-only quando recebido. `gate_search_question_registry.csv` registra 17 perguntas de busca mapeadas a gates G1–G7, com `blocking_if_unanswered=true` para perguntas de G1, G3, G4 e G7. `regional_request_priority_matrix.csv` consolida a prioridade de solicitação por região com forbidden_claim bloqueando declaração de ground truth, flood label, training label, flood detection, flood prediction e supervised training em todas as entradas. Protocolo B permanece BLOCKED e multimodal permanece HOLD em toda a camada.

A camada de referências observacionais candidatas (v1hq) inicia a primeira prova documental de eventos observados: `observed_event_reference_candidate_registry.csv` registra 9 eventos (3 por região) com G1/G2/G3 fechados documentalmente e G4 em triagem espacial — prova de existência do evento, fonte rastreável e temporalidade. `observed_event_reference_gap_registry.csv` cataloga as lacunas por evento (patch overlay não executado, revisão humana não feita, licença pendente, geometria ausente). `observed_event_reference_decision_registry.csv` registra uma decisão por evento com `can_promote_to_ground_reference=false` e `can_generate_training_label=false` em todas as linhas. `manual_external_evidence_needed_registry.csv` cataloga os dados externos que precisam ser trazidos manualmente por região, com `cannot_establish_ground_truth_alone=true` em todas as entradas. Ground truth operacional não está estabelecido. G4 nesta etapa é apenas triagem espacial, não overlay patch-level. Protocolo B permanece BLOCKED. Multimodal permanece HOLD. DINO permanece SUPPORT_ONLY.

A camada de pré-ligação evento–patch (v1hr) prepara as condições para patch-linking sem executá-lo: `event_patch_linking_preflight_registry.csv` organiza o escopo regional de triagem para cada um dos 9 eventos com `promotion_allowed=false`, `can_create_training_label=false`, `protocol_b_status=BLOCKED` em todas as linhas. `manual_geocoding_target_registry.csv` lista 22 localidades a geocodificar manualmente (7 Recife, 10 Petrópolis, 5 Curitiba) com `geocoding_status=NOT_GEOCODED` ou `NEEDS_MANUAL_REVIEW`, sem coordenadas criadas. `event_sentinel_temporal_window_registry.csv` define janelas temporais metadata-only de pré/evento/pós-evento para busca futura de assets Sentinel, com `acquisition_status=NOT_ACQUIRED` em todas. `patch_linking_dependency_registry.csv` documenta 48 dependências (SOURCE_GEOMETRY, MANUAL_GEOCODING, LICENSE_PROVENANCE, SENTINEL_TEMPORAL_SEARCH, HUMAN_REVIEW, mais PHENOMENON_SEPARATION para Petrópolis) com `current_status=OPEN` e `required_before_ground_reference=true` para todas. Nenhum overlay foi executado. Nenhuma geocodificação foi realizada. Nenhuma coordenada foi criada. Nenhum dado bruto foi baixado.

A triagem de eventos candidatos (v1hn) organiza eventos candidatos reais por região em três registros metadata-only: `event_candidate_screening_registry.csv` lista cinco eventos candidatos (Recife 2021, Recife 2022, Petrópolis fev/2022, Curitiba 2022, Curitiba 2023) com status (EVENT_SEARCH_TARGET ou PENDING_SOURCE_REVIEW), prioridade de busca e gates potencialmente endereçáveis. `event_source_search_backlog.csv` organiza as fontes a pesquisar por evento candidato, com referência cruzada ao tracker (v1hm). `event_patch_screening_scope.csv` registra quais patches do corpus DINO estão no perímetro de busca de cada evento candidato — com `spatial_overlap_assessed=false` e `promotion_allowed=false` em todas as linhas. Nenhum dado foi baixado, nenhum gate foi fechado e DINOv2 permanece review-only.

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
