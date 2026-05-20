# REV-P

## Visão geral

O REV-P é um protocolo auditável para organizar e inspecionar evidências físico-ambientais, geoespaciais e visuais sobre patches urbanos associados a suscetibilidade a inundação e alagamento. O repositório concentra manifests, scripts, testes e documentação técnica do pipeline DINO Sentinel-first.

## Escopo científico

O projeto está em estágio de revisão e auditoria estrutural. Não há classificação supervisionada de suscetibilidade, rótulos binários de enchente observada, alvos de treinamento ou afirmações preditivas.

O DINO é usado exclusivamente como encoder visual congelado para extração de características estruturais de patches Sentinel. O índice GIS (v1gq–v1gt) é um proxy interpretável para comparação e triagem — não é verdade de campo nem alvo supervisionado.

## Estrutura do repositório

```
configs/          Configurações de exemplo (parâmetros de extração DINO)
manifests/        Manifests CSV/JSON auditáveis de patches, preflight e validação
scripts/          Scripts do pipeline (trilha DINO e preparação de treinamento)
tests/            Testes automatizados de cada estágio do pipeline
docs/             Protocolo técnico, registro de comandos e estado metodológico
requirements.txt  Dependências Python do projeto
```

## O que não está versionado

Dados brutos, GeoTIFFs, shapefiles, GeoJSONs convertidos, embeddings `.npz`, outputs locais em `local_runs/`, caches, modelos pesados e arquivos locais de desenvolvimento não são versionados nem enviados ao repositório público.

## Linhagem dos patches

Os patches são recortes territoriais pré-existentes sobre áreas urbanas de Curitiba (14), Petrópolis (27) e Recife (18), com bounding boxes originadas de bases externas anteriores ao pipeline DINO. O DINO opera sobre imagens Sentinel associadas a esses patches — não define nem requalifica os limites territoriais.

Detalhes em [docs/patch_lineage_and_grounding.md](docs/patch_lineage_and_grounding.md).

## Trilha DINO Sentinel-first

O pipeline segue a ordem:

1. Manifesto Sentinel (v1fu) — inventário de 128 TIFs Sentinel elegíveis nas três regiões
2. Preflight local (v1fv) — verificação de quais referências são acessíveis no workspace privado
3. Execução smoke de embeddings (v1fx) — leitura real de pixels, extração local
4. Análise estrutural (v1fy–v1gi) — PCA, clustering, vizinhos, outliers, proveniência
5. Auditorias operacionais (v1gn–v1gp) — saúde, orquestração, prontidão para release
6. Auditorias GIS (v1gq–v1gt) — baseline multicritério, uso do solo, cobertura de fontes

Todos os outputs de execução ficam exclusivamente em `local_runs/`.

## Travas metodológicas

- Sem labels ou targets supervisionados
- Sem treinamento supervisionado
- Sem afirmações preditivas de vulnerabilidade
- Sem ativação multimodal (em espera)
- Índice GIS não é ground truth
- DINO não prediz vulnerabilidade
- `review_only=true`

## Datasets auditáveis e artefatos de pesquisa

O projeto produziu manifests públicos, registros de corpus e documentação de
evidências externas sem versionar dados pesados (rasters, embeddings, shapefiles).

- [`datasets/`](datasets/) — registros estruturados de datasets, corpora e evidências externas
- [`datasets/dataset_registry.csv`](datasets/dataset_registry.csv) — inventário geral de artefatos
- [`datasets/patch_corpus_registry.csv`](datasets/patch_corpus_registry.csv) — corpora de patches por estágio
- [`datasets/external_evidence_registry.csv`](datasets/external_evidence_registry.csv) — evidências GIS por região
- [`datasets/contextual_reference_layer_registry.csv`](datasets/contextual_reference_layer_registry.csv) — camada de referência contextual: status de evidência e claims por patch
- [`datasets/ground_reference_evidence_source_registry.csv`](datasets/ground_reference_evidence_source_registry.csv) — inventário de fontes de referência categorizado pelo Protocolo C
- [`docs/metodologia_cientifica/research_datasets_and_artifacts.md`](docs/metodologia_cientifica/research_datasets_and_artifacts.md) — narrativa metodológica dos datasets

## Protocolo C — construção de referência operacional

A camada de referência contextual foi refinada pelo Protocolo C, que organiza evidências externas, critérios de promoção e bloqueadores de operacionalização de forma auditável. O protocolo distingue explicitamente contexto, proxy, candidato de referência e validação operacional — sem declarar ground truth onde ele não existe. Ground truth operacional continua bloqueado no estado atual.

- [docs/metodologia_cientifica/protocolo_c_construcao_referencia_operacional.md](docs/metodologia_cientifica/protocolo_c_construcao_referencia_operacional.md) — Protocolo C: critérios de promoção, bloqueadores e relação com a literatura
- [docs/metodologia_cientifica/protocolo_c_aquisicao_ground_reference.md](docs/metodologia_cientifica/protocolo_c_aquisicao_ground_reference.md) — etapa de aquisição: registro metadata-only de eventos candidatos e vínculos patch-evento-fonte
- [docs/metodologia_cientifica/protocolo_c_plano_aquisicao_evidencias_observacionais.md](docs/metodologia_cientifica/protocolo_c_plano_aquisicao_evidencias_observacionais.md) — plano de aquisição de evidências observacionais por região (v1hl): fontes-alvo, prioridades, força metodológica e readiness regional
- [docs/metodologia_cientifica/protocolo_c_pacote_operacional_aquisicao_evidencias.md](docs/metodologia_cientifica/protocolo_c_pacote_operacional_aquisicao_evidencias.md) — pacote operacional de aquisição (v1hm): princípios, fluxo, intake, licenciamento, staging local e bloqueios
- [docs/metodologia_cientifica/protocolo_c_runbook_aquisicao_evidencias.md](docs/metodologia_cientifica/protocolo_c_runbook_aquisicao_evidencias.md) — runbook passo a passo para coleta futura de evidências (v1hm)
- [docs/metodologia_cientifica/protocolo_c_fechamento_evidencias_ground_reference.md](docs/metodologia_cientifica/protocolo_c_fechamento_evidencias_ground_reference.md) — etapa de fechamento: gates de promoção, níveis de evidência e matriz de lacunas por região
- [docs/metodologia_cientifica/protocolo_c_revisao_humana_referencia.md](docs/metodologia_cientifica/protocolo_c_revisao_humana_referencia.md) — protocolo de revisão humana: decisões possíveis, critérios de bloqueio e registro obrigatório
- [docs/metodologia_cientifica/protocolo_c_triagem_eventos_candidatos.md](docs/metodologia_cientifica/protocolo_c_triagem_eventos_candidatos.md) — triagem de eventos candidatos por região (v1hn): status, prioridade de busca, backlog de fontes, escopo por patch e gates do Protocolo C
- [docs/metodologia_cientifica/protocolo_c_dossies_eventos_candidatos.md](docs/metodologia_cientifica/protocolo_c_dossies_eventos_candidatos.md) — dossiês de evidência por evento candidato (v1ho): pacote mínimo de evidências, estados do dossiê, critérios de bloqueio e decisões de continuidade
- [docs/metodologia_cientifica/protocolo_c_busca_externa_solicitacao_regional.md](docs/metodologia_cientifica/protocolo_c_busca_externa_solicitacao_regional.md) — busca externa e solicitação regional (v1hp): planos de busca por região, pacotes de solicitação, perguntas por gate e matriz de prioridade regional
- [docs/metodologia_cientifica/protocolo_c_referencias_observacionais_candidatas.md](docs/metodologia_cientifica/protocolo_c_referencias_observacionais_candidatas.md) — referências observacionais candidatas (v1hq): primeira camada documental de eventos observados candidatos, diferenciação de níveis de evidência, gates G1–G4 em triagem, bloqueios e inventário de dados externos
- [docs/metodologia_cientifica/protocolo_c_diagnostico_dados_externos_validos.md](docs/metodologia_cientifica/protocolo_c_diagnostico_dados_externos_validos.md) — diagnóstico de dados externos válidos (v1hq): o que buscar manualmente por região, estrutura local-only e o que vai e não vai para o GitHub
- [docs/metodologia_cientifica/protocolo_c_pre_ligacao_evento_patch.md](docs/metodologia_cientifica/protocolo_c_pre_ligacao_evento_patch.md) — pré-ligação evento–patch (v1hr): diferença entre G4 em triagem e patch-linking real, geocodificação manual controlada, janelas temporais Sentinel metadata-only e dependências para overlay futuro
- [docs/metodologia_cientifica/camada_referencia_contextual_validada.md](docs/metodologia_cientifica/camada_referencia_contextual_validada.md) — hierarquia de status e guardrails por patch

O Protocolo C agora inclui pacote operacional de aquisição/intake (v1hm), camada de busca externa e solicitação regional (v1hp), primeira camada documental de eventos observados candidatos (v1hq) e camada de pré-ligação evento–patch (v1hr). A v1hq registra 9 eventos (3 por região) com G1/G2/G3 fechados documentalmente e G4 em triagem espacial. A v1hr prepara as condições para patch-linking sem executá-lo: organiza escopos de preflight, alvos de geocodificação manual (22 localidades), janelas temporais Sentinel metadata-only e dependências para overlay futuro. Nenhum overlay foi executado, nenhuma geocodificação foi realizada, nenhuma coordenada foi criada, nenhum dado bruto foi baixado. Ground truth operacional não está estabelecido. Protocolo B permanece bloqueado. Multimodal permanece em hold. O GitHub continua contendo apenas metadados públicos seguros.

### Datasets das etapas de aquisição, fechamento, triagem, busca e referências observacionais

- [`datasets/regional_external_search_plan.csv`](datasets/regional_external_search_plan.csv) — planos de busca externa por região (v1hp): fonte-alvo, gate, modo, prioridade e status (`forbidden_use` bloqueia ground truth e labels)
- [`datasets/source_request_package_registry.csv`](datasets/source_request_package_registry.csv) — pacotes de solicitação formal a instituições (v1hp): instituição, tipo de solicitação, status e `cannot_establish_ground_truth_alone=true`
- [`datasets/gate_search_question_registry.csv`](datasets/gate_search_question_registry.csv) — perguntas de busca por gate e região (v1hp): `current_answer_status`, blocking_if_unanswered e forbidden_if_unanswered
- [`datasets/regional_request_priority_matrix.csv`](datasets/regional_request_priority_matrix.csv) — matriz de prioridade regional de solicitação (v1hp): `protocol_b_status=BLOCKED`, `multimodal_status=HOLD`
- [`datasets/observed_event_reference_candidate_registry.csv`](datasets/observed_event_reference_candidate_registry.csv) — 9 eventos observados candidatos (v1hq): G1/G2/G3 fechados documentalmente, G4 em triagem, `operational_ground_truth_status=NOT_ESTABLISHED`, `protocol_b_status=BLOCKED`, `can_be_used_as_training_label=false`
- [`datasets/observed_event_reference_gap_registry.csv`](datasets/observed_event_reference_gap_registry.csv) — lacunas metodológicas por evento (v1hq): o que falta para avançar à ligação patch-evento e ground reference
- [`datasets/observed_event_reference_decision_registry.csv`](datasets/observed_event_reference_decision_registry.csv) — decisões metodológicas por evento (v1hq): `can_promote_to_ground_reference=false`, `can_generate_training_label=false`, `can_reopen_protocol_b=false` para todos
- [`datasets/manual_external_evidence_needed_registry.csv`](datasets/manual_external_evidence_needed_registry.csv) — inventário de dados externos necessários por região (v1hq): o que buscar manualmente, `cannot_establish_ground_truth_alone=true` para todos
- [`datasets/event_patch_linking_preflight_registry.csv`](datasets/event_patch_linking_preflight_registry.csv) — preflight de pré-ligação evento–patch (v1hr): escopo regional, status de overlay, bloqueios e guardrails por linha (`promotion_allowed=false`, `can_create_training_label=false`, `protocol_b_status=BLOCKED` para todos)
- [`datasets/manual_geocoding_target_registry.csv`](datasets/manual_geocoding_target_registry.csv) — alvos de geocodificação manual (v1hr): 22 localidades a geocodificar por evento, tipo, fonte e status (`geocoding_status=NOT_GEOCODED` ou `NEEDS_MANUAL_REVIEW`, `cannot_establish_ground_truth_alone=true` para todos)
- [`datasets/event_sentinel_temporal_window_registry.csv`](datasets/event_sentinel_temporal_window_registry.csv) — janelas temporais Sentinel por evento (v1hr): períodos pré/evento/pós metadata-only, relevância de sensor e status de aquisição (`acquisition_status=NOT_ACQUIRED`, `cannot_establish_ground_truth_alone=true` para todos)
- [`datasets/patch_linking_dependency_registry.csv`](datasets/patch_linking_dependency_registry.csv) — dependências para patch-linking real (v1hr): o que precisa ser resolvido antes de overlay, human review e ground reference (`current_status=OPEN` para todas)
- [`datasets/event_evidence_dossier_registry.csv`](datasets/event_evidence_dossier_registry.csv) — dossiês de evidência (v1ho): status do dossiê, lacunas, decisão de continuidade (`can_support_ground_reference_candidate=false` no estado atual)
- [`datasets/event_evidence_requirements_registry.csv`](datasets/event_evidence_requirements_registry.csv) — requisitos mínimos de evidência por evento candidato (v1ho): status por gate, blocking_if_missing e forbidden_if_missing
- [`datasets/event_dossier_decision_registry.csv`](datasets/event_dossier_decision_registry.csv) — decisões de continuidade por dossiê (v1ho): `can_reassess_protocol_b=false`, `can_start_multimodal=false`
- [`datasets/event_candidate_screening_registry.csv`](datasets/event_candidate_screening_registry.csv) — triagem de eventos candidatos (v1hn): status, prioridade de busca e gates por evento (`promotion_allowed=false` no estado atual)
- [`datasets/event_source_search_backlog.csv`](datasets/event_source_search_backlog.csv) — backlog de fontes a pesquisar por evento candidato (v1hn): fonte, família, status da busca
- [`datasets/event_patch_screening_scope.csv`](datasets/event_patch_screening_scope.csv) — escopo de triagem por patch (v1hn): perímetro de busca, `spatial_overlap_assessed=false` e `promotion_allowed=false`
- [`datasets/flood_event_candidate_registry.csv`](datasets/flood_event_candidate_registry.csv) — eventos candidatos por região (`eligible_for_reference_search=false` no estado atual)
- [`datasets/patch_event_reference_link_registry.csv`](datasets/patch_event_reference_link_registry.csv) — vínculos patch-evento-fonte com alinhamentos e bloqueadores (`promotion_allowed=false` no estado atual)
- [`datasets/ground_reference_gap_matrix.csv`](datasets/ground_reference_gap_matrix.csv) — matriz de lacunas: gates abertos, evidência faltante e próximo passo por região (`promotion_blocked=true` no estado atual)
- [`datasets/human_reference_review_registry.csv`](datasets/human_reference_review_registry.csv) — registry de revisões humanas ou placeholders (`promotion_allowed=false` no estado atual)
- [`datasets/reference_promotion_decision_registry.csv`](datasets/reference_promotion_decision_registry.csv) — decisões formais de promoção (`promotion_allowed=false`, `protocol_b_reassessment_allowed=false` no estado atual)

## Documentação técnica

- [docs/metodologia_cientifica/camada_referencia_contextual_validada.md](docs/metodologia_cientifica/camada_referencia_contextual_validada.md) — hierarquia de evidências e claims permitidos/proibidos por status
- [docs/metodologia_cientifica/patch_lineage_and_grounding.md](docs/metodologia_cientifica/patch_lineage_and_grounding.md) — linhagem territorial dos patches, vinculação Sentinel, claims permitidos e proibidos
- [docs/metodologia_cientifica/dino_sentinel_embedding_protocol.md](docs/metodologia_cientifica/dino_sentinel_embedding_protocol.md) — protocolo completo do pipeline DINO
- [docs/metodologia_cientifica/dino_command_registry.md](docs/metodologia_cientifica/dino_command_registry.md) — registro de comandos para reprodução local
- [docs/metodologia_cientifica/dino_sentinel_scientific_evidence_summary.md](docs/metodologia_cientifica/dino_sentinel_scientific_evidence_summary.md) — resumo de evidências científicas
- [docs/estado_metodologico_revp.md](docs/estado_metodologico_revp.md) — estado e limitações metodológicas atuais
