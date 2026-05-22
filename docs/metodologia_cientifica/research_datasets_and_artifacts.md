# Datasets e artefatos de pesquisa do REV-P

## O que o projeto produziu

O REV-P produziu quatro tipos de artefatos com papéis metodológicos distintos:

**Manifests auditáveis** — tabelas CSV/JSON que descrevem corpora, assets e decisões
metodológicas. São commitados neste repositório porque documentam o método sem
depender de arquivos pesados. O manifest Sentinel v1fu registra 128 candidatos com
campos explícitos de `label_status=NO_LABEL`, `target_status=NO_TARGET` e
`claim_scope=REVIEW_ONLY_NO_PREDICTIVE_CLAIM`.

**Corpus de patches Sentinel** — 59 patches territoriais (14 Curitiba, 27 Petrópolis,
18 Recife) com bounding boxes derivadas de bases externas pré-DINO. Os GeoTIFFs
correspondentes existem no workspace privado, somam múltiplos gigabytes e não são
versionados. O que é público é a rastreabilidade: qual patch tem qual designação de
TIF, qual QA foi executado, qual estado de vinculação persiste.

**Embeddings DINO** — representações visuais extraídas pelo encoder DINOv2 com
registros (congelado) sobre patches Sentinel. O corpus operacional tem 12 embeddings
balanceados (4 por região). Arquivos `.npz` ficam em `local_runs/` e não são
versionados. São evidência de execução técnica, não dataset para treinamento.

**Evidências externas GIS** — fontes institucionais por região (PE3D/Recife,
SGB/Petrópolis, GeoCuritiba/Curitiba) indexadas no pacote de validação externa. Os
arquivos pesados ficam locais; o pacote público contém CSV/JSON de evidência e guardrails.

---

## Por que os patches Sentinel são uma contribuição científica

Os 59 patches não foram selecionados aleatoriamente. A seleção passou por:

1. Grounding territorial em áreas com histórico de inundação e alagamento documentado
   em fontes institucionais (Defesa Civil, SGB/CPRM, PE3D, GeoCuritiba)
2. Derivação de bounding boxes a partir de bases externas pré-existentes — o DINO
   não define os limites, opera sobre o que já estava territorialmente consolidado
3. Separação metodológica de três regiões com papéis distintos: Recife como caso de
   evidência forte, Petrópolis como caso de complexidade de processo, Curitiba como
   região de contraste metodológico
4. Auditoria de vinculação Sentinel por patch (v1fm–v1fo), com documentação explícita
   de cada estado de resolução — incluindo os estados não resolvidos

O corpus não é um dataset de treinamento. É um conjunto territorial auditável para
análise exploratória e revisão estrutural.

---

## Por que os manifests são auditáveis

Um manifest é auditável quando qualquer leitor pode verificar, a partir do arquivo
público, que uma decisão metodológica foi tomada e quais guardrails ela impõe.

O manifest v1fu satisfaz isso: cada linha tem `label_status`, `target_status`,
`pixel_read_status` e `claim_scope` preenchidos com valores fixos que proíbem
classificação supervisionada. O script que gerou o manifest tem um QA automatizado
que falha se qualquer desses campos for diferente do esperado. Isso é rastreabilidade
por código, não apenas por documentação.

O pacote de validação externa tem um CSV de guardrails (`external_validation_master_claims_guardrail_v1.csv`)
com claims `ALLOWED` e `FORBIDDEN` explicitamente separados. Qualquer redação futura
pode ser verificada contra essa lista.

---

## Por que os dados pesados ficam fora do GitHub

Os GeoTIFFs Sentinel têm resolução de 10 m, cobrem patches de ~1,6 km² e somam
entre 10 MB e 200 MB por arquivo. Os 128 candidatos totalizam múltiplos gigabytes.
Versionar esses arquivos:

- tornaria o repositório impraticável para clonar;
- não adicionaria valor científico: as imagens Sentinel são de acesso público via
  Copernicus Open Access Hub e podem ser reproduzidas a partir dos metadados dos
  manifests;
- exporia caminhos e estruturas do workspace privado.

O que o repositório versiona é suficiente para verificar a metodologia: manifests com
referências de caminho relativo, QA passado, configuração do encoder, guardrails
explícitos. Um revisor pode confirmar que o método é auditável sem acesso aos rasters.

---

## Como os datasets se conectam à metodologia

A sequência abaixo é a cadeia de rastreabilidade do projeto:

**Grounding territorial**
Seleção de três regiões com histórico documentado. Bounding boxes derivadas de bases
externas. O DINO não participou dessa etapa.

↓

**Sentinel-first**
128 GeoTIFFs Level-2A inventariados. Decisão de priorizar Sentinel sobre multimodal
documentada em v1ft com justificativa quantitativa: 1 stack Recife disponível contra
37 candidatos Sentinel — desequilíbrio que bloqueia multimodal até recuperação.

↓

**Auditabilidade dos patches**
Designação TIF por patch (v1fm): 20 patches com candidato; 32 sem resolução; 7
placeholder. Estado documentado, não encoberto. Reconciliação de naming Recife ext/bg
(v1fo): 18 patches com problema de nomenclatura documentado e não resolvido.

↓

**Manifests**
v1fu: 128 entradas Sentinel com `label_status=NO_LABEL`, `target_status=NO_TARGET`,
QA PASS. Nenhum pixel lido na construção do manifest.

↓

**QA**
18 guardrails auditados a zero antes de qualquer extração. Pacote de validação
externa: 11 QA checks passados, 0 falhas.

↓

**Embeddings DINO**
DINOv2 com registros, encoder congelado. Extração local em v1fx (5 patches, smoke)
e v1fz (12 patches, corpus balanceado). Embeddings em `local_runs/` — não versionados,
não labels, não targets. Diagnósticos estruturais: kNN, clustering, outliers,
robustez, proveniência, triagem de revisão humana.

↓

**Contextualização GIS**
Índice multicritério v1gq sobre os 12 patches do corpus: distância ao rio, uso do
solo, densidade viária. Não é ground truth. Não é alvo supervisionado. É proxy
interpretável para comparação estrutural com os embeddings.

↓

**Revisão humana**
Pacote de evidências externas por região: PE3D/Recife (evidência forte de terreno),
SGB/Petrópolis (terreno com separação de processo obrigatória), GeoCuritiba (contraste
metodológico). Estado: mastered for review — pacote pronto, revisão humana não executada.

↓

**Plano de aquisição de evidências observacionais (v1hl)**
Protocolo C evolui de aquisição de eventos candidatos para plano real de aquisição de
evidências por região. Documento `protocolo_c_plano_aquisicao_evidencias_observacionais.md`
organiza fontes-alvo (Defesa Civil, CPRM, Sentinel pós-evento, GFM, anotação especializada),
classifica pela força metodológica e mapeia quais gates cada fonte pode fechar. Registros
`observational_evidence_acquisition_plan.csv` (12+ linhas por região; HIGH, MEDIUM, LOW,
METHOD_REFERENCE_ONLY) e `regional_ground_reference_readiness.csv` (uma linha por região)
documentam prontidão para cada gate, lacunas críticas, risco metodológico e allowed/forbidden
claims. Estado: metadata-only, nenhuma aquisição executada, nenhuma evidência baixada. Plano
não treina, prediz ou declara ground truth.

↓

**Pacote operacional de aquisição e intake (v1hm)**
O Protocolo C agora tem camada operacional de aquisição. Documento
`protocolo_c_pacote_operacional_aquisicao_evidencias.md` define princípios (metadata-first,
public registry local-only, licença documentada, nenhuma promoção automática) e fluxo de 9
passos. Runbook `protocolo_c_runbook_aquisicao_evidencias.md` é guia operacional passo a passo
para coleta futura. Templates de solicitação formal e checklist de triagem em `docs/templates/`.
Registros `evidence_acquisition_tracker.csv`, `evidence_source_intake_registry.csv` e
`evidence_license_provenance_registry.csv` rastreiam estado, intake e licença/proveniência por
fonte. GitHub contém apenas metadados públicos seguros; dados brutos permanecem local-only;
licença e proveniência insuficientes bloqueiam promoção; multimodal permanece em hold.

↓

**Triagem de eventos candidatos (v1hn)**
Triagem metadata-only de eventos candidatos de inundação/alagamento por região. Cinco eventos candidatos organizados em `event_candidate_screening_registry.csv`: Recife 2021 e 2022 (EVENT_SEARCH_TARGET, HIGH), Petrópolis fev/2022 (PENDING_SOURCE_REVIEW, HIGH), Curitiba 2022 (EVENT_SEARCH_TARGET, HIGH) e 2023 (EVENT_SEARCH_TARGET, MEDIUM). `event_source_search_backlog.csv` conecta cada evento às fontes-alvo já rastreadas em v1hm. `event_patch_screening_scope.csv` registra quais patches do corpus DINO estão no perímetro de busca de cada evento candidato — com `spatial_overlap_assessed=false`, `temporal_alignment_assessed=false` e `promotion_allowed=false` em todas as entradas. DINOv2 permanece review-only e não fecha gate de evento, temporalidade, espacialidade ou ground truth. Nenhum dado foi baixado.

↓

**Dossiês de evidência por evento candidato (v1ho)**
Para cada um dos cinco eventos candidatos da triagem, um dossiê especifica o pacote mínimo de evidências necessário. `event_evidence_dossier_registry.csv`: cinco dossiês com status (DOSSIER_OPEN para EVENT_SEARCH_TARGET; DOSSIER_PARTIAL para PENDING_SOURCE_REVIEW), lacunas de evidência mínima e `can_support_ground_reference_candidate=false`. `event_evidence_requirements_registry.csv`: 25 requisitos mínimos (cinco por evento: EVENT_CONFIRMATION, TEMPORAL_EVIDENCE, SPATIAL_EVIDENCE, HUMAN_REVIEW, PROMOTION_DECISION) com `current_status=MISSING` ou `PARTIAL` e `blocking_if_missing=true` para requisitos críticos. `event_dossier_decision_registry.csv`: uma decisão por dossiê (CONTINUE_SOURCE_SEARCH, REQUEST_FORMAL_SOURCE, READY_FOR_SOURCE_REVIEW ou WAIT_FOR_ACQUISITION) com `can_reassess_protocol_b=false` e `can_start_multimodal=false`. Nenhum gate foi fechado. Nenhum dado foi baixado.

↓

**Busca externa e solicitação regional (v1hp)**
Os dossiês são transformados em ação concreta por quatro registros metadata-only. `regional_external_search_plan.csv`: onze planos de busca por região (Recife 5, Petrópolis 3, Curitiba 3) com fonte-alvo, gate alvo, modo de busca (PUBLIC_PORTAL_REVIEW ou FORMAL_REQUEST), prioridade e status atual — todos com `forbidden_use` bloqueando declaração de ground truth, flood label, training label, flood detection e flood prediction. `source_request_package_registry.csv`: sete pacotes de solicitação formal a instituições (COMPDEC Recife, CPRM, Defesa Civil Petrópolis, Defesa Civil Curitiba, GeoCuritiba) com `cannot_establish_ground_truth_alone=true` em todas as linhas e dado bruto local-only quando recebido. `gate_search_question_registry.csv`: 17 perguntas de busca distribuídas por região e gate (Recife 6, Petrópolis 6, Curitiba 5), com `blocking_if_unanswered=true` para perguntas de G1, G3, G4 e G7, e `forbidden_if_unanswered` bloqueando promoção em todas as perguntas críticas. `regional_request_priority_matrix.csv`: quatro entradas consolidando a prioridade de solicitação por evento e região, com `protocol_b_status=BLOCKED` e `multimodal_status=HOLD` em todas as linhas. Nenhum dado foi baixado. Nenhuma fonte foi acessada. Nenhum gate foi fechado.

↓

**Referências observacionais candidatas (v1hq)**
Primeira camada documental de eventos observados candidatos. `observed_event_reference_candidate_registry.csv`: 9 eventos (3 por região) com G1/G2/G3 fechados documentalmente por fonte primária rastreável e G4 em triagem espacial — prova de existência do evento, fonte e temporalidade; não prova ground truth operacional, não cria label. Todos têm `operational_ground_truth_status=NOT_ESTABLISHED`, `protocol_b_status=BLOCKED`, `multimodal_status=HOLD`, `dino_usage_status=SUPPORT_ONLY`, `can_be_used_as_training_label=false` e `can_reopen_protocol_b=false`. `observed_event_reference_gap_registry.csv`: 43 lacunas organizadas por evento — patch overlay não executado, revisão humana não feita, licença pendente, geometria ausente, separação de fenômenos pendente em Petrópolis. `observed_event_reference_decision_registry.csv`: uma decisão por evento com `can_promote_to_ground_reference=false` e `can_generate_training_label=false` em todas as linhas; PET_2024_03_21_28 tem NEEDS_MORE_SPATIAL_EVIDENCE por G4 parcial. `manual_external_evidence_needed_registry.csv`: inventário de dados externos que precisam ser trazidos manualmente por região — decretos, boletins, mapas, shapefiles, séries pluviométricas, fotos oficiais, separação de fenômenos — com `cannot_establish_ground_truth_alone=true` em todas as entradas. G4 nesta etapa é apenas triagem espacial, não overlay patch-level. Nenhum dado foi baixado.

↓

**Governança multimodal**
Multimodal explicitamente em hold. Condição de desbloqueio: recuperação do stack
Recife, balanceamento regional, aprovação de revisor. Decisão documentada em v1ft,
não inferida.

---

## O que o projeto não produziu

- Rótulos de inundação observada (não existem)
- Dataset de treinamento supervisionado (não existe)
- Ground truth de suscetibilidade (não existe)
- Métricas de desempenho preditivo (não existem)
- Classificador supervisionado (não existe)

O resultado atual é um corpus territorial auditável com representações estruturais
exploratórias e evidências externas documentadas por região. Isso é a contribuição
desta fase — não mais, não menos.

---

## Protocolo C e construção de referência operacional

A camada de referência contextual validada foi refinada pelo Protocolo C. O protocolo organiza a diferença entre evidência contextual, proxy auditável, candidato forte de referência e validação operacional — e registra, por patch e por fonte, onde cada evidência se situa. Ground truth operacional continua bloqueado: faltam, para todos os patches, evento observado documentado, alinhamento temporal, sobreposição espacial confirmada e revisão humana ou de especialista.

O inventário de fontes (`ground_reference_evidence_source_registry.csv`) classifica cada fonte por grau de observação, tipo, allowed_use e forbidden_use. Fontes não adquiridas localmente são marcadas como METHODOLOGICAL_REFERENCE_ONLY e não podem ser usadas como referência aplicada. A distinção importa porque fontes metodológicas (como Sen1Floods11 ou Copernicus GFM) informam o design do Protocolo C sem substituir evidência local.

### Etapa de aquisição — qualificação de candidatos (metadata-only)

A etapa de aquisição do Protocolo C cria dois registros adicionais que organizam os candidatos a ground reference sem declarar evidência operacional:

- `flood_event_candidate_registry.csv` — eventos candidatos por região (Recife, Petrópolis, Curitiba), todos com `eligible_for_reference_search=false` no estado atual. Referências metodológicas externas (Sen1Floods11, Kuro Siwo, UFO, Copernicus GFM) registradas como METHOD_REFERENCE_ONLY.
- `patch_event_reference_link_registry.csv` — vínculos patch-evento-fonte com status de alinhamento temporal, espacial, CRS, cobertura e bloqueadores ativos. Todos os vínculos têm `promotion_allowed=false`. Vínculos com DINO têm `dino_used_as_support_only=true`.

Esta etapa é explicitamente metadata-only: não baixa rasters, não executa pipeline espacial, não gera labels supervisionados e não desbloqueia o Protocolo B. Veja [`protocolo_c_aquisicao_ground_reference.md`](protocolo_c_aquisicao_ground_reference.md) para a justificativa metodológica completa.

### Etapa de fechamento — gates de promoção e decisão formal (metadata-only)

A etapa de fechamento do Protocolo C define como as lacunas identificadas na etapa de aquisição serão resolvidas e como uma referência pode eventualmente ser promovida. Ela adiciona três registros:

- `ground_reference_gap_matrix.csv` — matriz de lacunas: para cada região, quais gates (G0–G9) estão abertos, qual evidência falta, qual é o risco metodológico e quais são os próximos passos permitidos e proibidos. Todas as linhas têm `promotion_blocked=true` no estado atual.
- `human_reference_review_registry.csv` — registry de revisões humanas ou placeholders: decisão (BLOCK_OPERATIONAL_PROMOTION para todos os placeholders), materiais revisados, consistency checks e claims. Todas as linhas têm `promotion_allowed=false` no estado atual.
- `reference_promotion_decision_registry.csv` — decisões formais de promoção: gates satisfeitos/falhados, `final_reference_status`, `promotion_allowed=false` e `protocol_b_reassessment_allowed=false` em todas as linhas atuais.

O Protocolo C agora inclui fechamento de evidências, revisão humana e decisão de promoção, formando uma trilha auditável para eventual ground reference. Ground truth operacional permanece não estabelecido. O objetivo desta etapa é identificar lacunas reais para aquisição futura — não treinar modelo. Veja [`protocolo_c_fechamento_evidencias_ground_reference.md`](protocolo_c_fechamento_evidencias_ground_reference.md) e [`protocolo_c_revisao_humana_referencia.md`](protocolo_c_revisao_humana_referencia.md) para as justificativas metodológicas.

### Pré-ligação evento–patch (v1hr)

A etapa v1hr prepara as condições para patch-linking sem executá-lo. É metadata-only: nenhum overlay foi executado, nenhuma geocodificação foi realizada, nenhuma coordenada foi criada, nenhum dado bruto foi baixado.

- `event_patch_linking_preflight_registry.csv` — 9 linhas REGION_LEVEL (uma por evento observado candidato) com `promotion_allowed=false`, `can_create_training_label=false`, `protocol_b_status=BLOCKED` e `multimodal_status=HOLD` em todas. Status de pré-ligação: READY_FOR_MANUAL_GEOCODING (Recife e Curitiba), READY_FOR_SOURCE_GEOMETRY_REVIEW (Petrópolis 2022) ou NOT_READY_FOR_PATCH_LINKING (Petrópolis 2024, G4=PARTIAL).
- `manual_geocoding_target_registry.csv` — 22 localidades a geocodificar manualmente por evento e região: 7 Recife (Jardim Uchôa, Areias, Sítio dos Macacos, Milagres, Rio Tejipió, CAIC Barro, Jardim Monte Verde), 10 Petrópolis (Alto da Serra, Centro, Chácara Flora, Morin, Caxambu, São Sebastião, Quitandinha, Castelânea, Valparaíso, Floresta), 5 Curitiba (Caximba, Tatuquara, Cajuru, Rua Miguel Pedro Abib, Rua Alice Vilas Boas da Conceição). Todas com `geocoding_status=NOT_GEOCODED` ou `NEEDS_MANUAL_REVIEW`, sem coordenadas, `requires_official_confirmation=true`, `cannot_establish_ground_truth_alone=true`.
- `event_sentinel_temporal_window_registry.csv` — 9 janelas temporais (uma por evento): pré-evento (14 dias antes), evento e pós-evento (14 dias depois). `sentinel_1_relevance=HIGH` para todos (SAR penetra nuvem), `sentinel_2_relevance=MEDIUM`, `expected_cloud_risk=HIGH`, `acquisition_status=NOT_ACQUIRED`, `cannot_establish_ground_truth_alone=true`. Nenhum asset foi buscado nem baixado.
- `patch_linking_dependency_registry.csv` — 48 dependências metodológicas: 5 por evento para Recife e Curitiba (SOURCE_GEOMETRY, MANUAL_GEOCODING, LICENSE_PROVENANCE, SENTINEL_TEMPORAL_SEARCH, HUMAN_REVIEW), 6 por evento para Petrópolis (mais PHENOMENON_SEPARATION). Todas com `current_status=OPEN` e `required_before_ground_reference=true`. Nenhuma dependência fechada nesta etapa.

O Protocolo B permanece BLOCKED. O pipeline multimodal permanece HOLD. DINO permanece SUPPORT_ONLY. Esta etapa não cria as condições para promoção a ground reference, label de treino, supervised training, flood detection, flood prediction ou reabertura do Protocolo B.

### Consolidação de referências observacionais candidatas (v1ib)

A v1ib transforma o Protocolo C de camada de bloqueio em camada de promoção positiva. Ela lê os registros de evidência das etapas anteriores e sintetiza, por evento, o que foi confirmado positivamente — o que sustenta a candidatura de cada evento como referência observacional.

- `observational_reference_promotion_registry.csv` — 9 linhas com nível de promoção (LEVEL_0–LEVEL_6), decisão (PROMOTE_TO_STRONG / PROMOTE_TO_SECONDARY / KEEP_AS_CONTEXTUAL / HOLD / BLOCK) e guardrails invariantes (`can_be_called_ground_truth_operational=false`, `can_create_training_label=false`, `can_reopen_protocol_b=false`, `multimodal_status=HOLD`, `dino_usage_status=SUPPORT_ONLY` para todos)
- `protocolo_c_event_evidence_level_matrix.csv` — 9 linhas com estado de cada gate G1–G7 por evento

Resultados: PET_2022_02_15 como candidata forte (LEVEL_5_SPATIAL_TRIAGE_READY) — a evidência mais robusta do corpus, com 3 fontes SOURCE_CONFIRMED, data 15/02/2022, fenômeno MIXED, 6 localidades, bloqueada para geocodificação por separação de fenômeno (PKG_FR_PET_001 pendente). PET_2024_03_21_28 como candidata secundária (LEVEL_6_READY_FOR_CONTROLLED_GEOCODING) — 1 fonte, 2 alvos READY_FOR_TRIAGE_ONLY. REC_2022_05_24_30 em LEVEL_4 (aguarda COMPDEC). Demais 6 eventos em LEVEL_0 ou LEVEL_1. Ground truth operacional não está estabelecido. Nenhum label criado. Protocolo B permanece BLOCKED.

### Separação assistida de fenômeno por localidade (v1ic)

A v1ic executa a separação fenomenológica máxima alcançável com fontes textuais disponíveis para PET_2022_02_15 — o evento mais forte do corpus e o principal bloqueio pendente para geocodificação.

- `event_locality_phenomenon_separation_registry.csv` — 8 linhas com `phenomenon_class`, `phenomenon_confidence`, `source_ids`, `hydrological_terms_found`, `mass_movement_terms_found`, `blocks_controlled_geocoding=true`, `can_create_training_label=false`, `multimodal_status=HOLD`, `dino_usage_status=SUPPORT_ONLY` para todas
- `event_phenomenon_separation_decision_registry.csv` — 1 linha com decisão consolidada: `phenomenon_separation_status=PARTIAL_SEPARATION`, `can_advance_to_controlled_geocoding_future=false`, `required_next_action`, `forbidden_claim` explícito

Fontes: DRM-RJ / NADE / Thalweg (ACQ_PET_2022_02_15_007, PDF 57p.) e NHESS / Copernicus (ACQ_PET_2022_02_15_008, HTML). Resultados: 0 HYDROLOGICAL_CONFIRMED, 3 MASS_MOVEMENT_CONFIRMED, 5 MIXED_CONFIRMED. Chácara Flora e Caxambu têm deslizamento documentado com HIGH confidence por texto técnico direto. Rio Quitandinha transbordou em 15/02/2022 (achado hidrológico confirmado), mas a localidade Quitandinha permanece MIXED por sobreposição geotécnica. Nenhuma localidade atingiu HYDROLOGICAL_CONFIRMED — o bloqueio de geocodificação controlada se mantém. A etapa seguinte requer PKG_FR_PET_001 (documento DRM-RJ completo com separação cartográfica por localidade). Ground truth operacional não está estabelecido. Nenhum label criado. Protocolo B permanece BLOCKED.

### Registro de pacote cartográfico e auditoria de ground reference (v1id + v1ie)

A v1id registra PKG_FR_PET_001 como REQUIRED_NOT_INGESTED. A v1ie executa busca local real e audita 10 candidatos com geometria (SIRGAS 2000, SGB/CPRM + FBDS) usando pyshp:

- `observed_reference_source_package_registry.csv` — 1 linha: PKG_FR_PET_001, `source_status=REQUIRED_NOT_INGESTED`, `operational_ground_truth_status=BLOCKED`
- `ground_reference_evidence_registry.csv` — 10 linhas: candidatos auditados, todos `operational_ground_truth_status=BLOCKED`, `ml_label_status=BLOCKED_UNTIL_SPLIT_AND_LEAKAGE_PROTOCOL`; Gate 6 (event_date_compatible) FAIL para todos

Achado crítico da v1ie: `Inundacao_A.shp` (SGB/CPRM, 617 feições) é suscetibilidade modelada (`CLASSE=Baixa/Média/Alta`), não ocorrência do evento de 15/02/2022. `Cicatriz_Area_A.shp` (444 cicatrizes) não tem campo de data — não pode ser vinculado ao evento. PKG_FR_PET_001 permanece não encontrado.

## Referências internas

- [`datasets/dataset_registry.csv`](../../datasets/dataset_registry.csv) — registro geral de datasets
- [`datasets/patch_corpus_registry.csv`](../../datasets/patch_corpus_registry.csv) — corpora de patches por estágio
- [`datasets/external_evidence_registry.csv`](../../datasets/external_evidence_registry.csv) — evidências externas por região
- [`datasets/contextual_reference_layer_registry.csv`](../../datasets/contextual_reference_layer_registry.csv) — status de referência e claims por patch
- [`datasets/ground_reference_evidence_source_registry.csv`](../../datasets/ground_reference_evidence_source_registry.csv) — inventário de fontes de referência pelo Protocolo C
- [`datasets/flood_event_candidate_registry.csv`](../../datasets/flood_event_candidate_registry.csv) — registro de eventos candidatos (etapa de aquisição)
- [`datasets/patch_event_reference_link_registry.csv`](../../datasets/patch_event_reference_link_registry.csv) — vínculos patch-evento-fonte (etapa de aquisição)
- [`datasets/ground_reference_gap_matrix.csv`](../../datasets/ground_reference_gap_matrix.csv) — matriz de lacunas por região (etapa de fechamento)
- [`datasets/human_reference_review_registry.csv`](../../datasets/human_reference_review_registry.csv) — registry de revisões humanas (etapa de fechamento)
- [`datasets/reference_promotion_decision_registry.csv`](../../datasets/reference_promotion_decision_registry.csv) — decisões formais de promoção (etapa de fechamento)
- [`datasets/event_candidate_screening_registry.csv`](../../datasets/event_candidate_screening_registry.csv) — eventos candidatos por região (v1hn)
- [`datasets/event_source_search_backlog.csv`](../../datasets/event_source_search_backlog.csv) — backlog de fontes a pesquisar por evento candidato (v1hn)
- [`datasets/event_patch_screening_scope.csv`](../../datasets/event_patch_screening_scope.csv) — escopo de triagem por patch (v1hn)
- [`datasets/event_evidence_dossier_registry.csv`](../../datasets/event_evidence_dossier_registry.csv) — dossiês de evidência por evento candidato (v1ho)
- [`datasets/event_evidence_requirements_registry.csv`](../../datasets/event_evidence_requirements_registry.csv) — requisitos mínimos de evidência por evento (v1ho)
- [`datasets/event_dossier_decision_registry.csv`](../../datasets/event_dossier_decision_registry.csv) — decisões de continuidade por dossiê (v1ho)
- [`datasets/regional_external_search_plan.csv`](../../datasets/regional_external_search_plan.csv) — planos de busca externa por região (v1hp)
- [`datasets/source_request_package_registry.csv`](../../datasets/source_request_package_registry.csv) — pacotes de solicitação formal a instituições (v1hp)
- [`datasets/gate_search_question_registry.csv`](../../datasets/gate_search_question_registry.csv) — perguntas de busca por gate e região (v1hp)
- [`datasets/regional_request_priority_matrix.csv`](../../datasets/regional_request_priority_matrix.csv) — matriz de prioridade regional de solicitação (v1hp)
- [`docs/metodologia_cientifica/protocolo_c_dossies_eventos_candidatos.md`](protocolo_c_dossies_eventos_candidatos.md) — Protocolo C: dossiês de evidência por evento candidato (v1ho)
- [`docs/metodologia_cientifica/protocolo_c_busca_externa_solicitacao_regional.md`](protocolo_c_busca_externa_solicitacao_regional.md) — Protocolo C: busca externa e solicitação regional (v1hp)
- [`datasets/observed_event_reference_candidate_registry.csv`](../../datasets/observed_event_reference_candidate_registry.csv) — 9 eventos observados candidatos (v1hq)
- [`datasets/event_patch_linking_preflight_registry.csv`](../../datasets/event_patch_linking_preflight_registry.csv) — preflight de pré-ligação evento–patch (v1hr)
- [`datasets/manual_geocoding_target_registry.csv`](../../datasets/manual_geocoding_target_registry.csv) — alvos de geocodificação manual (v1hr)
- [`datasets/event_sentinel_temporal_window_registry.csv`](../../datasets/event_sentinel_temporal_window_registry.csv) — janelas temporais Sentinel por evento (v1hr)
- [`datasets/patch_linking_dependency_registry.csv`](../../datasets/patch_linking_dependency_registry.csv) — dependências para patch-linking real (v1hr)
- [`docs/metodologia_cientifica/protocolo_c_pre_ligacao_evento_patch.md`](protocolo_c_pre_ligacao_evento_patch.md) — Protocolo C: pré-ligação evento–patch (v1hr)
- [`docs/metodologia_cientifica/protocolo_c_triagem_eventos_candidatos.md`](protocolo_c_triagem_eventos_candidatos.md) — Protocolo C: triagem de eventos candidatos (v1hn)
- [`docs/metodologia_cientifica/protocolo_c_construcao_referencia_operacional.md`](protocolo_c_construcao_referencia_operacional.md) — Protocolo C: formulação completa
- [`docs/metodologia_cientifica/protocolo_c_aquisicao_ground_reference.md`](protocolo_c_aquisicao_ground_reference.md) — etapa de aquisição: justificativa e registros metadata-only
- [`docs/metodologia_cientifica/protocolo_c_fechamento_evidencias_ground_reference.md`](protocolo_c_fechamento_evidencias_ground_reference.md) — etapa de fechamento: gates de promoção e matriz de lacunas
- [`docs/metodologia_cientifica/protocolo_c_revisao_humana_referencia.md`](protocolo_c_revisao_humana_referencia.md) — protocolo de revisão humana
- [`docs/metodologia_cientifica/camada_referencia_contextual_validada.md`](camada_referencia_contextual_validada.md) — hierarquia de status e guardrails por patch
- [`docs/metodologia_cientifica/patch_lineage_and_grounding.md`](patch_lineage_and_grounding.md) — linhagem territorial dos patches
- [`docs/metodologia_cientifica/protocolo_c_aquisicao_fontes_observacionais_publicas.md`](protocolo_c_aquisicao_fontes_observacionais_publicas.md) — aquisição local-only de fontes observacionais públicas (v1hs): política local-only, estados de aquisição, relação com ground reference
- [`docs/metodologia_cientifica/protocolo_c_relatorio_aquisicao_fontes_observacionais_v1hs.md`](protocolo_c_relatorio_aquisicao_fontes_observacionais_v1hs.md) — relatório de aquisição v1hs/v1ht: fontes tentadas, 6 adquiridas, falhas SSL/403, pendências formais e buscas manuais
- [`docs/metodologia_cientifica/protocolo_c_revisao_assistida_fontes_observacionais.md`](protocolo_c_revisao_assistida_fontes_observacionais.md) — metodologia da revisão assistida v1hu: o que extrai, gates avaliados, força de evidência, limites
- [`docs/metodologia_cientifica/protocolo_c_relatorio_revisao_assistida_fontes_v1hu.md`](protocolo_c_relatorio_revisao_assistida_fontes_v1hu.md) — relatório de execução v1hu: 38 fontes revisadas, 4 HTMLs lidos, 2 STRONG, candidatos G1–G4, lacunas e próximos passos
- [`docs/metodologia_cientifica/protocolo_c_matriz_decisao_gates_evento.md`](protocolo_c_matriz_decisao_gates_evento.md) — metodologia da matriz de decisão por evento v1hv: lógica de agregação por gate, readiness, próximas ações, restrições absolutas
- [`docs/metodologia_cientifica/protocolo_c_relatorio_matriz_decisao_gates_v1hv.md`](protocolo_c_relatorio_matriz_decisao_gates_v1hv.md) — relatório de execução v1hv: 36 linhas de matriz, PET_2022_02_15 candidato forte, 9 readiness, 30 ações, can_promote=false para todos
- [`docs/metodologia_cientifica/protocolo_c_validacao_assistida_fonte_evento.md`](protocolo_c_validacao_assistida_fonte_evento.md) — metodologia da validação assistida fonte–evento v1hw: estados de fonte, evento, temporal, fenômeno, localidade, licença, decisões e restrições absolutas
- [`docs/metodologia_cientifica/protocolo_c_relatorio_validacao_assistida_fonte_evento_v1hw.md`](protocolo_c_relatorio_validacao_assistida_fonte_evento_v1hw.md) — relatório de execução v1hw: 38 fontes, 3 SOURCE_CONFIRMED, PET_2022_02_15 REQUEST_PHENOMENON_SEPARATION, PET_2024_03_21_28 KEEP_AS_PRIORITY, 2 eventos selecionados para geocodificação; can_execute_overlay_now=false; can_promote=false para todos
- [`docs/metodologia_cientifica/protocolo_c_relatorio_execucao_aquisicao_dirigida_v1ht.md`](protocolo_c_relatorio_execucao_aquisicao_dirigida_v1ht.md) — relatório de execução v1ht: script endurecido, 22 alvos de busca dirigida, 11 solicitações formais, 13 lacunas que bloqueiam ground reference; nenhum novo arquivo adquirido nesta fase de catalogação
- [`docs/metodologia_cientifica/protocolo_c_resolucao_assistida_lacunas_v1hx.md`](protocolo_c_resolucao_assistida_lacunas_v1hx.md) — metodologia da resolução assistida de lacunas v1hx: classes de lacuna, WEB_* status codes, critério de sucesso, relação com próximas etapas
- [`docs/metodologia_cientifica/protocolo_c_relatorio_resolucao_lacunas_v1hx.md`](protocolo_c_relatorio_resolucao_lacunas_v1hx.md) — relatório de execução v1hx: 8/9 portais adquiridos, backend PDF ALL_MISSING, 12 pacotes formais DRAFT, re-execução v1hw com 46 fontes (3 DOWNGRADE_TO_CONTEXTUAL_ONLY); can_promote=false para todos
- [`docs/metodologia_cientifica/protocolo_c_revisao_profunda_pre_geocodificacao_v1hy.md`](protocolo_c_revisao_profunda_pre_geocodificacao_v1hy.md) — metodologia v1hy: revisão profunda de portais adquiridos, suporte a PDF via pypdf, fechamento pré-geocodificação com guardrails explícitos
- [`docs/metodologia_cientifica/protocolo_c_relatorio_revisao_profunda_pre_geocodificacao_v1hy.md`](protocolo_c_relatorio_revisao_profunda_pre_geocodificacao_v1hy.md) — relatório v1hy: pypdf instalado (v6.12.0), 1729 links extraídos/0 event-specific, pre_geocoding_closure_registry com 3 eventos selecionados, todos guardrails false/BLOCKED/HOLD/SUPPORT_ONLY
- [`docs/metodologia_cientifica/protocolo_c_pacote_geocodificacao_controlada_v1hz.md`](protocolo_c_pacote_geocodificacao_controlada_v1hz.md) — metodologia v1hz: escopo de geocodificação controlada, classificação de localidades, bloqueios espaciais, fila futura; nenhuma coordenada criada
- [`docs/metodologia_cientifica/protocolo_c_relatorio_pacote_geocodificacao_controlada_v1hz.md`](protocolo_c_relatorio_pacote_geocodificacao_controlada_v1hz.md) — relatório v1hz: 13 alvos (6 bloqueados/7 prontos), 13 bloqueios (1 CRITICAL), 5 fila futura, todos guardrails false/BLOCKED/HOLD/SUPPORT_ONLY
- [`docs/metodologia_cientifica/protocolo_c_inventario_fontes_espaciais_autoritativas_v1ia.md`](protocolo_c_inventario_fontes_espaciais_autoritativas_v1ia.md) — metodologia v1ia: inventário de fontes por alvo, classes e autoridade de fonte, preflight de execução, lacunas autoritativas; invariantes permanentes
- [`docs/metodologia_cientifica/protocolo_c_relatorio_inventario_fontes_espaciais_v1ia.md`](protocolo_c_relatorio_inventario_fontes_espaciais_v1ia.md) — relatório v1ia: 15 fontes (13 PRIMARY_OFFICIAL/TECHNICAL), 2 alvos READY_FOR_TRIAGE_ONLY (PET_2024), 11 bloqueados, 1 lacuna CRITICAL, 5 formal requests pendentes; nenhuma geocodificação executada
- [`docs/metodologia_cientifica/protocolo_c_consolidacao_referencias_observacionais_candidatas.md`](protocolo_c_consolidacao_referencias_observacionais_candidatas.md) — metodologia v1ib: consolidação positiva de evidências observacionais candidatas; níveis LEVEL_0–LEVEL_6; promoção graduada por evento; ground truth operacional não estabelecido; invariantes permanentes
- [`docs/metodologia_cientifica/protocolo_c_relatorio_consolidacao_referencias_observacionais.md`](protocolo_c_relatorio_consolidacao_referencias_observacionais.md) — relatório v1ib: PET_2022_02_15 como candidata forte (LEVEL_5), PET_2024_03_21_28 como secundária (LEVEL_6), demais contextuais ou bloqueados; nenhum label criado; Protocolo B permanece BLOCKED
- [`docs/metodologia_cientifica/protocolo_c_separacao_fenomeno_petropolis_2022_v1ic.md`](protocolo_c_separacao_fenomeno_petropolis_2022_v1ic.md) — metodologia v1ic: separação assistida de fenômeno por localidade; classes de fenômeno; critério de avanço; bloqueio de geocodificação; fontes utilizadas
- [`docs/metodologia_cientifica/protocolo_c_relatorio_separacao_fenomeno_petropolis_2022_v1ic.md`](protocolo_c_relatorio_separacao_fenomeno_petropolis_2022_v1ic.md) — relatório v1ic: PARTIAL_SEPARATION; 0 HYDRO/3 MASS/5 MIXED; Chácara Flora e Caxambu HIGH; rio Quitandinha; PKG_FR_PET_001 como próxima ação
- [`docs/metodologia_cientifica/protocolo_c_pacote_referencia_cartografica_petropolis_2022_v1id.md`](protocolo_c_pacote_referencia_cartografica_petropolis_2022_v1id.md) — metodologia v1id: registro e preparação de ingestão de PKG_FR_PET_001 (DRM-RJ cartográfico); bloqueio de geocodificação até cartografia disponível
- [`docs/metodologia_cientifica/protocolo_c_relatorio_pacote_referencia_cartografica_petropolis_2022_v1id.md`](protocolo_c_relatorio_pacote_referencia_cartografica_petropolis_2022_v1id.md) — relatório v1id: PKG_FR_PET_001 registrado como REQUIRED_NOT_INGESTED; bloqueios operacionais mantidos
- [`docs/metodologia_cientifica/protocolo_c_ingestao_auditoria_ground_reference_petropolis_2022_v1ie.md`](protocolo_c_ingestao_auditoria_ground_reference_petropolis_2022_v1ie.md) — metodologia v1ie: busca local real e auditoria de 10 candidatos SGB/CPRM+FBDS; distinção suscetibilidade vs. ocorrência de evento; Gate 6 FAIL; PKG_FR_PET_001 não encontrado; BLOCKED mantido
- [`docs/metodologia_cientifica/protocolo_c_relatorio_ingestao_auditoria_ground_reference_petropolis_2022_v1ie.md`](protocolo_c_relatorio_ingestao_auditoria_ground_reference_petropolis_2022_v1ie.md) — relatório v1ie: 10 candidatos auditados, todos BLOCKED; 0 HYDROLOGICAL_CONFIRMED; 0 ground reference auditado; inventário de suscetibilidade e hidrografia confirma contexto mas não event-specific ground truth
- [`docs/estado_metodologico_revp.md`](../estado_metodologico_revp.md) — estado e limitações metodológicas
- [`manifests/external_validation/`](../../manifests/external_validation/) — pacote de validação externa
