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
- [`docs/metodologia_cientifica/protocolo_c_dossies_eventos_candidatos.md`](protocolo_c_dossies_eventos_candidatos.md) — Protocolo C: dossiês de evidência por evento candidato (v1ho)
- [`docs/metodologia_cientifica/protocolo_c_triagem_eventos_candidatos.md`](protocolo_c_triagem_eventos_candidatos.md) — Protocolo C: triagem de eventos candidatos (v1hn)
- [`docs/metodologia_cientifica/protocolo_c_construcao_referencia_operacional.md`](protocolo_c_construcao_referencia_operacional.md) — Protocolo C: formulação completa
- [`docs/metodologia_cientifica/protocolo_c_aquisicao_ground_reference.md`](protocolo_c_aquisicao_ground_reference.md) — etapa de aquisição: justificativa e registros metadata-only
- [`docs/metodologia_cientifica/protocolo_c_fechamento_evidencias_ground_reference.md`](protocolo_c_fechamento_evidencias_ground_reference.md) — etapa de fechamento: gates de promoção e matriz de lacunas
- [`docs/metodologia_cientifica/protocolo_c_revisao_humana_referencia.md`](protocolo_c_revisao_humana_referencia.md) — protocolo de revisão humana
- [`docs/metodologia_cientifica/camada_referencia_contextual_validada.md`](camada_referencia_contextual_validada.md) — hierarquia de status e guardrails por patch
- [`docs/metodologia_cientifica/patch_lineage_and_grounding.md`](patch_lineage_and_grounding.md) — linhagem territorial dos patches
- [`docs/estado_metodologico_revp.md`](../estado_metodologico_revp.md) — estado e limitações metodológicas
- [`manifests/external_validation/`](../../manifests/external_validation/) — pacote de validação externa
