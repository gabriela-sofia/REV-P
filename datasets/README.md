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

## Protocolo C e camada de referência

A camada de referência contextual foi refinada pelo Protocolo C, que organiza a distinção entre evidência contextual, proxy auditável, candidato de referência e validação operacional. Ground truth operacional continua bloqueado no estado atual.

O `contextual_reference_layer_registry.csv` registra o status de evidência e os claims permitidos/proibidos por patch.

O `ground_reference_evidence_source_registry.csv` é o inventário de fontes de referência: classifica cada fonte por family (ex.: HYDROGEOMORPHOLOGICAL_CONTEXT, OPERATIONAL_FLOOD_PRODUCT), grau de observação (CONTEXTUAL, OPERATIONAL_ALGORITHMIC, EXPERT_INTERPRETED), e registra o allowed_use e forbidden_use de cada fonte. Fontes não adquiridas localmente são marcadas como NOT_ACQUIRED ou METHODOLOGICAL_REFERENCE_ONLY e não podem ser usadas como referência aplicada a patches.

Veja [`docs/metodologia_cientifica/protocolo_c_construcao_referencia_operacional.md`](../docs/metodologia_cientifica/protocolo_c_construcao_referencia_operacional.md) para a formulação completa do Protocolo C.

A etapa de fechamento de evidências adiciona mais três registros metadata-only: `ground_reference_gap_matrix.csv` mapeia os gates de promoção abertos por região com evidência faltante e risco metodológico; `human_reference_review_registry.csv` organiza revisões humanas executadas ou placeholders com decisão e claims; e `reference_promotion_decision_registry.csv` registra decisões formais de promoção com `protocol_b_reassessment_allowed=false` em todas as linhas atuais. O Protocolo C agora inclui fechamento de evidências, revisão humana e decisão de promoção — formando trilha auditável para eventual ground reference. Ground truth operacional permanece não estabelecido. Veja [`protocolo_c_fechamento_evidencias_ground_reference.md`](../docs/metodologia_cientifica/protocolo_c_fechamento_evidencias_ground_reference.md) e [`protocolo_c_revisao_humana_referencia.md`](../docs/metodologia_cientifica/protocolo_c_revisao_humana_referencia.md).

A etapa de aquisição adiciona dois registros metadata-only: `flood_event_candidate_registry.csv` organiza eventos candidatos por região (com status de confirmação e elegibilidade), e `patch_event_reference_link_registry.csv` registra os vínculos entre patches, eventos e fontes com alinhamentos e bloqueadores explícitos. No estado atual, nenhum evento tem `eligible_for_reference_search=true` e nenhum vínculo tem `promotion_allowed=true`. Veja [`docs/metodologia_cientifica/protocolo_c_aquisicao_ground_reference.md`](../docs/metodologia_cientifica/protocolo_c_aquisicao_ground_reference.md) para a justificativa metodológica desta etapa.

A camada de plano de aquisição (v1hl) transforma metodologia em roteiro real: `observational_evidence_acquisition_plan.csv` organiza fontes-alvo por região, classifica pela força metodológica, documenta prioridades de aquisição e mapeia quais gates cada fonte pode fechar. `regional_ground_reference_readiness.csv` registra a prontidão regional para cada gate, identifica a evidência mais forte já disponível, descreve as lacunas críticas, e documenta allowed/forbidden claims por região. Essa camada continua metadata-only e não treina, prediz ou declara ground truth.

A camada de aquisição operacional (v1hm) coloca o plano em prática: `evidence_acquisition_tracker.csv` rastreia o estado atual de cada fonte-alvo, com acquisition_status, license_status, current_blocker e forbidden_use. `evidence_source_intake_registry.csv` registra fontes acessadas ou em processo com decisão de intake (ACCEPT_METADATA_ONLY, BLOCK_USE, REQUEST_MORE_INFORMATION). `evidence_license_provenance_registry.csv` documenta licença, redistribuição e proveniência para cada fonte, garantindo que raw_data_publication_allowed=false quando redistribuição não for explicitamente pública e que use_for_operational_ground_truth_allowed=FALSE em todas as linhas atuais. O GitHub continua contendo apenas metadados públicos seguros — dados brutos permanecem local-only.

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
