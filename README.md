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
- [docs/metodologia_cientifica/camada_referencia_contextual_validada.md](docs/metodologia_cientifica/camada_referencia_contextual_validada.md) — hierarquia de status e guardrails por patch

O Protocolo C agora inclui pacote operacional de aquisição/intake (v1hm), que define como receber, registrar, licenciar e triar evidências. O GitHub continua contendo apenas metadados públicos seguros — dados brutos permanecem local-only. Licença e proveniência bloqueiam promoção quando insuficientes. Multimodal permanece em hold. Ground truth operacional permanece não estabelecido.

### Datasets das etapas de aquisição e fechamento

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
