# Protocolo C — trilha de aquisição de evidência (datasets/)
Protocolo C é o processo de busca, aquisição e adjudicação de evidência de ground truth (geometria oficial de evento, série hidrometeorológica, revisão humana) para as três regiões do projeto. Foi construído em 59 etapas sequenciais (`v1uc` a `v2bm`), cada uma com script, saída de dados e nota metodológica próprios.

**Resultado final da linhagem**: nenhuma das três regiões tem negativo formal ou label operacional aceito — Recife chega a `PROTOCOL_VALIDATED_CANDIDATE_REFERENCE` (produto cartográfico oficial, score 0,76), Curitiba a `PROTOCOL_VALIDATED_TEMPORAL_REFERENCE` (evidência temporal local forte, score 0,7), Petrópolis a `PROTOCOL_VALIDATED_CONTEXTUAL_REFERENCE` (proxy regional, score 0,55). Ver relatório completo: [`docs/protocolo_c/v2bm_cross_region_reapplication/reports/protocol_c_cross_region_reapplication_report.md`](../../docs/protocolo_c/v2bm_cross_region_reapplication/reports/protocol_c_cross_region_reapplication_report.md).

## Auditoria de 2026-08-19

Os arquivos de bookkeeping por etapa (`*_next_actions_registry.csv`, `*_guardrail_regression.csv`, `*_orchestrator_manifest.csv`, `*_versionable_artifacts_manifest.csv`, `*_completion_report.csv`, `*_next_programming_target_ranker.csv`) e as 21 variações de `*_ground_reference_blocker_matrix.csv` pareciam bookkeeping puro removível — mas checando o código (`scripts/protocolo_c/*_common.py`) cada um é lido de volta (`load_csv(...)`) pela própria automação da etapa seguinte, como parte de um padrão "carrega se existir, senão recalcula". Apagar esses arquivos quebraria a possibilidade de re-executar/auditar o pipeline. Por isso **nenhum arquivo foi removido desta pasta** — os 639 originais continuam todos aqui, intactos.

O que foi adicionado (sem remover nada): uma tabela de conveniência que junta as 21 variações de `*_ground_reference_blocker_matrix.csv` num único lugar pra leitura humana, sem duplicar esforço de abrir 21 arquivos: [`protocolo_c_ground_reference_blocker_matrix_consolidado.csv`](protocolo_c_ground_reference_blocker_matrix_consolidado.csv) (271 linhas, todas as 21 fontes — os originais continuam existindo e sendo usados pelo pipeline).

## Índice das 59 etapas

| Etapa | Região | Tópico | Arquivos de evidência única (fora do bookkeeping) |
|---|---|---|---|
| `v1uc` | Geral | acceptance audit | 1 |
| `v1ud` | Geral | real source acquisition | 5 |
| `v1ue` | Geral | event specific evidence deepening | 7 |
| `v1uf` | Geral | station resolved acquisition | 8 |
| `v1ug` | Geral | (sem título — ver arquivos) | 7 |
| `v1uh` | Geral | formal response intake | 8 |
| `v1ui` | Geral | public official discovery | 11 |
| `v1uj` | Geral | focused public source deepening | 12 |
| `v1uk` | Recife | recife ckan schema deep audit | 10 |
| `v1ul` | Recife | recife candidate review router | 6 |
| `v1um` | Recife | recife human review locality only | 9 |
| `v1un` | Recife | recife human review evidence consolidation | 8 |
| `v1uo` | Multi-região | multiregion replication engine | 9 |
| `v1up` | Petrópolis | petropolis public geometry deepening | 12 |
| `v1uq` | Petrópolis | petropolis phenomenon separation deep audit | 10 |
| `v1ur` | Petrópolis | petropolis public geodata path recovery | 11 |
| `v1us` | Geral | event patch package linkage engine | 10 |
| `v1ut` | Recife | recife coordinate recovery | 10 |
| `v1uu` | Recife | recife contextual coordinate layer consolidation | 10 |
| `v1uv` | Curitiba | curitiba event registry public source discovery | 10 |
| `v1uw` | Curitiba | curitiba public evidence deepening | 9 |
| `v1ux` | Curitiba | curitiba public evidence download schema audit | 10 |
| `v1uy` | Curitiba | curitiba public geodata deepening | 9 |
| `v1uz` | Curitiba/Multi-região | curitiba context only hold multiregion rerank | 7 |
| `v2aa` | Multi-região | sentinel date recovery | 9 |
| `v2ab` | Geral | event patch package schema hardening | 9 |
| `v2ac` | Geral | event patch schema migration | 9 |
| `v2ad` | Geral | event patch v2 qa harness | 8 |
| `v2ae` | Multi-região | multiregion registry hardening | 9 |
| `v2af` | Geral | event patch v2 qa automation | 8 |
| `v2ag` | Geral | sentinel date crosswalk discovery | 9 |
| `v2ah` | Geral | completion report | 6 |
| `v2ai` | Geral | completion report | 6 |
| `v2aj` | Geral | completion report | 6 |
| `v2al` | Geral | (sem título — ver arquivos) | 7 |
| `v2am` | Geral | (sem título — ver arquivos) | 11 |
| `v2an` | Geral | ground reference validation sprint | 12 |
| `v2ap` | Geral | patch geometry sentinel crosswalk | 11 |
| `v2aq` | Geral | event geometry patch link | 10 |
| `v2as` | Geral | official geometry deep probe | 10 |
| `v2at` | Geral | evidence fact hardening | 11 |
| `v2au` | Geral | source resolution plan | 10 |
| `v2av` | Geral | official source terms snapshot | 7 |
| `v2aw` | Geral | public data observational acquisition | 7 |
| `v2ax` | Geral | hydrometeorological temporal evidence | 9 |
| `v2ay` | Geral | hydromet series ingestion | 8 |
| `v2ba` | Geral | official geometry search and digitization | 10 |
| `v2bb` | Geral | secondary evidence adjudication | 9 |
| `v2bc` | Curitiba | curitiba ground truth seed | 8 |
| `v2bd` | Curitiba | curitiba candidate reference | 9 |
| `v2be` | Curitiba | curitiba sentinel crosswalk resolution | 9 |
| `v2bf` | Curitiba | curitiba patch asset lineage | 9 |
| `v2bg` | Recife | recife hydro geomorphic grounding pack | 5 |
| `v2bh` | Recife | recife charter 758 product audit | 8 |
| `v2bi` | Recife | recife charter temporal intake | 10 |
| `v2bj` | Recife | recife candidate gate reconciliation | 4 |
| `v2bk` | Recife | recife human review dossier | 5 |
| `v2bl` | Recife | protocol validated candidate reference | 7 |
| `v2bm` | Curitiba/Petrópolis/Multi-região | cross region reapplication | 7 |

## Como navegar

- Cada etapa tem documentação narrativa em prosa em `docs/metodologia_cientifica/protocolo_c_<etapa>_*.md` (etapas `v1uc`–`v2aj`) ou `docs/protocolo_c/<etapa>_<tópico>/README.md` (etapas `v2an`–`v2bm`).
- Os arquivos de dados desta pasta seguem o padrão `<etapa>_<conteúdo>.csv` — o prefixo indica a etapa que gerou o arquivo.
- Para o estado final consolidado por região, ver a seção "Resultado final da linhagem" acima e o relatório `v2bm`.
