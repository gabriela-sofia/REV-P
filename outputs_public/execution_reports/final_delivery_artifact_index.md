# Índice de artefatos públicos da entrega

Todos os artefatos listados aqui são resultados estruturais destinados à revisão humana. Nenhum deles constitui ground truth operacional, rótulo de treinamento, predição, confirmação de evento observado ou classificação supervisionada. Para a declaração formal sobre ausência de modelo operacional, ver `outputs_public/model/NO_OPERATIONAL_TRAINED_MODEL.md`.

Legenda da coluna de uso: `apêndice/artigo` = figura ou tabela candidata ao artigo ou apêndice; `apoio` = documentação, log ou registro de processo.

Todos os caminhos abaixo apontam para arquivos ou pastas que existem nesta entrega. Categorias ainda não materializadas como pasta estão descritas ao final, em texto, sem link.

Os códigos `v1*` e `v2*` que aparecem em nomes de artefato são **rastreabilidade técnica de etapas**, não narrativa obrigatória de leitura. A banca pode navegar pelos agrupamentos abaixo; a linha do tempo legível dos estágios está em `docs/metodologia_cientifica/revp_indice_etapas.md`.

---

## Figuras finais (`outputs_public/figures/`)

| Artefato | Caminho | Função no projeto | Uso |
|---|---|---|---|
| dino_similarity_heatmap_publication.png | `outputs_public/figures/dino_similarity_heatmap_publication.png` | Mapa de calor de similaridade entre embeddings DINOv2. | apêndice/artigo |
| dino_pca_projection_publication.png | `outputs_public/figures/dino_pca_projection_publication.png` | Projeção PCA dos embeddings. | apêndice/artigo |
| dino_knn_neighbor_network_publication.png | `outputs_public/figures/dino_knn_neighbor_network_publication.png` | Rede de vizinhos mais próximos (k-NN). | apêndice/artigo |
| dino_medoids_outliers_publication.png | `outputs_public/figures/dino_medoids_outliers_publication.png` | Medoids e outliers por região. | apêndice/artigo |
| dino_region_neighbor_matrix_publication.png | `outputs_public/figures/dino_region_neighbor_matrix_publication.png` | Matriz de vizinhança entre regiões. | apêndice/artigo |
| fig_dino_input_corpus_publication.png | `outputs_public/figures/fig_dino_input_corpus_publication.png` | Corpus de entrada DINOv2. | apêndice/artigo |
| fig_corpus_counts_by_region_status.png | `outputs_public/figures/fig_corpus_counts_by_region_status.png` | Contagem de corpus por região e estado. | apêndice/artigo |
| fig_recife_main_publication_v15_final.png | `outputs_public/figures/fig_recife_main_publication_v15_final.png` | Figura principal de Recife. | apêndice/artigo |
| fig_petropolis_main_publication_v17.png | `outputs_public/figures/fig_petropolis_main_publication_v17.png` | Figura principal de Petrópolis. | apêndice/artigo |
| fig_curitiba_main_publication_v17.png | `outputs_public/figures/fig_curitiba_main_publication_v17.png` | Figura principal de Curitiba. | apêndice/artigo |
| fig_recife_pe3d_mde_publication.png | `outputs_public/figures/fig_recife_pe3d_mde_publication.png` | Modelo digital de elevação (PE3D/MDE) de Recife. | apêndice/artigo |
| fig_recife_sentinel_technical_publication.png | `outputs_public/figures/fig_recife_sentinel_technical_publication.png` | Render técnico Sentinel de Recife. | apêndice/artigo |
| fig_evidence_layer_availability.png | `outputs_public/figures/fig_evidence_layer_availability.png` | Disponibilidade de camadas de evidência. | apêndice/artigo |
| fig_local_context_coverage.png | `outputs_public/figures/fig_local_context_coverage.png` | Cobertura de contexto local. | apêndice/artigo |
| fig_methodological_contribution_matrix.png | `outputs_public/figures/fig_methodological_contribution_matrix.png` | Matriz de contribuição metodológica. | apêndice/artigo |
| fig_regional_roles_summary.png | `outputs_public/figures/fig_regional_roles_summary.png` | Resumo dos papéis regionais. | apêndice/artigo |

## Figuras auxiliares de auditoria visual (`outputs_public/appendix_visual_audit/original_auxiliary_figures/`)

A pasta `outputs_public/appendix_visual_audit/original_auxiliary_figures/` guarda as versões auxiliares das figuras DINOv2 e de corpus, mais `fig_decision_trace_summary.png` (resumo do traço de decisões), usadas como registro diagnóstico da auditoria visual. Uso: apoio.

## Tabelas centrais (`outputs_public/tables/`)

| Artefato | Caminho | Função no projeto | Uso |
|---|---|---|---|
| table_corpus_summary.csv | `outputs_public/tables/table_corpus_summary.csv` | Resumo do corpus (59 patches, 128 assets, 12 embeddings). | apêndice/artigo |
| table_patch_distribution_by_region.csv | `outputs_public/tables/table_patch_distribution_by_region.csv` | Distribuição de patches por região. | apêndice/artigo |
| table_dino_embedding_inventory.csv | `outputs_public/tables/table_dino_embedding_inventory.csv` | Inventário dos 12 embeddings (SHA256, encoder congelado, sem rótulo). | apêndice/artigo |
| table_dino_quantitative_summary_by_region.csv | `outputs_public/tables/table_dino_quantitative_summary_by_region.csv` | Similaridade intra-região, medoid e outlier por região. | apêndice/artigo |
| table_dino_similarity_matrix.csv | `outputs_public/tables/table_dino_similarity_matrix.csv` | Matriz de similaridade entre embeddings. | apêndice/artigo |
| table_dino_nearest_neighbors.csv | `outputs_public/tables/table_dino_nearest_neighbors.csv` | Vizinhos mais próximos por embedding. | apêndice/artigo |
| table_dino_pca_coordinates.csv | `outputs_public/tables/table_dino_pca_coordinates.csv` | Coordenadas PCA dos embeddings. | apêndice/artigo |
| table_dino_medoids.csv | `outputs_public/tables/table_dino_medoids.csv` | Medoids por região. | apêndice/artigo |
| table_dino_outliers.csv | `outputs_public/tables/table_dino_outliers.csv` | Outliers por região. | apêndice/artigo |
| table_dino_region_neighbor_matrix.csv | `outputs_public/tables/table_dino_region_neighbor_matrix.csv` | Matriz de vizinhança regional. | apêndice/artigo |
| table_protocol_c_summary.csv | `outputs_public/tables/table_protocol_c_summary.csv` | Estado dos gates do Protocolo C (C4 bloqueado, sem negativos formais). | apêndice/artigo |
| table_external_evidence_summary.csv | `outputs_public/tables/table_external_evidence_summary.csv` | Resumo de evidência externa por região. | apêndice/artigo |
| table_claims_guardrails_summary.csv | `outputs_public/tables/table_claims_guardrails_summary.csv` | Resumo de afirmações permitidas e proibidas. | apêndice/artigo |
| table_artifact_index.csv | `outputs_public/tables/table_artifact_index.csv` | Índice tabular dos artefatos públicos. | apoio |

As demais tabelas de registro (registries do Protocolo C, scorecards e tabelas da cadeia forense `v2es`–`v2ff`) também residem em `outputs_public/tables/`. Tabelas intermediárias de sprint estão em `outputs_public/tables/saidas_intermediarias/`.

## Métricas descritivas (`outputs_public/metrics/`)

A pasta `outputs_public/metrics/` contém `dino_similarity_summary.csv`, `dino_cluster_summary.csv`, `dino_pca_summary.csv`, `dino_robustness_summary.csv`, `ablation_or_sensitivity_summary.csv`, `qa_metrics_summary.csv` e `readiness_summary.csv`. São métricas descritivas — **não representam desempenho de modelo operacional**. Uso: apêndice/artigo.

## Relatórios finais e de rastreabilidade (`outputs_public/execution_reports/`)

| Artefato | Caminho | Função no projeto | Uso |
|---|---|---|---|
| final_dino_structural_analysis_report.md | `outputs_public/execution_reports/final_dino_structural_analysis_report.md` | Análise estrutural consolidada dos embeddings. | apoio |
| final_execution_report.md | `outputs_public/execution_reports/final_execution_report.md` | Relatório geral de execução. | apoio |
| final_qa_report.md | `outputs_public/execution_reports/final_qa_report.md` | Relatório de QA. | apoio |
| final_guardrails_report.md | `outputs_public/execution_reports/final_guardrails_report.md` | Relatório de travas metodológicas. | apoio |
| final_traceability_report.md | `outputs_public/execution_reports/final_traceability_report.md` | Relatório de rastreabilidade. | apoio |
| final_figures_selection_report.md | `outputs_public/execution_reports/final_figures_selection_report.md` | Decisão de seleção de figuras. | apoio |
| final_figures_selection.csv | `outputs_public/execution_reports/final_figures_selection.csv` | Matriz da decisão de figuras. | apoio |
| figures_regeneration_report.md | `outputs_public/execution_reports/figures_regeneration_report.md` | Registro de regeneração de figuras. | apoio |
| final_validation_summary.json | `outputs_public/execution_reports/final_validation_summary.json` | Resumo de validação em JSON. | apoio |

## Cadeia de auditoria de continuidade e recuperação (`outputs_public/execution_reports/`)

| Artefato | Caminho | Função no projeto | Uso |
|---|---|---|---|
| revp_v2es_to_v2ey_integrated_report.md | `outputs_public/execution_reports/revp_v2es_to_v2ey_integrated_report.md` | Relatório integrado da tentativa de recuperação controlada. | apoio |
| revp_v2es_to_v2ey_scientific_summary.md | `outputs_public/execution_reports/revp_v2es_to_v2ey_scientific_summary.md` | Resumo científico da recuperação. | apoio |
| revp_v2ez_to_v2ff_relatorio_integrado.md | `outputs_public/execution_reports/revp_v2ez_to_v2ff_relatorio_integrado.md` | Relatório integrado da auditoria forense de recuperabilidade. | apoio |
| revp_v2ez_to_v2ff_resumo_cientifico.md | `outputs_public/execution_reports/revp_v2ez_to_v2ff_resumo_cientifico.md` | Resumo científico da auditoria forense. | apoio |
| revp_relatorio_painel_perda_recuperacao_base_original_v2ff.md | `outputs_public/execution_reports/revp_relatorio_painel_perda_recuperacao_base_original_v2ff.md` | Painel de perda e recuperabilidade da base original. | apoio |

Os demais relatórios da cadeia (`revp_relatorio_*_v2ez`–`v2fe`, checklist de entrega) residem na mesma pasta.

## Relatórios da curadoria pública

| Artefato | Caminho | Função no projeto | Uso |
|---|---|---|---|
| revp_auditoria_curadoria_repositorio_publico.md | `outputs_public/execution_reports/revp_auditoria_curadoria_repositorio_publico.md` | Auditoria da curadoria da camada pública. | apoio |
| revp_relatorio_limpeza_linguagem.md | `outputs_public/execution_reports/revp_relatorio_limpeza_linguagem.md` | Relatório de limpeza de linguagem. | apoio |
| revp_plano_organizacao_estado_git.md | `outputs_public/execution_reports/revp_plano_organizacao_estado_git.md` | Plano de organização do estado Git. | apoio |
| revp_relatorio_validacao_curadoria_publica.md | `outputs_public/execution_reports/revp_relatorio_validacao_curadoria_publica.md` | Validação da curadoria pública. | apoio |

## Registros resumidos de execução e QA (`outputs_public/logs_summary/`)

A pasta `outputs_public/logs_summary/` reúne `pytest_summary.txt`, `guardrail_validation_summary.txt`, os resumos de limites e testes da cadeia forense (`revp_v2ez_to_v2ff_resumo_limites.csv`, `revp_v2ez_to_v2ff_resumo_testes.csv`) e os summaries de status do Protocolo C (`protocol_c_current_status_summary.md`, `protocol_c_cross_region_status_summary.md`). Uso: apoio.

## Declaração e documentação de leitura

| Artefato | Caminho | Função no projeto | Uso |
|---|---|---|---|
| NO_OPERATIONAL_TRAINED_MODEL.md | `outputs_public/model/NO_OPERATIONAL_TRAINED_MODEL.md` | Declaração formal de ausência de modelo operacional. | apoio |
| README.md | `outputs_public/README.md` | Documentação dos resultados públicos. | apoio |
| final_delivery_artifact_index.md | `outputs_public/execution_reports/final_delivery_artifact_index.md` | Este índice. | apoio |

---

## Categorias planejadas (ainda não materializadas como pasta)

As categorias abaixo descrevem agrupamentos conceituais previstos para versões futuras da entrega. **Não existem como pastas nesta versão** e por isso aparecem apenas como descrição textual, sem caminho:

- **Figuras de apêndice dedicadas** — uma pasta `figuras_apendice/` separando figuras estritamente de apêndice (fluxo metodológico, fluxo do Protocolo C, bandas/índices Sentinel) das figuras principais. Hoje as figuras de apoio equivalentes estão em `outputs_public/appendix_visual_audit/original_auxiliary_figures/`.
- **Correspondência com o artigo** — uma pasta `correspondencia_artigo/` com o mapeamento explícito figura-do-artigo → artefato público. Hoje a decisão de figuras está em `final_figures_selection_report.md` e `final_figures_selection.csv`.
- **Pacote de revisão** — uma pasta `pacote_revisao/` com guia de navegação para o avaliador (o que checar primeiro). Hoje a orientação de leitura está no `README.md` da raiz e em `outputs_public/README.md`.

Quando (e se) essas pastas forem criadas, devem ser adicionadas às tabelas acima com caminho real.
