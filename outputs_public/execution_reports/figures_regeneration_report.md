# Relatorio de regeneracao das figuras auxiliares

## Objetivo

As figuras auxiliares de `outputs_public/figures/` foram regeneradas para melhorar leitura, consistencia visual e comunicacao metodologica em GitHub, PDF e apresentacoes. A regeneracao utilizou somente tabelas, metricas e relatorios publicos existentes.

## Figuras preservadas

As cinco figuras principais definidas como fora do escopo foram preservadas sem sobrescrita:

- `outputs_public/figures/fig_recife_main_publication_v15_final.png`
- `outputs_public/figures/fig_curitiba_main_publication_v17.png`
- `outputs_public/figures/fig_petropolis_main_publication_v17.png`
- `outputs_public/figures/fig_recife_pe3d_mde_publication.png`
- `outputs_public/figures/fig_recife_sentinel_technical_publication.png`

## Figuras regeneradas

Foram regeneradas 12 figuras auxiliares, mantendo os caminhos publicos. As versoes anteriores foram copiadas para `outputs_public/appendix_visual_audit/original_auxiliary_figures/`.

## Dados usados

- `dino_knn_neighbor_network_publication.png`: `outputs_public/tables/table_dino_nearest_neighbors.csv`.
- `dino_medoids_outliers_publication.png`: `outputs_public/tables/table_dino_medoids.csv`, `outputs_public/tables/table_dino_outliers.csv`.
- `dino_pca_projection_publication.png`: `outputs_public/tables/table_dino_pca_coordinates.csv`, `outputs_public/tables/table_dino_medoids.csv`, `outputs_public/tables/table_dino_outliers.csv`, `outputs_public/metrics/dino_pca_summary.csv`.
- `dino_region_neighbor_matrix_publication.png`: `outputs_public/tables/table_dino_region_neighbor_matrix.csv`.
- `dino_similarity_heatmap_publication.png`: `outputs_public/tables/table_dino_similarity_matrix.csv`.
- `fig_corpus_counts_by_region_status.png`: `outputs_public/tables/table_patch_distribution_by_region.csv`, `outputs_public/tables/table_corpus_summary.csv`.
- `fig_decision_trace_summary.png`: `outputs_public/tables/table_protocol_c_summary.csv`, `outputs_public/metrics/readiness_summary.csv`.
- `fig_dino_input_corpus_publication.png`: `outputs_public/tables/table_dino_embedding_inventory.csv`, `outputs_public/tables/table_dino_quantitative_summary_by_region.csv`.
- `fig_evidence_layer_availability.png`: `outputs_public/tables/table_external_evidence_summary.csv`.
- `fig_local_context_coverage.png`: `outputs_public/tables/table_external_evidence_summary.csv`, `outputs_public/tables/table_patch_distribution_by_region.csv`, `outputs_public/metrics/readiness_summary.csv`.
- `fig_methodological_contribution_matrix.png`: `outputs_public/README.md`, `outputs_public/tables/table_claims_guardrails_summary.csv`, `outputs_public/metrics/qa_metrics_summary.csv`, `outputs_public/metrics/readiness_summary.csv`.
- `fig_regional_roles_summary.png`: `outputs_public/tables/table_patch_distribution_by_region.csv`, `outputs_public/tables/table_external_evidence_summary.csv`.

## Problemas visuais corrigidos

- Rotulos tecnicos extensos foram substituidos por portugues comunicavel.
- Titulos, subtitulos, notas, margens, cores e hierarquia tipografica foram padronizados.
- Matrizes e graficos DINOv2 passaram a explicitar o papel exploratorio e review-only.
- Resumos de corpus, decisao, evidencias, contribuicoes e papeis regionais foram convertidos em composicoes editoriais mais diretas.

## Guardrails mantidos

- Nenhuma figura apresenta DINOv2 como classificador, detector ou preditor.
- Nenhuma figura sugere ground truth operacional, label binario, classe positiva/negativa ou acuracia operacional.
- A analise DINOv2 permanece baseada em encoder congelado, embeddings 768D e revisao humana.
- A validacao operacional e a transicao C4 permanecem bloqueadas conforme os artefatos publicos.
- Nenhum dado bruto pesado, GeoTIFF, shapefile, `.npz`, modelo, cache ou copia de `local_runs/` foi incluido.
- Os arquivos nao atribuem autoria a ferramentas automaticas.

## Rastreabilidade

O manifesto `outputs_public/tables/figure_regeneration_manifest.csv` registra o status, os dados de origem, o papel cientifico e os limites de interpretacao de cada figura principal preservada e auxiliar regenerada.
