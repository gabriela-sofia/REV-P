# Artefatos públicos do REV-P

Este diretório reúne os artefatos finais da entrega do REV-P: figuras, tabelas, métricas, relatórios e registros de validação. O projeto não apresenta detector operacional, classificador supervisionado ou preditor de inundação — os resultados documentam evidências contextuais, suporte territorial externo e análise visual-estrutural destinada a revisão.

## Estrutura

- `figures/`: figuras de publicação e resultados estruturais DINOv2.
- `tables/`: tabelas com contagens canônicas, inventários, vizinhos k-NN, coordenadas PCA, medoides, outliers e limites de interpretação.
- `metrics/`: métricas descritivas de similaridade, PCA, agrupamentos exploratórios, robustez, QA, prontidão e sensibilidade.
- `logs_summary/`: registros resumidos das validações executadas durante a entrega.
- `execution_reports/`: relatórios de execução, rastreabilidade, QA, análise DINOv2 e restrições metodológicas.
- `model/`: declaração da ausência de modelo supervisionado operacional.
- `article_crosswalk/`: correspondência entre as figuras do artigo e os artefatos públicos.
- `appendix_figures/`: figuras de apêndice organizadas por função metodológica.
- `appendix_visual_audit/`: figuras diagnósticas e justificativas de curadoria visual.
- `selection_audit/`: matriz e relatório das decisões de seleção de patches.
- `review_package/`: navegação para banca, orientação e avaliação.

## Resultados principais

- Corpus: 59 recortes territoriais/contextuais — 32 coerentes, 27 parcialmente coerentes.
- Distribuição regional: Recife 18, Curitiba 14, Petrópolis 27.
- Manifesto Sentinel-first: 128 assets candidatos.
- DINOv2: 12 embeddings reais, quatro por região, 768 dimensões, codificador visual congelado.
- Figura principal validada de Recife: `figures/fig_recife_main_publication_v15_final.png`, patch principal `REC_00205`.

## O que não está aqui

GeoTIFFs, vetores brutos, PE3D/MDE, embeddings `.npz`, modelo DINO, ambientes virtuais, caches e `local_runs/` permanecem locais. Este diretório contém apenas manifests, tabelas resumidas, métricas e figuras derivadas.

## Reprodução parcial

```bash
python scripts/repository/build_outputs_public_delivery.py
python -m pytest tests
python scripts/repository/build_outputs_public_delivery.py --validate-only
```

O índice completo dos artefatos está em [`execution_reports/final_delivery_artifact_index.md`](execution_reports/final_delivery_artifact_index.md).

## Regeneração das figuras auxiliares

As figuras auxiliares em `figures/` foram regeneradas com os mesmos caminhos públicos, usando as tabelas e métricas desta entrega. As figuras principais de patches e regiões foram preservadas sem alteração.

- Relatório: [`execution_reports/figures_regeneration_report.md`](execution_reports/figures_regeneration_report.md)
- Manifesto: [`tables/figure_regeneration_manifest.csv`](tables/figure_regeneration_manifest.csv)
- Versões anteriores: [`appendix_visual_audit/original_auxiliary_figures/`](appendix_visual_audit/original_auxiliary_figures/)
