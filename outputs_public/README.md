# Artefatos finais publicos do REV-P

Este diretorio reune os artefatos finais utilizados na entrega do REV-P. Os resultados documentam evidencias contextuais, suporte territorial externo e analise visual-estrutural destinada a revisao. O projeto nao apresenta detector operacional, preditor ou classificador supervisionado de inundacao.

## Estrutura

- `figures/`: figuras utilizadas no artigo e na apresentacao, alem de resultados estruturais DINOv2.
- `tables/`: tabelas auxiliares com contagens canonicas, inventarios, vizinhos, PCA, medoides, valores atipicos e limites de interpretacao.
- `metrics/`: metricas descritivas de similaridade, PCA, agrupamentos exploratorios, robustez, QA, estado de prontidao e sensibilidade.
- `logs_summary/`: registros resumidos das validacoes executadas durante a preparacao da entrega.
- `execution_reports/`: relatorios de execucao, rastreabilidade, QA, selecao de figuras, DINO e limites metodologicos.
- `model/`: declaracao da ausencia de modelo supervisionado operacional.

## Resultados comprovados

- Corpus: 59 recortes territoriais/contextuais; 32 coerentes e 27 parcialmente coerentes.
- Distribuicao regional: Recife 18, Curitiba 14, Petropolis 27.
- Manifesto Sentinel-first: 128 assets candidatos.
- DINOv2: 12 embeddings reais, quatro por regiao, 768 dimensoes, codificador congelado.
- Figura principal validada de Recife: `figures/fig_recife_main_publication_v15_final.png`, com patch principal `REC_00205`.

## Dados mantidos fora

GeoTIFFs, arquivos vetoriais, PE3D/MDE bruto, embeddings `.npz`, modelo DINO, ambientes virtuais, caches, `local_runs/` completo e logs brutos permanecem locais. A entrega publica apenas manifestos, hashes, tabelas resumidas e figuras derivadas.

## Reproducao parcial

```powershell
python scripts/repository/build_outputs_public_delivery.py
python -m pytest tests
python scripts/repository/build_outputs_public_delivery.py --finalize
python scripts/repository/build_outputs_public_delivery.py --validate-only
```

O indice principal dos artefatos esta em [`execution_reports/final_delivery_artifact_index.md`](execution_reports/final_delivery_artifact_index.md).
