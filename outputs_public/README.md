# Artefatos públicos do REV-P

Este diretório reúne os artefatos finais da entrega do REV-P: os resultados da linha causal por região (Recife, Curitiba, Petrópolis), a frente externa UK/Copernicus EMS, e a análise estrutural DINOv2 como evidência auxiliar. Ver [`README.md`](../README.md) na raiz do repositório para o estado atual completo por região.

## Estrutura

- `data/susc_20*`: núcleo causal — aquisição de evento, engenharia de features físico-hidrológicas, modelagem Firth, validação, motor de inferência e API, por região.
- `figures/`: figuras finais de publicação — mapas regionais e análise estrutural DINOv2.
- `tables/`: tabelas consolidadas citáveis (corpus, distribuição regional, evidência do Protocolo C, inventário DINOv2).
- `metrics/`: métricas descritivas reais (similaridade, PCA, agrupamento, robustez, QA).
- `execution_reports/`: índice de entrega, relatório de restrições metodológicas e análise estrutural DINOv2.
- `model/`: estado do modelo por região — ver `model/ESTADO_DO_MODELO.md`.

## Resultados principais

- Recife: Firth penalizado, n=278 eventos reais, LOO-AUC = 0,68, motor de inferência + API entregues.
- Curitiba: modelo treinado, colapsa em holdout temporal real 2026 — resultado negativo diagnosticado e documentado.
- Petrópolis: bloqueado por mistura enchente/deslizamento não separada nas fontes.
- Frente externa: piloto UK (AUC 0,79, 201 eventos) e multirregião Copernicus EMS (25.249 pontos, 119 áreas).
- DINOv2: 12 embeddings reais (4 por região, 768 dimensões), testados como feature causal e descartados — mantidos como análise estrutural auxiliar.

## O que não está aqui

GeoTIFFs, vetores brutos, dados de elevação (PE3D/MDE), embeddings `.npz`, ambientes virtuais, caches e execuções locais (`local_runs/`, `local_only/`) permanecem apenas na máquina local. Este diretório contém relatórios, tabelas resumidas, métricas e figuras derivadas — o suficiente para verificar a cadeia metodológica sem reproduzir os dados brutos.

## Reprodução

```bash
conda env create -f ../environment.yml
conda activate revp-susc
python -m pytest tests -q
```
