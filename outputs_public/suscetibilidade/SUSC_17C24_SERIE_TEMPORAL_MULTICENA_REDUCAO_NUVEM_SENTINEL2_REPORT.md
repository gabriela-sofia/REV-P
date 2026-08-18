# SUSC-17C24 - Serie temporal pre-evento multi-cena e reducao de nuvem Sentinel-2

## Objetivo
Reduzir a dependencia da cena unica de 2022-04-27 (cloud cover 42,32%) materializando recortes leves reais de multiplas cenas Sentinel-2 pre-evento por patch e consolidando features temporais robustas, mantendo tudo scientific_review_only conforme o 17C23.

## Selecao e materializacao multicena
- Patches processados: 5.
- Cenas pre-evento disponiveis: 60; selecionadas: 15 (>= 3 por patch, datas distintas).
- Patches com pilha multicena: 5.
- Artefatos leves criados: 45 (band_stats.csv + preview.png + stats.json por patch x cena).
- Materializacao via Earth Search/STAC/COG com leitura windowed COG, sem baixar produto completo.

## Estatisticas e features temporais
- Estatisticas por banda: 240; por indice: 180.
- Features multitemporais consolidadas: 40 (mean/median/min/max/std + best_cloud + scene_count + cloud_cover_min/median).
- Matriz robusta a nuvem criada: True (5 linhas).

## Guardrails
- Cena durante/pos-evento usada como pre-evento: 0.
- Features promovidas a treino: 0; produtos completos baixados: 0; raster pesado commitado: 0.
- Deltas nao entram na matriz robusta; fallback Earth Search continua scientific_review_only; Ground Reference: 0; ground truth: 0; label: 0; score v6 intacto; score v7 inexistente; 17B bloqueado.

## minimum_success_achieved: True

## Proximo marco recomendado
SUSC-17C25 Consolidacao multimodal (Sentinel-2 multitemporal + CHIRPS runtime) para dossie sensorial scientific review-only por patch
