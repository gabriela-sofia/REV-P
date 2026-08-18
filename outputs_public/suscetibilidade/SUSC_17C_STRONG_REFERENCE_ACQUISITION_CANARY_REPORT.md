# SUSC-17C - Strong Reference Acquisition Canary (review-only)

Status: review-only. `trainable=false`; `ground_truth=false`; `score_v6_changed=false`; `score_v7_created=false`; `strong_reference_created_count=0`; `technical_footprint_created_count=0`; `readiness_status=SAR_CANARY_READY`.

O SUSC-17C prepara a aquisicao de referencias fortes datadas e verificaveis review-only. Ele ancora em evento oficial datado e planeja o caminho Sentinel-1/SAR como produtor de footprint tecnico, nao como score. Nao cria benchmark 17B, nao altera o score v6 oficial, nao cria score v7, nao cria treino, modelo, label ou ground truth, nao transforma footprint em ground truth, nao transforma ausencia documental em negativo, nao chama alerta de ocorrencia e nao chama area de risco de evento ocorrido.

## 1. Por que o 17B estava bloqueado apos o 17A

O 17A consolidou 65 referencias fortes patch-linked, mas vindas de apenas 2 footprints, em Curitiba, com 0 `event_date`. Sem datas nao ha janela pre/pos para avaliacao temporal; sem diversidade de fontes e regioes, um mini-benchmark de evento seria fragil. O 17C ataca exatamente esse gargalo.

## 2. Estrategia: de registros fracos para registros fortes

A esteira liga inventario de fontes -> fila de eventos candidatos -> resolucao de datas -> viabilidade SAR -> priorizacao de 3-5 canaries -> plano Sentinel-1 -> fila de QA humano. Nada vira benchmark, score v7, treino ou ground truth nesta etapa.

## 3. Por que evento oficial datado e a ancora

A data oficial define a janela temporal pre/pos que torna a deteccao de mudanca (SAR) e a avaliacao patch-level possiveis. Sem data, o caso fica bloqueado. Por isso o 17C so promove candidatos com `exact_day` ou `date_range`.

## 4. Por que SAR entra como produtor de footprint tecnico, nao como score

O Sentinel-1/SAR produz uma geometria de inundacao candidata (footprint tecnico) por deteccao de mudanca pre/pos, com mascaras de agua permanente, HAND/slope e guarda de ambiguidade urbana. Esse footprint e evidencia observacional candidata para avaliar o score v6 - nunca um score, nunca ground truth, sempre com QA humano.

## 5. Fontes internas aproveitadas

11 fontes inventariadas (somente reais). Destaques: International Charter (evento datado 2022-05-24 com geometria oficial coarse), SUSC-16A sentinel event window plan (161 janelas Recife datadas, S1-elegiveis), Defesa Civil, e os artefatos internos 16A/16C/17A. Fontes prioritarias ausentes nos artefatos (CEMADEN, APAC, ANA/Hidroweb, S2iD autonomo) ficam como lacuna de aquisicao, nao foram fabricadas.

## 6. Quantos candidatos foram encontrados

174 eventos candidatos no target pack.

## 7. Quantos tem data utilizavel

163 com data resolvida (`exact_day`=161, `date_range`=2, `month_only`=0, `unknown/not_available`=11).

## 8. Quantos sao viaveis para SAR

162 candidatos SAR; 5 com geometria oficial; 162 candidatos fortes (data + geometria oficial OU data + viabilidade SAR), todos review-only.

## 9. Quais 3-5 eventos foram priorizados

1. `S17C_E_SUSC13A_00001` (Recife, 2022-05-24, SRC_INTERNATIONAL_CHARTER) -> official_observed_event_polygon; verificar geometria oficial e QA humano; opcionalmente confirmar via SAR
2. `S17C_W_S16AWIN_00003` (recife, 2014-01-15, SRC_16A_SENTINEL_WINDOW_PLAN) -> technical_remote_sensing_flood_footprint; executar Sentinel-1/SAR canary para produzir footprint tecnico candidato
3. `S17C_W_S16AWIN_00004` (recife, 2014-01-16, SRC_16A_SENTINEL_WINDOW_PLAN) -> technical_remote_sensing_flood_footprint; executar Sentinel-1/SAR canary para produzir footprint tecnico candidato
4. `S17C_W_S16AWIN_00005` (recife, 2014-01-21, SRC_16A_SENTINEL_WINDOW_PLAN) -> technical_remote_sensing_flood_footprint; executar Sentinel-1/SAR canary para produzir footprint tecnico candidato
5. `S17C_W_S16AWIN_00006` (recife, 2014-01-23, SRC_16A_SENTINEL_WINDOW_PLAN) -> technical_remote_sensing_flood_footprint; executar Sentinel-1/SAR canary para produzir footprint tecnico candidato

## 10. O que ainda bloqueia o benchmark 17B

["strong_candidates_concentrated_in_one_region", "no_sar_footprint_executed_yet_technical_footprint_created_count_0"]

Nenhum footprint tecnico foi realmente produzido (`technical_footprint_created_count=0`): o SAR foi planejado, nao executado. A diversidade regional dos candidatos fortes ainda e baixa.

## 11. Proximo passo

SUSC-17C2 Sentinel-1/SAR Footprint Execution para produzir footprints tecnicos datados a partir dos canaries priorizados. Se o gargalo for execucao, seguir para `SUSC-17C2 Sentinel-1/SAR Footprint Execution`; se for geometria oficial, seguir para `SUSC-17C2 Official Geometry Acquisition`; `SUSC-17B Event-Based Mini-Benchmark` so quando houver datas, diversidade e patch-links fortes suficientes.
