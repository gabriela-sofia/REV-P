# SUSC-17C34 - Extracao pre-evento de features locais para patches canarios

## Objetivo
Criar a matriz multimodal LOCAL pre-evento por patch canario (17C33) e contrastar review-only com o patch original / mismatch control. Janela pre-evento: 2022-04-24 a 2022-05-23 (evento: 2022-05-24 a 2022-05-30; pos-evento nao entra como feature).

## Features extraidas (reais/derivadas)
- Canarios consumidos: 11; controles originais: 5.
- Sentinel-2 pre-evento (STAC+COG por ponto): 11 patches; indices espectrais: 11.
- CHIRPS trigger region-level (17C25 herdado): 11; terrain (SRTM30m OpenTopoData): 11; drenagem (OSM proxy): 7; landcover/built-up proxy (NDBI): 11.
- Familias de features disponiveis: 6 (>=3). Completude media: 0.9394.

## Score v6 replay (honesto)
- Replay policy criada: True.
- Full replay computavel: False; partial: False.
- Bloqueio: score v6 usa normalizacao robust_minmax RELATIVA A POPULACAO dos patches oficiais; sem essa distribuicao e sem HAND/hidrografia oficial, um score comparavel NAO e reproduzivel. Score v6 oficial intacto; nenhum score numerico enganoso emitido; score v7 inexistente.

## Contraste original vs canarios
- Linhas de contraste: 11 (features BRUTAS pre-evento: NDWI/MNDWI/NDBI/CHIRPS/elevation/slope/distancia a drenagem).
- Metricas de aderencia observacional: 7 (review-only; sem AUC/treino/odds).

## Resposta cientifica (Resultado B)
Features locais pre-evento reais foram extraidas para os 11 canarios (>=3 familias). A comparacao QUANTITATIVA por score v6 fica DEFERIDA: a normalizacao do score v6 e populacional (patches oficiais) e HAND/hidrografia/landcover oficiais faltam. O contraste e apresentado em features brutas review-only; a ancora observacional (ocorrencia oficial hidrologica) segue favorecendo os canarios como AOIs mais relevantes, sem afirmar suscetibilidade supervisionada.

## Guardrails
- Somente features pre-evento/estaticas; pos-evento nao usado como feature; ocorrencia oficial NAO e feature do score; sem label de evento; sem ground truth/treino; sem score v7; score v6 oficial intacto; patches originais preservados; canario nao e positivo supervisionado; controle nao e negativo verdadeiro; CHIRPS e trigger, nao evento; SAR metadata 17C31 nao vira feature de score.

## minimum_success_achieved: True | result_class: B_features_extracted_score_replay_deferred

## Proximo marco recomendado
SUSC-17C35 Adquirir HAND (pipeline DEM+drenagem) e hidrografia/landcover oficiais para os canarios, e obter a distribuicao/normalizacao dos patches oficiais para um score v6 replay comparavel review-only; manter score v6 oficial intacto e 17B fail-closed.
