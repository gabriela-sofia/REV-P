# Cartao fisico/topografico S17C6_CANARY_REC_00001

## Identificacao

- Canario: `S17C6_CANARY_REC_00001`
- Evento: `S17C_REF_0063`
- Geometria/link: `S17C5_GEOM_0063`
- Regiao/cidade: REC / Recife
- bbox: `-34.944464634,-8.00476035,-34.935481481,-7.995866764`

## Features espectrais herdadas (17E, reais pre-evento)

- rain_idx=0.4932; urban_spectral_idx=0.3679; vegetation_idx=0.4943; moisture_idx=0.6488

## Features de chuva herdadas (17E, CHIRPS pre-evento)

- consolidadas no indice de chuva (rain_idx) do 17E

## Features fisicas encontradas

- HAND=5.2599 (referencia recife_00552); elevation=8.0504 (referencia recife_00552); slope=2.1454 (referencia recife_00552); distance_to_water=1490.9467 (referencia recife_00552); TWI=123.5202 (referencia recife_00552); flow_accumulation=1.6799 (referencia recife_00552)

## Features fisicas ausentes (diretas do canario)

- HAND;elevation;slope;distance_to_water;TWI;flow_accumulation

## Fonte da feature fisica

- Patch oficial de referencia: `recife_00552` a 984.4000 m; sobreposicao=0.0000
- Tipo: referencia_comparativa_review_only
- Direta ou comparativa: comparativa

## Incerteza

- Penalidade de incerteza aplicada ao componente fisico: 0.8359 (referencia distante, sem sobreposicao)

## Impacto na prontidao de calibracao

- Score exploratorio sem fisico: 0.4499 (classe medium)
- Score exploratorio com fisico comparativo penalizado: 0.5315 (classe medium)
- Pode calibracao forte agora: false

## Decisao 17F

- feature_source_mode: referencia_comparativa_review_only
- Calibracao forte permanece bloqueada por referencia comparativa distante; extracao direta enfileirada.

## Por que ainda nao e ground truth

Referencia comparativa de patch vizinho nao confirma ocorrencia no canario; sem verdade de referencia observacional nao ha ground truth.

## Por que ainda nao e treinavel

Sem rotulo validado e sem feature fisica direta, o canario nao alimenta treino supervisionado.

## Por que nao altera o score_v6

O componente fisico e comparativo, penalizado e review-only; o score exploratorio nunca substitui nem recalibra o score_v6 oficial.
