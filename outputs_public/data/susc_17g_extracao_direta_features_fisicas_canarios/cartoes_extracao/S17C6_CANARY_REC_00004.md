# Cartao de extracao S17C6_CANARY_REC_00004

## Identificacao

- Canario: `S17C6_CANARY_REC_00004`
- Evento: `S17C_REF_0063`
- bbox: `-34.935481481,-7.986973178,-34.926498328,-7.978079592` (CRS EPSG:4326)

## Insumos encontrados

- Modelo de elevacao Copernicus GLO-30 do dominio de bacia (cobre a area do canario)
- Hidrografia oficial (faixas-marginais) para proximidade hidrica

## Insumos ausentes

- nenhum insumo obrigatorio ausente

## Features diretas extraidas

- elevation_mean: 49.3445
- slope_mean: 7.4668
- HAND_mean: 19.9464
- TWI_mean: 7.8824
- flow_accumulation_mean: 1.1544
- distance_to_water_min_m: 19.4700

## Features ainda ausentes

- nenhuma

## Qualidade da extracao

- media_metodo_reconstruido_resolucao_92m
- Metodo reconstruido (17C36), resolucao aproximada de 92 m, review-only, nao provado equivalente ao oficial.

## Impacto na calibracao

- Descritor topografico direto (review-only): 0.2469
- Score exploratorio sem fisico: 0.3550 (classe low)
- Pode calibracao forte review-only: true

## Decisao 17G

- feature_source_mode: direta_por_dem_e_hidrografia_local
- Com features fisicas diretas + espectral + chuva reais, a calibracao forte review-only fica possivel.

## Por que ainda nao e ground truth

Feature fisica direta descreve o terreno, nao confirma ocorrencia de inundacao no canario; sem verdade de referencia observacional nao ha ground truth.

## Por que ainda nao e treinavel

Sem rotulo validado, mesmo com features fisicas diretas, o canario nao alimenta treino supervisionado.

## Por que nao altera o score_v6

A extracao usa metodo reconstruido review-only; alimenta apenas score exploratorio, nunca substitui nem recalibra o score_v6 oficial.
