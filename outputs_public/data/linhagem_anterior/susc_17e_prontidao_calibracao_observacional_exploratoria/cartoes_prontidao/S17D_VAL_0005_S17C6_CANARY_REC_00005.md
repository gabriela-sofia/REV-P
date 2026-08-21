# Cartao de prontidao S17D_VAL_0005

## Identificacao do item

- Item de validacao: `S17D_VAL_0005`
- Evento candidato: `S17C_REF_0063`
- Patch canario: `S17C6_CANARY_REC_00005`
- Geometria: `S17C5_GEOM_0063`
- Fonte: `S17C31_TFC_0001` (`technical_remote_sensing_flood_footprint`)
- Cidade/regiao: Recife / REC
- Fenomeno: flood_inundation_alagamento
- Data do evento: 2022-05-24..2022-05-30

## Vinculo espacial

- Classe de vinculo: `exact_polygon_overlap`
- Patch oficial mais proximo: `recife_00552`
- Incerteza espacial: requires_human_review

## Estado de avaliacao herdado do 17D

- Aceito para avaliacao review-only: true
- Candidato a calibracao no 17D: false (bloqueado por features)

## Features disponiveis (reais, pre-evento)

- NDVI=0.2494 (espectral)
- NDWI=-0.2230 (espectral)
- MNDWI=-0.1110 (espectral)
- NDBI=-0.1148 (espectral)
- CHIRPS_3d=20.5343 (chuva_gatilho)
- CHIRPS_7d=20.5343 (chuva_gatilho)
- CHIRPS_30d=139.6736 (chuva_gatilho)

## Features ausentes

- HAND (fisico)
- elevation (fisico)
- slope (fisico)
- distance_to_water (fisico)
- TWI (fisico)
- flow_accumulation (fisico)
- urban_prop (urbano_territorial)
- vegetation_prop (urbano_territorial)
- water_prop (urbano_territorial)
- MapBiomas (urbano_territorial)
- imperviousness_proxy (urbano_territorial)
- runoff_context_7d (chuva_gatilho)

## Contradicoes

- Nenhuma contradicao critica detectada (vegetacao baixa, sinal hidrico MNDWI positivo).

## Bloqueio principal e acao minima

- Bloqueio principal (para calibracao forte): bloqueado_por_features
- Bloqueios secundarios: amostra_regional_pequena_evento_unico;incerteza_espacial_requer_revisao
- Acao minima para desbloqueio: obter_feature_fisica

## Decisao de prontidao

- Prontidao de calibracao: **pronto_para_calibracao_exploratoria_review_only**
- Entra em calibracao forte: false
- Entra em calibracao exploratoria review-only: true
- Score exploratorio review-only (baseline): 0.3459 (classe low)
- Referencia score_v6 (patch oficial mais proximo): 0.6048 (classe medium)

## Por que ainda nao e ground truth

Aceite review-only nao confirma ocorrencia no patch; sem verdade de referencia observacional o item nunca vira ground truth.

## Por que ainda nao e treinavel

Sem rotulo validado e sem verdade de referencia, o item nao pode alimentar treino supervisionado.

## Por que nao altera o score_v6

O score exploratorio cobre apenas parte dos componentes (espectral e chuva), sem topografia/hidrologia; e review-only e nunca substitui nem recalibra o score_v6 oficial.
