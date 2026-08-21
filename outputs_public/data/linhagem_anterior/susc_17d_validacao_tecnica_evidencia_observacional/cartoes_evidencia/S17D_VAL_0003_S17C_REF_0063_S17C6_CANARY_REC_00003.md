# Cartao tecnico S17D_VAL_0003

## Resumo do evento

- Candidato: `S17C_REF_0063`
- Cidade/regiao: Recife / REC
- Fenomeno: flood_inundation_alagamento
- Data: 2022-05-24..2022-05-30

## Fonte e geometria

- Fonte: `S17C31_TFC_0001`
- Tipo de fonte: `technical_remote_sensing_flood_footprint`
- Geometria: `S17C5_GEOM_0063`
- Patch: `S17C6_CANARY_REC_00003`
- Classe de vinculo: `exact_polygon_overlap`

## Comparacao espacial e temporal

- Pontuacao geometrica: 85
- Pontuacao de vinculo com patch: 65
- Pontuacao temporal: 80
- Risco de vazamento temporal: 80

## Comparacao de features e score

- score_v6: disponivel=false; valor=not_available; consistencia=nao_avaliada
- score_v6_class: disponivel=false; valor=not_available; consistencia=nao_avaliada
- HAND: disponivel=false; valor=not_available; consistencia=nao_avaliada
- slope: disponivel=false; valor=not_available; consistencia=nao_avaliada
- elevation: disponivel=false; valor=not_available; consistencia=nao_avaliada
- distance_to_water: disponivel=false; valor=not_available; consistencia=nao_avaliada
- TWI: disponivel=false; valor=not_available; consistencia=nao_avaliada
- flow_accumulation: disponivel=false; valor=not_available; consistencia=nao_avaliada
- urban_prop: disponivel=false; valor=not_available; consistencia=nao_avaliada
- vegetation_prop: disponivel=false; valor=not_available; consistencia=nao_avaliada
- water_prop: disponivel=false; valor=not_available; consistencia=nao_avaliada
- NDVI: disponivel=false; valor=not_available; consistencia=nao_avaliada
- MNDWI: disponivel=false; valor=not_available; consistencia=nao_avaliada
- NDBI: disponivel=false; valor=not_available; consistencia=nao_avaliada
- CHIRPS_3d: disponivel=false; valor=not_available; consistencia=nao_avaliada
- CHIRPS_7d: disponivel=false; valor=not_available; consistencia=nao_avaliada
- CHIRPS_30d: disponivel=false; valor=not_available; consistencia=nao_avaliada
- runoff_context_7d: disponivel=false; valor=not_available; consistencia=nao_avaliada

## Bloqueios e decisao tecnica

- Bloqueios criticos: nenhum
- Confianca tecnica: media
- Status: aceito_para_avaliacao_review_only
- Pode entrar em avaliacao review-only: true
- Pode entrar em prontidao de calibracao futura: false

## Justificativa tecnica

fonte=85; temporal=80; geometria=85; vinculo=65; features do patch candidato nao disponiveis; ausencia nao reprova; sem leitura de feature temporal do patch candidato

## Por que nao e ground truth

validacao tecnica review-only; nao e verdade de referencia, nao e treino e nao autoriza score_v7
