# Cartao regional - Recife

- Total de patches: 100
- Cobertura fisica media: 1.0000
- Cobertura territorial media: 0.3333
- Cobertura espectral media: 1.0000
- Cobertura de chuva media: 1.0000
- Cobertura documental media: 0.0500
- Cobertura observacional media: 0.0500
- Patches com evidencia observacional: 5

## Lacunas
- Territorial parcial: faltam MapBiomas, exposed_soil_prop, water_prop e impervious_proxy.
- Evidencia documental/observacional restrita aos patches destacados.

## Relacao com o 18H
Recife tem 5 patches com evidencia observacional forte review-only (canarios), coerente com o 18H. E a referencia mais completa.

## Por que nao e ground truth
As features sao de referencia review-only, sem geometria de ocorrencia confirmada por patch.

## Por que nao e treino
Nenhuma coluna e alvo; eligible_for_training e falso em toda a matriz.

## Por que nao cria score_v7
A matriz organiza cobertura; nao gera score oficial. score_v6 permanece intacto.

## Proximo passo
consolidar features territoriais faltantes e ampliar eventos.
