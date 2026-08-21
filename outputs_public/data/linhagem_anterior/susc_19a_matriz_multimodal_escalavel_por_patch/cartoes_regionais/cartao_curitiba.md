# Cartao regional - Curitiba

- Total de patches: 100
- Cobertura fisica media: 1.0000
- Cobertura territorial media: 0.3333
- Cobertura espectral media: 1.0000
- Cobertura de chuva media: 1.0000
- Cobertura documental media: 0.0000
- Cobertura observacional media: 0.0200
- Patches com evidencia observacional: 2

## Lacunas
- Territorial parcial: faltam MapBiomas, exposed_soil_prop, water_prop e impervious_proxy.
- Evidencia documental/observacional restrita aos patches destacados.

## Relacao com o 18H
Curitiba e segunda regiao tecnica: 2 patches com overlay tecnico SAR (curitiba_01050 e curitiba_01101). A geometria oficial segue ausente (18D aguardando). patch_stats SAR e pos-evento e nao entra como feature.

## Por que nao e ground truth
As features sao de referencia review-only, sem geometria de ocorrencia confirmada por patch.

## Por que nao e treino
Nenhuma coluna e alvo; eligible_for_training e falso em toda a matriz.

## Por que nao cria score_v7
A matriz organiza cobertura; nao gera score oficial. score_v6 permanece intacto.

## Proximo passo
ingerir a resposta oficial (18D) e consolidar a referencia tecnica SAR.
