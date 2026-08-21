# Cartao territorial - Petropolis

## Cobertura territorial 19A/19B
Antes do 19F a cobertura territorial era 0.3333 (2 de 6 features:
urban_prop e vegetation_prop). O 19B manteve 0.3333 e preparou o
pacote MapBiomas/GEE.

## Resultado MapBiomas 19F
MapBiomas 19F: 100 patches preenchidos (colecao 9, ano 2022); impervious_proxy medio 0.1397, water_prop medio 0.0028.

## Classes majoritarias
Classe majoritaria por patch registrada em `matriz_territorial_mapbiomas_19f.csv`
(codigo MapBiomas e rotulo). Predominio urbano nas areas construidas.

## water_prop / exposed_soil / impervious
Proporcoes derivadas do histograma de classes MapBiomas por patch (agua, solo exposto,
area urbanizada). Valores reais ingeridos.

## Lacunas restantes
nenhuma (territorial completo)

## Por que nao e ground truth
Uso/cobertura do solo e feature territorial escalavel review-only; MapBiomas nao e
verdade de campo de enchente e nao vira ground truth.

## Por que nao e treino
eligible_for_training=false; a feature territorial nao habilita treino supervisionado.

## Por que nao cria score_v7
score_v7 permanece bloqueado (amostra, benchmark e validacao); preencher territorial
nao autoriza score_v7 e nao altera o score_v6.
