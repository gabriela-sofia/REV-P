# Cartao territorial - Petropolis

## Lacuna 19A
Territorial parcial: presentes urban_prop e vegetation_prop; faltam water_prop,
exposed_soil_prop, impervious_proxy e MapBiomas_class_majority.

## Fontes encontradas
Nenhuma fonte local cobre os alvos territoriais dos patches base. As tabelas de
landcover locais reembrulham urban/vegetation ou cobrem apenas canarios; a
planilha MapBiomas e por estado/bioma e esta em quarentena.

## Preenchimento 19B
Sem preenchimento local. Cobertura territorial de 0.3333
mantida (0.0000 de variacao). Extracao encaminhada ao pacote MapBiomas/GEE.

## Lacunas restantes
MapBiomas_class_majority, exposed_soil_prop, water_prop e impervious_proxy.

## Acao minima
executar pacote MapBiomas/GEE para os 100 patches da regiao.

## Por que nao e ground truth
Features territoriais de suscetibilidade escalavel, sem geometria de ocorrencia.

## Por que nao e treino
eligible_for_training e falso em toda a matriz.

## Por que nao cria score_v7
A sprint organiza cobertura territorial; nao gera score oficial. score_v6 intacto.
