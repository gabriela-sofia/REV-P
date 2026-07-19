# SUSC-19B - Auditoria e preenchimento de lacunas territoriais

## Estado herdado do 19A

O 19A consolidou 300 patches com fisico, espectral e chuva completos, mas
territorial parcial (apenas urban_prop e vegetation_prop). Estado do 17B:
`17B_APROXIMACAO_COM_SEGUNDA_REGIAO_TECNICA` (17B nao criado).

## Lacuna territorial real

Faltam, nos 300 patches base: MapBiomas_class_majority, MapBiomas_class_distribution,
exposed_soil_prop, water_prop e impervious_proxy. A cobertura territorial media
permanece em 0.3333 (2 de 6 features esperadas).

## Fontes encontradas

| Fonte | Status | Motivo |
| --- | --- | --- |
| LOCAL_18A_STORE | insuficiente | as 300 linhas oficiais apenas reembrulham urban_prop/vegetation_prop de susc_features_by_patch_v1; water_prop not_available; sem MapBiomas nem exposed_soil |
| LOCAL_17C35 | insuficiente | cobre apenas 11 canarios (S17C33), fora do universo base de 300 patches |
| LOCAL_17C34 | insuficiente | cobre apenas canarios; nao cobre os 300 patches base |
| LOCAL_NDBI_PROXY | insuficiente | built_up_proxy_ndbi e proxy espectral derivado de NDBI; nao substitui landcover MapBiomas nem preenche exposed_soil/water/classe |
| LOCAL_MAPBIOMAS_XLSX | insuficiente | nivel estado/bioma (DHN250), nao por patch; arquivo em quarentena local_only, nao versionavel e nao copiavel |
| MAPBIOMAS_GEE | pacote_externo_necessario | MapBiomas Colecao 9 via Earth Engine cobre os 300 patches; requer execucao externa autenticada pelo usuario |

## Houve preenchimento local?

Nao. Nenhuma fonte local cobre os alvos territoriais dos 300 patches base: as
tabelas de landcover reembrulham urban/vegetation ou cobrem apenas canarios, e a
planilha MapBiomas e por estado/bioma e esta em quarentena. Nenhum valor foi
inventado.

## Pacote MapBiomas/GEE

Como nao ha fonte local utilizavel, foi criado um pacote executavel em
`pacote_gee/`: script Earth Engine, documentacao, manifesto de export e schema de
saida esperada. O pacote nao contem credenciais e nao baixa raster pesado.

## Impacto na cobertura multimodal

| Regiao | Patches | Territorial 19A | Territorial 19B | Delta |
| --- | --- | --- | --- | --- |
| curitiba | 100 | 0.3333 | 0.3333 | 0.0000 |
| petropolis | 100 | 0.3333 | 0.3333 | 0.0000 |
| recife | 100 | 0.3333 | 0.3333 | 0.0000 |

A cobertura territorial nao muda nesta sprint porque o preenchimento depende da
execucao externa do pacote MapBiomas/GEE. A fila `fila_extracao_territorial_19b.csv`
lista as tarefas executaveis.

## Lacunas restantes

MapBiomas_class_majority, MapBiomas_class_distribution, exposed_soil_prop,
water_prop e impervious_proxy, em todas as regioes.

## Por que nao e ground truth, nem treino, nem score_v7

As features territoriais sao de suscetibilidade escalavel, sem geometria de
ocorrencia. eligible_for_training e falso; nenhum score oficial e criado; o
coverage mede completude, nao suscetibilidade; o score_v6 permanece intacto.

## Proximo marco recomendado

**SUSC-19C - Avaliacao observacional review-only**: comparar Recife e Curitiba
tecnica com score e features, apos encaminhar a extracao territorial via MapBiomas/GEE.
