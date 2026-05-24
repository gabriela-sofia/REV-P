# Protocolo C v1iy - Qualidade local do patch Sentinel no anchor oficial

Esta etapa audita os patches Sentinel-2 gerados na v1ix para o anchor oficial CPRM do ANEXO-II, Moinho Preto, Petropolis, vistoria de 19-02-2022, movimento de massa.

A v1ix foi um avanço material porque saiu do bloqueio operacional do GEE e gerou dois patches reais, centrados no anchor oficial, com bandas B02, B03, B04, B08, B11 e B12. Os patches ficaram locais, com metadados publicos em registry, sem criar label, target, treino ou ground truth operacional.

## Por que auditar localmente

A cena pre-evento selecionada tem `CLOUDY_PIXEL_PERCENTAGE=90.359181`. Esse valor descreve a cena inteira no metadado do Sentinel-2, nao necessariamente o recorte de 96 x 96 pixels no entorno do anchor. Por isso, ele nao deve aceitar nem rejeitar sozinho o patch local.

A v1iy separa duas perguntas:

- o patch local tem bandas, geometria, pixels validos e variacao espectral suficientes para revisao?
- existe mascara local de nuvem/sombra no arquivo baixado para medir nuvem no recorte?

No arquivo local v1ix foram baixadas apenas as bandas espectrais exigidas. As bandas SCL/QA60 nao estao presentes. Portanto, a fracao local de nuvem nao foi medida diretamente e o status de nuvem local fica `CLOUD_MASK_NOT_AVAILABLE`.

## Resultado local

O patch pre-evento passou nos checks locais de estrutura:

- bandas completas: B02, B03, B04, B08, B11, B12;
- shape: 96 x 96;
- CRS: EPSG:32723;
- valid_pixel_fraction: 1.000000;
- nodata_fraction: 0.000000;
- NaN/inf: ausente;
- ranges e variancia por banda: OK;
- NDWI aproximado e NDBI aproximado: computaveis.

O patch pos-evento tambem passou nos checks locais de estrutura, com as mesmas bandas, shape, CRS, pixels validos e indices espectrais computaveis.

## Decisao

O par pre/pos fica como `PRE_PATCH_CLOUD_RISK_HIGH`.

Isso significa que a qualidade local minima foi suficiente para manter o par como candidato de patch de referencia e candidato de revisao multimodal, mas a cena pre-evento carrega risco alto de nuvem porque o metadado global e elevado e nao ha mascara local SCL/QA60 no patch baixado.

Essa decisao nao transforma o patch em label, target, dado de treino ou ground truth operacional. Ela apenas documenta que o par pode ser revisado como referencia visual/espectral, com ressalva explicita sobre incerteza de nuvem no pre-evento.

## Proximo requisito minimo

Antes de qualquer uso mais forte, o minimo recomendado e obter uma mascara local SCL/QA60 para o mesmo recorte ou escolher uma cena pre-evento alternativa com menor risco de nuvem local. Mesmo assim, qualquer interpretacao deve permanecer separada de label e treino supervisionado.
