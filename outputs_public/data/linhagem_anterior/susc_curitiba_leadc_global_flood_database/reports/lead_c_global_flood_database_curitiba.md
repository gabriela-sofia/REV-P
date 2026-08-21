# Lead C (Curitiba) -- Global Flood Database (Tellman et al. 2021): mirror sem GEE, evento real encontrado

**Status**: concluído -- mesmo mirror público já usado no Lead C de Recife
(`lead_c_global_flood_database.md`, SUSC-20A), agora aplicado a Curitiba, com um
evento MODIS-validado real encontrado com pixels de inundação genuína sobre o
município.

## Método (idêntico ao Lead C Recife)

1. Catálogo completo DFO (`github.com/cloudtostreet/MODIS_GlobalFloodDatabase`,
   `data/shp_files/dfo_polys_20191203.shp`, 4.825 registros globais, 1985-2019),
   baixado via `raw.githubusercontent.com`, sem autenticação.
2. Filtro: `Country == 'BRAZIL'` e polígono intersectando bbox do Paraná
   (lon -54.6/-48.0, lat -26.8/-22.5) -- **28 candidatos**.
3. Para cada candidato com ano >= 2000, verificação de existência real no bucket
   público `gs://gfd_v1_4` (só os 913 eventos finais MODIS-validados existem lá;
   o catálogo DFO bruto tem candidatos não-validados).
4. **DFO_4276** (2015-07-10 a 2015-07-21, "Heavy Rain", 3 mortos, 1000 desalojados,
   Brasil/Paraguai/Argentina/Uruguai -- evento regional de chuva do sul do continente)
   **existe no bucket** -- baixado (28,4 MB), recortado exatamente na bbox metropolitana
   de Curitiba (lon -49.45/-49.10, lat -25.65/-25.30).

## Resultado real (não fabricado)

- **54 pixels com `flooded=1`** dentro da bbox de Curitiba.
- **0 pixels de água permanente** (`jrc_perm_water=1`) nessa mesma área -- ou seja,
  **todos os 54 pixels são inundação nova genuína** (diferente de Recife, onde 167
  pixels brutos viraram 13 após excluir água permanente).
- Duração: 1-3 dias.
- Centróide dos pixels: lon=-49.3674, lat=-25.5166.
- Geocodificação reversa (Nominatim): **bairro São Miguel, Curitiba/PR**
  (CEP 81452-070).

## Novo registro candidato

| point_id | data | lat | lon | tier |
|---|---|---|---|---|
| LEADC_CTBA_2015_0001 | 2015-07-10 | -25.516635 | -49.367363 | `global_flood_database_modis_event_extent_centroid` (centróide de cluster de pixels MODIS reais, não endereço/bairro-oficial -- mesmo tier já usado para o ponto de Recife, `LEADC_2008_0001`) |

## Limitações explícitas (idênticas às já documentadas para o ponto de Recife)

- Isto é um evento MODIS-validado real com pixels de inundação genuínos, **não** uma
  confirmação administrativa local (decreto/SEDEC) -- é um tier de evidência diferente,
  já rotulado como tal no dataset v12 (`negative_source_type`/tier system).
- O evento é regional (bacia do Prata, chuva de inverno de 2015), não o mesmo evento de
  15-17/01/2022 já documentado via fonte administrativa em
  `SUSC_18C_AQUISICAO_GEOMETRIA_OFICIAL_CURITIBA.md` (S17C_REF_0060) -- são dois eventos
  reais distintos, ambos válidos, não devem ser confundidos.
- Resolução MODIS (~250m/pixel) é grosseira para delimitação urbana fina -- o centróide
  aponta o bairro, não uma geometria de ocorrência precisa (mesma ressalva já feita para
  Recife).
- Nenhum label foi criado; este é um candidato a evidência, sujeito ao mesmo processo de
  adjudicação/QA já usado para os pontos SEDEC do v12 antes de entrar em qualquer dataset
  de treino.

## Arquivos

- Raster completo baixado nesta sessão (28,4 MB, gs://gfd_v1_4/DFO_4276_...) -- **não
  commitado** (raster bruto, seguindo a regra do projeto de nunca versionar `.tif`/dados
  brutos); reprodutível a partir da URL pública documentada acima.
