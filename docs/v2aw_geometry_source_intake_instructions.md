# v2aw - Geometry Source Intake instructions

## 1. Objetivo
A v2aw cria o canal auditavel para inserir geometrias reais de boundary de patch (e, quando
aplicavel, geometrias observadas de evento) sem contaminar o projeto. Ela comeca pelos 55
patches Recife P1 priorizados pela fila de recuperacao da v2av.

## 2. Por que o REV-P precisa de boundary vetorial de patch
A v2au mostrou que existem 172 pacotes evento-patch, mas 0 overlay, porque nao existe
geometria vetorial de patch. Sem um poligono de boundary (com CRS), nao ha como calcular
`patch ∩ evento`. A v2aw nao inventa geometria: ela pede a geometria real.

## 3. Formatos aceitos
- `bbox`: `minx,miny,maxx,maxy`
- `wkt`: `POLYGON((x y, x y, ...))`
- `geojson_inline`: um objeto GeoJSON Polygon em uma celula
- `geojson_file`: caminho relativo para um arquivo `.geojson`

### Exemplos
- bbox (EPSG:4326): `-34.95,-8.10,-34.90,-8.05`
- WKT (EPSG:3857): `POLYGON((-3888000 -893000, -3887000 -893000, -3887000 -892000, -3888000 -892000, -3888000 -893000))`
- GeoJSON inline: `{"type":"Polygon","coordinates":[[[-34.95,-8.10],[-34.90,-8.10],[-34.90,-8.05],[-34.95,-8.05],[-34.95,-8.10]]]}`

## 4. Campos obrigatorios (patch)
`geometry_source_id, linked_patch_id, source_type, crs, provenance_note` (alem de
`geometry_value` OU `geometry_path`). Sem CRS, a geometria e bloqueada.

## 5. Como preencher o template Recife
1. abra `datasets/v2aw_patch_geometry_sources_template.csv` (55 linhas Recife P1, `source_type=missing`);
2. para cada patch com geometria real, preencha `source_type`, `geometry_value`/`geometry_path`, `crs`,
   `provenance_type`, `provenance_note`, `digitized_by`, `digitized_at`, `source_document`,
   `source_confidence`, `license_status` e mude `review_status` para `provided_unreviewed`;
3. NAO invente geometria; deixe `missing` o que nao tiver dado real.

## 6. Como evitar erro metodologico
- Um ponto (centroide) NAO e boundary de patch. Boundary precisa ser poligono.
- Um ponto de evento (ex.: ponto CPRM) e `observed_event_point_anchor`, NAO um overlay.
- Geometria de contexto/risco NAO promove C4.
- CRS e obrigatorio; CRS desconhecido bloqueia.
- Geometria nunca e inventada; ausencia vira blocker, nunca negativo.

## 7. Por que ponto nao serve como boundary
Um overlay `patch ∩ evento` precisa de area. Um ponto tem area zero; usa-lo como boundary
fabricaria uma area falsa. Por isso `allow_point_as_patch_boundary=false`.

## 8. Por que CRS e obrigatorio
Sem CRS nao da para reprojetar nem calcular area/intersecao de forma confiavel. CRS aceitos:
EPSG:4326, EPSG:3857, EPSG:31982, EPSG:31983.

## 9. Fluxo depois do preenchimento
1. preencher `datasets/v2aw_patch_geometry_sources_template.csv`;
2. salvar como `datasets/v2av_patch_geometry_sources.csv` (mapeando `linked_patch_id` -> `patch_id`,
   `source_type`/`geometry_value`/`crs` iguais) ou alimentar o motor v2av conforme previsto;
3. rodar a v2av (`python scripts/run_v2av_patch_boundary_geometry_builder.py`) para gerar os GeoJSON;
4. apontar os GeoJSON gerados no manifesto da v2au (`datasets/v2au_geometry_sources.csv`) como
   `patch_boundary`, junto com a geometria observada de evento (poligono digitalizado);
5. rodar a v2au (`python scripts/run_v2au_patch_event_overlay_geometry.py`);
6. revisar o C4 candidate manualmente.

## 10. Isto NAO cria label automaticamente
Nenhum passo acima cria label operacional, ground truth final ou treina modelo. O resultado
maximo de um overlay confirmado e `C4_CANDIDATE_REQUIRES_HUMAN_REVIEW`, sempre sob revisao humana.
