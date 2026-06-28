# observed_event_sources_susc13a

Diretorio de entrada manual para o SUSC-13A.

Coloque aqui apenas arquivos oficiais/tecnicos pequenos e rastreaveis que possam
fortalecer evidencias observacionais reais de alagamento/inundacao:

- CSV, TSV ou XLSX com data, tipo de evento e lat/lon.
- GeoJSON, KML, KMZ, GPKG ou ZIP de SHP vetorial com CRS documentado.
- WKT ou TXT pequeno contendo geometria explicita.
- PDF pequeno quando trouxer tabela/texto rastreavel de ocorrencia, data e local.

Preferir arquivos que contenham: data ou periodo do evento, municipio/regiao,
coordenada ou poligono/bbox, tipo de evento, nome da fonte, instituicao e URL ou
referencia de origem. Nao colocar raster, Sentinel bruto, Google Maps, geocoding
generico, chaves de API ou arquivos acima de 100MB.

Todos os arquivos permanecem `review_only=true`, `can_be_ground_truth=false` e
`allowed_for_training=false`.
