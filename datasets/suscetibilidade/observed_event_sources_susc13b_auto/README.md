# observed_event_sources_susc13b_auto

Diretorio de aquisicao automatica/controlada do SUSC-13B-AUTO.

Conteudo:
- Subpastas por metodo (`ckan/`, `arcgis/`, `wfs/`, `portal/`) recebem arquivos
  baixados automaticamente quando `SUSC_13B_NETWORK=1` e a fonte foi marcada
  como `download_candidate=true` na descoberta.
- Voce tambem pode colocar manualmente arquivos oficiais/pequenos aqui (CSV,
  XLSX, GeoJSON, JSON, KML/KMZ, WKT, TXT, XML/GML, GPKG, ZIP vetorial, PDF
  pequeno) com data, tipo de evento e coordenada/poligono.

Bloqueado: raster, Sentinel bruto, executavel, Google Maps, geocoding generico,
chave de API e arquivos acima de 250MB. Todos os artefatos permanecem
`review_only=true`, `can_be_ground_truth=false`, `allowed_for_training=false`.
