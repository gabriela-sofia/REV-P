# Instrucoes de ingestao da resposta oficial Curitiba

Coloque a resposta oficial em:

`local_runs/suscetibilidade/18d_resposta_oficial_curitiba`

Formatos aceitos: CSV com `lat`/`lon`, CSV com `bbox`, GeoJSON, JSON, WKT e shapefile
quando houver suporte local. Campos minimos: `candidate_event_id`, `data_evento`,
`tipo_fenomeno`, `geometry_type`, `geometry_source` e `crs`.

Depois execute:

`python scripts\suscetibilidade\ingest_susc_18d_resposta_oficial_curitiba.py`

O resultado esperado e a normalizacao para EPSG:4326, overlay com os 43 poligonos CUR
e geracao de patch-links. Bairro, rua, centroide, area administrativa, alerta e risco
continuam bloqueados.
