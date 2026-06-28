# observed_event_sources_susc13c_live

Dados oficiais adquiridos ao vivo pelo SUSC-13C-LIVE (rede habilitada via
`SUSC_13B_NETWORK=1`) a partir de CKAN, ArcGIS REST, WFS/GeoServer e portais
HTML oficiais.

Subpastas: `ckan/`, `arcgis/`, `wfs/`, `html/`.

**Os arquivos brutos NAO sao versionados** (dado oficial pesado; ver `.gitignore`).
A proveniencia completa — URL, content-type, tamanho e SHA256 de cada arquivo —
fica em `manifests/suscetibilidade/susc_13c_live_download_manifest_v1.csv`, o que
permite reproduzir a aquisicao. Os eventos parseados (leves) ficam em
`datasets/suscetibilidade/susc_13c_live_observed_events_parsed_v1.csv`.

Governanca: tudo `review_only=true`, `can_be_ground_truth=false`,
`allowed_for_training=false`. Sem raster, sem Sentinel bruto, sem chave de API,
sem arquivos acima de 250MB.
