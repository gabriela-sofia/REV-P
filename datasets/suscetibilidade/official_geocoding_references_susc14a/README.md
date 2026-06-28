# official_geocoding_references_susc14a

Camadas oficiais/rastreaveis de referencia espacial (logradouros, eixos de via,
bairros, enderecos, pontos criticos de alagamento, drenagem) usadas pelo
SUSC-14A para tentar resgatar o vinculo espacial de ocorrencias oficiais de
cheia registradas sem lat/lon.

A aquisicao live so ocorre com `SUSC_13B_NETWORK=1`. Camadas oficiais ja
adquiridas pelo SUSC-13C (em `../observed_event_sources_susc13c_live/`) sao
reutilizadas sem copia.

**Os arquivos brutos NAO sao versionados** (dado oficial pesado; ver `.gitignore`).
A proveniencia completa fica em
`manifests/suscetibilidade/susc_14a_official_reference_download_manifest_v1.csv`.

Governanca: tudo `review_only=true`, `can_be_ground_truth=false`,
`allowed_for_training=false`. Sem raster, sem Sentinel bruto, sem chave de API,
sem Google Maps, sem geocoding generico, sem arquivos acima de 250MB.
