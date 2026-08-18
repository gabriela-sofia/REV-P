# flood_footprints_susc14a

Footprints publicos de cheia/inundacao (vetor) descobertos pelo SUSC-14A:
Copernicus EMS Rapid Mapping, Dartmouth Flood Observatory, Global Flood Database,
SGB/CPRM, INEA, Defesa Civil RJ, APAC, GeoCuritiba, quando disponiveis em formato
vetorial publico.

Aquisicao live so com `SUSC_13B_NETWORK=1`. **Sem raster pesado**: se a unica
forma disponivel for raster/preview, registra-se `footprint_unavailable_vector_required`.

**Os arquivos brutos NAO sao versionados** (ver `.gitignore`). Proveniencia em
`manifests/suscetibilidade/susc_14a_flood_footprint_manifest_v1.csv`.

Governanca: tudo `review_only=true`, `can_be_ground_truth=false`,
`allowed_for_training=false`.
