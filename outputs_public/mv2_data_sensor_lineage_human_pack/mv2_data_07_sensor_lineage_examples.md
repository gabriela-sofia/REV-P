# DATA-07 - Exemplos (ilustrativos, nao preencher o template com estes valores)

## Exemplo elegivel espectral (promove)
- sensor_family: SENTINEL_2
- source_asset_ref: S2A_MSIL2A_..._T24MTV (produto de origem)
- source_asset_type: SENTINEL_2_L2A
- sensor_source_ref: manifesto de aquisicao CDSE (registro auditavel)
- spectral_eligible: true
- support_only: false

## Exemplo suporte (nao baseline)
- sensor_family: SENTINEL_1
- sensor_source_ref: metadado GRD verificavel
- spectral_eligible: false
- support_only: true

## Exemplos bloqueados
- sensor_family: DINO_DERIVED -> bloqueado (nao e raster espectral).
- sensor_family: PNG_RENDER -> bloqueado (render nao e raster espectral).
- sensor_family: NPZ_EMBEDDING -> bloqueado (embedding nao e raster espectral).
- sensor_family: UNKNOWN ou CONFLICT -> bloqueado.
- SENTINEL_2 sem sensor_source_ref -> bloqueado (sem cadeia rastreavel).
