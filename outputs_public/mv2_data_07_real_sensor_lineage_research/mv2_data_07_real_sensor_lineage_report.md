# DATA-07 - pesquisa de lineage sensorial real

## Estado
- targets investigados: 10
- familia Sentinel-2 documentada (committed): 5
- distribuicao por familia: {"SENTINEL_2": 5, "UNKNOWN": 5}
- distribuicao por review: {"NEEDS_REVIEW": 5, "UNKNOWN_BLOCKED": 5}
- lineage completo forte: 0
- input local criado (nao versionado): False

## Evidencia
- REC (5): v2at declara sentinel_sensor_family=SENTINEL2_MSI (obs 2022-05-24) e v1fu marca
  SENTINEL_TIF_ASSET -> familia Sentinel-2 documentada por registries committed. Falta o
  product_id/scene_id Sentinel-2 explicito (sensor_source_ref) -> MEDIUM / NEEDS_REVIEW.
- PET (5): v1fu marca SENTINEL_TIF_ASSET, mas v2at registra sensor UNKNOWN. Sem declaracao
  Sentinel-2 explicita -> UNKNOWN_BLOCKED (nao inferido por nome de asset).

## Regras respeitadas
- so SENTINEL_2 pode ser spectral_eligible; SENTINEL_1 seria support_only;
- DINO/PNG/NPZ/UNKNOWN bloqueiam; sensor nunca inferido por nome visual;
- paths locais absolutos redigidos; render nunca usado como raster.

## Side effects
- chamadas/downloads/rasters/crops: 0/0/0/0.
