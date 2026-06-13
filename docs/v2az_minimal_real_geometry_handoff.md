# v2az - Handoff minimo de geometria real

Preencha primeiro o patch `REC_00019` e o evento `REC_2022_05_24_30` selecionados de forma auditavel.
Campos minimos: `source_type`, `geometry_value` ou `geometry_path`, `crs`, proveniencia, licenca e
review_status valido. Patch aceita bbox/WKT/GeoJSON; evento aceita WKT/GeoJSON polygon. Rode dry_run,
leia blockers e corrija antes de replay. Nao use ponto/centroide.
