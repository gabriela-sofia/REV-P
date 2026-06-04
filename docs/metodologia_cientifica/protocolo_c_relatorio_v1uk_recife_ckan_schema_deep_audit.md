# Protocolo C v1uk - Recife CKAN Schema Deep Audit

## Scope
- Event: REC_2022_05_24_30
- Event window: 2022-05-24 to 2022-05-30
- Source: CKAN Recife assets downloaded in v1uj
- No web search, no geocoding, no centroid, no overlay, no labels.

## Findings
- Audited assets: 64
- Occurrence-like tables profiled: 59
- Total table rows profiled: 2744469
- Rows in event window: 11163
- Rows with flood/rain/landslide terms: 45145
- Rows with neighborhood or address evidence: 4202901
- Rows with coordinates: 436824
- Event-window row matches: 25668
- Coordinate review candidates: 0
- Locality-only review candidates: 6672

## Registro de Atendimentos da Defesa Civil
- Attendance CSVs were parsed as documented occurrence tables.
- They expose date, occurrence/request, address, neighborhood, locality, and risk/action fields.
- Public registries contain only hashes/flags/counts, not raw sensitive values.

## Coordinate Evidence
- Coordinate audit classes: {'INFRASTRUCTURE_CONTEXT': 20, 'NO_COORDINATES': 40, 'REGIONAL_CONTEXT_POINTS': 2, 'OCCURRENCE_COORDINATES_CANDIDATE': 2}
- Defesa Civil GeoJSON Coordenadas geograficas da Regiao Sul: REGIONAL_CONTEXT_POINTS.
- Contextual layers are not promoted to occurrence geometry.

## REC_2022 Status
- Can advance to human review: true
- Can advance to overlay preflight now: false
- can_create_ground_reference=false
- can_create_training_label=false
- no_overlay_executed=true

## Next Action
- v1ul - Recife Locality-Only Human Review and Non-Overlay Evidence Package