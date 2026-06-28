# SUSC-09A — Workspace de Revisão Humana

> O SUSC-09A não executa nem substitui a revisão humana. Ele cria um workspace auditável para que uma pessoa revise os casos espaciais/documentais, preservando o status review-only e impedindo que aderência espacial seja confundida com ground truth.

Workspace em `outputs_public/suscetibilidade/SUSC_09A_human_review_workspace` com **13 casos**, 13 dossiês, 13 mapas SVG e 3 camadas GeoJSON (patch bbox: 6 features; event geometry: 9 features).

## Conteúdo

- `SUSC_09A_review_index.md/.csv` — índice navegável dos casos.
- `case_dossiers/<case_id>.md` — dossiê por caso (identificação, geometria, features, limitações, perguntas).
- `maps_svg/<case_id>.svg` — mapa simples patch×evento (review-only).
- `geojson/` — camadas para QGIS: patch bbox, geometria de evento rastreável, casos.
- `forms/` — formulário pré-preenchido (`approved_for_ground_truth=false`) + instruções.

## Governança

Todos os artefatos: `can_be_ground_truth=false`, `allowed_for_training=false`, `review_only=true`. O workspace organiza a revisão; não a executa nem cria ground truth.

## Limitações

Mapas SVG e bboxes são auxílio visual review-only; geometria de evento é candidata/rastreável, não footprint validado. Aderência espacial ≠ ocorrência confirmada por patch.

> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.
