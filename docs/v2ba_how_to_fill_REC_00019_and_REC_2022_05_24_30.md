# Fill the minimal Recife pair

Put a real patch boundary in `datasets/external_sources/recife_minimal_tp/patch_boundary_REC_00019/`
or fill `FILL_THIS_PATCH_BOUNDARY.csv`. Put a real observed-event polygon in
`event_polygon_REC_2022_05_24_30/` or fill `FILL_THIS_EVENT_POLYGON.csv`.

Fill `source_type`, either `geometry_value` or `geometry_path`, explicit `crs`, `provenance_note`,
`source_document`, `source_public`, `access_status`, and `review_status`. Accepted examples are bbox
`minx,miny,maxx,maxy`, polygon WKT, and polygon GeoJSON. Run v2ba `validate`, then v2az `dry_run`;
use v2az `replay` only when feeds are valid.
