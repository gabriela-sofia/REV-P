# v2be TP1 patch boundary integration gate

`REC_00019` was recovered by v2bd from explicit Sentinel asset bounds in `EPSG:32725` and exported as a candidate GeoJSON in `EPSG:4326`. The back-projected normalized geometry matches the preserved original bounds. It is a TP1 candidate because direct lineage, CRS, bounds, provenance and hash exist, but the raster payload is absent and human review remains mandatory.

The candidate is not an event polygon, label, final ground truth, or automatic C4 promotion.
