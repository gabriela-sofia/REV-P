# REV-P v2cq - external geospatial QA

This milestone checks whether local external evidence can be treated as a
geospatial candidate for TP2 review. Accepted vector formats are GeoJSON,
GeoPackage, Shapefile and WKT CSV with explicit CRS. GeoTIFF rasters can be
context only. PNG, JPEG, PDF, HTML, TXT, Markdown and files without CRS are not
validated as observed geometry.

If geospatial dependencies are unavailable, QA blocks instead of inventing a
result. A validated external geometry remains candidate-only.

Execution:

```powershell
python scripts\multimodal\revp_v2cq_external_geospatial_qa.py --force
```
