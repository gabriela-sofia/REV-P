# REV-P external evidence intake

This directory stores controlled metadata and, only when explicitly allowed,
raw external evidence files for the v2cn-v2cr sprint.

Rules:

- default execution is offline;
- `sources_registry_v2co.csv` is the only download allowlist;
- raw downloads, when allowed, go under `raw/`;
- acquisition and manifest metadata go under `metadata/`;
- raw external files are never written to `outputs_public`;
- registered or downloaded evidence remains review-only and candidate-only.

Missing license, missing CRS, missing hash, missing provenance or missing patch
boundary must remain blocked.
