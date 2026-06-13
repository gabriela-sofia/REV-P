# v2be TP1 review and replay instructions

Open `datasets/external_sources/recife_minimal_tp/derived/patch_boundary_REC_00019_from_lineage.geojson` in QGIS and confirm its location, extent, source CRS lineage and correspondence with the expected Sentinel patch. Review the separate `FILL_THIS_PATCH_BOUNDARY.autofill_tp1_candidate_v2be.csv`; do not overwrite the original `FILL_THIS_PATCH_BOUNDARY.csv` before review.

After human confirmation, copy the reviewed values into the manual intake and run:

1. `python scripts/run_v2ba_minimal_real_geometry_acquisition_workbench.py --mode validate`
2. `python scripts/run_v2az_turning_point_replay_orchestrator.py --mode dry_run`

Only run ingest/replay after explicit review. Never promote the candidate to a label or final truth.
