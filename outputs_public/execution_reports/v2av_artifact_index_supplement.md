# v2av artifact index supplement

Additive supplement to `final_delivery_artifact_index.md`. Nothing existing was removed or
rewritten; only v2av artifacts were added. The v2at and v2au registries were NOT overwritten.

| Artifact | Path | Function |
|---|---|---|
| Patch boundary source manifest | `datasets/v2av_patch_boundary_source_manifest.csv` | Per-patch spatial provenance and build feasibility. |
| Patch boundary geometry registry | `datasets/v2av_patch_boundary_geometry_registry.csv` | Built patch boundaries (WKT/GeoJSON/bbox) or blockers. |
| Patch boundary build audit | `datasets/v2av_patch_boundary_build_audit.csv` | 10 build gates per patch. |
| Patch boundary recovery queue | `datasets/v2av_patch_boundary_recovery_queue.csv` | Prioritised geometry recovery/digitization queue. |
| Built GeoJSON | `datasets/geometries/patch_boundaries/patch_boundary_<id>.geojson` | One file per built boundary (none when no metadata). |
| Report | `outputs_public/execution_reports/v2av_patch_boundary_geometry_builder_report.md` | v2av methodological report. |
| Summary | `outputs_public/execution_reports/v2av_patch_boundary_geometry_builder_summary.json` | v2av machine-readable summary. |

Methodological status: **PATCH_BOUNDARY_RECOVERY_READY_FOR_OVERLAY_NOT_FOR_TRAINING**
(`can_train_model=false`, `can_create_operational_labels=false`).
