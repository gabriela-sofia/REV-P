# v2aw artifact index supplement

Additive supplement to `final_delivery_artifact_index.md`. Nothing existing was removed or
rewritten; only v2aw artefacts were added. v2at/v2au/v2av were NOT overwritten.

| Artifact | Path | Function |
|---|---|---|
| Patch geometry source template | `datasets/v2aw_patch_geometry_sources_template.csv` | Fillable intake for 55 Recife P1 patch boundaries. |
| Event geometry source template | `datasets/v2aw_event_geometry_sources_template.csv` | Fillable intake for observed event geometries (+ CPRM point anchors). |
| Geometry source validation registry | `datasets/v2aw_geometry_source_validation_registry.csv` | Validates provided/existing geometry sources. |
| Recife P1 readiness | `datasets/v2aw_recife_p1_geometry_readiness.csv` | Per-patch readiness for v2av/v2au. |
| Intake instructions | `docs/v2aw_geometry_source_intake_instructions.md` | How to fill the templates safely. |
| Synthetic examples | `datasets/examples/v2aw_geometry_intake/` | Synthetic format examples (no real ids). |
| Report | `outputs_public/execution_reports/v2aw_geometry_source_intake_report.md` | v2aw methodological report. |
| Summary | `outputs_public/execution_reports/v2aw_geometry_source_intake_summary.json` | v2aw machine-readable summary. |

Methodological status: **GEOMETRY_SOURCE_INTAKE_READY_NOT_FOR_TRAINING**
(`can_train_model=false`, `can_create_operational_labels=false`).
