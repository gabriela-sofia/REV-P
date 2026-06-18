# v2au artifact index supplement

Additive supplement to `final_delivery_artifact_index.md`. Nothing existing was removed or
rewritten; only v2au artifacts were added. The v2at registries were NOT overwritten.

| Artifact | Path | Function |
|---|---|---|
| Geometry inventory | `datasets/v2au_geometry_inventory.csv` | Every geometry that actually exists (patch/event/context/point), CRS-validated. |
| Overlay registry | `datasets/v2au_patch_event_overlay_registry.csv` | patch ∩ event intersection, ratio, overlay status. |
| Package overlay update | `datasets/v2au_event_patch_package_overlay_update.csv` | Derived delta of v2at package status (never overwrites v2at). |
| Overlay gate audit | `datasets/v2au_overlay_gate_decision_audit.csv` | 12 overlay gates per package. |
| Overlay review queue | `datasets/v2au_overlay_review_queue.csv` | Prioritised geometry digitization/review queue. |
| Report | `outputs_public/execution_reports/v2au_patch_event_overlay_geometry_report.md` | v2au methodological report. |
| Summary | `outputs_public/execution_reports/v2au_patch_event_overlay_geometry_summary.json` | v2au machine-readable summary. |

Methodological status: **GEOMETRY_OVERLAY_READY_FOR_HUMAN_REVIEW_NOT_FOR_TRAINING** (max decision `C4_CANDIDATE_REQUIRES_HUMAN_REVIEW`;
`can_train_model=false`, `can_create_operational_labels=false`).
