# v2at artifact index supplement

Additive supplement to `final_delivery_artifact_index.md`. Nothing existing was removed or
rewritten; only v2at artifacts were added.

| Artifact | Path | Function |
|---|---|---|
| Source catalog | `datasets/v2at_external_evidence_source_catalog.csv` | Canonical hierarchy of external evidence sources. |
| Evidence observations | `datasets/v2at_evidence_observation_registry.csv` | Derived observational evidence registry (fail-closed). |
| Event-patch packages | `datasets/v2at_event_patch_package_registry.csv` | Event-patch packages with typing, window, score and promotion decision. |
| Promotion gate audit | `datasets/v2at_promotion_gate_decision_audit.csv` | 15 promotion gates per package. |
| Reviewer queue seed | `datasets/v2at_reviewer_queue_seed.csv` | Prioritised human-review queue. |
| Operational-label blocklist | `datasets/v2at_operational_label_blocklist.csv` | Everything that must NOT become a training label. |
| Report | `outputs_public/execution_reports/v2at_evidence_registry_event_patch_report.md` | v2at methodological report. |
| Summary | `outputs_public/execution_reports/v2at_evidence_registry_event_patch_summary.json` | v2at machine-readable summary. |

Methodological status: **EVIDENCE_SYSTEM_READY_FOR_HUMAN_REVIEW_NOT_FOR_TRAINING**
(`can_train_model=false`, `can_create_operational_labels=false`).
