# REV-P - Current project state after PT-BR curation

Date: 2026-06-18
Repository path: `C:\Users\gabriela\Documents\REV-P`
Current branch: `curadoria/repositorio-publico-ptbr`

## Summary

The repository is now in a public PT-BR curation state. The main public layer is coherent, technically conservative, and aligned with the REV-P methodological boundary: review-only evidence preparation, no operational ground truth, no formal labels, no formal negatives, and no supervised training.

The project should not move into a new modeling sprint. The next correct technical step is controlled manual recovery or restoration of the missing original candidate base `v2dz-v2ef`, followed by schema/count/hash/provenance validation and human review. Without operational ground truth, MV1 and supervised training remain blocked.

## Public narrative state

Primary public files:

- `README.md`.
- `docs/estado_metodologico_revp.md`.
- `docs/metodologia_cientifica/revp_narrativa_cientifica_consolidada.md`.
- `docs/metodologia_cientifica/revp_indice_etapas.md`.
- `docs/metodologia_cientifica/revp_guia_estilo_nomenclatura.md`.
- `outputs_public/README.md`.
- `outputs_public/execution_reports/final_delivery_artifact_index.md`.
- `outputs_public/execution_reports/final_guardrails_report.md`.
- `outputs_public/model/NO_OPERATIONAL_TRAINED_MODEL.md`.

These files are suitable as the public reading layer. Stage codes and English technical identifiers remain valid as internal traceability.

## Scientific status

Confirmed current status:

- `ground_truth_operational_status = ABSENT`.
- Formal labels: absent.
- Formal negatives: absent.
- Supervised training: blocked.
- DINOv2: frozen exploratory encoder only.
- Protocol C: evidence chain for human review, not operational validation.
- MV1: depends on operational ground truth and must remain blocked.
- Fallback: unavailable and not a substitute for original base.
- Textual references: clues only, not recovered content.
- `ORIGINAL_BASE_REQUIRES_MANUAL_RESTORE`: recoverability decision, not recovery.

## Real pipeline today

| Step | Current role | Status |
|---|---|---|
| Territorial corpus | 59 patches across Recife, Petropolis and Curitiba | Consolidated |
| Sentinel-first inventory | 128 candidate assets | Consolidated |
| DINOv2 embeddings | 12 real embeddings, 768D, frozen encoder | Exploratory/review-only |
| Protocol C | External evidence adjudication | Candidate/contextual/temporal only |
| Ground truth search | Official geometry and formal negatives | Blocked |
| TP2/external evidence chain | Candidate prioritization, digitization, QA, replay | Review-only, blocked for labels |
| `v2dz-v2ef` original base | Missing previous working base | Requires manual restore |
| PT-BR curation | Public README, narrative, stage index and style guide | Coherent; in progress branch |

## Critical files for defense, paper and slides

- `README.md`.
- `docs/estado_metodologico_revp.md`.
- `docs/metodologia_cientifica/revp_narrativa_cientifica_consolidada.md`.
- `docs/metodologia_cientifica/revp_indice_etapas.md`.
- `outputs_public/execution_reports/final_delivery_artifact_index.md`.
- `outputs_public/execution_reports/final_guardrails_report.md`.
- `outputs_public/model/NO_OPERATIONAL_TRAINED_MODEL.md`.
- `outputs_public/tables/table_corpus_summary.csv`.
- `outputs_public/tables/table_dino_embedding_inventory.csv`.
- `outputs_public/tables/table_protocol_c_summary.csv`.
- `outputs_public/tables/table_claims_guardrails_summary.csv`.
- `outputs_public/figures/*.png`.

## Historical or auxiliary files

- `outputs_public/execution_reports/arquivo_etapas/*`.
- `outputs_public/tables/saidas_intermediarias/*`.
- `outputs_public/logs_summary/*`.
- `docs/tcc_exports/*`.
- Granular `docs/metodologia_cientifica/protocolo_c_*`.
- Stage-specific `revp_v1*` and `revp_v2*` documents.
- `scripts/ground_truth/*` and related tests.

## Interpretation risks

- Treating candidate/reference evidence as operational ground truth.
- Treating DINOv2 embeddings as detection or prediction.
- Treating Protocol C scores as operational validation.
- Treating `ORIGINAL_BASE_REQUIRES_MANUAL_RESTORE` as restoration completed.
- Treating fallback or textual references as recovered original base.
- Translating internal status/enums and breaking tests.

## Recommended next technical step

Perform a controlled manual restoration attempt for `v2dz-v2ef` from diff, Git object, reflog, local backup, or equivalent external source. Any candidate must pass validation before it is used: expected schema, expected row counts, hashes or provenance, no private path leakage, no automatic promotion to ground truth, and human review.
