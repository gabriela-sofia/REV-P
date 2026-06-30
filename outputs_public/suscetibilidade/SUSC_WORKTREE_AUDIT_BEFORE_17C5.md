# SUSC worktree audit before SUSC-17C5

Branch: `marco/pre-unificacao-gates-mv1`

HEAD antes do SUSC-17C5: `ff66ef1c8299f3dad70d75fa73ef2f981f9874d7 feat: ingere artefatos oficiais candidatos SUSC-17C4`.

Status: auditoria pre-programacao para `SUSC-17C5 - Patch Grid Expansion Review`, sprint review-only e fail-closed para avaliar se a geometria oficial real do Charter 758, hoje fora da grade SUSC Recife, pode orientar uma futura expansao/alinhamento de grade sem criar patch oficial, patch-link oficial, score v7, benchmark 17B, treino, modelo, label ou ground truth.

## Comandos executados

- `git branch --show-current`, `git rev-parse HEAD`, `git diff --cached --name-only`
- `git status --short`, `git diff --name-only`, `git ls-files --others --exclude-standard`
- `git hash-object datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv`
- `Test-Path datasets/suscetibilidade/susc_score_v7_candidate_by_patch_v1.csv`
- leitura dos outputs 17C4: `susc_17c4_candidate_geometries.geojson`, `susc_17c4_extracted_reference_candidates.csv`, `susc_17c4_candidate_patch_links.csv`, `susc_17c4_summary.json`
- leitura dos outputs 17C3: `susc_17c3_patch_coverage_audit.csv`, `susc_17c3_next_action_decision_table.csv`
- localizacao preliminar da grade SUSC em `datasets/suscetibilidade/susc_features_by_patch_v1.csv`
- busca preliminar por registros historicos do Protocolo C contendo `REC_00019`
- preflight validators 17C4/17C3/17C2/17C/17A/16D

## Estado objetivo

- Branch: `marco/pre-unificacao-gates-mv1`.
- Staged: vazio (`0` arquivos).
- Sujeira tracked preexistente: `11` arquivos modificados fora do escopo SUSC-17C5, preservados.
- Untracked preexistentes: `473` arquivos por `git ls-files --others --exclude-standard`, preservados.
- Score v6 hash antes do SUSC-17C5: `a41d0983db97ca4b71e932313200a2d6f3c3a6f7`.
- Score v7 antes do SUSC-17C5: inexistente.

## Preflight validators

- `python scripts/suscetibilidade/validate_susc_17c4_official_artifact_ingestion.py` -> PASSED.
- `python scripts/suscetibilidade/validate_susc_17c3_official_source_acquisition.py` -> PASSED.
- `python scripts/suscetibilidade/validate_susc_17c2_sar_footprint_execution.py` -> PASSED.
- `python scripts/suscetibilidade/validate_susc_17c_strong_reference_acquisition.py` -> PASSED.
- `python scripts/suscetibilidade/validate_susc_17a_reference_evidence_protocol.py` -> PASSED.
- `python scripts/suscetibilidade/validate_susc_16d_calibration_candidate.py` -> PASSED.

## Fatos de entrada observados

- 17C4 encontrou `1` geometria candidata real do Charter 758 para `REC_2022_05_24_30`.
- A geometria e `MultiPolygon`, CRS EPSG:4326, `qa_status=needs_review`, `review_only=true`, `ground_truth=false`.
- 17C4 gerou `0` patch-links candidatos.
- 17C3 registrou `0` intersecoes contra a grade SUSC Recife e `nearest_patch_distance_m=1398.4`.
- `REC_00019` aparece em registros historicos do Protocolo C, mas nao pode ser assumido como patch SUSC `recife_*`.

## Decisao fail-closed para esta sprint

Prosseguir apenas com inventario e revisao metodologica. Nao alterar `datasets/suscetibilidade/susc_features_by_patch_v1.csv`, nao criar patch oficial, nao criar patch-link oficial, nao recalcular score v6, nao criar score v7, nao executar 17B, nao executar SAR runtime e nao baixar raster pesado. Todo artefato novo deve permanecer `review_only=true`, `trainable=false`, `ground_truth=false`.
