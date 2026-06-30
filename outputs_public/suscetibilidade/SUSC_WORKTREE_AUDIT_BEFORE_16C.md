# SUSC worktree audit before SUSC-16C

Branch: `marco/pre-unificacao-gates-mv1`

HEAD antes do SUSC-16C: `ef04037 feat: avalia score v6 com footprints SUSC-16B`.

Status: auditoria pre-programacao para executar `SUSC-16C - Diagnostico profundo de divergencia e calibracao review-only do proxy`.

## Comandos executados

- `git branch --show-current`
- `git status --short`
- `git diff --name-only`
- `git diff --stat`
- `git ls-files --others --exclude-standard`
- `git log --oneline --decorate -10`
- `git diff --cached --name-only`

## Resultado objetivo

- Branch atual confirmada: `marco/pre-unificacao-gates-mv1`.
- Area staged antes do 16C: vazia.
- Arquivos SUSC-16A/SUSC-16B relevantes ao 16C modificados localmente: nenhum.
- Decisao: prosseguir com SUSC-16C sem tocar na sujeira fora do escopo.

## Classificacao dos arquivos sujos

### A - arquivos SUSC-16B/16A que afetariam o 16C

Nenhum arquivo SUSC-16A/SUSC-16B de entrada do 16C apareceu como modificado localmente.

Entradas verificadas:

- `datasets/suscetibilidade/susc_16b_footprint_event_control_dataset_v1.csv`
- `datasets/suscetibilidade/susc_16a_footprint_patch_linkage_v1.csv`
- `datasets/suscetibilidade/susc_16a_observed_footprint_catalog_v1.csv`
- `outputs_public/suscetibilidade/SUSC_16B_score_v6_footprint_evaluation_summary.json`
- `outputs_public/suscetibilidade/SUSC_16B_feature_contrast_against_footprints.csv`
- `outputs_public/suscetibilidade/SUSC_16B_footprint_evidence_quality_audit.csv`
- `outputs_public/suscetibilidade/SUSC_16B_proxy_calibration_recommendations.csv`
- validadores e testes SUSC-16A/SUSC-16B usados no preflight.

### B - arquivos SUSC nao relacionados e possivelmente relevantes

Nao foram identificados arquivos `datasets/suscetibilidade`, `outputs_public/suscetibilidade`, `scripts/suscetibilidade` ou `tests/suscetibilidade` sujos antes do 16C fora das entradas verificadas.

### C - arquivos antigos de docs/revp_v2e/outputs fora do escopo

Existiam modificacoes locais tracked fora do escopo SUSC-16C:

- `docs/metodologia_cientifica/revp_v2es_readonly_sibling_artifact_inspector.md`
- `docs/metodologia_cientifica/revp_v2et_recovery_candidate_validator.md`
- `docs/metodologia_cientifica/revp_v2eu_controlled_artifact_recovery_copier.md`
- `outputs_public/execution_reports/revp_controlled_artifact_recovery_report_v2eu.md`
- `outputs_public/execution_reports/revp_readonly_sibling_artifact_inspection_report_v2es.md`
- `outputs_public/execution_reports/revp_recovery_candidate_validation_report_v2et.md`
- `outputs_public/tables/revp_controlled_artifact_recovery_manifest_v2eu.csv`
- `outputs_public/tables/revp_readonly_sibling_artifact_inspection_v2es.csv`
- `outputs_public/tables/revp_recovered_base_count_summary_v2ev.csv`
- `outputs_public/tables/revp_recovered_base_verification_v2ev.csv`
- `outputs_public/tables/revp_recovery_candidate_validation_v2et.csv`

Esses arquivos foram preservados e nao fazem parte do stage seletivo do SUSC-16C.

### D - dados brutos/cache gitignored

`git ls-files --others --exclude-standard` nao lista arquivos ignorados. Nenhum dado bruto/cache gitignored foi stageado ou inspecionado para inclusao.

### E - untracked possivelmente uteis

Existiam muitos arquivos untracked de ciclos MV2, curadoria externa, temporal assets, schemas e testes fora do escopo SUSC-16C, incluindo diretorios como:

- `outputs_public/mv2_*`
- `scripts/mv2_*`
- `tests/test_mv2_*`
- `outputs_public/execution_reports/revp_*`
- `outputs_public/tables/revp_*`
- `outputs_public/metrics/revp_*`
- `scripts/ground_truth/revp_*`
- `scripts/curadoria_externa/`

Eles foram preservados e nao foram usados como fonte cientifica para o 16C.

### F - risco de conflito

Risco baixo para o SUSC-16C: nao ha sujeira local nos arquivos SUSC-16A/SUSC-16B consumidos pelo marco. Risco operacional permanece para publicacao futura porque o worktree segue com muitos arquivos fora do escopo.

## Decisao fail-closed

Prosseguir com SUSC-16C e stagear somente os artefatos listados no marco. Se algum arquivo fora do escopo aparecer no staged area, o stage deve ser interrompido antes do commit.
