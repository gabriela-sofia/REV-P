# SUSC worktree audit before SUSC-16D

Branch: `marco/pre-unificacao-gates-mv1`

HEAD antes do SUSC-16D: `c345856 feat: diagnostica divergencia e calibracao review-only SUSC-16C`.

Status: auditoria pre-programacao para executar `SUSC-16D - Desenho controlado de calibracao candidata review-only` e a base minima do `Reference Evidence Protocol` (fundacao para SUSC-17A).

## Comandos executados

- `git branch --show-current`
- `git status --short`
- `git diff --name-only`
- `git diff --cached --name-only`
- `git ls-files --others --exclude-standard`
- `git log --oneline --decorate -8`
- preflight: `validate_susc_16a_observed_footprint_rescue.py`, `validate_susc_16b_score_v6_footprint_evaluation.py`, `validate_susc_16c_proxy_divergence_calibration.py`

## Resultado objetivo

- Branch atual confirmada: `marco/pre-unificacao-gates-mv1`.
- Area staged antes do 16D: vazia.
- Arquivos do SUSC-16C consumidos pelo 16D modificados localmente: nenhum.
- Nenhum arquivo de escopo SUSC-16D / Reference Evidence Protocol existia antes (pacote novo).
- Preflight 16A/16B/16C: todos PASSED antes de qualquer alteracao.
- Decisao: prosseguir com SUSC-16D sem tocar na sujeira fora do escopo.

## Entradas SUSC-16C verificadas (consumidas pelo 16D, nao modificadas)

- `datasets/suscetibilidade/susc_16c_unified_observational_analysis_table_v1.csv`
- `outputs_public/suscetibilidade/SUSC_16C_score_v6_component_decomposition.csv`
- `outputs_public/suscetibilidade/SUSC_16C_score_v6_component_decomposition_summary.json`
- `outputs_public/suscetibilidade/SUSC_16C_proxy_failure_modes.csv`
- `outputs_public/suscetibilidade/SUSC_16C_proxy_failure_modes_summary.json`
- `outputs_public/suscetibilidade/SUSC_16C_weight_sensitivity_review_only.csv`
- `outputs_public/suscetibilidade/SUSC_16C_weight_sensitivity_summary.json`
- `outputs_public/suscetibilidade/SUSC_16C_feature_direction_stability.csv`
- `outputs_public/suscetibilidade/SUSC_16C_unified_observational_analysis_summary.csv`
- `datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv` (score v6 oficial, somente leitura)

Nenhuma dessas entradas aparece em `git diff --name-only`.

## Classificacao dos arquivos sujos

### A - arquivos SUSC-16C/16B/16A que afetariam o 16D

Nenhum. As entradas do 16D (acima) nao aparecem como modificadas localmente.

### B - arquivos SUSC nao relacionados sujos

Nenhum arquivo em `datasets/suscetibilidade`, `outputs_public/suscetibilidade`, `scripts/suscetibilidade` ou `tests/suscetibilidade` aparece como modificado (`M`) antes do 16D.

### C - modificacoes tracked fora do escopo (preservadas)

Existem 11 arquivos tracked modificados fora do escopo SUSC-16D, todos do ciclo `revp_v2e*` de recuperacao de artefatos:

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

Esses arquivos sao preservados e nao fazem parte do stage seletivo do SUSC-16D.

### D - dados brutos/cache gitignored

`git ls-files --others --exclude-standard` nao lista dados brutos ignorados. Nenhum `.tif`, `.npz`, `.npy` ou `local_runs/` foi inspecionado para inclusao.

### E - untracked fora do escopo (preservados)

Existem 473 arquivos untracked de ciclos MV2, schemas `datasets/schemas/schema_mv2_*`, curadoria externa, temporal assets e relatorios `revp_*` fora do escopo SUSC-16D. Sao preservados e nao usados como fonte cientifica para o 16D.

### F - risco de conflito

Risco baixo para o SUSC-16D: nao ha sujeira local nos arquivos SUSC-16C consumidos pelo marco. Risco operacional de publicacao permanece porque o worktree segue com muitos arquivos fora do escopo, mas isso e tratado em marco de unificacao separado.

## Decisao fail-closed

Prosseguir com SUSC-16D e stagear somente os artefatos novos do pacote SUSC-16D e da base minima do Reference Evidence Protocol. Se algum arquivo fora do escopo aparecer no staged area, o stage deve ser interrompido antes do commit.
