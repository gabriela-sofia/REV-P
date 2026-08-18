# SUSC worktree audit before SUSC-17A

Branch: `marco/pre-unificacao-gates-mv1`

HEAD antes do SUSC-17A: `3e870e0 feat: desenha calibracao candidata review-only SUSC-16D`.

Status: auditoria pre-programacao para executar `SUSC-17A Reference Evidence Protocol`, formalizando o stub criado no SUSC-16D em protocolo de evidencia observacional validado e auditavel.

## Comandos executados

- `git branch --show-current`
- `git status --short`
- `git diff --name-only`
- `git diff --cached --name-only`
- `git ls-files --others --exclude-standard`
- `git log --oneline -1`
- inspecao de fontes 16A/16B/16C/16D e do stub 17A (schema, registry stub, class policy)

## Resultado objetivo

- Branch atual confirmada: `marco/pre-unificacao-gates-mv1`.
- Area staged antes do 17A: vazia.
- Arquivos consumidos pelo 17A modificados localmente: nenhum (entradas 16A/16B/16C/16D limpas).
- Decisao: prosseguir com SUSC-17A sem tocar na sujeira fora do escopo.

## Entradas consumidas pelo 17A (somente leitura, exceto o schema stub que sera formalizado)

- `datasets/suscetibilidade/susc_16a_observed_footprint_catalog_v1.csv` (catalogo de footprints; 12 elegiveis)
- `datasets/suscetibilidade/susc_16a_footprint_patch_linkage_v1.csv` (linkage_id por footprint-patch)
- `datasets/suscetibilidade/susc_16c_unified_observational_analysis_table_v1.csv` (65 links evento)
- `outputs_public/suscetibilidade/SUSC_16B_footprint_evidence_quality_audit.csv` (tiers de qualidade 16B)
- `outputs_public/suscetibilidade/susc_17a_reference_evidence_protocol_class_policy.json` (stub 16D)
- `outputs_public/suscetibilidade/susc_17a_reference_evidence_protocol_registry_stub.csv` (stub 16D, 0 registros)
- `schemas/suscetibilidade/susc_17a_reference_evidence_protocol_schema_v1.json` (stub 16D; sera formalizado no 17A)

Nenhuma dessas entradas aparece em `git diff --name-only`.

## Classificacao dos arquivos sujos

### A - arquivos que afetariam o 17A

Nenhum. As entradas do 17A nao aparecem como modificadas localmente.

### B - arquivos SUSC nao relacionados sujos

Nenhum arquivo em `datasets/suscetibilidade`, `outputs_public/suscetibilidade`, `scripts/suscetibilidade` ou `tests/suscetibilidade` aparece como modificado (`M`) antes do 17A.

### C - modificacoes tracked fora do escopo (preservadas)

11 arquivos tracked modificados fora do escopo SUSC-17A, todos do ciclo `revp_v2e*`:

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

Preservados; nao fazem parte do stage seletivo do SUSC-17A.

### D - dados brutos/cache gitignored

Nenhum dado bruto/cache gitignored foi inspecionado para inclusao.

### E - untracked fora do escopo (preservados)

473 arquivos untracked de ciclos MV2, schemas `schema_mv2_*`, temporal assets e relatorios `revp_*` fora do escopo. Preservados.

### F - risco de conflito

Risco baixo: nao ha sujeira local nas entradas do 17A. O schema stub `susc_17a_reference_evidence_protocol_schema_v1.json` sera formalizado dentro do escopo 17A (deliverable explicito).

## Nota de reprodutibilidade

A suite de testes 16C reexecuta o pipeline 16C e pode regenerar outputs tracked 16A/16B/16C por ordenacao nao determinista. No 17A o preflight usa apenas validators especificos (sem reexecutar pipelines), e qualquer arquivo fora do escopo que for sujado por testes sera restaurado com `git checkout --` antes do commit.

## Decisao fail-closed

Prosseguir com SUSC-17A e stagear somente os artefatos novos do pacote SUSC-17A (incluindo a formalizacao do schema 17A) e a auditoria 17A. Se algum arquivo fora do escopo aparecer no staged area, o stage deve ser interrompido antes do commit.
