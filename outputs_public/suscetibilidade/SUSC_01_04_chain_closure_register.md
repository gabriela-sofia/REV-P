# Registro Formal de Fechamento — Cadeia SUSC-01 → SUSC-04

> O SUSC-05 não calcula score, não treina modelo e não cria ground truth. Este registro apenas formaliza, dentro do repositório, o que a cadeia SUSC-01→04 produziu e consolidou localmente.

## 1. Objetivo da cadeia SUSC

Construir, de forma auditável e review-only, uma matriz multimodal de atributos associados à suscetibilidade urbana a enchentes por patch, com proveniência científica rastreável — sem criar ground truth de ocorrência e sem desbloquear treinamento supervisionado.

## 2. Estado anterior do projeto

A matriz operacional existia apenas em `PROJETO/data/dataset_final.csv` (300 patches × 211 colunas), fora do repositório público, sem schema formal, sem manifesto de proveniência e sem separação explícita entre feature física, score heurístico e label heurístico.

## 3. O que SUSC-01/02 formalizou

- Schema formal de 72 features: `schemas/suscetibilidade/susc_features_schema_v1.json`.
- Manifesto de proveniência: `manifests/suscetibilidade/susc_features_provenance_manifest_v1.csv`.
- Validador de schema e relatórios SUSC-01 (relatório de schema) e SUSC-02 (plano de migração).
- Governança gravada por feature: `allowed_for_training=false`, `can_be_used_as_ground_truth=false`, `review_only=true`.

## 4. O que SUSC-03 operacionalizou

- Migração auditável de 300×72 para `datasets/suscetibilidade/susc_features_by_patch_v1.csv` (somente colunas catalogadas; tokens originais preservados; SHA256 validado).
- Manifesto de artefato com SHA256 e governança.
- Diagnóstico exploratório (perfil por grupo, missingness=0, sumário numérico, sumário por região, screen de correlação — 0 alertas ≥0.90), sem treinar modelo.

## 5. O que SUSC-04 auditou

- Vocabulário científico de direção esperada: `schemas/suscetibilidade/susc_feature_scientific_direction_v1.json`.
- Scanner de proveniência read-only (5.534 arquivos), audit CSV e scan JSON.
- Matriz de decisão para score v6 e relatório de proveniência.
- Resultado: 20 `verified_script_and_source`, 10 `verified_script_only`, 3 `unresolved` (nomes solicitados ausentes da matriz). Todas `requires_manual_review=true`.

## 6. Commits locais e hashes

| Stage | Hash (curto) | Mensagem |
|-------|--------------|----------|
| SUSC-01/02 | `4b49eb5` | feat: formaliza schema de suscetibilidade SUSC-01-02 |
| SUSC-03 | `7986da4` | feat: migra matriz auditavel de suscetibilidade SUSC-03 |
| SUSC-04 | `3c49608` | feat: audita proveniencia cientifica das features SUSC-04 |

Hashes completos em `manifests/suscetibilidade/susc_chain_commits_manifest_v1.csv`.

## 7. Artefatos principais

- Schema: `schemas/suscetibilidade/susc_features_schema_v1.json`
- Direção científica: `schemas/suscetibilidade/susc_feature_scientific_direction_v1.json`
- Matriz: `datasets/suscetibilidade/susc_features_by_patch_v1.csv`
- Manifestos: `susc_features_provenance_manifest_v1.csv`, `susc_features_by_patch_v1_artifact_manifest.json`, `susc_feature_provenance_audit_v1.csv`, `susc_feature_provenance_scan_v1.json`
- Decisão v6: `outputs_public/suscetibilidade/SUSC_04_score_v6_feature_decision_matrix.csv`
- Relatórios: SUSC_01/02/03/04 em `outputs_public/suscetibilidade/`

## 8. Validações executadas

- `validate_susc_features_schema_v1.py` → PASSED (72 features)
- `validate_susc_features_by_patch_v1.py` → PASSED (300×72, SHA256 confere)
- `validate_susc_04_provenance_audit.py` → PASSED (unresolved∉v6, ambiguous∌high)
- `tests/suscetibilidade/test_susc_03_migration.py` + `test_susc_04_provenance_audit.py` → 22 passed

## 9. Estado da governança

Em todos os artefatos e features: `allowed_for_training=false`, `can_be_used_as_ground_truth=false`, `review_only=true`. Política anti-leakage do REV-P herdada e mantida.

## 10. Limites científicos preservados

- Suscetibilidade ≠ ocorrência confirmada de enchente.
- Score v5 e labels heurísticos são heurísticos, não verdade.
- SAR e DINO são evidência/representação complementar, não detector nem ground truth.
- Atribuição de fonte por feature ainda exige confirmação manual (todas `requires_manual_review=true`).

## 11. Por que ainda não é ground truth

Os labels (`label_evento_enchente_potencial_v5_core_regional_p75`, `label_confidence_v5`) derivam de scores compostos por limiar regional (p75), não de eventos observados. Nenhuma coluna tem `can_be_used_as_ground_truth=true`; os validadores falham-fechado caso isso mude.

## 12. Por que ainda não desbloqueia treino

Sem ground truth de ocorrência, treinar supervisionado introduziria vazamento (heurística → "verdade" circular). Todas as features mantêm `allowed_for_training=false`; a migração rejeita (fail-closed) qualquer coluna marcada para treino.

## 13. Próximo marco: SUSC-05

Feature cards científicos e registro formal da cadeia (este documento + cards por métrica), preparando a base metodológica para SUSC-06 (baseline GAM/SPGAM interpretável). Nenhum score, modelo ou ground truth é criado em SUSC-05.

## 14. Nota explícita

**Sem push.** Os três commits permanecem locais (`[ahead 3]` de `origin/marco/pre-unificacao-gates-mv1`). Nenhum envio ao remoto foi realizado.
