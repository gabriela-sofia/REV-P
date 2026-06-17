# REV-P v2ci - inventario TP2-ready de evidencia observacional candidata

## Objetivo

O marco `v2ci` cria um inventario auditavel de evidencias observacionais
candidatas que podem, futuramente, apoiar TP2. Ele nao fecha TP2, nao promove
evidencia candidata a ground truth operacional, nao cria labels, nao cria
negativos formais e nao autoriza treino supervisionado.

## Escopo metodologico

O REV-P permanece em modo review-only. O projeto nao possui ground truth
operacional patch-level, labels binarios, negativos formais ou classificador
supervisionado operacional.

Esta etapa separa explicitamente:

- evidencia textual;
- evidencia visual;
- geometria candidata;
- geometria observada validada;
- ground truth operacional.

Se houver duvida entre afirmar e bloquear, o inventario bloqueia.

## Entradas locais

O script le artefatos locais quando existem em `datasets/`, `docs/`,
`outputs_public/` e `manifests/`, com foco em Protocolo C, TP0-TP4, Charter,
patches, Curitiba, Recife e Petropolis. A etapa nao usa internet, nao baixa dados
externos e nao tenta completar campos ausentes por inferencia.

## Regras de status

- `TP2_BLOCKED`: ausencia de geometria observada validada, CRS, proveniencia ou
  hash suficiente.
- `TP2_CANDIDATE_ONLY`: existe evidencia candidata, mas ainda depende de revisao,
  georreferenciamento ou digitalizacao.
- `TP2_READY_FOR_REPLAY`: permitido apenas com geometria observada vetorial, CRS
  conhecido, proveniencia e hash.
- `TP3_READY`: permitido apenas com patch boundary validado, geometria observada
  validada e teste de intersecao possivel.

Qualquer status que indique promocao direta para ground truth operacional e proibido.

## Outputs

- `outputs_public/tables/revp_tp2_candidate_inventory_v2ci.csv`
- `outputs_public/tables/revp_tp2_patch_pair_candidates_v2ci.csv`
- `outputs_public/logs_summary/revp_tp2_candidate_guardrail_rollup_v2ci.csv`
- `outputs_public/execution_reports/revp_tp2_candidate_inventory_report_v2ci.md`
- `outputs_public/execution_reports/revp_tp2_candidate_commit_checklist_v2ci.md`

## Guardrails

O rollup de travas registra:

- `formal_labels_available=ABSENT`
- `formal_negatives_available=ABSENT`
- `training_ready=BLOCKED`
- `ground_truth_operational=ABSENT`
- `supervised_model_allowed=false`
- `prediction_claim_allowed=false`
- `intersection_claim_allowed=false`, exceto se houver geometria vetorial validada
  e teste explicito.

## Execucao

```powershell
python scripts\multimodal\revp_v2ci_tp2_candidate_inventory.py --force
```

## Validacao

```powershell
python -m pytest tests\test_revp_v2ci_tp2_candidate_inventory.py -q
```
