# v2bx — Protocolo formal de GT em modo dry-run e auditoria de prontidão anti-leakage

Versão: `v2bx`
Modo: auditoria metodológica autônoma estruturada, offline-determinística. Não
cria label, não preenche `gt_patch_flood_observed`, não cria negativo formal, não
libera treino.

## 1. Por que o v2bx existe

A cadeia v2bp→v2bw provou que **não existe footprint oficial poligonal revisado**
para o evento `REC_2022_05_24_30`. Continuar procurando o mesmo footprint local
já não agrega. O v2bx muda a pergunta: *o que aconteceria se o projeto adotasse
explicitamente a geometria QA-derived dos pontos da Defesa Civil como referência
operacional provisória?*

Ele monta um **protocolo formal em modo dry-run**, aplica os critérios em preview,
audita os resultados e bloqueia tudo que não pode avançar. O valor da etapa é
tornar visível e auditável a fronteira entre **candidato dry-run** e **label
formal de treino** — sem nunca falsificar ground truth.

## 2. Diferença entre dry-run candidate e label real

Um candidato dry-run é uma hipótese: "este patch *seria* positivo/negativo *se*
um protocolo formal fosse aprovado". Ele carrega a coluna
`would_label_if_protocol_approved`, mas:

- `label_created=false` (sempre);
- `gt_patch_flood_observed=NA` (sempre);
- `allowed_for_training=false` (sempre).

Um label real exigiria uma geometria de referência aprovada e um protocolo
aprovado — nenhum dos dois existe. O dry-run simula; não rotula.

## 3. Duas trilhas

- **Trilha A (oficial estrita)**: `BLOCKED_OFFICIAL_FOOTPRINT_NOT_FOUND` — exige
  footprint oficial revisado, que não existe.
- **Trilha B (referência QA-derived declarada)**:
  `PROTOCOL_DRY_RUN_ONLY_QA_DERIVED_REFERENCE` — usa a geometria derivada dos
  pontos Defesa Civil como referência provisória, **apenas em dry-run**, com a
  limitação explícita.

## 4. Por que REC_00276 é dry-run positive candidate

REC_00276 é QA-robusto (multi-método, incluindo geometrias tight: buffer_union_250
e cluster_envelope; razão de interseção 0.88, 4 alternativas intersectando). Sob a
Trilha B atende ao critério positivo dry-run, então
`would_be_positive_if_protocol_approved=true`. Mesmo assim **não é label**:
`gt_patch_flood_observed=NA`, `allowed_for_training=false`,
`blocked_reason=PROTOCOL_DRY_RUN_ONLY_OFFICIAL_FOOTPRINT_NOT_FOUND`. Há ainda um
cross-check defensivo contra a matriz de sensibilidade do v2bu: um positivo do
dossiê é rebaixado se a matriz não o confirmar como robusto (nunca promove nada).

## 5. Por que REC_00299 fica held

REC_00299 só intersecta métodos permissivos (convex_hull + buffer maior), sem
consenso tight. Permanece method-dependent e **não** é promovido a candidato
positivo (`METHOD_DEPENDENT_NOT_ROBUST_HELD`). No registro de candidatos aparece
como `dry_run_role=METHOD_DEPENDENT_HELD` / `DRY_RUN_HELD`.

## 6. Por que os negativos comparáveis são apenas dry-run

Os 14 patches comparáveis QA-only (mesma região, boundary recuperada disponível,
QA-noncompatible, dentro da banda de distância ≤8km) seriam negativos sob a
Trilha B, mas `formal_negative_label_created=false`
(`PROTOCOL_DRY_RUN_ONLY_NO_FORMAL_NEGATIVE_APPROVAL`).

## 7. Por que noncompatible não é negativo automaticamente

Não-compatibilidade e ausência **não são** negativos por si. Patches longe demais
(>8km, 21 deles) são excluídos com `DISTANCE_TOO_FAR_FOR_COMPARABLE_NEGATIVE`.
Patch sem boundary tentando virar negativo é bloqueado
(`MISSING_BOUNDARY_CANNOT_BE_NEGATIVE`). O conflict audit detecta ainda: patch
como positivo e negativo ao mesmo tempo, method-dependent virando negativo, e
qualquer `gt_patch_flood_observed` não-NA vazando (deve ser zero).

## 8. Como o anti-leakage split foi desenhado

Os patches são agrupados por **evento, região, source family, spatial block e o
`split_group`/tile existente** — nunca por random split simples. Co-membros de um
grupo devem permanecer no mesmo split (treino *ou* teste, nunca divididos). Com
apenas **1 positivo dry-run**, o treino supervisionado não é estatisticamente
viável: o plano fica `SPLIT_BLOCKED_TOO_FEW_POSITIVES` /
`SPLIT_PLAN_QA_ONLY_NOT_TRAINABLE`, com `recommended_split_role=HELD_NOT_ASSIGNED`.
O group audit marca grupos que misturam positivo e negativo como risco de leakage
e manda mantê-los juntos.

## 9. Por que treino segue bloqueado

Sem labels formais, com pouquíssimos positivos aprovados e referência QA-only,
tanto `can_train_supervised_model` quanto `can_train_dry_run_model` são `false`.
`label_creation_allowed=false`, `allowed_for_training_count=0`.

## 10. O que falta para liberar ground truth formal

1. Um footprint oficial revisado **ou** uma decisão explícita e documentada de
   adotar a geometria QA-derived como referência formal (com a limitação).
2. Um protocolo positivo formal aprovado.
3. Um protocolo negativo formal aprovado.
4. Positivos aprovados suficientes para um split seguro contra leakage.

## Outputs

`local_runs/ground_truth/v2bx/`:

- `formal_gt_protocol_dry_run_summary_v2bx.json`
- `positive_protocol_dry_run_v2bx.csv`
- `negative_protocol_dry_run_v2bx.csv`
- `dry_run_label_candidate_registry_v2bx.csv`
- `dry_run_label_conflict_audit_v2bx.csv`
- `anti_leakage_split_plan_v2bx.csv`
- `anti_leakage_group_audit_v2bx.csv`
- `label_readiness_gate_v2bx.json`
- `training_readiness_gate_v2bx.json`
- `protocol_dry_run_guardrails_v2bx.json`
- `protocol_dry_run_report_v2bx.md`

## Próxima etapa recomendada

Levar o protocolo dry-run a uma decisão metodológica: ou formalizar a adoção da
geometria QA-derived como referência declarada (Trilha B, com limitação escrita) e
desenhar os protocolos positivo/negativo correspondentes, ou adquirir o footprint
oficial revisado (Trilha A). Enquanto qualquer das duas não for aprovada, o
projeto pode simular o protocolo, mas ainda não pode treinar.
