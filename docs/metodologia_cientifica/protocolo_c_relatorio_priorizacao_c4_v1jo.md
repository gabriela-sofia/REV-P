# Relatorio v1jo - priorizacao C4 e fila de evidencias

## Resultado principal

summary_decision = C3_LAYER_STABLE_C4_BLOCKED_BY_NEGATIVE_EVIDENCE

A camada C3 permanece estavel com 9 eventos oficiais CPRM em C3_EVENT_PATCH_LINKED. Nenhum evento foi promovido a C4.

## Blockers C4

- 1. FORMAL_NEGATIVES_ZERO: afeta 9 eventos; bloqueia C4=true.
- 2. POSITIVE_LABEL_GATE_NOT_OPERATIONAL: afeta 9 eventos; bloqueia C4=true.
- 3. SPLIT_LEAKAGE_NOT_READY: afeta 9 eventos; bloqueia C4=true.
- 4. S1_PARTIAL_COVERAGE: afeta 8 eventos; bloqueia C4=false.
- 5. EXTERNAL_VALIDATION_OPTIONAL: afeta 9 eventos; bloqueia C4=false.

## Negativos formais

O blocker primario e FORMAL_NEGATIVES_ZERO. A fila v1jo exige evidencia explicita de ausencia/estabilidade, vistoria sem ocorrencia, area controle oficial ou classificacao oficial sem indicio de movimento de massa. Ausencia de registro nao e negativo.

## S1

S1 completo existe para 1 anchor; 8 anchors seguem parciais. Completar S1 melhora robustez multimodal e fortalece C3, mas would_unlock_c4=false em toda a fila.

## Split e leakage

Split/leakage permanece precondicionado a labels formais. A regra de split deve proteger unidade documental, anchor, par pre/post, buffer espacial e coerencia temporal, mas so fica acionavel para treino quando positivos e negativos formais existirem.

## Usos permitidos e proibidos

Permitido: revisao cientifica C3, fila de busca metadata-only, intake de evidencia oficial e fortalecimento S1 de C3. Proibido: label operacional, negativo por pseudo-ausencia, treino supervisionado, claim cientifico de modelo e descongelamento de DINO.
