# v1oz — DINO Review-Only Representation Queue

## Objetivo

Gerar fila para representação visual/embedding DINO review-only. Inclui patches/eventos com valor contextual mas sem label. Fila pode ser vazia se não houver linkage suficiente.

## Resultado

Entradas na fila DINO: 2. Bloqueados/não incluídos: 4. Status: DINO_QUEUE_POPULATED_REVIEW_ONLY.

## Papel do DINO

DINO é usado exclusivamente para representação estrutural visual. Nenhum target é criado. Nenhum label é derivado. dino_allowed_use=REVIEW_ONLY_REPRESENTATION.

## Guardrails

dino_can_create_label=false, dino_can_train_model=false, dino_target_field_created=false em todos os registros.
