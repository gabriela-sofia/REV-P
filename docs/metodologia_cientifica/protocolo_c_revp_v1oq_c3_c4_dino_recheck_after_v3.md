# Protocolo C v1oq — C3/C4/DINO Recheck After Scene Date v3

## Objetivo

v1oq revalida C3+, C4 e fila DINO após o resolver de scene_date v3 (v1oo/v1op).

## Regras

- DINO: sempre REVIEW_ONLY_REPRESENTATION. Nunca cria label ou target.
- Fila DINO: só patches com scene_date confirmada E ainda review-only.
- C3+: somente com scene_date confirmada + regra temporal satisfeita + formal negative.
- C4: fechado se formal_negative_count == 0.
- can_create_operational_label e can_train_model sempre false.

## Resultado

C3+ review candidates: 0. DINO queue: 0. C4 open: false. DINO status: REVIEW_ONLY_REPRESENTATION.
