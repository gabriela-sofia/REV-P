# v1pj — DINO Similarity Neighbor Graph

## Objetivo

Calcular vizinhos top-k por similaridade de cosseno e matriz long-form sobre embeddings válidos review-only. Com menos de 2 embeddings válidos, saídas vazias com header (fail-closed).

## Interpretação

Similaridade é sinal exploratório de coerência visual/semântica entre patches. `can_infer_same_event`, `can_create_label` e `can_train_model` são sempre false. Vizinhança não implica mesmo evento observado.

## Resultado

Embeddings válidos: 0. Pares de vizinhos: 0.
