# v1qk — Import Smoke Embeddings to Representation Layer

## Objetivo

Importar embeddings smoke (v1qj) para um feature store de representação review-only consolidado, validando 768D e deduplicando por (patch_id, path_hash, model_path_hash).

## Fronteira

Não mistura com rótulos. Não chama C3/C4. cluster_is_label=false.

## Status

**DINO_REPRESENTATION_WITH_SMOKE_EMPTY_FAIL_CLOSED**. Vetores válidos únicos: 0.
