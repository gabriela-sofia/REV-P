# v1qj — Controlled Real Smoke Embedding Executor

## Objetivo

Executar embeddings DINOv2 768D reais somente quando todos os gates passam (modelo local offline, assets prontos, dry-run=false, pixel read autorizado). Default: dry-run.

## Guardrails

Não baixa modelo. Não treina. Não cria rótulo/target/ground truth. Vetores são descritores visuais review-only.

## Status

**DINO_SMOKE_EMBEDDINGS_DRY_RUN_ONLY**. Vetores válidos 768D: 0.
