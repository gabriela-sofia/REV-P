# v1qv — DINOv2 Attention Rollout Visualizer (review-only)

## Objetivo

Extrair pesos de atenção REAIS (CLS -> tokens de patch) do DINOv2-with-registers local e renderizar rollout (Abnar & Zuidema, 2020) como heatmap. Auxílio de interpretabilidade apenas — nunca confirma evento, nunca cria rótulo.

## Guardrails

Fail-closed: requer REVP_DINO_DRY_RUN=false, REVP_DINO_PIXEL_READ_ALLOWED=true e modelo local offline. Default é dry-run.

## Resultado

**ATTENTION_ROLLOUT_READY_REVIEW_ONLY**. Renderizados: 3.
