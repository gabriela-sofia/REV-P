# v1qg — Local DINOv2 Model Offline Audit

## Objetivo

Auditar um modelo DINOv2 (with registers) local de forma offline, sem baixar nada e sem rodar inferência. Apenas presença de config/pesos/processor e leitura de hidden_size.

## Gates

Exige REVP_DINO_MODEL_PATH existente, REVP_DINO_ALLOW_DOWNLOAD=false e HF_HUB_OFFLINE=1. Qualquer falha resulta em status fail-closed.

## Status

**LOCAL_DINO_MODEL_MISSING_FAIL_CLOSED**. expected_dim=768.
