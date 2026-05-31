# v1pp — DINO Backend/Model Probe

## Objetivo

Detectar ambiente local sem baixar nada por default. REVP_DINO_ALLOW_DOWNLOAD=false (padrão).

## Fail-closed

Se modelo não existir localmente e download não autorizado, status = DINO_BACKEND_MODEL_UNAVAILABLE_FAIL_CLOSED.

## Resultado

Final status: DINO_BACKEND_MODEL_UNAVAILABLE_FAIL_CLOSED. can_execute=False.
