# v1pb - Protocol C End-to-End Orchestrator

## Objetivo

Verificar ou executar o pipeline Protocol C (v1og-v1pa) de forma ordenada
e reproduzivel. Default: check-only (verifica outputs existentes sem executar).

## Resultado

- Total de steps: 16
- Outputs presentes: 16
- Parciais: 0
- Ausentes: 0
- Modo: check-only
- Status final: ORCHESTRATION_CHECK_READY

## Modos

- `--dry-run`: lista ordem e arquivos esperados sem executar
- `--run`: executa scripts via subprocess
- `--check-only`: verifica outputs existentes (default)

## Nota

Nao usa internet. Nao baixa dados. Nao le pixels.
Nao altera ground truth ou decisoes cientificas.
