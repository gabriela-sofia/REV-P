# v1ry — Integration Test Runner Wrapper

## Objetivo

Gera plano de testes; por padrão (REVP_RUN_INTEGRATION_TESTS != true) não executa pytest. Quando env=true, executa cada suite com timeout individual.

## Resultado

Suites planejadas: 9. Executadas: 0.

## Como executar

`$env:REVP_RUN_INTEGRATION_TESTS='true'; python revp_v1ry_integration_test_runner_wrapper.py`
