# v2az - Turning Point Replay Orchestrator

`dry_run` valida intake, gera feeds e plano, mas nao executa subprocessos. `replay` so executa
quando ha precondicoes reais e usa workspace isolado. Feeds contem somente geometrias validas.
Ausencia vira blocker. A cadeia controlada e v2ax -> v2aw -> v2av -> v2au -> v2ay. Nada cria label.
