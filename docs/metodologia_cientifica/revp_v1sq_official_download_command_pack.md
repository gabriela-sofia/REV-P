# v1sq — Official Download Command Pack

## Objetivo

Empacotar a cadeia v1sg-v1sp de aquisicao oficial num command pack reproduzivel e fail-closed: CSV documentado, doc e um PowerShell seguro.

## Uso

Rode `configs/revp_official_download_commands_v1sq.ps1`. Por padrao a cadeia roda em modo queue-only (sem rede). Para baixar de verdade, descomente o bloco manual (Passo 2) e reexecute.

## Limites e seguranca

max_files=20, max_bytes_per_file=262144000, connect_timeout=15s, read_timeout=60s, retries=2. Downloads so para dominios .gov.br da allowlist; redirects para fora da allowlist sao bloqueados.

## Limitacoes

Os brutos vao para `data/external_raw/` (git-ignored) e nao devem ser commitados. Todos os outputs sao review-only: nao criam rotulos, targets, ground truth operacional nem negativos formais.
