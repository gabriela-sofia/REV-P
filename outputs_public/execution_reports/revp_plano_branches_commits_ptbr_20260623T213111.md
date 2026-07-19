# Plano de branches e commits PT-BR 20260623T213111

## Resumo executivo

Esta e uma proposta local, sem stage, commit, push, merge, rebase ou limpeza. A estrategia recomendada e fatiar a publicacao por marcos verificaveis e manter mensagens publicas em portugues tecnico.

## Mensagens PT-BR recomendadas

REV-P atual:

```text
feat: adiciona gates de pre-unificacao e nucleo dry-run MV2-16
```

Body recomendado, se houver amend autorizado:

```text
- Fecha DATA-05 como intake sem desbloqueio de dados.
- Adiciona DATA-06/07/08 em modo fail-closed.
- Adiciona nucleo MV2-16 dry-run.
- Mantem Dia 10 bloqueado.
- Nao executa chamadas, downloads, rasters ou crops.
```

MV2 atual:

```text
feat: consolida artefatos MV2-12 de readiness de dados
```

Body recomendado, se houver amend autorizado:

```text
- Consolida 16 artefatos MV2-12 Data Readiness.
- Mantem artefatos espectrais/STAC/crop fora do commit.
- Preserva Dia 10 bloqueado.
- Nao executa chamadas, downloads, rasters ou crops.
```

## Branches recomendadas

### `marco/pre-unificacao-gates-mv1`

- Escopo: Publicar d6a50ee apos decisao sobre amend PT-BR e criacao da branch remota.
- O que pode entrar: pre-unificacao, DATA-06/07/08, MV2-16 dry-run, schemas/testes/reports leves
- O que nao pode entrar: DATA-01..05, MV2-13/14/15, local_only, raster/crop, segredo
- Criterio de saida: commit com mensagem PT-BR, staged 0, branch remota criada conscientemente
- Quando pode ir para GitHub: apos amend PT-BR ou aceite explicito da mensagem atual
### `marco/mv2-12-readiness-dados`

- Escopo: Publicar o escopo MV2-12 Data Readiness sem arrastar MV2-01..MV2-11 sem auditoria.
- O que pode entrar: os 16 arquivos de 1d0d795 e relatorio de readiness
- O que nao pode entrar: reconstrucao espectral, STAC, crop, MV2-13/14/15
- Criterio de saida: branch fatiada/auditada ou decisao explicita de publicar historico acumulado
- Quando pode ir para GitHub: apos revisao do historico e estrategia de split
### `data/desbloqueio-sentinel-metadata-only`

- Escopo: Consolidar DATA-01..05 e API metadata-only sem raster/crop.
- O que pode entrar: planos, configs example, provas metadata-only, relatorios fail-closed
- O que nao pode entrar: api_config.local.json real, token, download raster, crop
- Criterio de saida: config local criada fora do Git e execucao metadata-only documentada
- Quando pode ir para GitHub: apos revisao humana de seguranca/config
### `data/lineage-temporal-sensor`

- Escopo: Resolver janela temporal e lineage source/sensor rastreavel.
- O que pode entrar: templates preenchidos, consenso de lineage, bloqueios documentados
- O que nao pode entrar: inferencias sem evidencia, labels, negativos, Dia 10 desbloqueado sem prova
- Criterio de saida: proveniencia rastreavel e Dia 10 ainda fail-closed ate evidencias reais
- Quando pode ir para GitHub: apos validacao dos campos humanos
### `marco/mv2-16-nucleo-unificado-dry-run`

- Escopo: Evoluir o nucleo unificado dry-run sem executar dados reais.
- O que pode entrar: core, policies, testes, summaries dry-run
- O que nao pode entrar: execucao operacional, raster/crop, claims de treino
- Criterio de saida: dry-run reproduzivel e sem efeitos colaterais
- Quando pode ir para GitHub: apos separacao do escopo pre-unificacao


## Decisoes por bloco

| Bloco | Decisao | Justificativa |
|---|---|---|
| REV-P `d6a50ee` | `PUSH_APOS_AMEND_PTBR` | Conteudo leve e focado, mas mensagem atual esta em ingles; branch remota ainda nao existe. Pode virar `PUSH_AGORA` se a mensagem atual for aceita explicitamente. |
| MV2 `1d0d795` | `SPLIT_NECESSARIO` | O commit isolado esta focado em 16 arquivos MV2-12, mas a branch atual levaria junto MV2-01..MV2-11. |
| Sequencia MV2-01..MV2-12 | `PUSH_APOS_REVISAO` | Sao 12 commits e 293 arquivos contra `origin/main`; ha paths historicos com `private`/`raster` que precisam decisao publica. |
| Untracked REV-P | `NAO_PUSHAR` | Inventario local, sem staging; exige revisao/split por categorias. |
| Untracked MV2 | `NAO_PUSHAR` | Inclui reconstrucao espectral, STAC e crop; fica para commit futuro revisado. |
| DATA antigos | `PUSH_APOS_REVISAO` | DATA-01..05 precisam separacao de escopo e verificacao metadata-only. |
| MV2-13/14/15 | `SPLIT_NECESSARIO` | Sao blocos futuros e nao entram no push atual. |
| Outputs publicos leves | `PUSH_APOS_REVISAO` | Podem ser publicaveis se PT-BR/escopo/guardrails passarem. |
| Arquivos com private/local_only/raster no nome | `NAO_PUSHAR` ou `PUSH_APOS_REVISAO` | Nunca publicar se forem reais/pesados/sensiveis; se forem apenas registros leves, revisar nome e conteudo antes. |

## Proximo comando recomendado

```powershell
git status --short; git diff --check
```

Depois de revisao humana, escolher um unico proximo ato: amend PT-BR do REV-P, criar branch fatiada MV2-12, ou auditar MV2-01..MV2-11 antes de qualquer push.
