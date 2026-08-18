# Plano operacional pós-pré-unificação 20260623T213111

## Escopo deste plano

Este documento organiza a continuidade do REV-P após o commit público de pré-unificação/MV2-16 dry-run. Ele não altera código, dados, testes, schemas ou resultados científicos. Ele registra uma política operacional local para próximos blocos de programação, publicação e separação de backlog.

Não houve push, commit, stage, merge, rebase, checkout amplo, git clean, chamada externa, download, raster ou crop nesta etapa de planejamento.

## Estado atual consolidado

- Worktree principal: `C:\Users\gabriela\Documents\REV-P`.
- Branch atual REV-P: `marco/pre-unificacao-gates-mv1`.
- Upstream publicado: `origin/marco/pre-unificacao-gates-mv1`.
- HEAD atual REV-P: `1c5744b feat: adiciona gates de pré-unificação e núcleo dry-run MV2-16`.
- Pré-unificação/MV2-16 dry-run: consolidada e publicada como fato metodológico.
- Staged esperado no REV-P: `0`.
- Arquivos soltos no REV-P antes deste relatório: `338`; estes arquivos não devem ser tratados como linha ativa automaticamente.
- Worktree MV2: `C:\Users\gabriela\Documents\REV-P-mv2-01-reconciliado`.
- Branch MV2: `marco/mv2-12-reconstrucao-espectral-sentinel-baseline`.
- HEAD MV2: `61ed547 feat: consolida artefatos MV2-12 de readiness de dados`.
- Arquivos soltos no MV2: `19`.
- MV2 permanece `SPLIT_NECESSARIO` e não deve ser publicado antes de split/auditoria da sequência acumulada.
- DATA-05: fechado.
- DATA-06: bloqueado por falta de template temporal preenchido.
- DATA-07: bloqueado por `source_sensor_lineage` desconhecido.
- DATA-08: bloqueado por falta de configuração metadata-only.
- Dia 10: `BLOCKED`.
- Chamadas/downloads/rasters/crops: `0`.

## Política de branches

- Usar nomes de branch em PT-BR humano, preferencialmente com prefixo de propósito:
  - `marco/` para entregas científicas consolidadas.
  - `dados/` para ingestão, normalização e validação metadata-only.
  - `auditoria/` para leituras, inventários e diagnósticos sem alteração de código.
  - `correcao/` para ajustes pequenos de publicação, nomenclatura ou documentação.
  - `backlog/` somente para organização local de pendências, sem presunção de publicação.
- Evitar nomes em inglês como linha pública nova, exceto termos técnicos estáveis quando indispensáveis.
- Não abrir microbranches para cada arquivo solto. Cada branch deve representar uma decisão metodológica ou bloco programável coerente.
- Não misturar MV2 acumulado com programação nova do REV-P principal.
- Não mover arquivos soltos para uma branch ativa sem inventário, classificação e autorização humana.
- A branch ativa recomendada para a próxima fase deve partir do estado publicado `marco/pre-unificacao-gates-mv1`, não do worktree MV2 acumulado.

## Política de commits

- Commits devem ser humanos, em PT-BR e factuais.
- Cada commit deve declarar o que foi consolidado e o que continua bloqueado.
- Não usar mensagem otimista que promova candidato, readiness ou dry-run para validação operacional.
- Não criar commit com conteúdo misto entre backlog, dados reais, programação pesada e documentação de publicação.
- Não incluir arquivos untracked por conveniência. Antes de qualquer commit, exigir:
  - staged area revisada e explícita;
  - `git diff --cached --name-only`;
  - auditoria de arquivos privados/pesados;
  - testes ou validações focadas;
  - `git diff --check`;
  - mensagem PT-BR final revisada.
- Se o conteúdo ainda é diagnóstico, escrever como diagnóstico. Se está bloqueado, manter `BLOCKED`.

## Política de push

- GitHub deve receber apenas fato consolidado, reproduzível e auditado.
- Push real só após confirmação humana explícita.
- Antes de push:
  - confirmar branch PT-BR;
  - confirmar HEAD e mensagem humana;
  - confirmar staged final `0` após commit;
  - confirmar que arquivos soltos permanecem fora;
  - confirmar ausência de arquivos privados/pesados;
  - rodar `git diff --check`;
  - rodar dry-run quando a ação criar ou alterar referência remota.
- Não publicar worktree MV2 enquanto permanecer `SPLIT_NECESSARIO`.
- Não publicar branch com raster real, crop real, `local_only` real, credencial, token, `.env`, configuração privada, cache pesado ou saída derivada de dado privado.

## Trilhas de trabalho

### Trilha A: governança GitHub e publicação

Mantém branches PT-BR, commits humanos, relatórios de readiness e política de push. Esta trilha não altera evidência científica.

### Trilha B: desbloqueio DATA metadata-only

Foca DATA-06, DATA-07 e DATA-08 com artefatos metadata-only, sem download, raster ou crop. Esta trilha deve produzir validações fail-closed e manter Dia 10 bloqueado enquanto qualquer entrada essencial estiver ausente.

### Trilha C: split/auditoria MV2

Lê e separa a cadeia MV2 acumulada em unidades publicáveis ou descartáveis, sem publicar MV2 inteiro. O objetivo é distinguir fatos consolidados, backlog, scripts experimentais, dependências de dado real e material que deve ficar local.

### Trilha D: programação pesada unificada

Só começa depois dos gates metadata-only mínimos. Deve integrar o núcleo MV2-16 de forma incremental, com testes focados e sem executar downloads/rasters/crops por padrão.

### Trilha E: dados reais e execução privada

Fica fora da publicação GitHub até existir política explícita de paths, hashes, proveniência, tamanho, privacidade e reprodutibilidade. Qualquer execução real deve ficar em local ignorado ou cache privado.

## O que está congelado

- Commit público `1c5744b`.
- Branch pública `marco/pre-unificacao-gates-mv1`.
- Pré-unificação/MV2-16 dry-run como marco metodológico publicado.
- DATA-05 como fechado.
- Decisão de não publicar MV2 acumulado.
- Decisão de manter Dia 10 `BLOCKED`.
- Chamadas/downloads/rasters/crops em `0` para esta etapa.

## O que está bloqueado

- DATA-06: falta template temporal preenchido.
- DATA-07: `source_sensor_lineage` permanece desconhecido.
- DATA-08: falta configuração metadata-only.
- Dia 10: `BLOCKED`.
- MV2: `SPLIT_NECESSARIO`.
- Crop/SCL: bloqueado até existir autorização de trilha, política de entrada/saída e critérios metadata-only ou execução privada.
- Treino, silver e negativos: proibidos enquanto não houver positivos formais, negativos formais, ground truth operacional, anti-leakage revisado e gates de separação aprovados.

## O que é backlog

- Arquivos soltos do REV-P não são linha ativa automaticamente.
- Arquivos soltos do MV2 não são linha publicável automaticamente.
- Artefatos MV2-13, MV2-14 e MV2-15 locais entram como backlog até auditoria específica.
- Scripts de API, raster canary, live probe, private raster validation, crop extraction e execução real entram como backlog ou execução privada, não como publicação padrão.
- Relatórios locais anteriores podem servir como evidência de diagnóstico, mas não substituem nova auditoria de stage/push.

## Próximo input humano

- Preencher ou fornecer o template temporal requerido para DATA-06.
- Indicar a fonte autorizada para resolver `source_sensor_lineage` de DATA-07.
- Confirmar a configuração metadata-only aceita para DATA-08.
- Confirmar se o próximo bloco deve priorizar DATA-06, DATA-07 ou DATA-08.
- Autorizar explicitamente qualquer nova branch.
- Manter MV2 bloqueado até decisão de split.

## Próximos 5 blocos de programação

1. **Inventário operacional de backlog**: criar uma matriz local que classifique arquivos soltos por trilha, risco, publicabilidade e dependência de dado real. Saída esperada: relatório e tabela metadata-only; sem stage automático.
2. **DATA-06 metadata-only**: implementar validador de template temporal preenchido, com estado `BLOCKED` quando campos essenciais estiverem ausentes. Não inferir datas.
3. **DATA-07 lineage metadata-only**: implementar resolução e auditoria de `source_sensor_lineage`, preservando `unknown` quando a origem não for demonstrável.
4. **DATA-08 configuração metadata-only**: criar preflight de configuração sem credenciais e sem execução live; falhar fechado se configuração mínima não existir.
5. **MV2-16 núcleo unificado controlado**: integrar apenas o orquestrador seco e os contratos já consolidados, condicionado à aprovação dos gates DATA-06/07/08. Sem crop/SCL real e sem execução pesada.

## Critérios para abrir MV2-16 metadata-only

- Branch PT-BR nova a partir de `marco/pre-unificacao-gates-mv1`.
- Staged area inicial `0`.
- Escopo limitado a contratos, schemas, registries, relatórios e testes metadata-only.
- Nenhum download, raster, crop, chamada externa ou arquivo privado.
- Dia 10 permanece `BLOCKED` se DATA-06/07/08 não estiverem resolvidos.
- Testes focados devem cobrir falha fechada, ausência de configuração, ausência de lineage e ausência de template temporal.
- Relatório final deve separar claramente fato consolidado, bloqueio e backlog.

## Critérios para abrir crop/SCL

- Não abrir crop/SCL como próxima branch ativa.
- Só abrir após:
  - DATA-06 temporal preenchido e validado;
  - DATA-07 lineage resolvido com proveniência;
  - DATA-08 configuração metadata-only validada;
  - política explícita de entrada/saída para dados reais;
  - diretórios privados/ignorados definidos;
  - critérios de tamanho, hash, CRS, geometria, cena e sensor documentados;
  - autorização humana explícita para qualquer execução real.
- Se crop/SCL for apenas planejamento metadata-only, a branch deve declarar isso no nome e nos commits.
- Se houver execução real, não publicar outputs privados/pesados no GitHub.

## Critérios que continuam proibindo treino, silver e negativos

- Ausência de ground truth operacional.
- Ausência de positivos formais.
- Ausência de negativos formais.
- `unknown` não vira negativo.
- Curitiba ou outro controle administrativo não vira negativo formal sem evidência formal.
- Evidência administrativa não vira label.
- DINOv2 ou similaridade representacional não prova inundação.
- Dados contextuais ou probes não viram labels.
- Sem anti-leakage e split auditados, não há treino.
- Sem contratos de classe, provenance e revisão humana, não há silver.

## Recomendação da próxima branch ativa

Nome recomendado:

```text
dados/desbloqueio-metadata-only-data-06-08
```

Justificativa:

- Ataca os bloqueios que impedem Dia 10 antes de programação pesada.
- Mantém o escopo metadata-only.
- Evita publicar MV2 acumulado.
- Permite commits humanos pequenos por DATA-06, DATA-07 e DATA-08.
- Não depende de raster, crop, download ou credenciais.

Alternativa se o objetivo imediato for apenas triagem:

```text
auditoria/inventario-backlog-pos-pre-unificacao
```

## Próximo prompt recomendado

```text
Você está no REV-P após o plano operacional pós-pré-unificação. Não faça push, commit, stage, merge, rebase, checkout amplo ou git clean.

Objetivo: abrir uma nova branch PT-BR para o próximo bloco metadata-only, validar staged zero e implementar somente o desbloqueio DATA-06/07/08 sem download, raster, crop, chamada externa, treino, silver ou negativos.

Branch recomendada: dados/desbloqueio-metadata-only-data-06-08.

Antes de editar, leia o estado Git, confirme staged zero, inventarie os arquivos soltos relevantes e proponha o escopo exato dos arquivos permitidos.
```
