# Diagnostico de publicacao GitHub 20260623T213111

## Resumo executivo

A publicacao direta continua bloqueada. O REV-P tem um commit local focado (`d6a50ee`) que pode ser publicado depois de decisao sobre mensagem PT-BR e criacao consciente da branch remota. O MV2 nao deve ser pushado como esta, porque a branch atual publica a sequencia acumulada MV2-01..MV2-12 contra `origin/main`, nao apenas `1d0d795`.

Dia 10 permanece `BLOCKED`. Nesta auditoria nao houve chamadas externas, downloads, rasters ou crops. Nao houve stage, commit, push, merge, rebase, checkout amplo ou limpeza.

## Estado atual do REV-P

- Branch: `analysis/temporal-asset-readiness-mv1`
- HEAD: `d6a50ee feat: add pre-unification gates and MV2-16 dry-run core`
- Branch remota correspondente: nao encontrada
- Commits locais fora de remotos: 1
- Arquivos no diff `origin/main..HEAD`: 56
- Arquivos soltos diagnosticados antes deste relatorio: 468

### Commits REV-P a publicar contra `origin/main`

- d6a50ee feat: add pre-unification gates and MV2-16 dry-run core

### O que ja esta commitado no REV-P

O commit `d6a50ee` contem 56 arquivos: `.gitignore`, exemplo de config, scripts de pre-unificacao/DATA-06/07/08/MV2-16, testes focados, schemas e outputs publicos leves `20260623T213111`.

### O que ainda esta solto no REV-P

| Categoria | Quantidade |
|---|---:|
| `CURADORIA_PUBLICA` | 37 |
| `DATA_01_05` | 151 |
| `DUPLICADO_REGENERADO` | 17 |
| `LEGADO_MV1` | 108 |
| `MV2_13` | 37 |
| `MV2_14` | 34 |
| `MV2_15` | 26 |
| `NAO_PUBLICAR_LOCAL` | 2 |
| `NAO_PUBLICAR_PESADO` | 1 |
| `NEEDS_REVIEW` | 5 |
| `PUBLICAVEL_APOS_REVISAO_PTBR` | 34 |
| `RELATORIO_PUBLICO_LEVE` | 16 |

## Estado atual do MV2

- Branch: `marco/mv2-12-reconstrucao-espectral-sentinel-baseline`
- HEAD: `1d0d795 feat: consolidate MV2-12 data readiness artifacts`
- Branch remota correspondente: nao encontrada
- Commits locais fora de remotos: 12
- Arquivos no diff `origin/main..HEAD`: 293
- Arquivos soltos diagnosticados antes deste relatorio: 29

### Commits MV2 a publicar contra `origin/main`

- 1d0d795 feat: consolidate MV2-12 data readiness artifacts
- 817ed17 MV2-11: rebalanceamento representacional regional label-free
- 9bd8a88 MV2-10: executor DINOv2 offline e hardening de confounder visual
- 1bf52bc MV2-09: expansao representacional com assets visuais canonicos
- b00dde2 MV2-08: pericia raster e canonizacao privada de assets
- 10265e0 MV2-07: recuperacao de assets e baseline espectral fail-closed
- 0ae770f MV2-06: adjudicacao IA e consenso de evidencias fail-closed
- b54c6a1 MV2-05: readiness review-only para negativos, silver e splits
- b6bb383 MV2-04: auditoria representacional label-free dos embeddings
- 31863f3 MV2-03: reconstrucao de lineage asset-scene Sentinel fail-closed
- fe649b2 MV2-02: manifesto temporal Sentinel por asset fail-closed
- c507f7a MV2-01: contrato observacional patch-asset-evento fail-closed

### O que ja esta commitado no MV2

O commit `1d0d795` contem exatamente 16 arquivos MV2-12 Data Readiness. A branch inteira, porem, contem tambem MV2-01..MV2-11 contra `origin/main`.

### O que ainda esta solto no MV2

| Categoria | Quantidade |
|---|---:|
| `NEEDS_REVIEW` | 15 |
| `PUBLICAVEL_APOS_REVISAO_PTBR` | 13 |
| `RELATORIO_PUBLICO_LEVE` | 1 |

## Diagnostico da evolucao desde o inicio das vertentes

- Ultimo commit estavel antes das vertentes: `NEEDS_REVIEW`. Candidatos observados: `67d8cfd` como base comum local das vertentes atuais; `5c3f977 origin/main` como estado publico remoto atual; `817ed17` como base imediata antes de MV2-12 no worktree MV2.
- DATA-01..DATA-05: aparecem como arquivos soltos no REV-P, sem commit local final nesta auditoria; precisam revisao e split por etapa.
- Pre-unificacao: entrou em `d6a50ee`, com DATA-06/07/08 e MV2-16 dry-run em modo fail-closed.
- MV2-12: entrou em `1d0d795`, mas a branch MV2 tambem carrega MV2-01..MV2-11.
- Blocos pendentes: DATA-01..05, DATA metadata-only/config local, MV2-13, MV2-14, MV2-15, reconstrucao espectral/STAC/crop no worktree MV2.
- Legado/curadoria: muitos arquivos REV-P soltos se classificam como `LEGADO_MV1`, `CURADORIA_PUBLICA`, `PROTOCOLO_C` ou `RELATORIO_PUBLICO_LEVE`; nenhum deve ser empurrado junto com o commit atual sem revisao.

## O que e publicavel como fato

- REV-P `d6a50ee`: conteudo tecnico leve e focado, publicavel como fato apos decisao de mensagem PT-BR/upstream.
- MV2 `1d0d795`: conteudo isolado publicavel como fato apos fatiamento de branch ou revisao da sequencia acumulada.
- Arquivos soltos: por politica fail-closed, ficam fora do push atual; os marcados `PUBLICAVEL_APOS_REVISAO_PTBR`, `RELATORIO_PUBLICO_LEVE`, `CURADORIA_PUBLICA`, `LEGADO_MV1` e `PROTOCOLO_C` exigem revisao humana antes de commit.

## O que precisa revisao humana

- Todos os arquivos soltos com `pode_ir_github=apos_revisao` no inventario.
- Arquivos com `private`, `raster`, `crop` ou `local_only` no nome, mesmo quando sao CSV/JSON/MD leves.
- A escolha do ultimo commit estavel antes das vertentes.
- A decisao sobre publicar ou fatiar a sequencia MV2-01..MV2-12.

## O que nao pode ir para GitHub agora

- Untracked REV-P e MV2 sem staging/revisao.
- Arquivos reais/pesados/sensiveis: `.env`, `api_config.local.json`, token, credencial, segredo, raster/crop/GeoTIFF, `local_only` real.
- Qualquer desbloqueio de Dia 10 sem evidencia real.

## Plano de commits futuros

1. Amend opcional PT-BR de `d6a50ee`, se autorizado.
2. Criar branch/publicacao limpa para pre-unificacao ou aceitar branch atual conscientemente.
3. Fatiar MV2-12 em branch sem arrastar MV2-01..MV2-11, ou auditar e publicar a sequencia acumulada como marco completo.
4. Separar DATA-01..05, MV2-13, MV2-14 e MV2-15 em commits independentes.
5. Revisar legado/curadoria/MV1 fora da trilha MV2.

## Decisoes para push

| Bloco | Decisao |
|---|---|
| REV-P `d6a50ee` | `PUSH_APOS_AMEND_PTBR` |
| MV2 `1d0d795` | `SPLIT_NECESSARIO` |
| Sequencia MV2-01..MV2-12 | `PUSH_APOS_REVISAO` |
| Untracked REV-P | `NAO_PUSHAR` |
| Untracked MV2 | `NAO_PUSHAR` |

## Proximo comando recomendado

```powershell
git status --short; git diff --check; git -C C:\Users\gabriela\Documents\REV-P-mv2-01-reconciliado status --short; git -C C:\Users\gabriela\Documents\REV-P-mv2-01-reconciliado diff --check
```

Depois disso, decidir explicitamente entre amend PT-BR do REV-P ou fatiamento da branch MV2-12.
