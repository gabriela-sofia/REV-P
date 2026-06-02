# REV-P v1og-v1ot — Recuperação de Proveniência Temporal Sentinel: Resultado Final

## Objetivo do bloco v1og-v1ot

Construir uma pipeline auditável de recuperação de datas de aquisição Sentinel
associadas a patches Recife, usando exclusivamente fontes de metadado legítimas:
SAFE product names, MTD XML, STAC JSON, IDs de produto Sentinel, grafo de
proveniência patch→asset. Documentar o resultado com rastreabilidade completa.

## O que foi tentado

1. **v1og** — grafo de proveniência patch→asset→produto, normalização de aliases.
2. **v1oh** — scan local de metadados/sidecars Sentinel sem leitura de pixel.
3. **v1oi** — resolução de data por ID de produto e tile MGRS.
4. **v1oj** — adjudicação v2 de scene_date por patch, com hierarquia de confiança.
5. **v1ok** — rematch temporal evento↔patch com datas confirmadas ou prováveis.
6. **v1ol** — recheck C3+/C4/DINO após recuperação de proveniência.
7. **v1om** — descoberta de sidecars SAFE/MTD/STAC/JSON.
8. **v1on** — parser de datas de produtos Sentinel com bloqueio explícito de fontes inválidas.
9. **v1oo** — resolver v3 patch→asset→produto, fail-closed, com filtro de fixture.
10. **v1op** — adjudicação temporal v3, fail-closed.
11. **v1oq** — recheck C3+/C4/DINO com resolver v3.
12. **v1or** — bundle e sumário v3.
13. **v1os** — auditoria de contaminação por fixture/test nos datasets.
14. **v1ot** — consolidação final, manifest, quality checks, sumário científico.

## Por que datas por nome/manifest/event window foram bloqueadas

A pipeline aplica hierarquia fechada de fontes:

- **Aceitas**: MTD XML (`SENSING_TIME`, `PRODUCT_START_TIME`), SAFE product name
  (`S2A_MSIL1C_YYYYMMDDTHHMMSS`), STAC `datetime`/`start_datetime`, ID de produto
  Sentinel com timestamp embutido.
- **Bloqueadas**: `manifestCreationDate` (data de empacotamento, não de aquisição),
  ID de evento REC (ex: `REC-20220415`), janela temporal de evento (estimativa,
  não data observada), nome derivado de patch (gerado pela pipeline interna),
  data de modificação de arquivo (mtime), data de execução da pipeline, `YYYYMMDD`
  isolado sem contexto Sentinel.

O princípio é: apenas datas rastreáveis a um produto oficial Sentinel com timestamp
de aquisição/sensoriamento podem confirmar `scene_date`.

## Por que fixture/test contamination foi auditada e bloqueada

Testes anteriores (v1og-v1ol) escreviam diretamente em `datasets/` usando IDs
sintéticos (`REC_00001`, `R1`, `C1`). O script v1os detecta essas contaminações
via heurísticas conservadoras: `resolution_id = R1/R2` e `candidate_id = C1`.
Scripts v1oo e v1op filtram via `is_fixture_row()` antes de processar qualquer
entrada. Todos os testes foram reescritos para usar `tmp_path` e env vars.

## Resultado final limpo

| Métrica | Valor |
|---|---|
| Patches avaliados | 2654 |
| Product dates confirmadas reais | 0 |
| can_unlock_temporal = true | 0 |
| C3+ candidates | 0 |
| C4 formal negatives | 0 |
| DINO queue | 0 |
| Fixture high-severity | 0 |
| **Status final** | **TEMPORAL_RECOVERY_FAIL_CLOSED** |

## Implicação para Protocolo C

A não-confirmação de `scene_date` via metadado oficial Sentinel não é falha
metodológica — é resultado esperado quando os ativos locais não possuem SAFE
completo, MTD XML ou metadado de produto com timestamp de aquisição rastreável.
A pipeline permaneceu auditável e documentada.

## Implicação para C3/C4

Sem `scene_date` confirmada → `can_unlock_temporal = false` → sem adjudicação
temporal forte → C3+ = 0. C4 permanece fechado: `formal_negative_count = 0`.
Ambas as condições são documentadas com rastreabilidade total.

## Implicação para DINO

DINO permanece `REVIEW_ONLY_REPRESENTATION`. A fila DINO está vazia porque não
há patches com `scene_date` confirmada aguardando revisão estrutural. Nenhum
label, target ou uso supervisionado é autorizado.

## Texto recomendado para inserir no TCC

> A etapa de recuperação de proveniência temporal Sentinel foi mantida em regime
> fail-closed. Embora tenham sido avaliadas relações entre patch, asset, produto
> e metadados auxiliares, nenhuma cadeia completa
> `patch → asset → produto Sentinel oficial → data de aquisição`
> foi confirmada de forma suficiente para liberar adjudicação temporal operacional.
> Assim, os resultados permanecem como evidência contextual/auditável, sem criação
> de rótulo operacional, sem ground truth e sem uso supervisionado.

## Limites e próximos passos

- Para desbloquear `scene_date` confirmada: disponibilizar arquivos SAFE completos
  com MTD XML, ou STAC JSON com campo `datetime` vinculado a produto Sentinel real.
- Para desbloquear C3+: além de scene_date confirmada, é necessário formal negative
  documentada (`formal_negative_count > 0`).
- Para desbloquear C4: formal negative de fontes oficiais comprovadas.
- DINO pode ser reavaliado após C3+ desbloqueado, sempre em modo REVIEW_ONLY.
