# REV-P — Diagnóstico do estado Git

Data: 2026-06-18
Fase: curadoria editorial — Fase 7

Este relatório registra o estado atual do repositório Git para referência e planejamento seguro de commit. Nenhuma ação destrutiva foi executada.

---

## 1. Branch atual

`chore/public-repository-curation` (histórico — mesclado em `main` via PR #21)

Branch da primeira fase da curadoria editorial em inglês. O presente trabalho de curadoria em português é realizado na branch `curadoria/repositorio-publico-ptbr`.

---

## 2. Arquivos modificados pela curadoria

Os seguintes arquivos foram alterados como parte da curadoria editorial (Fases 3–6):

| Arquivo | Tipo de mudança |
|---|---|
| `README.md` | Reestruturado para formato científico compacto |
| `outputs_public/execution_reports/final_delivery_artifact_index.md` | Remoção de 92 disclaimers repetidos |
| `outputs_public/execution_reports/final_guardrails_report.md` | Texto humanizado |
| `outputs_public/logs_summary/protocol_c_current_status_summary.md` | Status em inglês convertidos para prosa em português |
| `outputs_public/logs_summary/protocol_c_cross_region_status_summary.md` | Idem |

---

## 3. Arquivos criados pela curadoria (untracked)

| Arquivo | Função |
|---|---|
| `docs/metodologia_cientifica/revp_indice_etapas.md` | Índice público de estágios |
| `docs/metodologia_cientifica/revp_guia_estilo_nomenclatura.md` | Guia editorial canônico |
| `outputs_public/execution_reports/revp_auditoria_curadoria_repositorio_publico.md` | Auditoria editorial |
| `outputs_public/execution_reports/revp_relatorio_limpeza_linguagem.md` | Relatório de limpeza de linguagem |
| `outputs_public/execution_reports/revp_plano_organizacao_estado_git.md` | Este arquivo |
| `outputs_public/execution_reports/revp_relatorio_validacao_curadoria_publica.md` | Relatório de validação |
| `outputs_public/tables/revp_indice_etapas_publicas.csv` | Índice de estágios em CSV |
| `outputs_public/tables/revp_lista_arquivos_exportacao_publica.csv` | Lista de exportação pública |

---

## 4. Arquivos não relacionados à curadoria (estado limpo)

O worktree não contém arquivos modificados de sprints anteriores não relacionados à curadoria. O estado é limpo para os artefatos de sprint (datasets, scripts, testes) que não foram tocados.

---

## 5. Diretórios problemáticos por permissão

O diretório `docs/templates/protocolo_c_solicitacoes_preenchidas/` foi referenciado em commits anteriores como tendo arquivos de permissão problemática. **Este diretório não foi tocado durante a curadoria.**

Se houver erro de permissão ao fazer `git status -uall` neste diretório, usar `git status` sem a flag `-uall`.

---

## 6. Histórico recente relevante

```
efb1668  Merge pull request #19 (multimodal GT scaffold)
deec0fe  data: prepara aquisicao e QA geoespacial de evidencias externas
ca0765b  analysis: prepara priorizacao TP2 e replay bloqueavel
285ebd7  Merge pull request #18
08c5764  data: registra fontes externas reais e triagem conservadora
```

---

## 7. Plano seguro para commit da curadoria

### Arquivos seguros para stage

Todos os arquivos listados nas seções 2 e 3 podem ser staged com segurança:

```bash
git add README.md
git add outputs_public/execution_reports/final_delivery_artifact_index.md
git add outputs_public/execution_reports/final_guardrails_report.md
git add outputs_public/logs_summary/protocol_c_current_status_summary.md
git add outputs_public/logs_summary/protocol_c_cross_region_status_summary.md
git add docs/metodologia_cientifica/revp_indice_etapas.md
git add docs/metodologia_cientifica/revp_guia_estilo_nomenclatura.md
git add outputs_public/execution_reports/revp_auditoria_curadoria_repositorio_publico.md
git add outputs_public/execution_reports/revp_relatorio_limpeza_linguagem.md
git add outputs_public/execution_reports/revp_plano_organizacao_estado_git.md
git add outputs_public/execution_reports/revp_relatorio_validacao_curadoria_publica.md
git add outputs_public/tables/revp_indice_etapas_publicas.csv
git add outputs_public/tables/revp_lista_arquivos_exportacao_publica.csv
```

### O que NÃO deve ser staged

- Nenhum arquivo de `datasets/`, `scripts/`, `tests/`, `manifests/`, `configs/` foi alterado pela curadoria — não incluir.
- Não usar `git add .` ou `git add -A` — pode incluir arquivos não intencionais.

### Mensagem de commit sugerida

```
docs: organiza camada publica e linguagem metodologica do REV-P
```

---

## 8. Observações

- Não executar `git reset --hard` nem `git clean` sem validação explícita.
- O commit desta curadoria é separado dos commits de sprint (v2ca–v2cw) e não afeta o histórico deles.
- A primeira fase de curadoria foi entregue em `chore/public-repository-curation` e mesclada em `main` via PR #21. A curadoria em português prossegue em `curadoria/repositorio-publico-ptbr`.
