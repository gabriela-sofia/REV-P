# REV-P — Relatório de validação da curadoria pública

Data: 2026-06-18
Branch: `chore/public-repository-curation`
Fase: validação pós-curadoria editorial — Fase 9

---

## 1. Testes executados

### Conjunto compatível com o ambiente (sem numpy, sem dados locais)

```bash
python -m pytest tests/test_revp_v1fu_dino_sentinel_input_manifest.py \
  tests/test_revp_v1fv_dino_local_asset_preflight.py \
  tests/test_revp_v1fw_dino_embedding_extraction_scaffold.py \
  tests/test_revp_v1gp_dino_github_release_readiness_audit.py \
  tests/test_revp_v1gk_dino_reproducibility_closure.py \
  tests/test_revp_v1gv_external_evidence_coverage_matrix.py \
  tests/test_revp_v1if_official_observed_event_vector_acquisition_audit.py \
  tests/test_revp_v1ih_official_open_data_event_vector_discovery_validation.py \
  tests/test_revp_v1ii_targeted_official_repository_event_vector_mining.py \
  -q --tb=no
```

**Resultado**: 163 aprovados, 4 ignorados (skips esperados), 0 falhas

---

## 2. Falhas pré-existentes (não relacionadas à curadoria)

As falhas encontradas em outros testes são pré-existentes e causadas por:

| Causa | Testes afetados |
|---|---|
| `ModuleNotFoundError: No module named 'numpy'` | `v1fz`, `v1ge`, `v1gf`, `v1gh`, `v1gi` e outros que importam numpy/torch |
| Dados locais ausentes (embeddings, preflight, revisão longitudinal) | `v1gh`, `v1gi`, `v1hb` — requerem arquivos de `local_runs/` |
| Erros de coleta por dependências ausentes | 36 testes com `ERROR` durante `--collect-only` |

Nenhuma dessas falhas foi causada pelas alterações da curadoria. Os arquivos de código (`scripts/`, `tests/`) não foram modificados.

---

## 3. Arquivos alterados pela curadoria

Todos os arquivos modificados são documentos públicos (`.md`, `.csv`). Nenhum script Python, teste, dataset, manifesto ou configuração foi alterado.

### Arquivos editados

| Arquivo | Mudança |
|---|---|
| `README.md` | Reestruturação completa — formato científico compacto de 9 seções |
| `outputs_public/execution_reports/final_delivery_artifact_index.md` | Remoção de 92 disclaimers repetidos; cabeçalho com nota única |
| `outputs_public/execution_reports/final_guardrails_report.md` | Título humanizado; texto narrativo em português |
| `outputs_public/logs_summary/protocol_c_current_status_summary.md` | Status em inglês convertidos para prosa em português |
| `outputs_public/logs_summary/protocol_c_cross_region_status_summary.md` | Tabela clara com tradução dos tipos de referência |

### Arquivos criados

| Arquivo | Função |
|---|---|
| `outputs_public/execution_reports/revp_public_repository_curation_audit.md` | Auditoria editorial — Fase 1 |
| `docs/metodologia_cientifica/revp_style_and_naming_guide.md` | Guia editorial canônico — Fase 2 |
| `docs/metodologia_cientifica/revp_stage_index.md` | Índice de estágios com nomes públicos — Fase 4 |
| `outputs_public/tables/revp_public_stage_index.csv` | Versão CSV do índice de estágios — Fase 4 |
| `outputs_public/execution_reports/revp_language_cleanup_report.md` | Relatório de limpeza de linguagem — Fase 6 |
| `outputs_public/execution_reports/revp_git_state_cleanup_plan.md` | Diagnóstico Git — Fase 7 |
| `outputs_public/tables/revp_public_export_filelist.csv` | Lista de exportação pública — Fase 8 |
| `outputs_public/execution_reports/revp_public_curation_validation_report.md` | Este relatório — Fase 9 |

---

## 4. Verificação de integridade

```bash
git diff --check
# Saída: nenhum conflito de espaço em branco ou marcador de conflito
```

```bash
git status --short
# 5 arquivos modificados (M), 8 arquivos novos (??)
# Nenhuma exclusão, nenhum arquivo de código alterado
```

---

## 5. Riscos remanescentes

| Risco | Nível | Observação |
|---|---|---|
| `datasets/README.md` com possível problema de encoding | Baixo | Exibição garbled no output da ferramenta de leitura; verificar se o arquivo é UTF-8 válido antes de commit |
| Testes que dependem de numpy não instalado | Pré-existente | Instalar `pip install -r requirements.txt` no ambiente de execução resolve |
| Testes que dependem de dados em `local_runs/` | Pré-existente | Requerem execução local com dados completos |
| Relatórios de sprint individuais (v2at–v2bh) com títulos técnicos | Baixo | Cobertos pelo índice público de estágios; títulos internos preservados por compatibilidade |

---

## 6. Próximos passos recomendados

1. Verificar encoding de `datasets/README.md` (comando: `file -i datasets/README.md` no ambiente local).
2. Instalar dependências completas (`pip install -r requirements.txt`) para rodar a suíte completa.
3. Fazer stage seletivo dos arquivos listados na seção 3 e commitar com a mensagem sugerida.
4. Push para `chore/public-repository-curation`.

### Mensagem de commit sugerida

```
docs: organiza camada publica e linguagem metodologica do REV-P
```

---

## 7. Verificação de segurança metodológica

A curadoria não alterou:

- ✅ Nenhum fato científico ou número
- ✅ Nenhum status metodológico (bloqueado permanece bloqueado)
- ✅ Nenhum schema de CSV ou contrato de código
- ✅ Nenhuma rastreabilidade ou proveniência
- ✅ Nenhuma limitação ou restrição do projeto
- ✅ Nenhum arquivo de código Python
- ✅ Nenhum dataset, manifesto ou configuração
- ✅ Nenhuma licença ou hash de artefato
