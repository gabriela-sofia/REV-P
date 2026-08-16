# Plano de stage seletivo — marco MV1

> Plano review-only. Não faz `git add`, commit, push, merge, rebase, delete ou cleanup. Define, de forma auditável, o que deve entrar no stage/commit do marco MV1 e o que NÃO deve entrar, evitando `git add -A`.

## 1. Escopo

Consolidação do marco MV1 em um único commit coerente, review-only. Esta passada apenas planeja o stage seletivo; a execução do `git add`/commit é ação manual humana. Não cria artefatos científicos novos nem altera os existentes (exceto correção objetiva de consistência, que não foi necessária).

## 2. Branch

`marco/validacao-label-free-evidencia-estrutural-mv1`. Nenhuma nova branch é criada.

## 3. Estado do working tree

No início: nada staged, whitespace limpo (`git diff --check` exit 0), nenhum arquivo rastreado modificado. Todos os itens do marco estão como `untracked`. Caches (`__pycache__`, `.pytest_cache`), `local_only/` e o `.log` de logs_summary estão git-ignored (`!!`), portanto fora de risco de stage acidental.

## 4. Por que `git add -A` é proibido

`git add -A` capturaria, além do marco MV1: (a) os 8 artefatos da auditoria PT-BR que estão fora do escopo deste marco; (b) qualquer cache/temporário que escape do `.gitignore`; (c) eventuais arquivos não relacionados. O stage deve ser seletivo por grupo, explícito e auditável.

## 5. Critério de inclusão no stage

Entra no stage o artefato que: pertence a uma das 11 fases do marco MV1; é leve e público (relatório `.md`, tabela `.csv`, métrica `.json`, script `.py` ou teste); não contém bruto/pesado; e está coberto por guardrails review-only. Total: **113 arquivos candidatos**.

## 6. Critério de exclusão do stage

Fica fora: bruto externo, dados pesados, `local_only/`, `local_runs/`, caches, `__pycache__/`, `.pytest_cache/`, scripts/logs temporários, outputs não relacionados à branch, os 8 artefatos da auditoria PT-BR fora de escopo e qualquer arquivo que não pertença ao marco MV1.

## 7. Grupos de arquivos que devem entrar

Por fase (detalhe em `revp_plano_stage_seletivo_marco_mv1.csv`):

- `restauracao_v2dz_v2ef` — 14 (registros v2dz–v2ef, restauração manual, orquestrador e teste).
- `auditoria_temporal` — 5.
- `validacao_label_free` — 12.
- `curadoria_externa` — 8.
- `navegacao_downloads_externos` — 25 (navegação, fechamento de downloads e integração externa).
- `normalizacao_evidencias_externas` — 9.
- `protocolo_fail_closed` — 10 (inclui as 3 políticas em `docs/metodologia_cientifica/`).
- `integracao_final` — 8 (matriz de maturidade, fila de revisão humana, gates, bloqueadores).
- `hardening_reprodutibilidade` — 9.
- `auditoria_critica` — 9 (auditoria de banca + auditoria de reprodutibilidade externa).
- `plano_stage` — 4 (este pacote).

## 8. Grupos de arquivos que não devem entrar

- **Auditoria PT-BR fora de escopo (8 arquivos):** `revp_auditoria_integral_estado_atual.md`, `revp_current_project_state_after_ptbr_curation.md`, `revp_diagnostico_linguagem_publica_ptbr.md`/`.csv`, `revp_arquivos_cruciais_para_defesa.csv`, `revp_arquivos_historicos_ou_auxiliares.csv`, `revp_estado_git_branches_curadoria_ptbr.csv`, `revp_mapa_pipeline_real_atual.csv` — avaliar em commit/branch separado de curadoria.
- **Padrões estruturais permanentes:** `local_only/`, `local_runs/`, `data/`, `*.tif/*.tiff/*.geotiff`, `*.npz/*.npy`, `__pycache__/`, `.pytest_cache/`, `outputs_public/logs_summary/*.log`, `scripts/**/_tmp_*.py`.

## 9. Comandos sugeridos de stage seletivo

Executar por grupo (PowerShell), nunca `git add -A`. A forma mais segura é stagear por diretório/prefixo do marco, **depois** despromover (`git reset`) os 8 PT-BR fora de escopo:

```powershell
# Fase 1 — restauracao v2dz-v2ef
git add outputs_public/execution_reports/revp_restauracao_manual_v2dz_v2ef.md
git add outputs_public/tables/revp_restauracao_manual_v2dz_v2ef_candidatos.csv
git add outputs_public/tables/revp_restauracao_manual_v2dz_v2ef_manifesto.csv
git add outputs_public/tables/revp_restauracao_manual_v2dz_v2ef_validacao.csv
git add outputs_public/tables/revp_observed_event_registry_v2dz.csv
git add outputs_public/tables/revp_evidence_packet_registry_v2ea.csv
git add outputs_public/tables/revp_patch_event_temporal_alignment_v2eb.csv
git add outputs_public/tables/revp_patch_event_spatial_binding_v2ec.csv
git add outputs_public/tables/revp_human_review_queue_v2ed.csv
git add outputs_public/tables/revp_formal_label_gate_evaluator_v2ee.csv
git add outputs_public/tables/revp_ground_truth_closure_dashboard_v2ef.csv
git add scripts/ground_truth/revp_v2dz_to_v2ef_common.py
git add scripts/ground_truth/revp_v2dz_to_v2ef_orchestrator.py
git add tests/test_revp_v2dz_to_v2ef_orchestrator.py

# Fase 2 — auditoria temporal
git add outputs_public/execution_reports/revp_auditoria_prontidao_temporal_assets_mv1.md
git add outputs_public/tables/revp_auditoria_prontidao_temporal_assets_mv1.csv
git add outputs_public/metrics/revp_auditoria_prontidao_temporal_assets_mv1.json
git add scripts/ground_truth/revp_auditoria_prontidao_temporal_assets_mv1.py
git add tests/test_revp_auditoria_prontidao_temporal_assets_mv1.py

# Fase 3 — validacao label-free
git add outputs_public/execution_reports/revp_validacao_label_free_evidencia_estrutural_mv1.md
git add outputs_public/execution_reports/revp_fechamento_marco_validacao_label_free_evidencia_estrutural_mv1.md
git add outputs_public/tables/revp_validacao_label_free_evidencia_estrutural_mv1.csv
git add outputs_public/tables/revp_matriz_topologica_cidades_mv1.csv
git add outputs_public/tables/revp_vizinhos_embeddings_label_free_mv1.csv
git add outputs_public/tables/revp_guardrails_validacao_label_free_mv1.csv
git add outputs_public/tables/revp_manifesto_marco_validacao_label_free_evidencia_estrutural_mv1.csv
git add outputs_public/tables/revp_proximos_passos_pos_marco_label_free_mv1.csv
git add outputs_public/metrics/revp_validacao_label_free_evidencia_estrutural_mv1.json
git add outputs_public/metrics/revp_fechamento_marco_validacao_label_free_evidencia_estrutural_mv1.json
git add scripts/ground_truth/revp_validacao_label_free_evidencia_estrutural_mv1.py
git add tests/test_revp_validacao_label_free_evidencia_estrutural_mv1.py

# Fase 4 — curadoria externa
git add outputs_public/execution_reports/revp_curadoria_evidencias_externas_mv1.md
git add outputs_public/metrics/revp_curadoria_evidencias_externas_mv1.json
git add outputs_public/tables/revp_manifesto_evidencias_externas_mv1.csv
git add outputs_public/tables/revp_auditoria_fontes_externas_mv1.csv
git add outputs_public/tables/revp_indice_eventos_externos_candidatos_mv1.csv
git add outputs_public/tables/revp_indice_geometrias_externas_candidatas_mv1.csv
git add scripts/curadoria_externa/revp_curadoria_evidencias_externas_mv1.py
git add tests/test_revp_curadoria_evidencias_externas_mv1.py

# Fase 5 — navegacao/downloads externos (relatorios, metricas, tabelas, scripts, testes)
git add outputs_public/execution_reports/revp_navegacao_downloads_evidencias_externas_mv1.md
git add outputs_public/execution_reports/revp_fechamento_downloads_evidencias_externas_mv1.md
git add outputs_public/execution_reports/revp_integracao_marco_label_free_evidencias_externas_mv1.md
git add outputs_public/execution_reports/revp_integracao_marco_label_free_evidencias_externas_navegacao_mv1.md
git add outputs_public/metrics/revp_navegacao_downloads_evidencias_externas_mv1.json
git add outputs_public/metrics/revp_fechamento_downloads_evidencias_externas_mv1.json
git add outputs_public/metrics/revp_integracao_marco_label_free_evidencias_externas_mv1.json
git add outputs_public/metrics/revp_integracao_marco_label_free_evidencias_externas_navegacao_mv1.json
git add outputs_public/tables/revp_manifesto_evidencias_externas_downloads_mv1.csv
git add outputs_public/tables/revp_manifesto_evidencias_externas_navegacao_mv1.csv
git add outputs_public/tables/revp_auditoria_fontes_externas_downloads_mv1.csv
git add outputs_public/tables/revp_auditoria_fontes_externas_navegacao_mv1.csv
git add outputs_public/tables/revp_log_downloads_evidencias_externas_mv1.csv
git add outputs_public/tables/revp_log_navegacao_downloads_evidencias_externas_mv1.csv
git add outputs_public/tables/revp_indice_arquivos_baixados_evidencias_externas_mv1.csv
git add outputs_public/tables/revp_indice_eventos_externos_candidatos_downloads_mv1.csv
git add outputs_public/tables/revp_indice_eventos_externos_candidatos_navegacao_mv1.csv
git add outputs_public/tables/revp_indice_geometrias_externas_candidatas_downloads_mv1.csv
git add outputs_public/tables/revp_indice_geometrias_externas_candidatas_navegacao_mv1.csv
git add outputs_public/tables/revp_integracao_marco_label_free_evidencias_externas_mv1.csv
git add outputs_public/tables/revp_integracao_marco_label_free_evidencias_externas_navegacao_mv1.csv
git add scripts/curadoria_externa/revp_navegacao_downloads_evidencias_externas_mv1.py
git add scripts/curadoria_externa/revp_fechamento_downloads_e_integracao_evidencias_externas_mv1.py
git add tests/test_revp_navegacao_downloads_evidencias_externas_mv1.py
git add tests/test_revp_fechamento_downloads_e_integracao_evidencias_externas_mv1.py

# Fase 6 — normalizacao evidencias externas
git add outputs_public/execution_reports/revp_normalizacao_evidencias_externas_publicas_mv1.md
git add outputs_public/metrics/revp_normalizacao_evidencias_externas_publicas_mv1.json
git add outputs_public/tables/revp_manifesto_publico_arquivos_externos_url_hash_mv1.csv
git add outputs_public/tables/revp_normalizacao_bloqueadores_evidencias_externas_mv1.csv
git add outputs_public/tables/revp_manifesto_evidencias_externas_normalizado_mv1.csv
git add outputs_public/tables/revp_auditoria_fontes_externas_normalizada_mv1.csv
git add outputs_public/tables/revp_recomendacoes_pacote_externo_normalizada_mv1.csv
git add scripts/curadoria_externa/revp_normalizacao_evidencias_externas_publicas_mv1.py
git add tests/test_revp_normalizacao_evidencias_externas_publicas_mv1.py

# Fase 7 — protocolo fail-closed (inclui as 3 politicas em docs/)
git add outputs_public/execution_reports/revp_protocolo_ground_truth_fail_closed_mv1.md
git add outputs_public/metrics/revp_protocolo_ground_truth_fail_closed_mv1.json
git add outputs_public/tables/revp_ontologia_estados_label_mv1.csv
git add outputs_public/tables/revp_gates_readiness_treino_mv1.csv
git add outputs_public/tables/revp_dashboard_bloqueio_treino_ground_truth_mv1.csv
git add docs/metodologia_cientifica/revp_ontologia_labels_ground_truth_mv1.md
git add docs/metodologia_cientifica/revp_politica_evidencia_negativa_mv1.md
git add docs/metodologia_cientifica/revp_politica_anti_leakage_mv1.md
git add scripts/ground_truth/revp_protocolo_ground_truth_fail_closed_mv1.py
git add tests/test_revp_protocolo_ground_truth_fail_closed_mv1.py

# Fase 8 — integracao final
git add outputs_public/execution_reports/revp_integracao_final_marco_mv1_maturidade_revisao_humana.md
git add outputs_public/metrics/revp_integracao_final_marco_mv1_maturidade_revisao_humana.json
git add outputs_public/tables/revp_matriz_maturidade_metodologica_mv1.csv
git add outputs_public/tables/revp_fila_revisao_humana_candidatos_mv1.csv
git add outputs_public/tables/revp_matriz_evidencias_externas_gates_mv1.csv
git add outputs_public/tables/revp_bloqueadores_finais_ground_truth_treino_mv1.csv
git add scripts/ground_truth/revp_integracao_final_marco_mv1_maturidade_revisao_humana.py
git add tests/test_revp_integracao_final_marco_mv1_maturidade_revisao_humana.py

# Fase 9 — hardening reprodutibilidade
git add outputs_public/execution_reports/revp_hardening_reprodutibilidade_publica_marco_mv1.md
git add outputs_public/metrics/revp_hardening_reprodutibilidade_publica_marco_mv1.json
git add outputs_public/tables/revp_plano_pacote_externo_reprodutibilidade_mv1.csv
git add outputs_public/tables/revp_dependencias_local_only_marco_mv1.csv
git add outputs_public/tables/revp_checklist_reproducao_publica_marco_mv1.csv
git add outputs_public/tables/revp_indice_reprodutibilidade_marco_mv1.csv
git add outputs_public/tables/revp_manifesto_mestre_marco_mv1.csv
git add scripts/ground_truth/revp_hardening_reprodutibilidade_publica_marco_mv1.py
git add tests/test_revp_hardening_reprodutibilidade_publica_marco_mv1.py

# Fase 10 — auditoria critica + reprodutibilidade externa
git add outputs_public/execution_reports/revp_auditoria_critica_banca_marco_mv1.md
git add outputs_public/metrics/revp_auditoria_critica_banca_marco_mv1.json
git add outputs_public/tables/revp_auditoria_critica_banca_marco_mv1.csv
git add outputs_public/tables/revp_checklist_pre_stage_marco_mv1.csv
git add outputs_public/execution_reports/revp_auditoria_reprodutibilidade_externa_marco_mv1.md
git add outputs_public/metrics/revp_auditoria_reprodutibilidade_externa_marco_mv1.json
git add outputs_public/tables/revp_verificacao_fontes_externas_marco_mv1.csv
git add outputs_public/tables/revp_verificacao_arquivos_local_only_marco_mv1.csv
git add outputs_public/tables/revp_recomendacoes_pacote_externo_marco_mv1.csv

# Fase 11 — plano de stage (este pacote)
git add outputs_public/execution_reports/revp_plano_stage_seletivo_marco_mv1.md
git add outputs_public/tables/revp_plano_stage_seletivo_marco_mv1.csv
git add outputs_public/tables/revp_exclusoes_stage_marco_mv1.csv
git add outputs_public/metrics/revp_plano_stage_seletivo_marco_mv1.json
```

## 10. Comandos de validação antes do commit

```powershell
git diff --check
git diff --cached --name-only    # deve listar os 113 do marco e NENHUM dos 8 PT-BR
git status --short               # os 8 PT-BR devem permanecer como ??
python -m pytest tests/ -q       # opcional: suite do marco
```

Conferir explicitamente que nenhum dos 8 artefatos PT-BR e nenhum caminho `local_only/`/`local_runs/`/cache aparece em `git diff --cached --name-only`.

## 11. Mensagem de commit sugerida

```text
Consolida marco MV1 review-only e reprodutibilidade externa
```

## 12. Guardrails preservados

- evidência externa não vira label;
- download não vira validação operacional;
- suscetibilidade não vira evento observado;
- landslide scar não vira flood extent;
- Curitiba não vira negativo formal;
- não liberar treino;
- não declarar ground truth operacional;
- `pode_virar_label_agora=false`;
- `git add -A` proibido — stage seletivo por grupo.

## 13. Próximo passo após stage

Após o stage seletivo: rodar as validações da seção 10, conferir `git diff --cached`, e então (ação manual humana) `git commit` com a mensagem sugerida. Push e PR permanecem ações manuais explícitas. Os 8 artefatos PT-BR ficam para um commit separado de curadoria; ground truth operacional segue ausente e treino bloqueado.
</content>
