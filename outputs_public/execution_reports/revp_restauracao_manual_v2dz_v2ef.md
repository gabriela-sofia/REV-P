# REV-P - Restauracao manual controlada v2dz-v2ef

Data: 2026-06-18
Pasta usada: `C:\Users\gabriela\Documents\REV-P`
Branch usada: `ground-truth/restauracao-v2dz-v2ef`
Fonte restaurada: `cf642b9b833e09fa8bbd6390ba676f8b754ac7c4` (`untracked files on chore/public-repository-curation`)

## Resultado

Status final: **RECUPERACAO_COMPLETA_COM_FONTE_GIT_UNTRACKED_VALIDADA**.

A tentativa manual encontrou conteudo real em um objeto Git do tipo commit de arquivos untracked (`cf642b9`). A fonte contem os sete CSVs esperados da base `v2dz-v2ef`, dois scripts e um teste relacionado. Os sete CSVs foram validados com 53 linhas cada, hashes SHA256 origem/destino equivalentes, decisoes humanas vazias, gates positivo e negativo fechados como `false`, e `ground_truth_operational_status=ABSENT` no dashboard `v2ef`.

Antes da restauracao, os destinos nao existiam no worktree atual; portanto o backup previo foi registrado como `NO_EXISTING_DESTINATION_TO_BACKUP`. Nenhum arquivo pesado, `__pycache__` ou artefato fora do escopo foi copiado.

## Fontes verificadas

- `revp_public_repository_curation.diff`: nao encontrado.
- `git log --all --name-only --oneline`: encontrou nomes alvo vinculados ao objeto `cf642b9`.
- `git reflog --date=iso`: registrou checkout para `ground-truth/restauracao-v2dz-v2ef` e commit de auditoria `afff542`; nao e fonte de conteudo por si so.
- `git fsck --no-reflogs --unreachable`: executado; commits unreachable inspecionados nao apresentaram nomes alvo.
- `git worktree list`: tres worktrees ativos verificados.
- `C:/Users/gabriela/Documents/REV-P*`: worktree atual e worktrees relacionados verificados.
- `C:/Users/gabriela/Downloads`, `C:/Users/gabriela/Desktop`, `C:/Users/gabriela/OneDrive`: sem pacote fonte completo encontrado.
- `%TEMP%`: encontrou saidas de pytest geradas, rejeitadas como fonte original.

## Arquivos restaurados

- `outputs_public/tables/revp_observed_event_registry_v2dz.csv`
- `outputs_public/tables/revp_evidence_packet_registry_v2ea.csv`
- `outputs_public/tables/revp_patch_event_temporal_alignment_v2eb.csv`
- `outputs_public/tables/revp_patch_event_spatial_binding_v2ec.csv`
- `outputs_public/tables/revp_human_review_queue_v2ed.csv`
- `outputs_public/tables/revp_formal_label_gate_evaluator_v2ee.csv`
- `outputs_public/tables/revp_ground_truth_closure_dashboard_v2ef.csv`
- `scripts/ground_truth/revp_v2dz_to_v2ef_common.py`
- `scripts/ground_truth/revp_v2dz_to_v2ef_orchestrator.py`
- `tests/test_revp_v2dz_to_v2ef_orchestrator.py`

## Validacao de contagens

- Eventos (`v2dz`): 53.
- Evidence packets (`v2ea`): 53.
- Alinhamento temporal (`v2eb`): 53.
- Vinculacao espacial (`v2ec`): 53.
- Fila de revisao humana (`v2ed`): 53.
- Gate formal de label (`v2ee`): 53.
- Dashboard de fechamento (`v2ef`): 53.
- Positive gates fechados: 0.
- Negative gates fechados: 0.
- Ground truth operacional: `ABSENT`.

## Guardrails

- Nenhum label final foi criado.
- Nenhum negativo formal foi criado.
- Nenhuma decisao humana foi preenchida.
- `ground_truth_operational_status=ABSENT` permanece preservado.
- Treino supervisionado e MV1 permanecem bloqueados.
- A restauracao recupera a base candidata de trabalho `v2dz-v2ef`; ela nao promove ground truth operacional.

## Tabelas geradas

- `outputs_public/tables/revp_restauracao_manual_v2dz_v2ef_candidatos.csv`.
- `outputs_public/tables/revp_restauracao_manual_v2dz_v2ef_validacao.csv`.
- `outputs_public/tables/revp_restauracao_manual_v2dz_v2ef_manifesto.csv`.

## Proximo passo recomendado

Executar os testes focados `v2dz-v2ef` restaurados e, se passarem, tratar esta base como recuperada para auditoria e continuidade metodologica, ainda sem liberar labels, negativos formais ou treino supervisionado.
