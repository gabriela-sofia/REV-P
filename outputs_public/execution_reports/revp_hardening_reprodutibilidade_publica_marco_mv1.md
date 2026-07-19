# Hardening de reprodutibilidade publica do marco MV1

## 1. Escopo
Este pacote documenta o que pode ser reproduzido publicamente no Git, o que pode ser verificado por hash e o que depende de arquivos locais, downloads externos ou solicitacao formal. O escopo permanece review-only e label-free.

## 2. O que significa reprodutibilidade publica no REV-P
Reprodutibilidade publica significa conseguir reexecutar scripts, validar schemas, comparar hashes e auditar proveniencia sem publicar dados brutos pesados ou dependencias de licenca incerta.

## 3. O que e reproduzivel so com Git
Artefatos reproduziveis no Git: 54. Incluem scripts, testes, CSVs, JSONs e relatorios publicos quando os insumos ja estao no repositorio.

## 4. O que e verificavel por hash
Artefatos com hash registrado: 75. O hash serve para integridade e rastreabilidade, nao para promover conclusao operacional.

## 5. O que depende de `local_only`
Dependencias locais registradas: 21. Elas incluem arquivos externos em quarentena e embeddings DINOv2 brutos mantidos fora do Git.

## 6. O que depende de download externo
Arquivos listados no indice de evidencias externas podem ser baixados novamente quando a fonte oficial permite. A reproducao integral depende de rede, licenca e disponibilidade da fonte.

## 7. O que depende de solicitacao formal
Fontes sem download publico direto devem permanecer bloqueadas ate haver resposta formal e hash do material recebido.

## 8. O que precisa de pacote externo futuro
O plano de pacote externo tem 26 itens. Ele prioriza embeddings DINOv2, indices, hashes, scripts e instrucoes; GeoTIFF pesado nao entra sem decisao explicita.

## 9. O que nao deve ser publicado no Git
Nao publicar GeoTIFF pesado, rasters brutos, arquivos em `local_only`, embeddings brutos `.npz`, caches locais ou arquivos com licenca incerta sem revisao.

## 10. Como reproduzir scripts principais
- `python scripts/ground_truth/revp_hardening_reprodutibilidade_publica_marco_mv1.py --repo-root .`
- `python -m pytest tests/test_revp_hardening_reprodutibilidade_publica_marco_mv1.py -q`
- `python -m pytest tests/test_revp_integracao_final_marco_mv1_maturidade_revisao_humana.py -q`
- `python -m pytest tests/test_revp_protocolo_ground_truth_fail_closed_mv1.py -q`
- `python -m pytest tests/test_revp_navegacao_downloads_evidencias_externas_mv1.py -q`
- `python -m pytest tests/test_revp_validacao_label_free_evidencia_estrutural_mv1.py -q`
- `python -m pytest tests/test_revp_auditoria_prontidao_temporal_assets_mv1.py -q`

## 11. Como validar CSVs/JSONs
Use o teste dedicado do hardening para verificar colunas obrigatorias, JSON parseavel, valores fixos conservadores e ausencia de linguagem proibida nos campos estruturados.

## 12. Como validar hashes
Compare os campos `sha256` do manifesto mestre e dos indices locais com os arquivos existentes. Para arquivos externos, use o indice publico de downloads como fonte de hash esperada.

## 13. Limite dos 12 embeddings
Os 12 embeddings DINOv2 sustentam um piloto exploratorio label-free. Eles nao sustentam promocao automatica, treino supervisionado ou fechamento de ground truth operacional.

## 14. Limite dos brutos externos
Arquivos externos baixados ajudam rastreabilidade e revisao humana futura, mas continuam como contexto ou dependencia. Redistribuicao exige licenca, tamanho e decisao de escopo.

## 15. Stage seletivo recomendado
Recomendacao: usar apenas `git add <lista explicita>` para os arquivos publicos revisados. `git add -A` permanece proibido neste marco.

## 16. Guardrails preservados
- Sem treino supervisionado.
- Sem labels novos.
- Sem positivos formais.
- Sem negativos formais.
- `local_only` e `local_runs` ficam como dependencias, nao como publico Git.
- `itens_podem_virar_label_agora` permanece 0.

## 17. Conclusao
Status: `PUBLICO_REPRODUTIVEL_PARCIAL_COM_DEPENDENCIAS_LOCAL_ONLY_E_EXTERNAS`. O marco pode ser auditado publicamente por scripts, tabelas, JSONs, hashes e indices, mas a reproducao integral depende de dependencias locais e externas controladas.

## Artefatos publicos existentes no Git
- `outputs_public/execution_reports/revp_restauracao_manual_v2dz_v2ef.md`
- `outputs_public/tables/revp_restauracao_manual_v2dz_v2ef_manifesto.csv`
- `outputs_public/tables/revp_restauracao_manual_v2dz_v2ef_validacao.csv`
- `scripts/ground_truth/revp_v2dz_to_v2ef_orchestrator.py`
- `tests/test_revp_v2dz_to_v2ef_orchestrator.py`
- `outputs_public/execution_reports/revp_auditoria_prontidao_temporal_assets_mv1.md`
- `outputs_public/tables/revp_auditoria_prontidao_temporal_assets_mv1.csv`
- `outputs_public/metrics/revp_auditoria_prontidao_temporal_assets_mv1.json`
- `scripts/ground_truth/revp_auditoria_prontidao_temporal_assets_mv1.py`
- `tests/test_revp_auditoria_prontidao_temporal_assets_mv1.py`
- `outputs_public/execution_reports/revp_validacao_label_free_evidencia_estrutural_mv1.md`
- `outputs_public/tables/revp_validacao_label_free_evidencia_estrutural_mv1.csv`
- `outputs_public/tables/revp_matriz_topologica_cidades_mv1.csv`
- `outputs_public/tables/revp_vizinhos_embeddings_label_free_mv1.csv`
- `outputs_public/tables/revp_guardrails_validacao_label_free_mv1.csv`
- `outputs_public/metrics/revp_validacao_label_free_evidencia_estrutural_mv1.json`
- `outputs_public/tables/table_dino_embedding_inventory.csv`
- `outputs_public/tables/table_dino_similarity_matrix.csv`
- `outputs_public/tables/table_dino_nearest_neighbors.csv`
- `scripts/ground_truth/revp_validacao_label_free_evidencia_estrutural_mv1.py`
- `tests/test_revp_validacao_label_free_evidencia_estrutural_mv1.py`
- `outputs_public/execution_reports/revp_protocolo_ground_truth_fail_closed_mv1.md`
- `outputs_public/tables/revp_ontologia_estados_label_mv1.csv`
- `outputs_public/tables/revp_gates_readiness_treino_mv1.csv`
- `outputs_public/tables/revp_dashboard_bloqueio_treino_ground_truth_mv1.csv`
- `outputs_public/metrics/revp_protocolo_ground_truth_fail_closed_mv1.json`
- `scripts/ground_truth/revp_protocolo_ground_truth_fail_closed_mv1.py`
- `tests/test_revp_protocolo_ground_truth_fail_closed_mv1.py`
- `outputs_public/tables/revp_manifesto_evidencias_externas_navegacao_mv1.csv`
- `outputs_public/tables/revp_indice_arquivos_baixados_evidencias_externas_mv1.csv`
- `outputs_public/tables/revp_log_navegacao_downloads_evidencias_externas_mv1.csv`
- `outputs_public/execution_reports/revp_navegacao_downloads_evidencias_externas_mv1.md`
- `outputs_public/metrics/revp_navegacao_downloads_evidencias_externas_mv1.json`
- `tests/test_revp_navegacao_downloads_evidencias_externas_mv1.py`
- `outputs_public/execution_reports/revp_integracao_final_marco_mv1_maturidade_revisao_humana.md`
- `outputs_public/tables/revp_matriz_maturidade_metodologica_mv1.csv`
- `outputs_public/tables/revp_fila_revisao_humana_candidatos_mv1.csv`
- `outputs_public/tables/revp_bloqueadores_finais_ground_truth_treino_mv1.csv`
- `outputs_public/metrics/revp_integracao_final_marco_mv1_maturidade_revisao_humana.json`
- `scripts/ground_truth/revp_integracao_final_marco_mv1_maturidade_revisao_humana.py`
- `tests/test_revp_integracao_final_marco_mv1_maturidade_revisao_humana.py`
- `outputs_public/execution_reports/revp_auditoria_critica_banca_marco_mv1.md`
- `outputs_public/tables/revp_auditoria_critica_banca_marco_mv1.csv`
- `outputs_public/metrics/revp_auditoria_critica_banca_marco_mv1.json`
- `outputs_public/tables/revp_checklist_pre_stage_marco_mv1.csv`
- `outputs_public/execution_reports/revp_hardening_reprodutibilidade_publica_marco_mv1.md`
- `outputs_public/tables/revp_manifesto_mestre_marco_mv1.csv`
- `outputs_public/tables/revp_indice_reprodutibilidade_marco_mv1.csv`
- `outputs_public/tables/revp_dependencias_local_only_marco_mv1.csv`
- `outputs_public/tables/revp_plano_pacote_externo_reprodutibilidade_mv1.csv`
- `outputs_public/tables/revp_checklist_reproducao_publica_marco_mv1.csv`
- `outputs_public/metrics/revp_hardening_reprodutibilidade_publica_marco_mv1.json`
- `scripts/ground_truth/revp_hardening_reprodutibilidade_publica_marco_mv1.py`
- `tests/test_revp_hardening_reprodutibilidade_publica_marco_mv1.py`
