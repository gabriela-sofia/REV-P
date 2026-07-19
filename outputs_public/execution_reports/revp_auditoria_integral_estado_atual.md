# REV-P - Auditoria integral do estado atual apos curadoria PT-BR

Data: 2026-06-18
Pasta auditada: `C:\Users\gabriela\Documents\REV-P`
Branch auditada: `curadoria/repositorio-publico-ptbr`
Escopo: auditoria documental e de continuidade metodologica, sem implementacao de sprint nova, sem stage, commit, push, merge, limpeza ou troca de branch.

## 1. Estado Git observado

Comandos iniciais executados:

```powershell
git branch --show-current
git branch --all
git status --short
git diff --cached --name-only
git log --oneline -15
git remote -v
git worktree list
```

Resultado consolidado:

- Branch atual: `curadoria/repositorio-publico-ptbr`.
- Staged area inicial: vazia.
- Arquivos rastreados modificados no inicio: nenhum.
- Arquivos untracked no inicio: nenhum.
- Remote: `origin https://github.com/gabriela-sofia/REV-P.git`.
- Worktrees ativos:
  - `C:/Users/gabriela/Documents/REV-P` em `curadoria/repositorio-publico-ptbr`, commit `67d8cfd`.
  - `C:/Users/gabriela/.codex/worktrees/0475/REV-P` em `public-docs-consolidation`, commit `4c7bad4`.
  - `C:/Users/gabriela/Documents/REV-P-v2cx-v2dd-clean-20260617_154005` em `feat/v2de-v2dk-trainability-mv1`, commit `8b8c278`.

Ultimos commits observados:

```text
67d8cfd docs: consolida curadoria publica e linha metodologica PT-BR
afff542 analise: audita recuperabilidade da base original v2dz-v2ef
58ed64a curadoria: organiza repositorio publico em portugues
af729bf curadoria: organiza repositorio publico em portugues
faa9156 chore: curadoria da camada publica do repositorio
efb1668 Merge pull request #19 from gabriela-sofia/feat/v1gu-v1gv-multimodal-gt-scaffold
deec0fe data: prepara aquisicao e QA geoespacial de evidencias externas
ca0765b analysis: prepara priorizacao TP2 e replay bloqueavel sem ground truth operacional
285ebd7 Merge pull request #18 from gabriela-sofia/feat/v1gu-v1gv-multimodal-gt-scaffold
08c5764 data: registra fontes externas reais e triagem conservadora de evidencias
6e34602 Merge pull request #17 from gabriela-sofia/feat/v1gu-v1gv-multimodal-gt-scaffold
edda0db docs: inventaria candidatos TP2 sem promover ground truth operacional
0ea189d Merge pull request #16 from gabriela-sofia/feat/v1gu-v1gv-multimodal-gt-scaffold
b439b44 docs: consolida cadeia de ground truth e estado metodologico do REV-P
4bfedd4 Merge pull request #15 from gabriela-sofia/feat/v1gu-v1gv-multimodal-gt-scaffold
```

## 2. Arquivos ligados a curadoria PT-BR

Arquivos centrais observados:

- `README.md`.
- `docs/estado_metodologico_revp.md`.
- `docs/metodologia_cientifica/revp_narrativa_cientifica_consolidada.md`.
- `docs/metodologia_cientifica/revp_indice_etapas.md`.
- `docs/metodologia_cientifica/revp_guia_estilo_nomenclatura.md`.
- `outputs_public/README.md`.
- `outputs_public/execution_reports/final_delivery_artifact_index.md`.
- `outputs_public/execution_reports/final_guardrails_report.md`.
- `outputs_public/execution_reports/revp_auditoria_curadoria_repositorio_publico.md`.
- `outputs_public/execution_reports/revp_relatorio_limpeza_linguagem.md`.
- `outputs_public/execution_reports/revp_relatorio_validacao_curadoria_publica.md`.
- `outputs_public/tables/revp_indice_etapas_publicas.csv`.
- `outputs_public/tables/revp_lista_arquivos_exportacao_publica.csv`.
- `tests/test_v1uc_v1ue_public_terminology.py`.

Diagnostico: a camada de entrada publica esta em portugues tecnico suficiente para defesa e leitura academica. Persistem nomes internos em ingles por compatibilidade com scripts, testes, manifests e historico de microversoes.

## 3. Arquivos ligados a ground truth

Arquivos estruturais observados:

- `docs/estado_metodologico_revp.md`.
- `docs/metodologia_cientifica/revp_indice_etapas.md`.
- `outputs_public/model/NO_OPERATIONAL_TRAINED_MODEL.md`.
- `outputs_public/logs_summary/revp_v2ez_to_v2ff_resumo_limites.csv`.
- `outputs_public/tables/revp_painel_perda_recuperacao_base_original_v2ff.csv`.
- `outputs_public/tables/revp_registro_decisao_recuperacao_forense_v2fe.csv`.
- `outputs_public/tables/revp_proximas_acoes_base_original_v2ff.csv`.
- `scripts/ground_truth/revp_v2ez_to_v2ff_orquestrador.py`.
- `scripts/ground_truth/revp_v2ez_indice_busca_forense_repositorio.py`.
- `scripts/ground_truth/revp_v2fa_extrator_artefatos_diff_patch.py`.
- `scripts/ground_truth/revp_v2fb_inspetor_objetos_git_reflog.py`.
- `scripts/ground_truth/revp_v2fc_planejador_busca_backups_locais.py`.
- `scripts/ground_truth/revp_v2fd_validador_candidatos_forenses.py`.
- `scripts/ground_truth/revp_v2fe_registro_decisao_recuperacao_forense.py`.
- `scripts/ground_truth/revp_v2ff_painel_perda_recuperacao_base_original.py`.
- `tests/test_revp_v2ez_to_v2ff_orquestrador.py`.
- `tests/test_revp_v2ff_painel_perda_recuperacao_base_original.py`.

## 4. Confirmacoes metodologicas

Estado confirmado nos documentos e registries:

- Ground truth operacional continua ausente: `ground_truth_operational_status = ABSENT`.
- Labels formais continuam ausentes.
- Negativos formais continuam ausentes.
- Treino supervisionado continua bloqueado: `training_ready = false` ou `training_ready = BLOCKED`, conforme contrato do artefato.
- DINOv2 e encoder visual congelado, exploratorio, sem ajuste, sem treino supervisionado e sem funcao de detector.
- MV1 depende de ground truth operacional; nao deve ser aberto antes de geometria oficial, rotulos formais e negativos formais.
- Fallback nao substitui a base original.
- Referencia textual nao recupera a base original.
- `ORIGINAL_BASE_REQUIRES_MANUAL_RESTORE` e decisao de recuperabilidade, nao recuperacao efetiva.

## 5. Pipeline real do REV-P hoje

1. Corpus territorial: 59 patches em Recife, Petropolis e Curitiba.
2. Inventario Sentinel-first: 128 assets candidatos.
3. DINOv2 congelado: 12 embeddings reais, 4 por regiao, 768 dimensoes, usados para similaridade, k-NN, PCA, medoids e outliers.
4. Protocolo C: organiza evidencia externa candidata, temporal e contextual, sem promover ground truth operacional.
5. Busca por ground truth: tentativas documentadas de geometria oficial, separacao de fenomeno, CRS e sobreposicao patch-evento.
6. Cadeia TP2 e evidencias externas: candidatos, digitalizacao manual, QA geoespacial e replay bloqueavel.
7. Auditoria da base `v2dz-v2ef`: recuperabilidade forense, sem restauracao automatica e sem fallback.
8. Curadoria publica PT-BR: README, narrativa, indice de etapas, guia editorial, relatorios e tabelas publicas.

## 6. Linha do tempo tecnica correta

- `v1f-v1g`: corpus territorial, manifesto Sentinel e DINOv2.
- `v1i-v2a`: Protocolo C e busca de evidencia externa.
- `v2an-v2bm`: validacao regional de referencias, sem ground truth.
- `v2bn-v2ca`: dry-run e cadeias de geometria/replay bloqueadas por ausencia de evidencia operacional.
- `v2ci-v2cm`: inventario, priorizacao TP2, digitalizacao, validacao de geometria e replay bloqueavel.
- `v2cn-v2cw`: evidencia externa multirregional, licenca, QA e prontidao regional.
- `v2dz-v2ef`: base candidata anterior referenciada, nao localizada.
- `v2es-v2ey`: tentativa de recuperacao controlada, bloqueada.
- `v2ez-v2ff`: auditoria forense, decisao `ORIGINAL_BASE_REQUIRES_MANUAL_RESTORE`.
- `curadoria/repositorio-publico-ptbr`: consolidacao publica em portugues brasileiro.

## 7. Riscos principais de interpretacao pela banca

- Confundir evidencia contextual ou referencia candidata com ground truth operacional.
- Ler DINOv2 como classificador, detector ou preditor.
- Interpretar pontuacoes do Protocolo C como validacao operacional de evento observado.
- Interpretar `ORIGINAL_BASE_REQUIRES_MANUAL_RESTORE` como base restaurada.
- Interpretar fallback, referencia textual ou candidato Git como substituto da base original.
- Confundir nomes internos em ingles com narrativa publica obrigatoria.
- Ler checklists historicos ou logs como resultado final.

## 8. Lacunas criticas de ground truth

- Geometria oficial de evento observado ainda insuficiente para fechar patch-evento.
- Negativos formais ausentes; ausencia de registro nao e negativo formal.
- Labels formais binarios ausentes.
- Separacao de fenomeno em Petropolis 2022 ainda exige fonte oficial/curadoria humana.
- CRS, proveniencia, hash e licenca seguem como gates fail-closed.
- Base `v2dz-v2ef` nao foi recuperada automaticamente.

## 9. Proximo passo tecnico correto

O proximo passo tecnico correto nao e iniciar sprint nova de modelo nem treinar classificador. O caminho correto e uma acao manual controlada: restaurar ou recuperar a base original `v2dz-v2ef` por diff, objeto Git, backup local ou fonte externa equivalente, e submeter qualquer candidato restaurado a validacao de schema, contagem, hash, proveniencia e revisao humana. Em paralelo, a frente cientifica deve continuar a obtencao de geometria oficial de evento e negativos formais; sem isso, MV1 e treino supervisionado permanecem bloqueados.

## 10. Checklist de correcoes recomendadas

- Manter arquivos ancora PT-BR como fonte principal de leitura publica.
- Padronizar apenas texto narrativo que ainda esteja misto ou robotico.
- Nao renomear scripts, testes, datasets, enums ou status tecnicos.
- Acrescentar notas explicativas em portugues quando status internos forem exibidos em tabelas.
- Verificar encoding exibido por ferramentas antes de inferir problema real no arquivo.
- Preservar explicitamente `ground_truth_operational_status = ABSENT`.
- Nao tratar fallback, referencia textual ou candidato Git como recuperacao de base.
- Nao abrir MV1 enquanto ground truth, labels e negativos formais estiverem ausentes.
