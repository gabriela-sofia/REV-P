# REV-P v2fg — DINOv2 como camada de governança ativa da API

**Data de execução**: 2026-08-21
**Escopo**: transformar o DINOv2 de evidência auxiliar passiva em camada de governança
ativa da API SUSC-20E, sem tocar no modelo físico de Firth.
**Método/detalhes**: `docs/metodologia_cientifica/revp_v2fg_dinov2_camada_governanca_api.md`.

---

## 1. Arquivos alterados

### Novos

| Arquivo | Conteúdo |
|---|---|
| `scripts/dino/revp_v2fg_dinov2_embedder.py` | `Dinov2Embedder` — `facebook/dinov2-with-registers-base`, 768D, L2, CPU/GPU, fail-closed sem pesos, mock só com opt-in. |
| `scripts/dino/revp_v2fg_dinov2_governance_engine.py` | `Dinov2GovernanceEngine` — cosseno, medoid territorial, gate OOD, bloco de auditoria. |
| `scripts/dino/revp_v2fg_build_dinov2_governance_corpus.py` | Pipeline E2/E3 — valida embeddings reais, calcula medoids, escreve manifesto. |
| `outputs_public/data/linha_causal/susc_20e_api_contrato_inferencia_recife/scripts/dino_governance_bridge.py` | Ponte da API: resolve o embedding de consulta e delega à engine. |
| `datasets/dinov2_governance_medoids_v2fg.json` | Manifesto: medoids + vetores + config do gate + diagnósticos + cross-check. |
| `datasets/dinov2_governance_corpus_v2fg.csv` | Auditoria candidato-a-candidato (134 linhas). |
| `datasets/dinov2_governance_summary_v2fg.csv` | Contagens (candidatos/válidos/bloqueados). |
| `datasets/schemas/dinov2_governance_corpus_v2fg_schema.csv` | Schema do CSV de corpus. |
| `tests/test_revp_v2fg_dinov2_governance_layer.py` | 48 testes. |
| `docs/metodologia_cientifica/revp_v2fg_dinov2_camada_governanca_api.md` | Documentação do fluxo. |

### Modificados

| Arquivo | Mudança |
|---|---|
| `.../scripts/contract_schema.py` | +`DinoGovernance`, `DinoMedoidSimilarity`, `DinoGovernanceAudit`; +`ScoreResponse.dino_governance`; +`ScoreRequest.visual_patch_id` (opcional). |
| `.../scripts/app.py` | Carrega a engine no startup; chama `_governance()` nos dois caminhos de resposta; adiciona linha em `limitations` quando o gate OOD dispara ou há divergência territorial. |
| `.../reports/RELATORIO_susc_20e_api_contrato_inferencia.md` | Seção nova com a validação end-to-end desta rodada. |

Nenhum arquivo do motor físico (`susc_20d_score_engine.py`, `engine_bridge.py`, `gates.py`,
`region_registry.py`) foi tocado.

---

## 2. Decisões (e por quê)

**2.1 — Corpus vem de `datasets/`, não de `local_runs/`.** Os únicos embeddings DINOv2 reais
persistidos e versionados no repositório são os 8 CSVs `datasets/dino_*embedding*.csv`
(backbone `dinov2-with-registers-base`, 768D, `l2_normalized=true`). As tabelas públicas em
`outputs_public/tables/` foram geradas por uma rodada local não persistida. Governança precisa
de artefato conferível em disco, então o corpus é o de `datasets/`.

**2.2 — Limiar OOD derivado do corpus, não arbitrado.** `0,305554` = percentil 5% da
similaridade de cada embedding válido ao medoid regional mais próximo (n=97). A base de
cálculo fica escrita no manifesto e aparece na auditoria de toda resposta. Configurável por
`REVP_DINOV2_OOD_THRESHOLD` ou por argumento.

**2.3 — Não usei `is_fixture_patch()` do v1pg/v1pm.** Aquele helper usa
`^(REC|PET|CWB)_0{3}\d{2}$`, que casa com patches **reais** de número < 100 — entre eles
`REC_00019` (patch de linhagem TP1) e `CUR_00038` (medoid de Curitiba publicado). Aplicá-lo
teria excluído dois patches reais e documentados. Este pipeline usa só a triagem textual por
termo de fixture/mock. Desvio registrado em comentário no próprio código.

**2.4 — Resolução do embedding de consulta em 3 níveis.** `visual_patch_id` explícito → bbox
do patch Sentinel (`REVP_SUSC20D_SENTINEL_DIR`, privado) → nada (`no_visual_evidence`). O
contrato não recebe imagem, então não inventei um endpoint de upload; `visual_patch_id` é a
adição mínima e retrocompatível que torna a camada exercitável no repositório público.

**2.5 — Divergência territorial é observação, não veredito.** A concordância medida no
corpus é 65/97 = 0,6701. O número está no manifesto e é exposto em
`audit.territorial_concordance_in_corpus`, para que quem lê a resposta saiba quanto o sinal
vale.

**2.6 — Governança roda em todos os caminhos de resposta**, inclusive `insufficient_data` e
`region_not_supported`. Nenhum gate fica invisível.

**2.7 — Front-end: não alterado.** Não existe front-end no estado atual do projeto (o próprio
`RELATORIO_susc_20e` registra "Interface web — não iniciado"), então nada foi criado.

---

## 3. Números reais produzidos

| | |
|---|---|
| Fontes reais lidas | 8 CSVs (hash SHA-256 curto de cada um no manifesto) |
| Candidatos | 134 |
| Válidos | 97 |
| Bloqueados | 37 — todos `DUPLICATA_IDENTICA_DE_PATCH_JA_ACEITO` |
| Por região | RECIFE 52, CURITIBA 24, PET 21 |
| Medoids | CURITIBA `CUR_00402` (0,870973), PET `PET_00614` (0,810092), RECIFE `REC_00292` (0,636340), CORPUS `REC_00529` (0,712836) |
| Limiar OOD default | 0,305554 (p5; min 0,2079 / p50 0,8422 / p95 0,9154) |
| Concordância territorial | 65/97 = 0,6701 |
| Cross-check vs. matriz pública | 11 patches em comum, 110 pares, max \|Δ\| 0,11934, média \|Δ\| 0,047523 |

O pipeline é determinístico: reexecutar produz CSV byte-a-byte idêntico; no JSON só
`generated_at` muda (verificado nesta sessão).

---

## 4. Testes executados

**Novos** — `tests/test_revp_v2fg_dinov2_governance_layer.py`: **48 passed**.

Cobertura, item a item do pedido:

| Exigência | Testes |
|---|---|
| dimensionalidade / normalização | `test_embedder_mock_produz_768d_l2_normalizado_e_deterministico`, `test_embedder_sem_l2_nao_normaliza`, `test_l2_normalize_recusa_vetor_degenerado`, `test_validate_embedding` (6 casos) |
| similaridade | `test_cosseno_*` (3) |
| threshold OOD | `test_gate_ood_dispara_para_vetor_ortogonal_ao_corpus`, `test_limiar_ood_e_configuravel_e_muda_o_estado`, `test_limiar_ood_pode_vir_do_ambiente`, `test_limiar_default_vem_do_manifesto_com_base_declarada` |
| seleção de medoid | `test_medoid_e_o_patch_de_maior_similaridade_media_no_recorte` (reexecuta a definição sobre o corpus e confere o vencedor), `test_todo_medoid_persistido_e_768d_l2_normalizado`, `test_medoid_de_recorte_unitario_nao_e_inventado` |
| incompatibilidade territorial | `test_incompatibilidade_territorial_e_explicita`, `test_sem_regiao_solicitada_nao_ha_veredito_territorial`, `test_concordancia_territorial_do_corpus_e_medida_e_registrada` |
| contrato da API | `test_contrato_expoe_bloco_de_governanca_com_campos_obrigatorios`, `test_contrato_declara_que_a_governanca_nunca_soma_ao_score`, `test_contrato_recusa_status_de_governanca_desconhecido`, `test_request_aceita_patch_visual_opcional_sem_quebrar_chamadas_antigas`, `test_resultado_da_engine_valida_contra_o_contrato`, `test_auditoria_nao_emite_url`, `test_api_devolve_governanca_em_resposta_bloqueada`, `test_api_score_fisico_nao_muda_com_a_governanca` |
| sem embeddings/medoids válidos | `test_sem_manifesto_a_governanca_reporta_indisponivel`, `test_manifesto_de_versao_incompativel_e_recusado`, `test_manifesto_com_medoid_invalido_e_recusado`, `test_patch_fora_do_corpus_nao_inventa_evidencia`, `test_sem_patch_e_sem_vetor_o_estado_e_explicito`, `test_embedding_malformado_e_rejeitado_com_estado_proprio` |
| mock nunca vira corpus | `test_embedder_mock_exige_optin_explicito`, `test_embedder_sem_backend_real_nao_inventa_vetor`, `test_corpus_nao_contem_mock_nem_fixture` |

**Regressão** — `pytest tests/ -k "dino or susc_20e" --continue-on-collection-errors`,
comparando o mesmo comando com e sem as mudanças (via `git stash`):

| | antes | depois |
|---|---|---|
| passed | 258 | 306 (+48) |
| failed | 160 | 160 |
| errors | 31 | 31 |
| skipped | 6 | 6 |

A lista nominal de falhas/erros é **idêntica** (`diff` vazio entre os dois runs). As 191
falhas/erros são pré-existentes e independentes desta etapa: dependem de artefatos privados
em `local_runs/` (não versionados) e de módulos removidos nos commits de limpeza
`ec88ad0`/`bf2f79a`. Não foram tocados aqui.

`tests/test_susc_20e_region_registry_schema.py`: 7 passed (inalterado).

**Validação end-to-end da API** (TestClient sobre `app.py`, nesta sessão):

| Caso | status | score | governança |
|---|---|---|---|
| Curitiba + `visual_patch_id=CUR_00402` | `insufficient_data` | null | `in_domain`, cos=1.0, `CURITIBA`, `match` |
| Ponto real de Recife, sem evidência visual | `ok` | 0,7737 CI [0,6692; 0,8718] | `no_visual_evidence` |
| Mesmo ponto + `visual_patch_id=CUR_00402` | `ok` | 0,7737 CI [0,6692; 0,8718] — **idêntico** | `in_domain`, `mismatch` RECIFE≠CURITIBA |

O 0,7737 bate com o valor já documentado no audit do SUSC-20D para o mesmo ponto.

---

## 5. Limitações (reais, não contornadas)

1. **A matriz de similaridade pública não é reproduzível a partir de `datasets/`.**
   Max \|Δ\| = 0,11934, concentrado nos patches de Recife. As tabelas públicas vieram de uma
   rodada em `local_runs/` que não está no repositório. Registrado em
   `cross_check_published`; nenhuma tabela pública foi alterada para casar.
2. **Os medoids de v2fg diferem dos publicados** (`CUR_00402`/`PET_00614`/`REC_00292` vs.
   `CUR_00038`/`PET_00104`/`REC_00205`) por dois motivos somados: recorte diferente (97 vs. 12
   patches) e vetores de rodadas diferentes (item 1). Os dois conjuntos ficam rastreáveis lado
   a lado no manifesto.
3. **`PET_00016` está no inventário público mas não tem vetor em `datasets/`** — fora do
   corpus de governança, sem substituto inventado.
4. **Concordância territorial de 0,6701** — um terço dos patches reais tem medoid mais
   próximo de outra região. `suggested_region` não é classificador territorial.
5. **A API não recebe imagem.** Sem `REVP_SUSC20D_SENTINEL_DIR` configurado e sem
   `visual_patch_id`, toda requisição cai em `no_visual_evidence`. `Dinov2Embedder` está
   pronto, mas o caminho imagem→embedding→governança em tempo de request não existe no
   contrato atual.
6. **`Dinov2Embedder` não foi executado com pesos reais nesta sessão** — não há pesos DINOv2
   locais neste ambiente e `REVP_DINO_ALLOW_DOWNLOAD` é `false` por padrão em todo o pipeline
   DINO do projeto. O backend real está implementado nos mesmos moldes do executor v1qj já
   validado, mas **o que foi exercitado em teste aqui foi o caminho mock (marcado como tal) e
   o caminho fail-closed**. Corpus e medoids não dependem disso: vêm dos CSVs reais.
7. **Só 1 patch (`REC_00019`) tem geometria pública real** — a resolução geometria→patch
   depende do diretório Sentinel privado.
8. **191 falhas/erros pré-existentes** na suíte `dino`/`susc_20e` continuam de pé; não são
   desta etapa e não foram corrigidos aqui.

---

## 6. Próximos passos (não executados)

1. Rodar `Dinov2Embedder` com pesos locais reais (`REVP_DINO_MODEL_PATH`) e confirmar que um
   embedding recém-extraído de um patch já no corpus reproduz o vetor persistido — fecha a
   lacuna 6.
2. Reconstruir a matriz/medoids públicos a partir de `datasets/` **ou** persistir os vetores
   da rodada `local_runs/v1ge` — hoje as duas fontes divergem e nenhuma é canônica (lacunas
   1 e 2). Decisão metodológica, precisa de revisão humana.
3. Expandir a geometria pública de patch para além de `REC_00019`, para que a resolução
   geometria→patch funcione sem o diretório privado (lacuna 7).
4. Investigar a concordância territorial de 0,6701: é limite do backbone auto-supervisionado
   para essa tarefa, ou efeito de composição do corpus (Recife com 52 patches heterogêneos,
   cos médio intra-região de só 0,636)?
5. Se um front-end vier a existir, consumir `dino_governance.status`,
   `territorial_match` e `audit` — os campos já existem e são estáveis.
