# SUSC-GT07 - Protocolo de QA Humano para Canários Observacionais

## 1. Escopo do marco

Este marco prepara o **protocolo de QA humano** (QA = Quality Assurance, controle de
qualidade) para os canários selecionados no GT06, **sem executar o QA real** e sem marcar
nenhum canário como aceito/rejeitado. É um pacote **offline e review-only** (uso restrito a
revisão): não busca internet, não consulta STAC nem API, não roda SAR nem GEE, não baixa
Sentinel nem raster, não abre imagem externa, não cria footprint, não cria geometria real,
não executa QA de verdade, não altera o `score_v6`, não cria `score_v7`, não treina modelo
e não promove nada a ground truth nem a `positive_strong`.

## 2. Relação com GT01 a GT06

O GT01 definiu a política; o GT02 aplicou-a; o GT03 montou a fila; o GT04 resolveu datas; o
GT05 preparou geometria; o GT06 selecionou os canários. O GT07 monta a **estrutura de
revisão humana** desses canários.

## 3. Por que QA humano é necessário

QA humano é etapa **obrigatória** antes de qualquer promoção observacional forte: só o
revisor humano confirma compatibilidade fenomenológica, temporal e espacial, qualidade da
fonte e do footprint futuro, exclusões e ausência de leakage temporal. Sem QA, nenhuma
evidência vira referência forte.

## 4. Por que este marco não executa QA real

O QA real exige o **artefato visual/footprint** do evento, que só existirá após a execução
SAR futura. Sem imagem, o revisor não pode concluir a avaliação. Este marco cria a fila, o
formulário, o checklist, a matriz de decisão e os critérios — mas a revisão final fica para
um marco posterior.

## 5. Entradas usadas

Pacote e canários prioritários do GT06 (mais elegibilidade, requisitos e exclusões), lidos
dos outputs versionados sem regeração. Canários na fila: **5**.

## 6. Canários incluídos na fila

| qa_queue_id | event_id | patch_id | cidade/bairro | data | preparacao | status atual |
| --- | --- | --- | --- | --- | --- | --- |
| QAQ_0547 | REC_2022_05_24_30 | S18A_PATCH_0301 | recife/Pina | 2022-05-24 | pronto_para_qa_futuro | prepared_for_future_review |
| QAQ_0548 | REC_2022_05_24_30 | S18A_PATCH_0302 | recife/Imbiribeira | 2022-05-24 | pronto_para_qa_futuro | prepared_for_future_review |
| QAQ_0549 | REC_2022_05_24_30 | S18A_PATCH_0303 | recife/Afogados | 2022-05-24 | pronto_para_qa_futuro | prepared_for_future_review |
| QAQ_0550 | REC_2022_05_24_30 | S18A_PATCH_0304 | recife/Areias | 2022-05-24 | pronto_para_qa_futuro | prepared_for_future_review |
| QAQ_0551 | REC_2022_05_24_30 | S18A_PATCH_0305 | recife/Areias | 2022-05-24 | pronto_para_qa_futuro | prepared_for_future_review |

## 7. Estados de preparação QA

`pronto_para_qa_futuro` (dados mínimos presentes, falta artefato visual/footprint),
`aguardando_footprint_futuro`, `aguardando_geometria_ou_aoi`,
`aguardando_separacao_fenomenologica`, `bloqueado_para_qa` e `fora_da_fila_qa`. Distribuição:

| qa_preparation_status | canários | QA possível agora | pendentes | aceitos agora | rejeitados agora |
| --- | --- | --- | --- | --- | --- |
| pronto_para_qa_futuro | 5 | 0 | 5 | 0 | 0 |

## 8. Formulário de QA futuro

`susc_gt07_formulario_qa_canarios.csv` traz uma linha por canário com todos os campos de
revisão (fenômeno, temporalidade, espacialidade, fonte, exclusões, leakage, incerteza,
decisão) **em branco/placeholder** (`a_preencher`/`nao_verificado`), porque o QA real não
foi executado. O `qa_status_after_review` inicial é `not_reviewed`.

## 9. Checklist por canário

`susc_gt07_checklist_qa_por_canario.csv` detalha os itens por grupo (`fenomeno`,
`temporalidade`, `espacialidade`, `fonte`, `footprint_futuro`, `exclusoes`, `leakage`,
`incerteza`, `decisao`), todos com `current_status=pendente`.

## 10. Critérios de aceitação, rejeição e ambiguidade

`susc_gt07_criterios_aceitacao_rejeicao.csv` define, por critério, o valor que leva a
aceitar, rejeitar ou marcar como ambíguo, sempre bloqueando `positive_strong` e benchmark
futuros até a revisão.

## 11. Matriz de decisão QA

`susc_gt07_matriz_decisao_qa.csv` mapeia condições para as opções de decisão futura
(`decision_option`). **Nenhuma** regra habilita treino, ground truth supervisionado ou
`score_v7` (`can_enable_training=false`, `can_enable_ground_truth=false`,
`can_enable_score_v7=false`). Apenas `aceitar_como_referencia_observacional_forte_review_only`
habilita avaliação review-only futura.

## 12. Bloqueios atuais

Em `susc_gt07_bloqueios_qa.csv`: todos os canários estão bloqueados para conclusão do QA por
**ausência de footprint SAR/artefato visual** (`prevents_qa_completion_now=true`), que
depende de execução futura.

## 13. Como o QA futuro poderá liberar avaliação review-only

Quando o revisor humano aceitar um canário
(`aceitar_como_referencia_observacional_forte_review_only`), ele liberará **apenas**
avaliação review-only da aderência observacional — nunca treino, ground truth ou `score_v7`.

## 14. Por que QA não libera treino nem ground truth supervisionado

Aceitar um canário no QA significa que a evidência é uma referência observacional de
qualidade suficiente para **revisão**, não um rótulo supervisionado. Todas as linhas mantêm
`eligible_for_training=false`, `eligible_for_ground_truth=false` e `score_v7_candidate=false`.

## 15. Tratamento de leakage temporal

O checklist e os critérios exigem confirmar que o **footprint pós-evento não é usado como
feature pré-evento** — o item de `leakage` bloqueia a aceitação se essa checagem falhar.

## 16. Petrópolis e eventos mistos

Canários de Petrópolis ou eventos mistos (0 na fila) entram como
`aguardando_separacao_fenomenologica` e não seguem para aceitação sem separar inundação de
deslizamento.

## 17. Confirmação explícita dos bloqueios

Este marco **não** usou internet, **não** consultou STAC/API, **não** executou SAR
(`sar_executado=0`), **não** rodou GEE, **não** baixou raster nem
Sentinel (`sentinel_baixado=0`), **não** criou footprint
(`footprint_criado=0`), **não** criou geometria real, **não**
executou QA real (`qa_executado=0`), **não** marcou nenhum canário
como aceito/rejeitado (`accepted_now=0`,
`rejected_now=0`), **não** treinou modelo, **não** produziu ground
truth, **não** criou `score_v7`, **não** alterou o `score_v6`
(`score_v6_changed=false`) e **não** promoveu nenhum
canário a `positive_strong`
(`positive_strong_promovidos=0`). Contagens de
controle: `eligible_for_training=true` → 0;
`eligible_for_ground_truth=true` → 0;
`score_v7_candidate=true` → 0.

O REV-P não prevê enchentes operacionalmente: produz análise estrutural review-only com
evidência observacional auditável.

## 18. Próximo passo recomendado

**GT08 - Busca Controlada de Cenas Sentinel-1 sem Download**. Como o QA está bloqueado pela ausência de footprint/artefato
visual, o próximo passo natural é a busca controlada de cenas Sentinel-1 (sem download) para
viabilizar o footprint técnico futuro que permitirá concluir o QA.
