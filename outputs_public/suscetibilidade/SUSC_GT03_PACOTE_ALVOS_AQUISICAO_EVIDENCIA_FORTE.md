# SUSC-GT03 - Pacote de Alvos para Aquisição de Evidência Forte

## 1. Escopo do marco

Este marco transforma o resultado do replay do SUSC-GT02 em uma **fila auditável de
alvos** para aquisição futura de evidência forte. Ele **não** promove nenhum estado,
**não** busca internet, **não** baixa dado novo, **não** roda SAR nem GEE, **não** cria
footprint, **não** altera o `score_v6`, **não** cria `score_v7`, **não** treina modelo e
**não** altera as decisões do GT02 (apenas as lê). Tudo permanece **review-only** (uso
restrito a revisão).

## 2. Relação com GT01 e GT02

O GT01 definiu a política (estados, requisitos, bloqueios). O GT02 aplicou a política aos
artefatos existentes e concluiu que o acervo atual tem 86 `positive_provisional`, 471
`no_data` e **0** `positive_strong`. O GT03 lê exatamente esses registros e diagnostica,
por alvo, **o que falta** para cada um virar candidato forte no futuro.

## 3. Por que o GT03 ainda não calibra o score

Sem referência forte (`positive_strong`) não há base para calibração nem benchmark. O
GT03 apenas organiza a aquisição futura; calibrar agora seria promover evidência
insuficiente a verdade — proibido pela política.

## 4. Por que o GT03 não busca internet nem roda SAR

Este é um marco de **planejamento determinístico**: monta a fila e o plano a partir do
que já existe no repositório. Aquisição real (datas, geometria oficial, footprint SAR,
QA humano) fica para os marcos seguintes, com execução controlada e auditável.

## 5. Entradas usadas

Foram lidos os outputs versionados do GT02 (sem regeração): registro de replay,
decisões por patch, bloqueios, resumo de estados e manifesto. Total de alvos gerados:
**557** (um por registro de replay do GT02).

## 6. Critérios de priorização

O `priority_score` é determinístico, de 0 a 100. Pontua positivamente a presença de
evento (`event_id`), data (`event_date`), cidade/região já trabalhada (Recife, Curitiba,
Petrópolis), fonte forte (`source_authority`), geometria (parcial ou forte), patch,
footprint e fenômeno de inundação, além de já ser `positive_provisional` no GT02 e de
faltarem poucos requisitos. Pontua negativamente fontes fracas (alerta, área de risco,
notícia, contexto municipal, mapa de suscetibilidade), ausência total de
data/geometria/patch, fenômeno misto sem separação e `no_data` sem ação clara. A
priorização **não** promove estado: mesmo um alvo de prioridade alta continua
`review_only` e não treinável.

## 7. Distribuição das prioridades

| priority_class | alvos | score médio | podem virar forte após aquisição | treino permitido |
| --- | --- | --- | --- | --- |
| prioridade_media | 80 | 62.4 | 80 | 0 |
| prioridade_baixa | 1 | 32.0 | 1 | 0 |
| bloqueado_sem_acao_imediata | 476 | 1.95 | 355 | 0 |

Leitura pública: *priority_class* é a classe de prioridade; *podem virar forte após
aquisição* conta alvos que, adquiridos os requisitos, poderiam alcançar `positive_strong`
no futuro; *treino permitido* é **sempre zero** por política.

## 8. Principais requisitos faltantes

| requisito | ocorrências |
| --- | --- |
| geometry_type | 557 |
| qa_status | 557 |
| patch_link_quality | 501 |
| uncertainty_m | 491 |
| patch_id | 481 |
| phenomenon_type | 467 |
| source_authority | 450 |
| event_date | 372 |
| pre_event_window | 1 |
| post_event_window | 1 |

## 9. Trilhas de aquisição propostas

| trilha | alvos |
| --- | --- |
| obter_geometria_oficial | 557 |
| estimar_incerteza_espacial | 557 |
| executar_qa_humano | 557 |
| resolver_patch_link | 501 |
| resolver_data_evento | 372 |
| separar_fenomeno | 13 |
| produzir_footprint_tecnico_sar | 1 |

Cada trilha explica objetivo, entrada e saída esperada em
`susc_gt03_trilhas_aquisicao.csv`, sempre com `no_network_in_this_milestone=true` e
`no_sar_execution_in_this_milestone=true`.

## 10. Top alvos por prioridade

| target_id | classe | score | região | estado GT02 | trilhas |
| --- | --- | --- | --- | --- | --- |
| TGT_0547 | prioridade_media | 70 | Pina | positive_provisional | obter_geometria_oficial;estimar_incerteza_espacial;executar_qa_humano;resolver_patch_link |
| TGT_0548 | prioridade_media | 70 | Imbiribeira | positive_provisional | obter_geometria_oficial;estimar_incerteza_espacial;executar_qa_humano;resolver_patch_link |
| TGT_0549 | prioridade_media | 70 | Afogados | positive_provisional | obter_geometria_oficial;estimar_incerteza_espacial;executar_qa_humano;resolver_patch_link |
| TGT_0550 | prioridade_media | 70 | Areias | positive_provisional | obter_geometria_oficial;estimar_incerteza_espacial;executar_qa_humano;resolver_patch_link |
| TGT_0551 | prioridade_media | 70 | Areias | positive_provisional | obter_geometria_oficial;estimar_incerteza_espacial;executar_qa_humano;resolver_patch_link |
| TGT_0552 | prioridade_media | 70 | Ipsep | positive_provisional | obter_geometria_oficial;estimar_incerteza_espacial;executar_qa_humano;resolver_patch_link |
| TGT_0553 | prioridade_media | 70 | Iputinga | positive_provisional | obter_geometria_oficial;estimar_incerteza_espacial;executar_qa_humano;resolver_patch_link |
| TGT_0554 | prioridade_media | 70 | Iputinga | positive_provisional | obter_geometria_oficial;estimar_incerteza_espacial;executar_qa_humano;resolver_patch_link |
| TGT_0555 | prioridade_media | 70 | Varzea | positive_provisional | obter_geometria_oficial;estimar_incerteza_espacial;executar_qa_humano;resolver_patch_link |
| TGT_0556 | prioridade_media | 70 | Varzea | positive_provisional | obter_geometria_oficial;estimar_incerteza_espacial;executar_qa_humano;resolver_patch_link |

## 11. Alvos bloqueados e por quê

Alvos em `bloqueado_sem_acao_imediata` ou `descartar_como_contexto` estão listados em
`susc_gt03_bloqueios.csv` com o motivo em português: fonte fraca (apenas contexto),
ausência de campos críticos (evento/data/geometria/patch) ou fenômeno misto sem
separação. Nenhum deles é tratado como negativo.

## 12. Petrópolis e eventos mistos

Alvos de Petrópolis (9 no total) com fenômeno de deslizamento ou misto exigem a
trilha `separar_fenomeno` antes de qualquer promoção e **nunca** recebem prioridade alta
sem essa separação. A separação obrigatória cobre flood, flash_flood, river_flood,
landslide, mixed_flood_landslide e insufficient_type.

## 13. Confirmação explícita dos bloqueios

Este marco **não** treinou modelo, **não** produziu ground truth supervisionado,
**não** criou `score_v7`, **não** alterou o `score_v6`
(`score_v6_changed=false`), **não** executou SAR,
**não** usou internet e **não** promoveu nenhum alvo a `positive_strong`
(`positive_strong_promovidos=0`). Contagens de
controle: `eligible_for_training=true` → 0;
`eligible_for_ground_truth=true` → 0;
`score_v7_candidate=true` → 0.

O REV-P não prevê enchentes operacionalmente: produz análise estrutural review-only com
evidência observacional auditável.

## 14. Próximo passo recomendado

**GT04 - Resolvedor de Datas e Janelas Pre/Pos-Evento**. O maior gargalo do acervo é a ausência de datas de
evento na maioria dos alvos promissores; resolver datas e janelas pré/pós-evento
desbloqueia as trilhas seguintes (geometria oficial, footprint SAR e QA humano).

| fonte | alvos | prioridade alta | bloqueados | requisito dominante |
| --- | --- | --- | --- | --- |
| COMPDEC/Defesa Civil PE | 1 | 0 | 1 | event_date |
| DRM-RJ/NADE | 1 | 0 | 1 | event_date |
| Defesa Civil Municipal Petropolis | 1 | 0 | 1 | event_date |
| Defesa Civil Petropolis / DRM-RJ (a confirmar) | 1 | 0 | 1 | event_date |
| International Charter Space and Major Disasters | 1 | 0 | 1 | event_date |
| Prefeitura Curitiba/IPPUC | 1 | 0 | 1 | event_date |
| Programa PE3D Pernambuco 3D | 1 | 0 | 1 | event_date |
| SGB/CPRM | 2 | 0 | 2 | event_date |
| defesa_civil_recife_occurrence | 11 | 0 | 0 | geometry_type |
| institutional_public | 11 | 0 | 11 | event_date |
| international_charter_space_and_major_disasters_official_remote_sensing | 1 | 0 | 0 | geometry_type |
| local_project_official_artifact | 75 | 0 | 6 | event_date |
| not_available | 450 | 0 | 450 | source_authority |
