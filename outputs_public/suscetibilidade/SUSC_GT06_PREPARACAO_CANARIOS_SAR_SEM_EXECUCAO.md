# SUSC-GT06 - Preparação de Canários SAR sem Execução

## 1. Escopo do marco

Este marco seleciona e prepara, **sem executar nada**, um conjunto pequeno de alvos
candidatos a **canário SAR** (Sentinel-1) para footprint técnico **futuro**. É um pacote
**offline e review-only** (uso restrito a revisão): não busca internet, não consulta STAC
nem API, não roda SAR nem GEE, não baixa Sentinel-1/2 nem raster, não cria footprint, não
cria geometria real, não geocodifica, não altera o `score_v6`, não cria `score_v7`, não
treina modelo e não promove nada a ground truth nem a `positive_strong`.

## 2. Relação com GT01, GT02, GT03, GT04 e GT05

O GT01 definiu a política; o GT02 aplicou-a; o GT03 montou a fila; o GT04 resolveu datas e
janelas; o GT05 preparou a geometria e apontou os alvos datados sem geometria oficial como
o gargalo. O GT06 usa esses alvos para escolher **poucos** canários SAR prioritários.

## 3. Por que SAR entra como canário, não como base central

O SAR não é a base central do REV-P: a generalização continua sustentada por **features
escaláveis por patch**. O canário SAR serve apenas para **testar a aderência observacional
em poucos eventos prioritários** — não é ground truth massivo, não é base de treino e não
altera o score. O footprint pós-evento, quando existir no futuro, será **referência de
avaliação** review-only, **nunca feature pré-evento**.

## 4. Por que este marco não executa SAR

A execução (busca de cena Sentinel-1, pré-processamento, máscara de água, filtros, QA) é
custosa e depende de rede e dados externos. Este marco apenas **prepara** a fila e o plano,
de forma determinística e auditável, deixando a execução para um marco futuro controlado.

## 5. Entradas usadas

Alvos e diagnóstico geométrico do GT05 e janelas do GT04 (por `target_id`), lidos dos
outputs versionados sem regeração. Total de alvos: **557**.

## 6. Critérios de elegibilidade SAR

Sete classes: `elegivel_canario_sar_prioritario` (datado, com janela, patch/AOI e fenômeno
de inundação compatível), `elegivel_canario_sar_secundario` (datado com lacunas),
`requer_geometria_antes_do_sar` (datado sem patch/AOI), `requer_separacao_fenomenologica`
(fenômeno misto/deslizamento), `bloqueado_temporalmente` (sem data/janela),
`bloqueado_por_contexto_fraco` (fonte só contextual) e `nao_elegivel_para_sar`.

## 7. Critérios de pontuação e seleção

O `sar_canary_priority_score` (0 a 100) pontua data exata/inferida, janela gerada,
necessidade de footprint, prioridade geométrica alta, patch, fonte forte, fenômeno de
inundação e cidade/região no escopo; penaliza precisão `unknown`, ausência de janela/patch,
fenômeno misto e fonte fraca. São selecionados no **máximo 5** canários
prioritários, por maior score, com diversidade de bairros quando possível.

## 8. Distribuição das elegibilidades

| sar_canary_eligibility | alvos | score médio | selecionados | podem preparar SAR |
| --- | --- | --- | --- | --- |
| elegivel_canario_sar_prioritario | 13 | 78.15 | 5 | 13 |
| elegivel_canario_sar_secundario | 2 | 26.0 | 0 | 2 |
| requer_geometria_antes_do_sar | 271 | 31.42 | 0 | 271 |
| requer_separacao_fenomenologica | 8 | 27.0 | 0 | 0 |
| bloqueado_temporalmente | 263 | 4.2 | 0 | 0 |

## 9. Quantos podem preparar SAR futuro

**286** alvos podem preparar SAR futuro (elegíveis ou
que precisam apenas de geometria/AOI). Nenhum pode executar SAR agora
(`can_execute_sar_now_true_count=0`).

## 10-11. Canários prioritários selecionados

Elegíveis prioritários: **13**; selecionados:
**5** (limite 5).

| rank | event_id | patch_id | cidade | bairro/regiao | data | score |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | REC_2022_05_24_30 | S18A_PATCH_0301 | recife | Pina | 2022-05-24 | 84 |
| 2 | REC_2022_05_24_30 | S18A_PATCH_0302 | recife | Imbiribeira | 2022-05-24 | 84 |
| 3 | REC_2022_05_24_30 | S18A_PATCH_0303 | recife | Afogados | 2022-05-24 | 84 |
| 4 | REC_2022_05_24_30 | S18A_PATCH_0304 | recife | Areias | 2022-05-24 | 84 |
| 5 | REC_2022_05_24_30 | S18A_PATCH_0305 | recife | Areias | 2022-05-24 | 84 |

## 12. Principais bloqueios para SAR futuro

Em `susc_gt06_bloqueios_canarios_sar.csv`: ausência de data/janela (bloqueio temporal),
fenômeno misto/deslizamento sem separação, fonte apenas contextual e ausência de patch/AOI.

## 13. Critérios de exclusão para execução futura

Em `susc_gt06_criterios_exclusao_sar.csv`: janela ausente, incompatibilidade de fenômeno,
evento misto, ausência de patch/AOI, fonte contextual e, para a execução futura, confusão
com água permanente, sombra/layover urbano, ambiguidade por vegetação, inconsistência de
declividade/HAND e ausência de QA.

## 14. Plano de execução SAR futura (sem execução agora)

Em `susc_gt06_plano_execucao_sar_futura.csv`, dez passos para cada canário prioritário:
confirmar AOI/patch; buscar cenas Sentinel-1 futuras; selecionar janela pré-evento;
selecionar janela pós-evento; pré-processar SAR; aplicar máscara de água permanente;
aplicar filtros HAND/slope; tratar ambiguidade urbana; gerar polígono candidato (referência
de avaliação); executar QA humano. Todos com `no_execution_now=true`.

## 15. Petrópolis e eventos mistos

Alvos de Petrópolis ou mistos (21 relacionados) entram como
`requer_separacao_fenomenologica` e **nunca** são selecionados como canário sem separar o
fenômeno (inundação vs deslizamento).

## 16. Confirmação explícita dos bloqueios

Este marco **não** usou internet, **não** consultou STAC/API, **não** executou SAR
(`sar_executado=0`), **não** rodou GEE, **não** baixou raster nem
Sentinel (`sentinel_baixado=0`), **não** criou footprint
(`footprint_criado=0`), **não** criou geometria real, **não**
treinou modelo, **não** produziu ground truth, **não** criou `score_v7`, **não** alterou o
`score_v6` (`score_v6_changed=false`) e **não**
promoveu nenhum alvo a `positive_strong`
(`positive_strong_promovidos=0`). Contagens de
controle: `eligible_for_training=true` → 0;
`eligible_for_ground_truth=true` → 0;
`score_v7_candidate=true` → 0.

O REV-P não prevê enchentes operacionalmente: produz análise estrutural review-only com
evidência observacional auditável.

## 17. Próximo passo recomendado

**GT07 - Protocolo de QA Humano para Canarios**. Com **5** canários
selecionados, o próximo passo natural é o protocolo de QA humano desses poucos canários
(ou a busca controlada de cenas Sentinel-1), sempre review-only e sem treino.
