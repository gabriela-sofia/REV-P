# SUSC-GT04 - Resolvedor de Datas e Janelas Pré/Pós-Evento

## 1. Escopo do marco

Este marco resolve, infere ou classifica **datas de evento** e gera **janelas
pré-evento e pós-evento** para aquisição futura, usando apenas os artefatos que já
existem no repositório. É um resolvedor **review-only** (uso restrito a revisão): não
busca internet, não baixa dado novo, não roda SAR nem GEE, não cria footprint, não
altera o `score_v6`, não cria `score_v7`, não treina modelo e não promove nada a ground
truth nem a `positive_strong`. As janelas geradas são apenas metadata; nenhuma é usada
para executar SAR agora.

## 2. Relação com GT01, GT02 e GT03

O GT01 definiu a política, o GT02 aplicou-a às evidências existentes e o GT03 montou o
pacote de alvos, apontando a **ausência de `event_date`** (data do evento) como maior
gargalo. O GT04 ataca exatamente esse gargalo, lendo os alvos do GT03 e o replay do GT02.

## 3. Por que resolver data vem antes de SAR, geometria e QA

Sem a data do evento não há janela temporal pré/pós-evento; sem janela não é possível
buscar geometria oficial por período, preparar canários SAR nem contextualizar o QA
humano. A data é o requisito **upstream** que desbloqueia os demais.

## 4. Entradas usadas

Alvos e requisitos do GT03 e o replay do GT02 (campos originais), lidos dos outputs
versionados sem regeração. Total de alvos processados: **557**.

## 5. Hierarquia de fontes de data

Ordem de confiança: (1) `event_date` explícito; (2) `event_date_candidate`; (3)
`date`/`source_date` com sinal de evento; (4) data inferida de
`event_id`/`candidate_event_id`; (5) data em nome de arquivo/`source_artifact`; (6) data
de publicação; (7) ano/mês parcial; (8) `unknown` (desconhecida). Cada resolução
registra o campo-fonte, o valor-fonte, o método, a precisão e a confiança.

## 6. Categorias de precisão temporal

`exact_day` (dia exato de campo explícito), `inferred_from_event_id` (dia inferido do
identificador — marcado como inferido, não confirmado), `publication_date_only` (apenas
data de publicação, **não** libera SAR), `date_range` (intervalo início–fim),
`month_only`, `year_only`, `unknown` e `invalid_or_conflicting` (datas inválidas ou
conflitantes).

## 7. Política de janelas pré/pós-evento

Para `exact_day`/`inferred_from_event_id`: pré-evento = [data−30 dias, data−1] e
pós-evento = [data, data+7 dias]. Para `date_range`: pré = [início−30,
início−1] e pós = [início, fim+7], com incerteza de intervalo. Para
`publication_date_only`: sem janela SAR, apenas contexto documental. Para
`month_only`/`year_only`/`unknown`/`invalid_or_conflicting`: sem janela operacional,
bloqueada por resolução insuficiente.

## 8-13. Distribuição da resolução

- Datas resolvidas exatas (`exact_day`): **185**.
- Datas inferidas (`inferred_from_event_id`): **109**.
- Intervalos (`date_range`): **0**.
- Apenas publicação (`publication_date_only`): **0**.
- Desconhecidas (`unknown`): **263**.
- Inválidas/conflitantes (`invalid_or_conflicting`): **0**.
- Janelas geradas (utilizáveis): **294**.
- Janelas bloqueadas: **263**.

| temporal_precision | alvos | confiança média | janelas utilizáveis | janelas bloqueadas |
| --- | --- | --- | --- | --- |
| exact_day | 185 | 100.0 | 185 | 0 |
| inferred_from_event_id | 109 | 75.0 | 109 | 0 |
| unknown | 263 | 0.0 | 0 | 263 |

## 14. Exemplos de resoluções

- **Dia exato** (`exact_day`): alvo `TGT_0088`, event_id `S17C_E_SUSC13A_00001`, data resolvida `2022-05-24`, confianca 100, método `campo_event_date_explicito`.
- **Inferida do identificador** (`inferred_from_event_id`): alvo `TGT_0076`, event_id `REC_2022_05_24_30`, data resolvida `2022-05-24`, confianca 75, método `inferencia_de_event_id`.
- **Apenas publicação** (`publication_date_only`): nenhum alvo neste nivel.
- **Desconhecida** (`unknown`): alvo `TGT_0001`, event_id `S16ALOCG_00070`, data resolvida `not_available`, confianca 0, método `nenhuma_data_util`.
- **Inválida/conflitante** (`invalid_or_conflicting`): nenhum alvo neste nivel.

## 15. Confirmação explícita dos bloqueios

Este marco **não** usou internet, **não** executou SAR nem GEE, **não** baixou raster,
**não** criou footprint, **não** treinou modelo, **não** produziu ground truth
supervisionado, **não** criou `score_v7`, **não** alterou o `score_v6`
(`score_v6_changed=false`) e **não** promoveu
nenhum alvo a `positive_strong`
(`positive_strong_promovidos=0`). Contagens de
controle: `eligible_for_training=true` → 0;
`eligible_for_ground_truth=true` → 0;
`score_v7_candidate=true` → 0.

O REV-P não prevê enchentes operacionalmente: produz análise estrutural review-only com
evidência observacional auditável.

## 16. Próximo passo recomendado

**GT05 - Pacote de Aquisicao de Geometria Oficial**. Com **294** alvos liberados
por terem data resolvida em nível operacional, o próximo gargalo passa a ser a geometria
oficial: a maioria dos alvos datados ainda precisa de ponto/polígono/bbox para virar
candidato forte no futuro.
