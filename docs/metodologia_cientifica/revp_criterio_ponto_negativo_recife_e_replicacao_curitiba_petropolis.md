# Critério de ponto negativo: o que Recife realmente usou, e como replicar em Curitiba e Petrópolis

**Escopo**: documentação de metodologia. Nenhum negativo foi amostrado para Curitiba ou
Petrópolis nesta rodada, e nem deve ser: com N=1 positivo por região não há base para o
pareamento que o próprio critério de Recife exige (seção 3). Ver
`revp_linhagem_coleta_curitiba_petropolis_paridade_recife.md`.

---

## 1. A regra que governa tudo: ausência de registro não é negativo

O critério de Recife foi construído em cima de uma proibição explícita, escrita no script que
gera o pool de candidatos:

> `local_runs/treino_exploratorio_diagnostico_v2/_scratch/build_recife_negative_candidates.py:11-12`
> — "Absence-of-flood-record is NOT used as negative evidence (explicitly forbidden per
> docs/recife_external_seced_geocoding_plan_v1.md)."

Ou seja: um ponto **não** é negativo por não aparecer na lista de enchentes. Um ponto é negativo
quando existe **registro positivo de outra coisa** — uma ocorrência real, datada e localizada, de
um tipo de evento causalmente independente da chuva, no mesmo sistema municipal que registra as
enchentes. O negativo é evidência de presença de outro fenômeno, nunca evidência de ausência.

Essa é a diferença que faz o critério ser replicável: não depende de cobertura completa do
cadastro de enchentes (que nunca existe), depende de existir um cadastro paralelo de ocorrências
não-hidrológicas.

## 2. As três coortes de negativos que compõem o v12

O `dataset_eventos_features_v12_final.csv` tem 278 linhas: 154 positivos e **124 negativos**, com
`negative_source_type = real_clean_sedec_negative` em 100% deles. Os 124 vêm de três coortes com
prefixo de `point_id` distinto:

| Coorte | n | Fonte | Script |
|---|---:|---|---|
| `REC_NEGV3_*` | 22 | SEDEC/Defesa Civil, ocorrências não-hidrológicas | `build_recife_negative_candidates.py` + `pipeline_recife_v2.py` |
| `NEWNEG_ILUM_*` | 14 | EMLURB 156, iluminação pública (snapshot de 3 dias) | `build_v7_new_points_features.py:49-54` |
| `NEWNEG_V9_BAIRROMATCH_*` | 88 | EMLURB 156, arquivo anual 2015-2024, reselecionado por bairro | `build_v9_bairro_matched_new_negatives.py:30-34` |

### 2.1 Coorte SEDEC (22 pontos) — filtro por categoria de ocorrência

Duas etapas encadeadas.

**Etapa A — seleção positiva de categorias não-hidrológicas.** O script varre as CSVs anuais da
Defesa Civil (886.264 linhas, 2014-2025) e rotula a ocorrência por termo textual real dos campos
`Ocorrencia`/`Solicitacao`/`Tipo_da_Acao`
(`build_recife_negative_candidates.py:35-42`, `NON_HYDRO_CATEGORY_LABELS`):

```
incendio, arvore_em_risco, muro_com_danos, desabamento_estrutural,
deslizamento_barreira, invasao_terreno
```

Exige ainda qualidade de endereço equivalente a `geocoding_ready_precise` (linha 144): número
provável presente, sem ofuscação de caracteres. Resultado: 354 candidatos.

**Etapa B — exclusão das categorias que ainda dependem de chuva.** Das 6 categorias, três são
descartadas na montagem do alvo limpo
(`local_runs/recife_modelo_oficial_v2/pipeline_recife_v2.py:59`, aplicado na linha 116):

```python
EXCLUDE_NEG_CATEGORIES = {"arvore_em_risco", "deslizamento_barreira", "desabamento_estrutural"}
clean_negatives = negatives[~negatives["classe"].isin(EXCLUDE_NEG_CATEGORIES)].copy()
```

O motivo é causal, não estatístico: queda de árvore, deslizamento/barreira e desabamento
estrutural são desencadeados por chuva forte — usá-los como "não-enchente" contaminaria o
contraste com o próprio fenômeno que se quer separar. Sobram `incendio`, `muro_com_danos` e
`invasao_terreno`.

**Data real, não data sintética.** `clean_target()` recupera a data de ocorrência real do campo
`Data` da fonte para esses 22 registros (`pipeline_recife_v2.py:119-126`), marcando
`reference_date_is_synthetic = False`. O docstring da função é explícito sobre por que isso
importa: esses registros **têm** data real na fonte, ao contrário de um "não-evento" verdadeiro
(linhas 99-102).

**Geocodificação.** Só entram candidatos com `confidence_tier` em `("strong", "medium")`
(`build_recife_v3_dataset.py:36`).

### 2.2 Coorte EMLURB 156 (14 + 88 pontos) — chamado de serviço elétrico

O v7 percebeu que o cadastro do 156 (central de atendimento municipal) traz **coordenada real
fornecida pela própria fonte**, sem geocodificação, e filtrou por grupo de serviço
(`build_v7_new_points_features.py:49`):

```python
ilum = df156[df156["GRUPOSERVICO_DESCRICAO"] == "ILUMINACAO PUBLICA"].copy()
```

O v8 ampliou para o arquivo anual completo (2012-2026, ~90-110 mil registros/ano), com os grupos
`ILUMINAÇÃO PÚBLICA`, `ILUMINAÇÃO RELUZ`, `ILUMINAÇÃO PROVISÓRIA`, `LUMINÁRIAS`
(`build_v8_new_negatives_base.py:3-9`). A justificativa é a mesma da etapa B do SEDEC:
manutenção elétrica/luminária é independente da chuva. Os grupos `DRENAGEM`, `PAVIMENTAÇÃO` e
`ARBORIZAÇÃO` do mesmo cadastro foram **excluídos** exatamente por poderem ter ligação com chuva
(`task_a_novas_categorias_report.md`, seção 3).

Amostragem determinística: 8 registros/ano × 10 anos (2015-2024), ordenação por hash estável, sem
aleatoriedade dependente de semente; 2014 excluído porque a coluna de coordenada não existe
naquele ano. Checagem de não-sobreposição com os negativos já existentes em raio de ~30 m
(0,0003°): 0 colisões.

### 2.3 A correção do v9: pareamento por bairro

O v8 entregou 80 negativos espalhados por 39 bairros, dos quais **apenas 3 coincidiam** com os 29
bairros onde havia positivo de enchente — contra 14/27 dos negativos originais do v7. Isso inflou
o AUC por confundimento geográfico (0,7032 no v8), não por sinal físico: o modelo estava
aprendendo "que parte da cidade é essa", não "esse lugar alaga".

A correção não relocou nem fabricou ponto nenhum. Do **mesmo** pool bruto de 200 candidatos já
baixados, reselecionou os que caem em bairro onde existe positivo real
(`build_v9_bairro_matched_new_negatives.py:30-34`):

```python
pos_bairros = set(dataset_v8.loc[dataset_v8.label == 1, "neighborhood"]
                  .dropna().str.upper().str.strip())
raw["bairro_norm"] = raw["bairro"].str.upper().str.strip()
raw["bairro_overlaps_positive_geography"] = raw["bairro_norm"].isin(pos_bairros)
matched = raw[raw["bairro_overlaps_positive_geography"]].copy()
```

88 dos 200 satisfizeram o critério; todos os 88 foram usados. LOO-AUC caiu de 0,7032 para 0,6578
— e essa queda é o resultado honesto, não uma regressão.

**Estado geográfico do v12 (contagem real sobre o dataset final)**: 35 bairros com positivo,
36 bairros com negativo, **23 bairros em comum**.

## 3. O critério, destilado em cinco condições

Um ponto só é negativo válido no padrão de Recife se satisfizer, simultaneamente:

1. **Presença de outro fenômeno**, não ausência de enchente. Registro real de ocorrência em
   cadastro municipal.
2. **Categoria causalmente independente de chuva.** Incêndio, muro com danos, invasão de terreno,
   manutenção de iluminação. Excluídos: árvore em risco, deslizamento/barreira, desabamento,
   drenagem, pavimentação, arborização.
3. **Coordenada de qualidade auditável.** Coordenada fornecida pela fonte, ou geocodificação com
   `confidence_tier` strong/medium e endereço sem ofuscação.
4. **Data real da ocorrência**, recuperada da fonte — nunca data sintética.
5. **Pareamento geográfico com os positivos**: o bairro do negativo tem que ser bairro onde existe
   positivo real. Sem isso o modelo separa geografia, não risco.

Rota alternativa existente e mais fraca: pseudo-ausência espaço-temporal
(`pipeline_recife_v2.py:60-61,161-162`) — ponto a ≥ 300 m de qualquer positivo, em data de chuva
abaixo do percentil 25. Foi usada só na análise **secundária** do v7, nunca na primária, e é
justamente o tipo de negativo cuja fragilidade o v7 documentou. Só deve ser considerada se a
rota de cadastro municipal falhar, e sempre rotulada como secundária.

## 4. Como isso se aplicaria a Curitiba e Petrópolis (não executado)

A condição 5 é o que trava a execução hoje: com **1 positivo por região**, o pareamento por bairro
produziria negativos em exatamente 1 bairro (Juvevê em Curitiba, Valparaíso em Petrópolis).
Amostrar agora seria trabalho descartável — todo o conjunto teria de ser refeito assim que o
segundo, o terceiro e o vigésimo positivo entrarem, porque a geografia-alvo muda a cada positivo
novo. Por isso o que segue é ordem de execução futura, com condições de parada, não tarefa
pendente.

### 4.1 Verificações que precisam vir antes de qualquer amostragem

Nenhuma delas foi feita nesta rodada. São verificações de existência de fonte, e cada uma pode
falhar — falhar aqui é resultado válido e deve ser registrado, não contornado.

| # | Verificação | Falha significa |
|---|---|---|
| V1 | Existe cadastro municipal de ocorrências **não-hidrológicas**, com endereço/coordenada e data, em série multianual? | rota de cadastro fechada; avaliar pseudo-ausência como secundária |
| V2 | O cadastro traz coordenada da própria fonte, ou vai exigir geocodificação? | se exigir, aplicar o mesmo corte strong/medium de Recife e medir a taxa de sucesso antes de contar com o N |
| V3 | Quais categorias do cadastro são de fato independentes de chuva **naquela cidade**? | não copiar a lista de Recife: Petrópolis é dominada por movimento de massa, e "muro com danos" lá provavelmente **é** consequência de chuva |
| V4 | Existe camada oficial de bairro/distrito para o pareamento? | sem ela, condição 5 é inexequível e a rota inteira trava |

Sobre V1, o único ponto de partida já tocado nesta linhagem é o **SIAC 156 de Curitiba**, que
aparece entre as fontes consultadas na busca de positivos
(`revp_linhagem_coleta_curitiba_petropolis_paridade_recife.md`, seção 3). Ele foi consultado para
**positivos**; sua estrutura de categorias, disponibilidade de coordenada e profundidade histórica
**não foram auditadas** para uso como negativo. Para Petrópolis não há candidato equivalente
identificado. Nada aqui afirma que essas fontes servem — afirma apenas onde começar a olhar.

### 4.2 Ordem de execução, quando o N permitir

1. Rodar V1-V4 por região e registrar o resultado real, inclusive falha.
2. Reconstruir a lista de categorias aceitas/excluídas **por cidade**, com justificativa causal
   escrita, à moda de `EXCLUDE_NEG_CATEGORIES` — sem herdar a de Recife por conveniência.
3. Só com o conjunto de positivos da região **fechado para aquela rodada**: extrair o conjunto de
   bairros com positivo e aplicar o filtro de sobreposição do v9.
4. Amostrar deterministicamente por ano (ordenação por hash estável), com checagem de
   não-sobreposição de coordenada (~30 m) contra positivos e contra negativos já existentes.
5. Registrar as contagens reais que o v9 registrou: candidatos brutos, quantos passaram no filtro
   de bairro, bairros de positivo, bairros de negativo, interseção.
6. Só então extrair features físico-hidrológicas nesses pontos — inclusive HAND/TWI, com os
   rasters de prontidão já gerados em SUSC-20G. Amostragem de ponto **não** faz parte do 20G.

### 4.3 Proporção positivo/negativo

Recife chegou a 154/124 (≈1,24:1) por acúmulo histórico, não por alvo de projeto. Não há aqui
recomendação de proporção fixa: o número de negativos é limitado pelo pareamento por bairro
(condição 5), e o EPV que governa quantas features cabem é calculado sobre o **menor** dos dois
grupos. Fixar proporção antes de saber quantos positivos existem seria inverter a ordem.

## 5. O que este documento não é

Não é autorização para amostrar negativo em Curitiba ou Petrópolis. Não é afirmação de que o SIAC
156 ou qualquer outra fonte serve. Não altera o estado das regiões: Curitiba e Petrópolis seguem
com N=1 positivo, abaixo do piso EPV, sem `model_version`.
