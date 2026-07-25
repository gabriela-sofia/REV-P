# Fase 2 -- Decisões de design do contrato de inferência (rascunho `txtpragab.docx`, revalidado em `revp_contrato_inferencia_v0_revalidacao_cientifica.md`)

**Status**: DECISAO_DE_DESIGN_NAO_CODIGO -- este documento decide, não implementa.
Implementação entra na Fase 3 (MVP) e Fase 5 (API), nessa ordem.

Pré-requisito cumprido: a Fase 1 (`revp_fase1_conclusao_dino_ab_test.md`) já rodou o
teste A/B que o próprio rascunho pede (seção "Processamento", item 72: "roda comparação
Modelo A vs Modelo B internamente, mas só usa o resultado se já validado como ganho
estável"). Resultado: **não validado como ganho estável** (LRT ingênuo p=0.0048 não
sobrevive ao controle cluster-robusto por patch, p=0.1752). Portanto, pela própria regra
que o rascunho propõe, `dino_embedding` **não entra no cálculo do score** hoje --
fica como `evidence` auxiliar na saída (o rascunho já previa isso: "SAR entra como
evidência contextual apenas se a região tiver modelo que já valide seu uso" -- a mesma
lógica agora se aplica ao DINO em Recife).

---

## 1. Semântica de `score.confidence_interval` -- decisão

**Decisão: bootstrap preditivo, não delta method.**

Mecânica: para uma nova região com features físicas x_novo, reamostrar o dataset de
treino N=1000 vezes (mesmo desenho já usado em `bootstrap_firth_coefs` de
`pipeline_v12_primary.py`: reamostragem estratificada por classe, positivos e negativos
separadamente, com reposição), reajustar o Firth em cada reamostra, projetar
`score_b = sigmoid(x_novo · beta_b)` para cada um dos 1000 conjuntos de coeficientes
`beta_b`, e tomar os percentis 2.5/97.5 da distribuição resultante de scores como
`confidence_interval`.

**Por quê, não delta method:**
- É uma extensão direta e já validada do que o projeto já faz (`primaria_v12_bootstrap_coefs.csv`,
  N=1000, seed fixa) -- não introduz uma segunda metodologia estatística no mesmo produto.
- Delta method depende da aproximação assintótica normal do log-odds propagada por
  Wald/Fisher information -- exatamente a aproximação que Firth foi adotado para evitar
  em primeiro lugar (viés de amostra pequena em regressão logística com poucos eventos).
  Usar delta method para o CI depois de usar Firth para o ponto estimado é
  metodologicamente inconsistente.
- Custo computacional (1000 reajustes de Firth por request) é aceitável para o volume
  esperado do MVP (Recife, sob demanda, não em tempo real de alta frequência); se isso
  virar gargalo real em produção, cache por patch/geometria é a otimização certa, não
  trocar de método estatístico.

**O que NÃO fazer**: usar diretamente `primaria_v12_bootstrap_coefs.csv` (CI dos
*coeficientes*) como se fosse CI do *score* -- são objetos estatísticos diferentes (a
confusão que a revalidação científica anterior já apontou, seção 3.5). O CI de
coeficiente responde "qual a incerteza do efeito de HAND?"; o CI de score responde
"qual a incerteza da predição para ESTA região?". Este documento fecha essa ambiguidade.

---

## 2. Mapeamento gate-do-contrato ↔ gate-do-SUSC-existente

O rascunho propõe 8 gates obrigatórios antes de qualquer inferência (seção "Gates
obrigatórios"). Auditoria linha a linha contra o que já existe:

| # | Gate do contrato | Existe hoje no SUSC/pipeline? | Onde | Reusar ou construir novo |
|---|---|---|---|---|
| 1 | Geometria válida (topologicamente correta, área mínima) | **Não** -- gates SUSC (`susc_18c_readiness_summary.json`: `trainability_gate_rechecked`, `label_contract_training_allowed`, `accepted_ground_reference_gate_passed`) auditam se o *dataset de treino* pode crescer/re-treinar, não se uma geometria *nova enviada por um usuário* é válida | -- | **Construir novo** -- validação de geometria (shapely `is_valid`, área mínima) é lógica de API, não existe no research pipeline porque o pipeline sempre trabalhou com os 278 pontos já fixos, nunca com polígono arbitrário de entrada |
| 2 | CRS conhecido e compatível | Parcial -- `_patch_bboxes_wgs84()` (usado em v1r4/v1r5) já faz reprojeção CRS→EPSG:4326 via `rasterio.warp.transform_bounds` para os patches Sentinel fixos | `scripts/dino/revp_v1r5_dino_v12_ab_test.py` | **Reusar o padrão** (rasterio transform_bounds), mas generalizar para CRS arbitrário de entrada, não só os patches já baixados |
| 3 | DEM cobrindo a área | **Não formalizado como gate** -- existe extração real de DEM (GLO-30) em `PROJETO/local_runs/recife_modelo_v7_otimizado` e no SUSC-20B (commit `d85f710`), mas como pipeline de construção de dataset, não como verificação de cobertura de uma área nova sob demanda | PROJETO (privado) | **Construir novo gate de cobertura**, reusando o downloader/reader de DEM já validado (não reinventar a extração, só envolver com um check de `bounds contains geometry`) |
| 4 | Declividade derivada do DEM | Calculada offline no dataset v12 (coluna `slope_deg`) | `dataset_v12_final.csv` | Reusar a fórmula/script de derivação; sem gate de "calculável" formal ainda |
| 5 | HAND calculável | Calculada offline (coluna `hand_m_dinf`, D-infinity, ver `improvement2_hand_twi_dinf_report.md`) | PROJETO (privado) | Reusar o script D-infinity já validado; falta o gate "é calculável para ESTA geometria" |
| 6 | TWI calculável | Idem HAND (coluna `twi_dinf`) | idem | idem |
| 7 | Chuva com período/fonte conhecidos, sem gap crítico | Parcial -- SUSC-20B já busca CHIRPS/ERA5-Land com período e fonte registrados por ponto (`rain_data_source`, `rain_max_24h_chirps`, `rain_decay_index_api_chirps` no v12), mas de novo para pontos fixos, não para checagem de gap sob demanda numa região nova | `dataset_v12_final.csv` colunas `rain_data_source`/`rain_*` | Reusar o fetcher CHIRPS/ERA5 já implementado; construir só o check de "gap crítico" que falta |
| 8 | Modelo estatístico válido para a região/domínio | **Existe, mas informal** -- hoje é conhecimento de projeto (só Recife tem `dataset_v12_final.csv` + coeficientes Firth), documentado em prosa em `RELATORIO_v12_MASTER.md` e no plano de ação (Fase 4), não como um campo de gate machine-readable | `PLANO_ACAO_produto_v1.md` seção 4 | **Construir gate novo**, mas trivial: um registro `{region: model_version_or_null}` -- hoje `{"recife": "v12", "curitiba": null, "petropolis": null}` |

**Achado principal desta auditoria**: os gates do SUSC existentes (`trainability_gate`,
`promotion_blockers`, `label_contract_training_allowed` etc.) respondem a uma pergunta
**diferente** da que o contrato faz. SUSC pergunta *"podemos criar mais rótulos/treinar
de novo?"* (gate de governança de pesquisa, sobre o pipeline de construção do dataset).
O contrato pergunta *"esta região nova enviada por um usuário do produto tem dados
suficientes para eu rodar o modelo JÁ treinado?"* (gate de disponibilidade de dados por
requisição). **Não há duplicação de auditoria a evitar** -- são camadas diferentes que
não se sobrepõem, exceto no gate #8 (modelo válido pra região), que é o único ponto de
contato real entre as duas linguagens de gate. Não existe hoje um módulo de "cobertura
de dados por geometria arbitrária" no SUSC -- ele precisa ser escrito na Fase 5 (API),
reusando os *fetchers/calculadores* já validados (DEM/HAND/TWI/CHIRPS), não a lógica de
gate em si.

---

## 3. Consequência prática para a Fase 3/5

- Fase 3 (MVP local) não precisa dos gates #1-7 (geometria arbitrária) porque vai operar
  só sobre `patch_id` **já conhecidos** dos 278 pontos do v12 -- não há geometria nova
  para validar. O único gate relevante na Fase 3 é o CI (decidido acima) e o #8 (região
  suportada = só Recife).
- Fase 5 (API) é onde os gates #1-7 precisam ser escritos de fato, e devem *importar* os
  calculadores físicos já validados em PROJETO (DEM/HAND/TWI D-infinity, CHIRPS/ERA5),
  não reimplementá-los.
