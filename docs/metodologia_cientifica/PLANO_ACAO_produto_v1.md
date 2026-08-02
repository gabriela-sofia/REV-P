# Plano de ação — do "motor científico" pro produto (rascunho txtpragab reorganizado)

**Status**: PLANO_DE_TRABALHO_NAO_CANONICO — documento de orientação entre sessões, não é
gate nem alteração de dado/modelo. Serve pra qualquer sessão futura (minha ou de outra
instância) se realinhar rápido com onde a gente já chegou.

**Regra que rege este plano inteiro**: cada fase só é considerada "feita" quando produz um
número real, um arquivo real ou um resultado real que pode ser conferido — nunca "deveria
funcionar" ou "provavelmente ajuda". Se uma fase não provar nada, ela é descartada ou
reformulada, não carimbada como concluída.

---

## Status de execução (atualizado 2026-07-23, sessão posterior a este rascunho)

Fases 1-4 já rodaram com prova real. Resumo pra realinhamento rápido (detalhe completo em
cada doc/pasta linkado):

- **Fase 1 -- FECHADA.** Teste A/B refeito contra o v12 (n=278→109 joined com DINO real).
  LRT ingênuo pareceu significante (p=0.0048), mas descobriu-se pseudorreplicação (DINO é
  por-patch, não por-ponto: 109 pontos em só 23 patches únicos). Erro-padrão
  cluster-robusto derrubou o achado (p=0.1752). **Conclusão: DINO não entra no score;
  fica como evidência visual auxiliar.** Ver
  `revp_fase1_conclusao_dino_ab_test.md`, `scripts/dino/revp_v1r5_dino_v12_ab_test.py`,
  `scripts/dino/revp_v1r6_dino_v12_cluster_robust_sensitivity.py`.
- **Fase 2 -- FECHADA.** CI do score decidido como bootstrap preditivo (não delta method).
  Gates do contrato mapeados contra os gates do SUSC existentes -- achado: são camadas
  diferentes (governança de treino vs. disponibilidade de dados por request), quase sem
  sobreposição real. Ver `revp_fase2_decisoes_design_contrato.md`.
- **Fase 3 -- FECHADA.** Motor de inferência local (SUSC-20D) construído, validado
  (reproduz coeficientes e bootstrap publicados do SUSC-20C a <1e-3) e auditado contra os
  269/278 rótulos reais (AUC in-sample=0.7107 vs LOO-AUC=0.6781 documentado -- números
  diferentes por desenho, não conflatados). Ver
  `outputs_public/data/susc_20d_motor_inferencia_local_mvp_recife/`.
- **Fase 4 -- PARCIALMENTE FECHADA.** Leads B (ANA) e C (Global Flood Database) para
  Curitiba concluídos com dado real e achados novos (corroboração hidrológica real em 2
  estações da RMC para o evento de 2022-01-15/16; evento MODIS-validado real DFO_4276/2015
  com 54 pixels de inundação genuína no bairro São Miguel). Lead A (Diário Oficial) com
  tentativa real registrada, decreto específico ainda não recuperado. Curitiba continua
  "evidência em processamento", não "análise disponível" -- geometria de ocorrência do
  evento de 2022 ainda ausente (bloqueio já conhecido de `SUSC-18C`). Ver
  `outputs_public/data/susc_curitiba_leadb_ana_estacoes_reais/reports/RELATORIO_fase4_curitiba_leads_abc.md`.
- **Fase 5 -- FEITA (MVP local).** API FastAPI real (`susc_20e_api_contrato_inferencia_recife/`)
  implementando o contrato do rascunho, gates completos, testada com 7 cenários reais.
  **Extensão SUSC-20F (2026-07-24)**: pipeline de geoprocessamento sob demanda fecha a
  limitação "só pontos conhecidos" -- terreno real amostrado do DTM PE3D merged
  (HAND/TWI match exato; elevação/declividade com ~2,7m/5,6° de diferença histórica
  documentada, raster original não recuperável) + chuva ao vivo via Open-Meteo
  ERA5-Land (match exato). API agora responde `ok` com score real pra qualquer
  coordenada dentro da cobertura real (~21km x 28km em Recife), não só os 269 pontos
  de treino. Ver `susc_20f_pipeline_geoprocessamento_sob_demanda_recife/`.
- **Fase 1b (SUSC-21/21b, 2026-08-01) -- FECHADA, DECISÃO: voltar o foco pro causal.**
  Depois da Fase 1 fechar "DINO não é feature", testou-se um segundo papel: DINO como
  refinador de evidência (fila de revisão priorizada por similaridade, adaptado do
  H2O-Net/arXiv:2010.05309 -- nunca cria label, nunca treina). Pré-registrado antes de
  rodar (φ_H=0,75/φ_L=0,25/suporte≥2/critério de utilidade fixo). Resultado nas 23 patches
  Recife com embedding: **nulo** -- separação semente-a-semente foi a pior das 70
  partições possíveis (enumeração exata p=1,0000), confundidores todos nulos. Ampliado
  pra 52 patches (100% cobertura Recife, 0 downloads novos) pra descartar falta de poder
  estatístico: **nulo idêntico confirmado** (mesmo -0,118337, mesma posição 70/70) --
  gargalo real é quantos patches têm ponto SEDEC suficiente pra virar semente, não
  cobertura de embedding. Decisão (2026-08-01, humana): não seguir pra triagem cross-city
  com esse sinal (seria misturar validação científica com continuidade de plano) nem
  pra troca de modelo geoespacial agora -- **linha DINO fechada por ora, foco volta pro
  SUSC-20 causal** (Curitiba/Petrópolis). Ver
  `docs/metodologia_cientifica/revp_v1r7_dino_evidence_refinement_recife.md` e
  `revp_v1r8b_dino_expanded_evidence_refinement_recife.md`.
- **Fase 1c (SUSC-22/v1r9, 2026-08-01) -- FECHAMENTO DEFINITIVO da linha de embedding/patch
  estático.** Testou-se trocar DINOv2 genérico por Clay (modelo geoespacial de propósito,
  pré-treinado em multiespectral real) sobre os mesmos 52 patches de Recife. Auditoria de
  viabilidade (`datasets/clay_feasibility_audit_v1r9.csv`) achou o bloqueio real ANTES de
  gastar esforço: os `.tif` locais são **composição mediana de 12 meses**
  (`export_sentinel.py`, `.filterDate(2024-01-01,2024-12-31).median()`), não uma cena de um
  instante real -- não existe `time` de aquisição pra declarar ao Clay, que é condicionado
  em semana/hora. **Diagnóstico de fundo (não é sobre qual modelo escolher)**: as três
  tentativas nesta linha (DINO A/B com pseudorreplicação, refinamento nulo em 23 e depois 52
  patches, Clay bloqueado por falta de instante de aquisição) são a mesma falha estrutural
  aparecendo de formas diferentes -- **patch estático (composição de vários meses) não carrega
  assinatura de evento pontual**, independente de qual encoder processa a imagem. A literatura
  confirma: mapeamento de enchente por change detection funciona comparando uma referência
  pré-evento contra uma imagem do evento (Prithvi-EO-2.0 e Clay são construídos multi-temporais
  exatamente pra isso), nunca por embedding de um único composto estático. O próprio projeto já
  validou essa abordagem certa em outro lugar -- SAR/NDWI/MNDWI ancorados em evento real
  (Curitiba Juvevê, Petrópolis Via B/C) -- só nunca foi aplicada aos 278 pontos SEDEC de Recife
  (aplicar exigiria ~278 aquisições individuais, escopo e custo que ficam para decisão futura,
  não desta rodada). **Decisão (2026-08-01, humana): fechar de vez a linha de
  embedding/patch estático como candidata a evidência ou feature -- não é mais revisitada.**
  Foco 100% no causal SUSC-20. Ver `datasets/clay_feasibility_audit_v1r9.csv`.
- **Frente causal SUSC-20 (SUSC-20O/20P, 2026-08-01/02) -- EM ABERTO, diagnóstico do colapso
  de AUC prospectivo em Curitiba.** SUSC-20O (2026-08-01) testou holdout temporal genuíno
  (treino 2023-2025, teste 2026-parcial nunca visto) na rota primária de 5 features: AUC caiu
  de 0,6459 (LOO-CV embaralhado) para 0,5246 -- quase indistinguível de aleatório fora da
  janela de treino. Duas hipóteses não-excludentes ficaram em aberto: (a) vazamento espacial
  (unidade do mesmo bairro em treino e teste do CV embaralhado infla o AUC) e (b) deriva
  temporal/administrativa (composição do SIAC 156 muda de ano pra ano; 2026 parcial tem
  sazonalidade não comparável). Pesquisa de literatura (2026-08-02) confirmou que ambos os
  mecanismos sao documentados na area de suscetibilidade a enchente -- random split sem
  spatial block CV infla AUC em 5-15%, e ha relato de modelos com CV quase perfeito falhando
  fora da amostra temporal. SUSC-20P (2026-08-02) isolou a hipotese (a): GroupKFold por bairro
  (73 bairros, nenhum aparece em treino e teste do mesmo fold) deu AUC medio 0,6442 (desvio
  0,032) -- estatisticamente igual ao CV embaralhado, muito acima do holdout temporal.
  Conclusao: vazamento espacial nao e a causa -- o modelo generaliza bem pra bairros nunca
  vistos. A explicacao do colapso fica concentrada na hipotese (b), ainda nao testada
  isoladamente -- proximo passo natural, nao iniciado nesta rodada (uma tarefa por vez). Ver
  outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/reports/
  (susc_20o_validacao_temporal_holdout_curitiba_report.md,
  susc_20p_validacao_blocos_espaciais_curitiba_report.md).

---

## 0. Achado que muda o ponto de partida do plano (descoberto agora, não estava nas minhas contas)

Fazendo o levantamento pra este plano, encontrei uma coisa que nem eu tinha me dado conta
até agora: **existem três frentes rodando em paralelo com maturidades bem diferentes**, e
o documento do seu colega já está ancorado numa delas que eu não tinha usado nesta sessão.

| Frente | Onde vive | O que é | Maturidade real |
|---|---|---|---|
| **v12 (modelo supervisionado real)** | `PROJETO/local_runs/recife_modelo_v12_extracao_final/` | Firth penalizado + bootstrap N=1000 + LOO/k-fold, **n=278 (154 pos / 124 neg)**, 6 features físicas, **LOO-AUC = 0,6781** | **A mais madura.** É o número que o rascunho do seu colega já cita (278/154/124/0,678) — ele já está olhando pra essa, não pro v5. |
| **SUSC_01→18C (pipeline auditado)** | `REV-P/outputs_public/suscetibilidade/` | Score v5 (circular, já documentado) + score v6 candidato (determinístico, não-supervisionado, 300 patches, `review_only=true`) + gates formais de treino | **Deliberadamente travada.** `SUSC_18C`: `accepted_ground_reference_count=0`, `label_contract_training_allowed=false`, `supervised_training_allowed_after_18c=false` — o SUSC nunca absorveu os eventos SEDEC reais como referência aceita, por desenho (respeita a regra do REV-P de nunca ter treino supervisionado). |
| **DINO x SEDEC (o que fiz nesta sessão)** | `REV-P/scripts/dino/` | Embeddings DINO reais testados contra `dataset_v4_features_finais.csv` — **só 163 pontos (141 pos / 22 neg)**, não os 278 do v12 | Real, mas **testado contra o dataset errado** — o v12 (278 pontos) é mais novo e mais forte, e eu não sabia da existência dele quando rodei o join. |

**Consequência prática**: a pergunta central do documento — "DINO agrega valor além da física?"
— ainda não foi respondida com o dado mais forte que existe. O LOO-AUC=0,490 (nível de
chance) que achei pro DINO sozinho foi contra um recorte de 78 pontos tirado dos 163
antigos. Isso é uma pista forte, não a resposta final. **Fase 1 abaixo resolve isso.**

---

## 1. Fase 1 — Refazer o teste A/B com o dado mais forte (prova, não suposição)

**Objetivo**: responder de vez "DINO agrega ao modelo físico?" usando o v12 (278 pontos),
não o recorte de 163.

**Passos concretos**:
1. Rejoinar os patches DINO já embeddados (52 patches Recife com Sentinel real) contra
   `dataset_v12_final.csv` em vez de `dataset_v4_features_finais.csv` — o join espacial
   (ponto-em-bbox) é o mesmo método já validado, só troca a fonte de eventos.
2. Modelo A = replicar exatamente as 6 features físicas do v12 (mesma metodologia Firth já
   usada, já validada, já com LOO-AUC=0,6781 documentado — não mexe nisso).
3. Modelo B = Modelo A + 1–2 componentes PCA do DINO, **só se o EPV permitir** (com o n que
   sobrar do join, calcular EPV antes de decidir quantos componentes DINO cabem — se não
   couber, reduzir features físicas a 2-3 mais fortes, como já indiquei na revalidação
   anterior).
4. Comparar A vs B por **razão de verossimilhança** (não ΔAUC bruto — já documentei por quê
   na revalidação científica anterior), e reportar ΔLOO-AUC só como descritivo complementar.

**Critério de prova**: só existe resposta quando esse script rodar e produzir um p-valor
real da razão de verossimilhança + um ΔLOO-AUC real. Enquanto isso não rodar, a resposta
"DINO não ajuda" continua sendo provisória (baseada no dado antigo), não definitiva.

**Duas saídas possíveis, ambas válidas como resultado**:
- Se B não bate A de forma consistente → decisão documentada: modelo do produto é
  Firth-só-física (v12), DINO vira evidência visual explicável na interface, nunca input
  do score. Fecha a pergunta do documento com prova.
- Se B bater A de forma estável → aí sim vale investir em generalização (fora da amostra,
  em Curitiba/Petrópolis) antes de aceitar DINO como feature do produto.

---

## 2. Fase 2 — Decisões de design que travam a API se não forem tomadas antes (já identificadas)

Da revalidação anterior do rascunho do contrato, dois pontos ficaram em aberto e **têm que
ser decididos antes de escrever qualquer código de API**, senão a implementação vai ter
que ser refeita no meio do caminho:

1. **Semântica de `confidence_interval`**: hoje o que existe é CI de coeficiente (bootstrap
   do v12). O contrato pede CI por score individual de uma região nova. Decidir entre
   bootstrap preditivo (reamostra + reajusta + projeta score, N vezes) ou delta method
   (erro padrão assintótico propagado pelo logito). São implementações diferentes.
2. **O contrato não pode reinventar gates que já existem**: o SUSC já tem uma linguagem de
   gate madura (`trainability_gate`, `promotion_blockers`, `readiness_summary`). Antes de
   escrever os "gates obrigatórios" do rascunho como código novo, mapear cada gate proposto
   (geometria válida, CRS, DEM, HAND, TWI, chuva, modelo válido pra região) contra o que já
   existe nesses arquivos do SUSC, pra não duplicar auditoria em dois lugares diferentes do
   projeto.

**Critério de prova**: essa fase produz uma decisão escrita (não código) — um parágrafo
dizendo qual dos dois métodos de CI foi escolhido e por quê, e uma tabela de
correspondência gate-do-contrato ↔ gate-do-SUSC-existente.

---

## 3. Fase 3 — MVP local, sem servidor ainda: o motor rodando ponta a ponta

**Objetivo**: provar que o contrato funciona antes de gastar esforço em API/backend/interface.

Um único script (não uma API) que recebe um `patch_id` já conhecido (dos ~278 do v12, ou
dos 52 com DINO), calcula/lê as 6 features físicas, roda os coeficientes Firth já treinados
do v12 (sem retreinar nada), devolve:
- score + intervalo (usando o método decidido na Fase 2),
- quais features mais pesaram,
- se houver embedding DINO pro patch, anexa como evidência visual separada (nunca somada
  ao score),
- limitações explícitas (n=278, região=Recife apenas, etc).

**Critério de prova, mensurável**: rodar esse script nos patches que já têm rótulo real
conhecido (os 278 do v12) e comparar o score gerado com o rótulo real — não para treinar
nada de novo (o modelo já está treinado), só para auditar se a saída do "motor" bate com o
que o v12 já reportou. Isso é literalmente a prova ponta-a-ponta que você está pedindo, sem
violar nenhuma regra (não cria label novo, não treina nada, só executa um modelo já
validado e confere a saída).

---

## 4. Fase 4 — Curitiba e Petrópolis, sem fingir simetria (o rascunho já acertou nisso)

- **Curitiba**: tem footprint SAR processado e corpus Sentinel, mas não tem um modelo Firth
  equivalente ao de Recife. Próximo passo real: replicar a extração de eventos reais
  (estilo v8/v9 do Recife — Diário Oficial, ANA, Global Flood Database) pra Curitiba antes
  de dar status "análise disponível". Enquanto isso não existir, status honesto = "evidência
  em processamento", exatamente como o rascunho propôs.
- **Petrópolis**: mistura enchente/deslizamento já é um bloqueio documentado — não avança
  pro produto até essa separação de fenômeno ser resolvida. Status honesto =
  "dados insuficientes para inferência".
- **SAR em Curitiba**: se algum dia voltar, entra como evidência de evento (par
  antes/depois de um evento real conhecido), não como feature estática recorrente — é o
  motivo técnico documentado do porquê foi descartado em Recife (ver revalidação anterior).

---

## 5. Fase 5 — Só agora a camada de API/contrato (rascunho original), com números reais por trás

Depois das Fases 1–3 terem rodado de verdade, o schema de entrada/saída do rascunho do seu
colega já pode virar código real, porque as duas ambiguidades que travavam a implementação
(CI, orçamento EPV pro DINO) já foram decididas com dado, não hipótese. A LLM entra só
nessa fase, só como camada de explicação sobre o payload estruturado — nunca decidindo o
score, exatamente como o rascunho já propôs (e como já documentei que é o padrão de
governança reconhecido — Model Cards / selective prediction — na revalidação anterior).

---

## 6. Ordem de execução recomendada (uma coisa por vez, cada uma só começa com a anterior provada)

1. Fase 1 — teste A/B real contra v12 (278 pontos). **Isso decide se DINO entra no score ou fica só como evidência visual.**
2. Fase 2 — duas decisões de design (CI; mapeamento de gates).
3. Fase 3 — MVP local auditável contra os 278 rótulos reais.
4. Fase 4 — Curitiba/Petrópolis, em paralelo mas sem prometer prazo.
5. Fase 5 — API/contrato/interface/LLM, só depois de 1–3 terem números reais.

Nenhuma fase promete prazo. Cada uma entrega um artefato conferível antes da próxima começar.
