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
  `outputs_public/data/linha_causal/susc_20d_motor_inferencia_local_mvp_recife/`.
- **Fase 4 -- PARCIALMENTE FECHADA.** Leads B (ANA) e C (Global Flood Database) para
  Curitiba concluídos com dado real e achados novos (corroboração hidrológica real em 2
  estações da RMC para o evento de 2022-01-15/16; evento MODIS-validado real DFO_4276/2015
  com 54 pixels de inundação genuína no bairro São Miguel). Lead A (Diário Oficial) com
  tentativa real registrada, decreto específico ainda não recuperado. Curitiba continua
  "evidência em processamento", não "análise disponível" -- geometria de ocorrência do
  evento de 2022 ainda ausente (bloqueio já conhecido de `SUSC-18C`). Ver
  `outputs_public/data/linhagem_anterior/susc_curitiba_leadb_ana_estacoes_reais/reports/RELATORIO_fase4_curitiba_leads_abc.md`.
- **Fase 5 -- FEITA (MVP local).** API FastAPI real (`outputs_public/data/linha_causal/susc_20e_api_contrato_inferencia_recife/`)
  implementando o contrato do rascunho, gates completos, testada com 7 cenários reais.
  **Extensão SUSC-20F (2026-07-24)**: pipeline de geoprocessamento sob demanda fecha a
  limitação "só pontos conhecidos" -- terreno real amostrado do DTM PE3D merged
  (HAND/TWI match exato; elevação/declividade com ~2,7m/5,6° de diferença histórica
  documentada, raster original não recuperável) + chuva ao vivo via Open-Meteo
  ERA5-Land (match exato). API agora responde `ok` com score real pra qualquer
  coordenada dentro da cobertura real (~21km x 28km em Recife), não só os 269 pontos
  de treino. Ver `outputs_public/data/linha_causal/susc_20f_pipeline_geoprocessamento_sob_demanda_recife/`.
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
- **Frente causal SUSC-20 (SUSC-20O/20P/20Q, 2026-08-01/02) -- diagnóstico do colapso de AUC
  prospectivo em Curitiba, causa isolada até onde o dado atual permite.** SUSC-20O
  (2026-08-01) testou holdout temporal genuíno (treino 2023-2025, teste 2026-parcial nunca
  visto): AUC caiu de 0,6459 (LOO-CV embaralhado) para 0,5246. SUSC-20P (2026-08-02) descartou
  vazamento espacial (GroupKFold por 73 bairros deu AUC=0,6442, igual ao embaralhado). Por
  pedido explícito de esgotar toda vertente de literatura, SUSC-20Q (2026-08-02) rodou mais 6
  diagnósticos sobre o dado já existente (nenhuma aquisição nova): (1) bootstrap CI do AUC
  holdout inclui 0,5/acaso mas exclui 0,6459 -- a queda é real, não ruído de amostra; (2)
  ablação terreno-só (AUC=0,5213) vs. chuva-só (AUC=0,4984) -- os dois colapsam juntos; (3)
  **walk-forward multi-corte mostrou que 2024 e 2025 generalizam bem prospectivamente
  (AUC 0,63 e 0,67) -- o colapso é específico de 2026, não falha geral de generalização
  temporal**; (4) holdout casado por estação (jan-jul→jan-jul) deu AUC=0,5219, quase idêntico
  ao original -- descarta sazonalidade não-comparável; (5) coeficiente Firth por ano mostrou
  as duas features de chuva como sinal forte e consistente em 2023-2025 (p<0,01) **ficando
  completamente nulas em 2026** -- mecanismo direto do colapso; (6) composição/metadado
  (confiança, categoria, cobertura de chuva) estável ano a ano, inclusive 2026 -- descarta
  deriva administrativa visível. **SÍNTESE**: depois de 7 diagnósticos no total, descartadas
  vazamento espacial, sazonalidade, ruído de amostra, deriva administrativa visível e falha
  geral de generalização temporal. O que resta: 2026 especificamente tem relação chuva↔queixa
  diferente dos 3 anos anteriores, por razão física (regime de chuva atípico) ou completude do
  ano parcial ainda em processamento -- nenhuma testável com o dado atual sem nova aquisição
  externa (ex.: índice ENSO/ONI, série de chuva total anual independente de queixa), não
  executada aqui. Redesenho de amostragem negativa (pareamento temporal positivo-negativo,
  positive-unlabeled learning -- literatura: "Contrast or Diversity" ScienceDirect
  S0022169425003919, PU-learning MDPI land11111971) é uma vertente real e distinta, documentada
  como opção de metodologia maior que ataca a limitação estrutural já conhecida
  (positivo=dia de queixa, negativo=data arbitrária não-hidrológica) -- **não é diagnóstico,
  é redesenho, exige aprovação explícita antes de rodar, não decidido aqui**. Ver
  outputs_public/data/linha_causal/susc_20k_siac156_curitiba_flood_candidates/reports/
  (susc_20o_validacao_temporal_holdout_curitiba_report.md,
  susc_20p_validacao_blocos_espaciais_curitiba_report.md,
  susc_20q_bateria_exaustiva_diagnosticos_temporais_curitiba_report.md).
- **Revisão de literatura ampliada (2026-08-02) -- referências novas pra atacar a lacuna do
  SUSC-20Q, nenhuma execução ainda, aguardando decisão sobre qual vertente seguir.** Pedido
  explícito de esgotar bases acadêmicas a partir das palavras-chave/cenário do projeto. Quatro
  referências novas, diretamente acionáveis:
  1. **Agostini, Pierson & Garg, "A Bayesian Spatial Model to Correct Under-Reporting in Urban
     Crowdsourcing" (AAAI-24, arXiv:2312.11754)** -- ataca exatamente o mesmo problema
     estrutural do SIAC 156 (dado de queixa cidadã: impossível distinguir evento que não
     ocorreu de evento que ocorreu mas não foi reportado), aplicado no mesmo domínio (queixas
     de alagamento em NYC). Usa correlação espacial pra inferir probabilidade real de
     ocorrência. Referência mais forte e mais no-alvo encontrada até agora pro "confundimento
     temporal estrutural" (positivo=dia de queixa, negativo=data arbitrária) já documentado
     desde o SUSC-20K/20N como limitação não resolvida.
  2. **PBLC -- "A Positive-Unlabeled Learning Algorithm for Urban Flood Susceptibility
     Modeling" (Land, 2022, DOI 10.3390/land11111971)** -- mesmo problema ("case-control
     sampling with contaminated controls", quase a mesma frase que já usamos pra descrever
     nosso próprio limite), método específico pra suscetibilidade a enchente (não genérico),
     testado em Guangzhou.
  3. **"Contrast or Diversity: Non-Flood sampling in urban flood susceptibility modelling"
     (J. Hydrology 656, 2025, ScienceDirect S0022169425003919)** -- framework de amostragem
     negativa por distância (DBS) balanceando contraste e diversidade; não resolve o
     confundimento temporal sozinho, mas informa como desenhar qualquer minerador de negativo
     futuro pra Curitiba/Recife melhor que o atual (cap fixo por ano, sem controle de
     contraste/diversidade espacial).
  4. **Hipótese ENOS/El Niño pro colapso específico de 2026** -- Nota Técnica SIMEPAR (fonte
     primária, 2026-06-11): El Niño confirmado pela NOAA/CPC em jun/2026, favorece "sistemas
     convectivos de mesoescala" e "episódios prolongados de chuva" no Paraná a partir do
     inverno 2026. Literatura de clima (consecutive wet days/CWD como indicador mais robusto
     de impacto de ENOS em precipitação que intensidade/frequência) sustentaria um novo
     feature físico-causal (dias consecutivos de chuva, agregado do mesmo CHIRPS já usado,
     sem dado novo) que os 2 features de chuva atuais (peak/decay) podem não capturar.
     **Ressalva de honestidade**: a linha do tempo real do ONI (La Niña 2020-início 2023 →
     El Niño forte 2023-mai/2024 → neutro com dip fraco de La Niña fim 2024/início 2025 →
     neutro até set/2025 → El Niño formando jun/2026) NÃO dá um corte limpo "2023-25 estável
     vs. 2026 anômalo" -- 2023 foi ele mesmo ano de El Niño forte e generalizou bem (SUSC-20Q,
     walk-forward). A hipótese ENOS não pode ser afirmada como explicação sem puxar o índice
     ONI real por período e correlacionar com o comportamento do coeficiente -- não feito
     aqui, é o próximo passo mais barato e mais concreto (série pública pequena, mesma
     categoria de dado que Open-Meteo/CHIRPS já em uso).
  **Decisão (2026-08-02, humana): seguir as duas vertentes em sequência.**
- **SUSC-20R (2026-08-02) -- ONI real testado, resultado NEGATIVO pra hipótese ENOS.** Índice
  ONI real (fonte: ggweather.com/enso/oni.htm, ONI v5 NOAA/CPC), média Jan-Jul por ano
  comparada com AUC de teste do walk-forward (SUSC-20Q): 2024 (El Niño forte, ONI médio
  +0,88) generalizou bem (AUC=0,6282); 2025 (neutro, ONI médio −0,07) generalizou ainda melhor
  (AUC=0,6652) -- **padrão oposto ao esperado se anomalia ENOS explicasse o colapso**. 2026
  sem valor de ONI publicado na fonte consultada (não estimado, documentado como limitação).
  **Conclusão: hipótese ENOS não sustentada pelos dados reais disponíveis** -- não é mais
  tratada como explicação provável do colapso de 2026. Ver
  reports/susc_20r_correlacao_oni_enso_curitiba_report.md.
- **SUSC-20S (2026-08-02) -- piloto PU bagging testado, resultado NEGATIVO pro redesenho de
  amostragem.** Segunda vertente: PU bagging (Mordelet & Vert, arXiv:1010.0772) -- positivos
  fixos, pool "negativo" tratado como não-rotulado e reamostrado por bag, 300 bags, escolhido
  sobre o modelo bayesiano espacial de sub-reporte (Agostini et al.) por risco desse último
  criar uma superfície de risco derivada da densidade de queixa -- próximo demais de um
  score/proxy proibido pelas regras fixas. Resultado: holdout temporal PU bagging=0,5245
  (baseline supervisionado=0,5246); spatial block CV PU bagging=0,6443
  (baseline=0,6442) -- **diferença nula nas duas validações**. PU bagging corrige viés de
  rótulo contaminado; não corrige mudança de distribuição/relação feature-rótulo ao longo do
  tempo -- e o nosso problema (SUSC-20Q diagnóstico 5: coeficiente de chuva cai a zero em
  2026) é do segundo tipo, não do primeiro. Rota primária de Curitiba não muda (continua
  Firth supervisionada, 5 features, SUSC-20N). Recife não tocado. Ver
  reports/susc_20s_piloto_pu_bagging_curitiba_report.md. **SÍNTESE FINAL das duas vertentes
  da revisão de literatura**: ambas testadas honestamente, ambas negativas -- isso não é
  beco sem saída, é eliminação rigorosa que fortalece a conclusão do SUSC-20Q: o colapso de
  2026 é uma propriedade real e ainda não explicada desse período específico, não um
  artefato de desenho metodológico corrigível com as ferramentas testadas até agora.
- **SUSC-20T (2026-08-02) -- mais 3 vertentes testadas (achado de noticia real + 2 tecnicas),
  as 3 negativas.** (1) Lancamento real do CuritibaApp (25/03/2026, app municipal unificado
  com IA absorvendo o 156) cai dentro da janela de teste -- testado se o colapso se concentra
  pos-lancamento: NAO (AUC pre-app=0,4341, pos-app=0,4656 -- pre e pior). (2)
  rain_max_24h_chirps (coluna ja no dataset, nunca usada) tem correlacao fraca com o rotulo
  em TODOS os anos, nao so 2026 -- nao e alternativa viavel (AUC=0,523 no holdout, igual ao
  baseline). (3) Peso de recencia (decaimento exponencial, meia-vida 1 e 2 anos) nao muda
  nada (AUC=0,5241 e 0,5223, quase identicos ao baseline 0,5246).
- **SUSC-20U (2026-08-02) -- diagnostico de nao-linearidade: PRIMEIRO RESULTADO POSITIVO da
  bateria inteira (20P-20U).** Pergunta ortogonal a tudo testado ate aqui: a relacao
  features-rotulo e nao-linear (limiares/interacoes), nao capturavel por modelo linear,
  independente da causa da mudanca ano a ano? Gradient boosting raso sobre as MESMAS 5
  features causais (sem feature nova): holdout temporal AUC=0,5888 (IC 95% [0,5188; 0,6589],
  exclui 0,5/acaso -- diferente do baseline linear, cujo IC incluia 0,5); spatial block CV
  AUC=0,6664 (vs. 0,6442 linear). Robusto: 92,6% de 27 combinacoes de hiperparametro tem
  AUC>0,55, mediana 0,586. Importancia de feature fisicamente coerente (chuva domina 67%,
  HAND segunda, twi_dinf ultima). Ressalvas importantes, nao escondidas: GBM nao respeita o
  piso de EPV que rege a rota Firth (mais parametros efetivos que 5 coeficientes);
  feature_importances_ nao sao coeficientes causais (sem sinal, sem IC, sem interpretacao
  "aumentar X muda risco em Y") -- conflita direto com a prioridade do projeto por
  interpretabilidade. NAO e proposta de rota primaria, e diagnostico -- decisao de
  aprofundar fica com orientacao humana. Ver
  reports/susc_20t_mais_3_diagnosticos_curitiba_report.md e
  susc_20u_diagnostico_nao_linearidade_curitiba_report.md.
- **Hipotese nao executada (documentada, nao perseguida)**: obras de drenagem reais em
  Curitiba (R$118 milhoes investidos em 2025 -- canal Vila Oficinas, bacia de detencao
  Pilarzinho) poderiam ser causa fisica real e valida -- mas cruzar datas/locais de obra
  especificos com os bairros do dataset exige esforco de matching nao trivial, nao executado
  por ora. Registrado como possivel linha futura, nao forcado nesta rodada.
- **SUSC-20V (2026-08-02) -- decomposicao do GBM + varredura de 8 classes de modelo, pedido
  explicito de testar tudo mesmo fora do filtro de interpretabilidade.** Ressalva mantida
  deliberadamente mesmo com o pedido ampliado: nenhuma feature nova derivada do label foi
  adicionada -- isso invalidaria a validacao por vazamento, e piso de validade cientifica, nao
  preferencia de estilo. Varredura (SVM-RBF, MLP, ExtraTrees, AdaBoost, RandomForest, GBM) nas
  MESMAS 5 features causais: **as 8 classes de modelo nao-linear ficam acima do baseline
  linear (0,5246)** -- de 0,5292 (MLP raso) a 0,5888 (GBM, SUSC-20U). Nao e acidente de um
  algoritmo especifico. Decomposicao via partial dependence (shap/xgboost nao instalaram no
  sandbox por timeout de rede, documentado, nao contornado por workaround arriscado): nenhuma
  das 5 features tem relacao monotonica simples (todas com multiplas mudancas de inclinacao),
  confirmando nao-linearidade real. 4 de 5 features mantem direcao geral consistente com o
  sinal causal ja estabelecido (chuva, HAND, declividade) -- o modelo nao-linear nao inventa
  fisica nova, capta limiares/interacoes dentro da mesma direcao causal conhecida. `twi_dinf`
  e excecao (direcao oposta ao esperado), consistente com nunca ter tido sinal robusto em
  nenhuma rodada linear anterior. **Ainda diagnostico, nao rota de producao** -- mesmas
  ressalvas de interpretabilidade/EPV do SUSC-20U seguem valendo. Ver
  reports/susc_20v_decomposicao_gbm_e_varredura_modelo_curitiba_report.md.
- **SUSC-20W (2026-08-02) -- checagem de robustez decisiva: a vantagem nao-linear NAO e
  especifica de 2026.** Rodou walk-forward multi-corte (2023->2024, 2023-24->2025,
  2023-25->2026) pras 4 classes de modelo (linear, GBM, AdaBoost, RandomForest). Resultado:
  8 de 9 comparacoes modelo x ano ficam acima do linear (unica excecao: GBM em 2025, por
  0,0065). Isso descarta que o ganho de 2026 seja overfitting ao ruido do teste --
  nao-linearidade real, presente em qualquer corte. AO MESMO TEMPO, 2026 continua sendo o
  ano mais dificil de prever pra QUALQUER classe de modelo (AUC absoluto mais baixo em todas
  as 4 linhas: 0,56-0,59 em 2026 vs. 0,63-0,68 em 2024 e 0,66-0,69 em 2025) -- a
  nao-linearidade compensa parcialmente a deriva ja diagnosticada no SUSC-20Q, mas nao a
  resolve. **Duas conclusoes independentes e complementares**: (1) existe nao-linearidade
  genuina e generalizavel nas 5 features causais; (2) a deriva especifica de 2026 continua
  sem explicacao causal identificada. Ver
  reports/susc_20w_walk_forward_nao_linear_curitiba_report.md.
- **SUSC-20X (2026-08-02) -- tentativa de traduzir a nao-linearidade pra termo interpretavel:
  NEGATIVA.** Testou se interacao-produto ou indicador de limiar (lidos direto da partial
  dependence do GBM, nao escolhidos por busca: hand_m_dinf<4, rain_decay>20,
  rain_peak<-4) adicionados ao modelo linear recuperam o sinal do GBM (0,5888). Resultado:
  nenhuma das 9 configuracoes passa de 0,53 (melhor: interacao rain_decay x rain_peak,
  0,5295) -- nao recupera nem de longe o patamar do GBM. Sugere que a nao-linearidade
  capturada nao e uma interacao produto suave nem um limiar aditivo simples, provavelmente
  estrutura mais complexa (interacao de ordem mais alta ou splits por subregiao) que um GLM
  com poucos termos nao replica facilmente. NAO invalida o achado SUSC-20U/20V/20W -- so
  estabelece que resolver a tensao interpretabilidade x performance nao e trivial com os
  recursos tentados. Rota primaria continua linear/Firth. Ver
  reports/susc_20x_tentativa_traducao_interpretavel_curitiba_report.md.
- **SUSC-20Y (2026-08-02) -- GAM aditivo (spline por feature) pra traduzir a nao-linearidade:
  MISTA.** Modelo aditivo por construcao (sem termo cruzado entre features, cada feature com
  seu proprio bloco de B-spline). Config default: AUC holdout 2026 = 0,5445; melhor da grade
  de 30 combos (nos x grau x C) = 0,575. Baseline linear = 0,5246; GBM (SUSC-20U) = 0,5888.
  Gap pro GBM caiu de 0,0593 (melhor tentativa SUSC-20X) pra 0,0138 -- reducao de ~77%, mas
  nenhuma configuracao iguala ou supera o GBM. Leitura: boa parte da nao-linearidade e
  por-feature (curva isolada, capturavel sem interacao), mas resta um residuo que so
  interacao real entre features explicaria -- consistente com a leitura do SUSC-20X. 3 de 5
  curvas de efeito batem com o sinal causal esperado (twi_dinf e rain_peak_residual nao
  batem; twi_dinf ja era anomalo no SUSC-20V/GBM). NAO invalida SUSC-20U/20V/20W/20X. Rota
  primaria continua linear/Firth. Ver
  reports/susc_20y_gam_splines_curitiba_report.md.
- **SUSC-20Z (2026-08-02) -- GAM + interacao tensor-spline (2D): NEGATIVA, encerra a vertente
  GAM.** Testou termo de interacao tensor-product (te()-like) nos mesmos 3 pares do SUSC-20X,
  somado ao GAM aditivo do SUSC-20Y. Melhor resultado (3 tensores + grade de hiperparametro) =
  0,5631 -- MENOR que o melhor GAM puramente aditivo do SUSC-20Y (0,575). Interacao explicita
  nao trouxe ganho liquido sobre so dar mais liberdade a parte aditiva; provavel overfitting
  (16-25 colunas extra por par, N treino=1179). Leitura consolidada da vertente GAM: maior
  parte da nao-linearidade do GBM e por-feature (capturavel sem interacao); residuo restante
  (~0,01-0,03 AUC) provavelmente estrutura de ordem >2 ou splits regionais, nao capturavel por
  GAM aditivo nem por tensor-spline 2D com os dados disponiveis. Vertente GAM/spline
  considerada exaurida com as ferramentas testadas. NAO invalida achados anteriores. Rota
  primaria continua linear/Firth. Ver
  reports/susc_20z_gam_tensor_interaction_curitiba_report.md.
- **SUSC-21A (2026-08-02) -- GBM com restricao monotonica causal: POSITIVA-PARCIAL, nova
  vertente (fora da familia GAM).** HistGradientBoostingClassifier com monotonic_cst forcando
  o sinal causal ja estabelecido (EXPECTED_SIGN, mesmo usado no Firth) em cada uma das 5
  features -- nao pode inverter direcao, so a forma (limiar/plato/taxa) e livre. Resultado:
  melhor config = 0,5561 (vs linear 0,5246, GAM 0,575, GBM irrestrito 0,5888) -- nao supera
  GAM nem GBM, mas fica acima do linear em 100% dos 27 combos testados. Custo de impor
  monotonicidade (mesma config): 0,0224 de AUC. Achado central: e o UNICO modelo nao-linear
  desta serie com 100% de conformidade causal garantida por construcao (5/5 features na
  direcao esperada, verificado via decision_function bruta, nao so PD agregada) -- GBM
  irrestrito (SUSC-20V) tinha twi_dinf anomalo (4/5), GAM aditivo (SUSC-20Y) tinha 2 features
  anomalas (3/5). Reformula a pergunta: em vez de "recuperar 100% do sinal do GBM" (dificil,
  SUSC-20X/Y/Z), "qual o melhor modelo nao-linear que nunca viola fisica conhecida" -- resposta
  = GBM monotonico. Nao resolve a lacuna original (colapso 2026); estabelece candidato mais
  defensavel cientificamente que GBM irrestrito, se decisao futura for usar nao-linear em
  producao. Rota primaria continua linear/Firth. Ver
  reports/susc_21a_gbm_monotonico_causal_curitiba_report.md.

---

## 0. Achado que muda o ponto de partida do plano

Ao consolidar este plano, identifiquei que **existem três frentes rodando em paralelo
com maturidades bem diferentes**, e uma nota de rascunho anterior já estava ancorada
numa delas sem considerar o dado mais atual disponível.

| Frente | Onde vive | O que é | Maturidade real |
|---|---|---|---|
| **v12 (modelo supervisionado real)** | `PROJETO/local_runs/recife_modelo_v12_extracao_final/` | Firth penalizado + bootstrap N=1000 + LOO/k-fold, **n=278 (154 pos / 124 neg)**, 6 features físicas, **LOO-AUC = 0,6781** | **A mais madura.** É o número que o rascunho anterior já cita (278/154/124/0,678) — a referência correta é essa, não o v5. |
| **SUSC_01→18C (pipeline auditado)** | `REV-P/outputs_public/suscetibilidade/` | Score v5 (circular, já documentado) + score v6 candidato (determinístico, não-supervisionado, 300 patches, `review_only=true`) + gates formais de treino | **Deliberadamente travada.** `SUSC_18C`: `accepted_ground_reference_count=0`, `label_contract_training_allowed=false`, `supervised_training_allowed_after_18c=false` — o SUSC nunca absorveu os eventos SEDEC reais como referência aceita, por desenho (respeita a regra do REV-P de nunca ter treino supervisionado). |
| **DINO x SEDEC** | `REV-P/scripts/dino/` | Embeddings DINO reais testados contra `dataset_v4_features_finais.csv` — **só 163 pontos (141 pos / 22 neg)**, não os 278 do v12 | Real, mas **testado contra o dataset errado** — o v12 (278 pontos) é mais novo e mais forte, e esse teste precede a consolidação do v12. |

**Consequência prática**: a pergunta central do documento — "DINO agrega valor além da física?"
— ainda não foi respondida com o dado mais forte que existe. O LOO-AUC=0,490 (nível de
chance) obtido pro DINO sozinho foi contra um recorte de 78 pontos tirado dos 163
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
