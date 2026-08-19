# Curitiba (SIAC 156) — cadeia diagnóstica completa

**Resultado principal**: o modelo físico (Firth, mesmo método de Recife) mostra AUC de 0,65 embaralhado (`v20m`/`v20n`), mas colapsa para 0,52 em holdout temporal real de 2026 (`v20o`). As etapas `v20p` a `v20x` são 7+ diagnósticos independentes tentando explicar esse colapso — todos descartaram a causa que testavam (vazamento espacial, sazonalidade, ruído de amostra, deriva administrativa, correlação com El Niño/La Niña). A rota declarada continua linear/interpretável (Firth); não-linearidade real foi confirmada (`v20u`-`v21a`, GBM/GAM) mas é tratada como diagnóstico, nunca como substituição de produção.

Comparação lado a lado dos coeficientes das 3 especificações de modelo (primário + 2 sensibilidades), embaralhado vs. holdout real: [`results/v20mn_comparativo_coeficientes_3_especificacoes.csv`](results/v20mn_comparativo_coeficientes_3_especificacoes.csv).

## Índice cronológico dos relatórios (`reports/`)

**[SUSC-20K2/K3 — Réplica completa do método SEDEC/Recife em Curitiba (SIAC 156)](reports/susc_20k2_k3_replica_completa_metodo_recife_report.md)**
RESULTADO REAL, POSITIVO. Curitiba deixa de ter N=1 e passa a ter **1238 positivos + 119 negativos**, minerados/geocodificados/pareados com o mesmo rigor metodológico usado no dataset v12 de Recife. N deixa de ser o gargalo do projeto para esta região.

**[SUSC-20K — SIAC 156 (Curitiba): equivalente administrativo ao SEDEC, achado e minerado](reports/susc_20k_siac156_curitiba_flood_candidates_report.md)**
RESULTADO REAL, POSITIVO, PARCIAL. Primeira fonte administrativa geocodificável real pra Curitiba — mesma estrutura que fez o modelo de Recife funcionar (SEDEC: 91,6% dos 154 positivos vieram de reclamação administrativa geocodificada, não de imagem de satélite). Não substitui adjudicação individual completa (mesmo padrão do SUSC-20A/Valparaíso/Juvevê); produz um pool de candidatos com triagem física real, pronto pra próxima rodada de decisão.

**[SUSC-20L — Engenharia de features físico-hidrológicas, Curitiba (SIAC 156)](reports/susc_20l_engenharia_features_curitiba_report.md)**
FEATURES EXTRAÍDAS, DATASET COMPLETO. Nenhum modelo treinado nesta etapa. **Escopo**: só a Tarefa 1 (features). A modelagem/validação (SUSC-20M, espelho do SUSC-20C) está travada atrás da decisão de EPV documentada na seção 6.

**[SUSC-20M — Modelagem e validação estatística, Curitiba (SIAC 156)](reports/susc_20m_modelagem_validacao_curitiba_report.md)**
RESULTADO REAL, FRACO. O modelo separa positivo de negativo acima do acaso (LOO-AUC 0,605), mas **o único preditor estável é chuva antecedente**; as três features de terreno têm intervalo de confiança cruzando zero. O resultado é reportado como veio.

**[SUSC-20N — Reforço de negativos e retest do gate EPV, Curitiba](reports/susc_20n_reforco_negativos_retest_epv_report.md)**
EXECUTADO. Negativos de 103 → 426 unidades, EPV com 6 features de 17,17 → **70,5** (passa o piso). Escopo restrito: só aumento de N e re-rodada. Nenhuma feature nova, nenhum `EXPECTED_SIGN` alterado, nenhum método trocado.

**[SUSC-20O — Validação temporal holdout, Curitiba](reports/susc_20o_validacao_temporal_holdout_curitiba_report.md)**
EXECUTADO. Treino 2023–2025 (1179 unidades, 315 neg/864 pos), teste 2026-parcial (279 unidades, 108 neg/171 pos). Rota primária do SUSC-20N (5 features, `elevation_m` fora — decisão causal mantida, não revisitada aqui).

**[SUSC-20P — Validação por blocos espaciais (bairro), Curitiba](reports/susc_20p_validacao_blocos_espaciais_curitiba_report.md)**
EXECUTADO. GroupKFold(n_splits=5) por bairro, 1471 unidades, 73 bairros. Rota primária do SUSC-20N/20M/20O (5 features, `elevation_m` fora — decisão causal mantida, não revisitada aqui).

**[SUSC-20Q — Bateria exaustiva de diagnósticos do colapso de AUC prospectivo, Curitiba](reports/susc_20q_bateria_exaustiva_diagnosticos_temporais_curitiba_report.md)**
EXECUTADO. 6 diagnósticos independentes, todos sobre dado já existente (nenhuma aquisição nova). Rota primária do SUSC-20N/20M/20O/20P (5 features, `elevation_m` fora).

**[SUSC-20R — Correlação ONI/ENOS real vs. colapso de AUC prospectivo, Curitiba](reports/susc_20r_correlacao_oni_enso_curitiba_report.md)**
EXECUTADO, resultado negativo (hipótese não sustentada pelo dado real).

**[SUSC-20S — Piloto de redesenho de amostragem negativa (PU bagging), Curitiba](reports/susc_20s_piloto_pu_bagging_curitiba_report.md)**
EXECUTADO, resultado negativo (redesenho não resolve o colapso de generalização temporal). Piloto, não substitui a rota primária de produção.

**[SUSC-20T — Mais 3 diagnósticos (lançamento de app real + 2 técnicas de literatura), Curitiba](reports/susc_20t_mais_3_diagnosticos_curitiba_report.md)**
EXECUTADO, três resultados negativos.

**[SUSC-20U — Diagnóstico de não-linearidade (GBM raso), Curitiba](reports/susc_20u_diagnostico_nao_linearidade_curitiba_report.md)**
EXECUTADO. **Primeiro resultado positivo desta linha de investigação inteira** (SUSC-20P a 20U). Diagnóstico, não proposta de rota primária — decisão de aprofundar ou não fica com a orientação humana.

**[SUSC-20V — Decomposição do GBM + varredura de classes de modelo, Curitiba](reports/susc_20v_decomposicao_gbm_e_varredura_modelo_curitiba_report.md)**
EXECUTADO. Fortalece o achado do SUSC-20U: não é um acidente de um algoritmo específico — praticamente toda classe de modelo não-linear testada supera o baseline linear.

**[SUSC-20W — Walk-forward multi-corte para 4 classes de modelo, Curitiba](reports/susc_20w_walk_forward_nao_linear_curitiba_report.md)**
EXECUTADO. Checagem de robustez decisiva para o achado do SUSC-20U/20V: a vantagem não-linear **não é específica de 2026** — aparece nos 3 cortes prospectivos disponíveis.

**[SUSC-20X — Tentativa de tradução interpretável da não-linearidade, Curitiba](reports/susc_20x_tentativa_traducao_interpretavel_curitiba_report.md)**
EXECUTADO, resultado negativo. Tentativa honesta de resolver a tensão interpretabilidade×performance flagueada no SUSC-20U/20V — não conseguiu.

**[SUSC-20Y — GAM aditivo com splines por feature, Curitiba](reports/susc_20y_gam_splines_curitiba_report.md)**
EXECUTADO, resultado misto (melhora parcial, não recupera totalmente o GBM).

**[SUSC-20Z — GAM + interação tensor-spline (2D), Curitiba — fecha a vertente GAM](reports/susc_20z_gam_tensor_interaction_curitiba_report.md)**
EXECUTADO, resultado negativo (não melhora sobre o GAM aditivo puro do SUSC-20Y). Encerra a vertente GAM/spline nesta série.

**[SUSC-21A — GBM com restrição monotônica causal, Curitiba](reports/susc_21a_gbm_monotonico_causal_curitiba_report.md)**
EXECUTADO. Resultado positivo-parcial: não supera o GBM irrestrito nem o melhor GAM em AUC, mas é o único modelo não-linear testado nesta série com 100% de conformidade causal garantida por construção.

## Estrutura da pasta

- `registries/` — datasets brutos/geocodificados/pareados (positivos e negativos SIAC 156).
- `results/` — saídas numéricas de cada etapa (coeficientes, CV, hiperparâmetros); ver o comparativo acima como ponto de entrada.
- `scripts/` — código de cada etapa, mesmo prefixo do relatório correspondente.
- `reports/` — os 18 relatórios acima, em prosa, cada um com decisão e status explícitos.
