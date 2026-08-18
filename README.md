# REV-P

O REV-P investiga vulnerabilidade urbana a enchentes em três regiões brasileiras — Recife, Curitiba e Petrópolis — a partir de uma base físico-hidrológica causal. O modelo não tenta "descobrir" o que causa enchente: parte de relações físicas já conhecidas (acúmulo de água, capacidade de escoamento, proximidade a corpos hídricos, chuva antecedente) e testa essas relações contra eventos reais registrados por fontes oficiais (Defesa Civil, ANA, Diário Oficial, bases internacionais de inundação). Dado orbital (Sentinel-1/2) entra só como evidência auxiliar — nunca como variável causal.

**Método principal**: regressão logística penalizada de Firth (rota interpretável, adequada a eventos raros) com GBM monotônico causal como diagnóstico de não linearidade, nunca como substituto do modelo interpretável.

A narrativa científica completa (problema, motivação, contribuição, limites) está em [`docs/metodologia_cientifica/revp_narrativa_cientifica_consolidada.md`](docs/metodologia_cientifica/revp_narrativa_cientifica_consolidada.md).

---

## Estado atual por região

| Região | Estado | Resultado |
|---|---|---|
| **Recife** | Modelo causal maduro, auditado ponta a ponta | Firth penalizado (`v12`), n=278 eventos reais (154 positivos / 124 negativos), **LOO-AUC = 0,68** (repetido 5-fold: 0,67 ± 0,01). Coerência física preservada nos 6 sinais de coeficiente. Motor de inferência local + API de contrato entregues. |
| **Curitiba** | Modelo treinado, mas não generaliza — reportado como tal | Firth (`SUSC-20N`) com AUC embaralhado de 0,65 colapsa para 0,52 em holdout temporal real de 2026. Sete diagnósticos independentes descartaram vazamento espacial, sazonalidade, ruído de amostra, deriva administrativa e correlação com El Niño/La Niña como causa. Não linearidade real confirmada (GBM = 0,59), mas não resolve o colapso. Rota declarada continua linear/interpretável. |
| **Petrópolis** | Bloqueado | Enchente e deslizamento não estão separados nas fontes disponíveis — dado insuficiente para inferência nesta entrega. |
| **Frente externa (Reino Unido / Copernicus EMS)** | Piloto internacional concluído | Piloto UK: 7.476 pontos, 201 eventos independentes, AUC agrupada 0,79. Multirregião Copernicus: 25.249 pontos em 119 áreas (serra e planície), transferência serra↔planície sem perda relevante de desempenho (0,77). Achado metodológico: a definição de negativo afeta a métrica mais do que o fenômeno em si — AUC alto pode medir o critério de amostragem, não a suscetibilidade real. |

Em nenhuma das três regiões brasileiras há negativo formal aceito (`C4_BLOCKED_NO_FORMAL_NEGATIVES`); a ausência de rótulo negativo oficial é uma condição declarada e auditável, não contornada por proxy.

---

## Onde estão os resultados

- **Recife** (pipeline completo — aquisição de evento, features físico-hidrológicas, modelagem, motor de inferência, API): [`outputs_public/data/susc_20a_aquisicao_eventos_reais_recife/`](outputs_public/data/susc_20a_aquisicao_eventos_reais_recife/) até [`susc_20f_pipeline_geoprocessamento_sob_demanda_recife/`](outputs_public/data/susc_20f_pipeline_geoprocessamento_sob_demanda_recife/). Relatório final: `susc_20c_modelagem_validacao_estatistica_rigorosa_recife/reports/RELATORIO_v12_master.md`.
- **Camadas físicas genéricas** (HAND/TWI, candidatos de água Sentinel-1/2, janelas de evento): [`outputs_public/data/susc_20g_hand_twi_dinfinity_generico/`](outputs_public/data/susc_20g_hand_twi_dinfinity_generico/) até `susc_20j_sentinel1_sar_water_candidates/`.
- **Curitiba** (aquisição, features, modelagem e a cadeia completa de diagnóstico do colapso temporal): [`outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/`](outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/) — relatórios `susc_20l` a `susc_21a` em sua pasta `reports/`.
- **Frente externa UK/Copernicus EMS** (script, tabela de pontos harmonizada, modelos piloto): [`scripts/externo/`](scripts/externo/) e [`docs/metodologia_cientifica/ext_tabela_unica_e_pool_harmonizado_v1.md`](docs/metodologia_cientifica/ext_tabela_unica_e_pool_harmonizado_v1.md).
- **Figuras finais** (mapas por região, matriz de vizinhança DINOv2, PCA, heatmap de similaridade): [`outputs_public/figures/`](outputs_public/figures/).
- **Embeddings DINOv2** (análise estrutural auxiliar — similaridade, k-NN, PCA, medoids, outliers; testados e descartados como feature causal via comparação A/B contra o modelo físico): [`docs/metodologia_cientifica/PLANO_ACAO_produto_v1.md`](docs/metodologia_cientifica/PLANO_ACAO_produto_v1.md), seção 0 e Fase 1.
- **Linhagem do pipeline SUSC-17→19** (versão anterior ao pivô causal SUSC-20, review-only): [`outputs_public/suscetibilidade/`](outputs_public/suscetibilidade/) e [`datasets/suscetibilidade/`](datasets/suscetibilidade/). Mantida integralmente no repositório como registro de como o projeto chegou à rota atual — não é o resultado principal, mas documenta a evolução metodológica.

---

## Metodologia

O projeto segue um princípio fixo: **a base causal é físico-hidrológica**. Variáveis orbitais (Sentinel-2, CBERS) e representações auto-supervisionadas (DINOv2) entram só como evidência auxiliar de apoio à revisão — nunca como variável causal do modelo, nunca como substituto de física conhecida. O modelo não deve "aprender" a enchente a partir de um padrão de imagem; ele testa hipóteses físicas já estabelecidas contra evento real.

Rota primária: regressão logística penalizada de Firth, escolhida por lidar bem com eventos raros e por manter coeficientes interpretáveis com significância estatística e sinal esperado. GBM monotônico entra apenas como diagnóstico complementar — para checar se há não linearidade real no fenômeno — nunca como modelo de produção, e é sempre restrito a manter a mesma direção causal esperada em cada feature.

Validação: leave-one-out cross-validation (LOO) e k-fold repetido, sempre reportando desvio-padrão, nunca um único número. Testes de coerência física (sinal e significância de cada coeficiente) fazem parte da validação, não são um passo opcional.

---

## Ambiente de execução

Duas linhas de dependência, não misturar:

- **Linha causal (SUSC — Firth + candidatos interpretáveis)**: ambiente conda dedicado, com trava real de versão de Python (`<3.11`) exigida por `firthlogist`/`interpret`. Configuração em `environment.yml` (conda, recomendado) ou `requirements-susc.txt` (pip/venv). Detalhes em `docs/ambiente_treino_susc.md`.
- **Linha DINOv2/embeddings** (auxiliar, análise estrutural): `requirements.txt`, ambiente Python padrão.

```bash
conda env create -f environment.yml
conda activate revp-susc
python -m pytest tests -q
```

Toda a modelagem foi rodada localmente, direto na máquina de desenvolvimento — sem serviço externo de treinamento.

---

## Estrutura do repositório

```text
REV-P/
├── docs/                     # Documentação metodológica e cronograma científico
├── datasets/                 # Registries, schemas e evidência estruturada do Protocolo C
│   └── suscetibilidade/      # Linhagem do pipeline SUSC-17→19 (anterior ao pivô causal)
├── outputs_public/
│   ├── data/susc_20*/        # Núcleo causal: eventos, features, modelagem, API (por região)
│   ├── figures/               # Figuras finais para artigo e apresentação
│   ├── tables/                # Tabelas consolidadas citáveis
│   ├── metrics/                # Métricas descritivas reais (similaridade, PCA, robustez)
│   ├── execution_reports/     # Índice de entrega e relatórios de restrição metodológica
│   ├── model/                 # Estado do modelo por região
│   └── suscetibilidade/      # Linhagem do pipeline SUSC-17→19 (anterior ao pivô causal)
├── scripts/externo/           # Frente externa UK/Copernicus EMS
├── tests/                     # Testes automatizados do pipeline
└── environment.yml            # Ambiente conda da linha causal (Firth)
```

---

## Limitações

- Nenhuma das três regiões brasileiras tem negativo formal aceito; a ausência é documentada, não contornada.
- Curitiba: o modelo físico não generaliza para o período temporal de 2026 apesar de diagnóstico extensivo; reportado como resultado negativo informativo, não escondido.
- Petrópolis: mistura enchente/deslizamento nas fontes impede inferência nesta entrega.
- Corpus de embeddings DINOv2 é pequeno (12 vetores reais) — suficiente para análise estrutural exploratória, não para validação estatística de desempenho.
- Fontes externas oficiais (COMPDEC, DRM-RJ, Defesa Civil, CPRM) têm solicitações formais pendentes de resposta.

---

## Próximos passos

- Obter geometria oficial de evento em Petrópolis (DRM-RJ) para destravar a região.
- Resolver a separação de fenômeno (enchente x deslizamento) em Petrópolis 2022.
- Consolidar a métrica final de Curitiba como resultado negativo documentado no artigo, em vez de buscar mais diagnósticos.
- Ampliar a tabela de pontos harmonizada da frente externa como evidência de transferência regional.
