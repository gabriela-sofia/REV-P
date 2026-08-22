# REV-P

Suscetibilidade urbana a enchentes com base causal físico-hidrológica, em três regiões brasileiras — Recife, Curitiba e Petrópolis. O produto é um modelo causal: parte de relações físico-hidrológicas conhecidas e testa essas relações contra eventos reais, em vez de deixar um modelo descobrir padrões em imagem de satélite. Recife é a entrega madura e auditada ponta a ponta; Curitiba e Petrópolis são resultados parciais, reportados como vieram — inclusive quando o resultado é negativo.

Este repositório está reduzido ao produto final. A infraestrutura histórica de aquisição e exploração (Protocolo C, linhagem pré-causal SUSC-01 a SUSC-19, DINOv2 aplicado, suítes de teste dessas frentes) foi retirada da árvore de trabalho e permanece recuperável pelo histórico do git.

---

## Estrutura

```text
REV-P/
├── docs/
│   ├── metodologia_cientifica/  # Narrativa científica consolidada
│   └── tcc_exports/             # Artigo (planejamento_entrega01) e pôster
├── outputs_public/
│   ├── data/susc_20*/           # Linha causal: 11 etapas, cada uma com scripts, dados e relatório
│   └── model/                   # Estado do modelo por região
├── scripts/dino/                # Governança DINOv2 (review-only, fora do modelo)
├── tests/                       # Regressão da linha causal
├── environment.yml              # Ambiente conda da linha causal (Firth)
└── requirements.txt             # Ambiente da linha DINOv2
```

Cada etapa em `outputs_public/data/susc_20*/` é autocontida: traz os scripts que a produziram, os dados curados de saída e um relatório em `reports/`.

| Etapa | O que entrega |
|---|---|
| `susc_20a` | Aquisição de eventos reais em Recife |
| `susc_20b` | Atributos físico-hidrológicos |
| `susc_20c` | Modelagem e validação estatística (Firth) |
| `susc_20d` | Motor de inferência local |
| `susc_20e` | Contrato de API por região |
| `susc_20f` | Geoprocessamento sob demanda |
| `susc_20g` | HAND e TWI por D-infinity |
| `susc_20h` / `susc_20j` | Candidatos de água por Sentinel-2 e Sentinel-1 |
| `susc_20i` | Janelas de evento 2023–2026 |
| `susc_20k` | Curitiba: candidatos, negativos, features e modelagem |

---

## O que sustenta o resultado

A base é uma tabela única harmonizada com **65.070 pontos elegíveis ao ajuste fluvial**, reduzidos a partir de seis fontes, todas na mesma cadeia de derivação de terreno D-infinity e com chuva de fonte única (Open-Meteo/ERA5-Land, cobertura 99,99%). O negativo é declarado em três níveis: observado (Copernicus EMS, 25.249 pontos em 119 AOIs, mais a ativação EMSR720 no Rio Grande do Sul com 216,55 km² na proporção 5,94:1), exclusão qualificada (Environment Agency/UK, 7.476 pontos — 3.738 / 3.738 — em 201 eventos independentes; e 114 pontos de Curitiba) e ausência de registro (Recife e Petrópolis).

A base do modelo é sempre físico-hidrológica: acúmulo de água, capacidade de escoamento, proximidade a corpo hídrico, chuva antecedente. Dado orbital e representação aprendida (DINOv2) nunca entram como causa. A checagem do DINOv2 foi formal, não descartada por suposição: a razão de verossimilhança deu significativa à primeira vista, mas a correção de pseudorreplicação por patch mostrou que o sinal era artefato de amostra — por isso ficou de fora, com o motivo estatístico documentado.

O segundo ponto é reportar o que sai, mesmo negativo: o colapso de generalização temporal em Curitiba (AUC 0,65 → 0,52) está mantido e documentado com sete diagnósticos que descartaram outras explicações.

| Região | Resultado |
|---|---|
| **Recife** | Firth, n=278 (154 positivos da SEDEC / 124 negativos), LOO-AUC = 0,68. Motor de inferência e contrato de API entregues. |
| **Curitiba** | 1.045 positivos do SIAC 156, 114 em exclusão qualificada, 1.471 unidades de validação. Firth não generaliza (AUC 0,65 → 0,52 em holdout real) e não sustenta holdout próprio: 114 negativos contra 1.238 positivos. Resultado negativo documentado. |
| **Petrópolis** | Zero linhas na tabela única. Grade servida por transferência sem referência local — predição, nunca afirmação de acerto. |
| **Frente externa (UK/Copernicus)** | Holdout temporal: 201 eventos em 110 datas (2000–2025), 8 cortes na faixa 0,70–0,88, AUC médio 0,7992. Ajuste por classe de relevo: serra 0,7916 (`hand_m` −1,44 [−3,11; −0,83]), planície 0,7245 (`hand_m` −2,10 [−2,78; −1,56]; `twi_dinf` +0,40 [+0,33; +0,45]), transferência planície→serra 0,7957. |

A grade de aplicação sai a 120 m nas três regiões, derivada da cadeia de terreno já existente: 56.666 células em Recife, 65.275 em Curitiba e 172.015 em Petrópolis. Célula fora do domínio do ajuste fica vazia no mapa em vez de receber escore baixo. O contrato de inferência roda como função pura com cinco portões em ordem declarada; falta o transporte HTTP.

A narrativa completa está em [`docs/metodologia_cientifica/revp_narrativa_cientifica_consolidada.md`](docs/metodologia_cientifica/revp_narrativa_cientifica_consolidada.md); o estado do modelo, em [`outputs_public/model/ESTADO_DO_MODELO.md`](outputs_public/model/ESTADO_DO_MODELO.md).

---

## Metodologia

Regressão logística penalizada de Firth é a rota primária — lida bem com eventos raros e mantém coeficientes interpretáveis. GBM monotônico entra só como diagnóstico de não linearidade, nunca como modelo de produção. Validação por leave-one-out e k-fold repetido, sempre com desvio-padrão. Os eventos vêm de fontes oficiais (Defesa Civil, ANA, Diário Oficial, bases internacionais). Nenhuma das três regiões tem negativo formal aceito, e essa ausência é declarada, não contornada por proxy.

---

## Executar

```bash
conda env create -f environment.yml
conda activate revp-susc
python -m pytest tests/test_susc_2*.py -q
```

Linha causal (Firth): ambiente conda com Python `<3.11`, ver `environment.yml`. Linha DINOv2: `requirements.txt`, Python padrão. Toda a modelagem roda localmente, sem serviço externo de treinamento.
