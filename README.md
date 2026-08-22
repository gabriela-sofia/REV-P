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

A base do modelo é sempre físico-hidrológica: acúmulo de água, capacidade de escoamento, proximidade a corpo hídrico, chuva antecedente. Dado orbital e representação aprendida (DINOv2) nunca entram como causa. A checagem do DINOv2 foi formal, não descartada por suposição: a razão de verossimilhança deu significativa à primeira vista, mas a correção de pseudorreplicação por patch mostrou que o sinal era artefato de amostra — por isso ficou de fora, com o motivo estatístico documentado.

O segundo ponto é reportar o que sai, mesmo negativo: o colapso de generalização temporal em Curitiba (AUC 0,65 → 0,52) está mantido e documentado com sete diagnósticos que descartaram outras explicações.

| Região | Resultado |
|---|---|
| **Recife** | Firth, n=278, LOO-AUC = 0,68. Motor de inferência e API entregues. |
| **Curitiba** | Firth não generaliza (AUC 0,65 → 0,52 em holdout real); resultado negativo documentado. |
| **Petrópolis** | Sem inventário local. Mapa de suscetibilidade servido por transferência sem referência local — predição, nunca afirmação de acerto. |
| **Frente externa (UK/Copernicus)** | Holdout temporal em 8 cortes entre 2000 e 2025, AUC médio 0,80. Ajuste por classe de relevo: serra 0,7916, planície 0,7245, transferência planície→serra 0,7957. |

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
