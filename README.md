# REV-P

O REV-P investiga vulnerabilidade urbana a enchentes em três regiões brasileiras — Recife, Curitiba e Petrópolis. O produto final é um modelo causal de suscetibilidade a enchente: parte de relações físico-hidrológicas já conhecidas e testa essas relações contra eventos reais, em vez de deixar um modelo "descobrir" padrões em imagem de satélite. Recife é a entrega madura e auditada ponta a ponta; Curitiba e Petrópolis são resultados parciais, reportados como vieram — inclusive quando o resultado é negativo.

A narrativa científica completa está em [`docs/metodologia_cientifica/revp_narrativa_cientifica_consolidada.md`](docs/metodologia_cientifica/revp_narrativa_cientifica_consolidada.md).

---

## Frentes de trabalho

O projeto se organiza em três frentes, metodologicamente separadas:

| Frente | O que é | Onde está |
|---|---|---|
| **Linha causal (SUSC-20)** | O produto central: modelo físico-hidrológico causal (Firth), por região, mais a frente externa de validação (UK/Copernicus EMS). É tudo que veio depois da consolidação da entrega científica. | [`outputs_public/data/susc_20*/`](outputs_public/data/), [`scripts/externo/`](scripts/externo/) |
| **Protocolo C** | Infraestrutura de aquisição e adjudicação de evidência (geometria oficial, série hidrometeorológica, revisão humana) que sustenta o ground truth da linha causal. | [`datasets/protocolo_c/`](datasets/protocolo_c/), [`docs/protocolo_c/`](docs/protocolo_c/) |
| **Linhagem anterior à consolidação** | Pipeline exploratório pré-causal (SUSC_01–19) e a análise estrutural DINOv2 — mantidos por rastreabilidade, não são o resultado principal. | [`outputs_public/suscetibilidade/`](outputs_public/suscetibilidade/), [`datasets/dino_README.md`](datasets/dino_README.md) |

---

## O diferencial da linha causal

A linha causal não maximiza acurácia — ela testa hipótese física. A base do modelo é sempre físico-hidrológica (acúmulo de água, capacidade de escoamento, proximidade a corpo hídrico, chuva antecedente); dado orbital e representação aprendida (DINOv2) nunca entram como causa, só como checagem auxiliar. Essa checagem foi feita de forma rigorosa, não descartada por suposição: um teste formal (razão de verossimilhança) comparando o modelo com e sem DINOv2 deu estatisticamente significativo à primeira vista, mas a checagem de pseudorreplicação por patch mostrou que o sinal era artefato de amostra, não causa real — por isso DINOv2 ficou de fora, com o motivo estatístico documentado, não por decisão arbitrária.

O segundo diferencial é reportar o que sai, mesmo quando o resultado é negativo: o colapso de generalização temporal em Curitiba (AUC 0,65 → 0,52) foi mantido e documentado com 7+ diagnósticos que descartaram outras explicações, em vez de escondido ou maquiado.

---

## Metodologia

Regressão logística penalizada de Firth é a rota primária (lida bem com eventos raros, mantém coeficientes interpretáveis). GBM monotônico causal entra só como diagnóstico de não linearidade, nunca como modelo de produção. Validação por leave-one-out e k-fold repetido, sempre com desvio-padrão. Eventos vêm de fontes oficiais (Defesa Civil, ANA, Diário Oficial, bases internacionais) via Protocolo C — nenhuma das três regiões tem negativo formal aceito ainda, e essa ausência é declarada, não contornada por proxy.

---

## Estado atual por região

| Região | Resultado |
|---|---|
| **Recife** | Firth, n=278, LOO-AUC = 0,68. Motor de inferência e API entregues. |
| **Curitiba** | Firth não generaliza (AUC 0,65 → 0,52 em holdout real); resultado negativo documentado. Na base harmonizada não sustenta holdout próprio: 114 negativos contra 1.238 positivos. |
| **Petrópolis** | Sem inventário local: zero linhas na tabela única. Mapa de suscetibilidade gerado (172.015 células), servido por transferência sem referência local — predição, nunca afirmação de acerto. |
| **Frente externa (UK/Copernicus)** | Piloto concluído. Holdout temporal (E4) fechado: 8 cortes entre 2000 e 2025, AUC médio 0,80. Ajuste por classe de relevo (E3) fechado: serra 0,7916, planície 0,7245, transferência planície→serra 0,7957. |


**Serviço (E6).** O contrato de inferência roda como função pura em `scripts/servico/`: cinco portões em ordem declarada, escore por área com IC de bootstrap de grupos, *model card* e explicação gerada por regras sobre o payload. Recife responde `mvp_local`, Curitiba por transferência caracterizada e Petrópolis por transferência sem referência local. A grade de suscetibilidade (E5) cobre as três regiões a 120 m, e célula fora do domínio do ajuste fica vazia no mapa em vez de receber escore baixo. Ver `docs/metodologia_cientifica/ext_servico_contrato_inferencia_v1.md`.
---

## Estrutura do repositório

```text
REV-P/
├── docs/                      # Documentação metodológica e narrativa científica
├── datasets/                  # Registries e evidência estruturada do Protocolo C
│   └── suscetibilidade/       # Linhagem anterior à consolidação
├── outputs_public/
│   ├── data/susc_20*/         # Linha causal: eventos, features, modelagem, API (por região)
│   ├── figures/                # Figuras finais
│   ├── tables/                 # Tabelas consolidadas citáveis
│   ├── model/                  # Estado do modelo por região
│   └── suscetibilidade/       # Linhagem anterior à consolidação
├── scripts/externo/            # Frente externa UK/Copernicus EMS
├── tests/                      # Testes automatizados
└── environment.yml             # Ambiente conda da linha causal (Firth)
```

---

## Ambiente de execução

```bash
conda env create -f environment.yml
conda activate revp-susc
python -m pytest tests -q
```

Linha causal (Firth): ambiente conda com Python `<3.11`, ver `environment.yml`/`docs/ambiente_treino_susc.md`. Linha DINOv2: `requirements.txt`, Python padrão. Toda a modelagem roda localmente, sem serviço externo de treinamento.
