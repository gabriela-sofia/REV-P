# REV-P

[![Status](https://img.shields.io/badge/Status-Em_Desenvolvimento_/_Auditoria-orange.svg)](https://github.com/gabriela-sofia/REV-P)
[![Mode](https://img.shields.io/badge/Mode-Review__Only-blue.svg)](https://github.com/gabriela-sofia/REV-P)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB.svg)](https://www.python.org/)

O **REV-P** é um pipeline auditável para organizar, validar e inspecionar evidências territoriais, geoespaciais e visuais associadas à suscetibilidade urbana a inundações e alagamentos.

O projeto combina imagens Sentinel, embeddings DINO, proxies GIS e documentação metodológica para apoiar revisão humana antes de qualquer etapa preditiva. Seu foco é construir uma base rastreável de evidências, com governança explícita de fontes, limites e critérios de validação.

---

## 🎯 Por que este projeto existe

Pesquisas aplicadas a risco urbano frequentemente dependem de bases heterogêneas, incompletas ou difíceis de validar em escala local. O REV-P nasce para organizar essa etapa anterior à modelagem: a auditoria das evidências.

Em vez de partir diretamente para classificação ou predição, o projeto estrutura um fluxo de validação que permite responder:

* quais patches territoriais existem;
* quais imagens Sentinel estão associadas a eles;
* quais evidências externas apoiam a interpretação físico-ambiental;
* quais fontes são rastreáveis;
* quais lacunas ainda impedem validação operacional;
* quais dados podem ser usados apenas como suporte contextual.

---

## ⚙️ O que o REV-P faz

O pipeline atualmente executa quatro funções principais:

* **Organização do corpus territorial:** estrutura patches urbanos de Recife, Petrópolis e Curitiba, preservando sua linhagem espacial.
* **Extração visual Sentinel-first:** usa DINO como encoder visual congelado para extrair embeddings estruturais de imagens Sentinel.
* **Análise estrutural dos patches:** aplica PCA, clustering, vizinhos próximos e detecção de outliers para inspecionar padrões visuais do corpus.
* **Validação contextual auditável:** usa Protocolo C, registries e proxies GIS para documentar fontes, lacunas, critérios de elegibilidade e limites de interpretação.

O REV-P não é apenas um conjunto de scripts. Ele funciona como uma camada de governança científica para impedir que evidências contextuais sejam confundidas com rótulos operacionais.

---

## 🗺️ Corpus Atual

O corpus consolidado reúne **59 patches urbanos** distribuídos em três regiões brasileiras:

| Região | Patches | Assets Sentinel candidatos |
| :--- | :---: | :---: |
| Recife | 18 | 37 |
| Petrópolis | 27 | 48 |
| Curitiba | 14 | 43 |
| **Total** | **59** | **128** |

Os patches representam unidades territoriais/contextuais. Os assets Sentinel representam imagens candidatas associadas ao pipeline de análise visual.

---

## 🧬 Trilha DINO Sentinel-first

O pipeline segue uma trilha sequencial, documentada por identificadores de estado metodológico:

```text
[v1fu] Manifesto Sentinel
   └──> [v1fv] Preflight Local
            └──> [v1fx] Execução Smoke (Embeddings)
                     └──> [v1fy–v1gi] Análise Estrutural
                              └──> [v1gn–v1gp] Auditorias Operacionais
                                       └──> [v1gq–v1gt] Auditorias GIS
```

1. **Manifesto Sentinel:** inventário inicial dos assets Sentinel elegíveis.
2. **Preflight local:** checagem de integridade e disponibilidade dos arquivos no workspace privado.
3. **Execução smoke:** leitura real de pixels e extração local de embeddings DINO.
4. **Análise estrutural:** PCA, clustering, vizinhos próximos, outliers e proveniência.
5. **Auditorias operacionais:** verificação de saúde do pipeline e prontidão de execução.
6. **Auditorias GIS:** cruzamento multicritério com evidências territoriais externas.

---

## 📂 Estrutura do Repositório

```text
REV-P/
├── configs/          # Parâmetros e configurações do pipeline
├── datasets/         # Registries CSV/JSON de corpus, evidências e auditorias
├── docs/             # Documentação técnica e metodológica
├── manifests/        # Manifests auditáveis de patches, preflight e validação
├── scripts/          # Scripts de extração, análise e orquestração
├── tests/            # Testes automatizados do pipeline
└── requirements.txt  # Dependências Python
```

---

## 🧪 Execução Local Básica

Para preparar o ambiente local:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

Os scripts principais ficam em [`scripts/`](./scripts/) e os testes automatizados ficam em [`tests/`](./tests/).

> A reprodução integral depende dos dados locais esperados pelo pipeline. Os arquivos pesados não são versionados no GitHub.

---

## 📦 O que não está versionado

Por controle de tamanho, licenciamento e segurança operacional, o repositório público não versiona:

* GeoTIFFs brutos;
* shapefiles e GeoJSONs pesados;
* embeddings `.npz`;
* caches locais;
* modelos pesados;
* outputs de execução;
* arquivos intermediários gerados em `local_runs/`.

O GitHub mantém apenas os artefatos leves e auditáveis: código, configurações, manifests, registries, documentação e testes.

---

## 🔍 Protocolo C

O **Protocolo C** é a camada de governança do REV-P. Ele organiza evidências externas, fontes documentais, critérios de elegibilidade e lacunas de validação antes que qualquer dado seja usado como referência observacional.

A função do Protocolo C é separar:

* evidência contextual;
* suporte territorial;
* registro observacional candidato;
* lacuna metodológica;
* referência que ainda exige validação humana.

Essa separação evita que uma heurística do próprio projeto seja tratada como verdade de campo.

Os registries associados ao Protocolo C estão organizados em [`datasets/`](./datasets/), com registros CSV/JSON de eventos candidatos, fontes, lacunas, decisões e prioridades de geocodificação.

---

## 🧭 Limites Metodológicos

O REV-P opera em modo `review_only=true`.

Isso significa que o projeto organiza, cruza e audita evidências, mas não transforma automaticamente essas evidências em rótulos operacionais.

No estado atual, o projeto:

* não declara *ground truth* patch-level;
* não treina classificador supervisionado operacional;
* não afirma detecção automática de inundação observada;
* não promove patches automaticamente a positivos ou negativos;
* não usa DINO como classificador físico-ambiental.

As evidências visuais, GIS e documentais sustentam interpretação contextual, auditoria e revisão humana.

---

## 📚 Documentação Técnica

A documentação completa está organizada em [`docs/`](./docs/) e os registros auditáveis em [`datasets/`](./datasets/).

Principais pontos de entrada:

* [`docs/estado_metodologico_revp.md`](./docs/estado_metodologico_revp.md) — estado metodológico consolidado.
* [`docs/metodologia_cientifica/dino_sentinel_embedding_protocol.md`](./docs/metodologia_cientifica/dino_sentinel_embedding_protocol.md) — protocolo de extração de embeddings DINO.
* [`docs/metodologia_cientifica/patch_lineage_and_grounding.md`](./docs/metodologia_cientifica/patch_lineage_and_grounding.md) — linhagem territorial dos patches.
* [`docs/metodologia_cientifica/camada_referencia_contextual_validada.md`](./docs/metodologia_cientifica/camada_referencia_contextual_validada.md) — diretrizes de claims e uso contextual das evidências.
* [`datasets/`](./datasets/) — registries CSV/JSON usados na auditoria.

---

## 📌 Estado do Projeto

O REV-P está em desenvolvimento e em fase de consolidação metodológica. O foco atual é fortalecer a rastreabilidade, a governança de evidências e a reprodutibilidade do pipeline antes de qualquer etapa de modelagem operacional.

---
