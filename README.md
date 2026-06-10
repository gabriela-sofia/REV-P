# REV-P — Repositório de Entrega dos Artefatos

Este repositório reúne os artefatos públicos do projeto **REV-P**, desenvolvido como um pipeline auditável para organização, validação e inspeção de evidências territoriais, geoespaciais e visuais associadas à suscetibilidade urbana a inundações e alagamentos.

A entrega disponibiliza código-fonte, configurações, documentação metodológica, registries, manifests, testes automatizados, tabelas auxiliares, métricas, figuras e relatórios finais de execução.

Arquivos brutos pesados e saídas locais completas não são versionados no GitHub. Quando aplicável, esses arquivos são representados por manifests, registries, hashes, relatórios resumidos e artefatos públicos derivados.

---

## 1. Objetivo do projeto

O **REV-P** tem como objetivo estruturar uma trilha auditável de evidências para apoiar análise físico-ambiental urbana com imagens Sentinel, evidências territoriais externas e embeddings visuais DINOv2.

O projeto não parte diretamente para uma classificação operacional. Antes disso, organiza:

- patches territoriais de Recife, Petrópolis e Curitiba;
- assets Sentinel candidatos;
- evidências GIS e territoriais externas;
- registries de fontes, lacunas e decisões metodológicas;
- embeddings visuais extraídos com encoder DINOv2 congelado;
- auditorias de QA, rastreabilidade e governança de claims;
- figuras, tabelas, métricas e relatórios finais para submissão acadêmica.

O foco principal é permitir **revisão humana**, **rastreabilidade científica** e **controle metodológico** antes de qualquer etapa supervisionada ou operacional.

---

## 2. Escopo científico e limites metodológicos

O REV-P opera em modo **review-only**.

No estado atual, o projeto:

- não entrega um detector operacional de inundação;
- não declara *ground truth* operacional em nível de patch;
- não cria labels binários de treinamento;
- não treina classificador supervisionado operacional;
- não usa DINOv2 como classificador físico-ambiental;
- não transforma evidência contextual em validação automática de evento observado;
- não transforma coerência externa em predição.

As imagens Sentinel, os embeddings DINOv2, as evidências GIS e as fontes externas são usados como suporte para análise estrutural, inspeção contextual, auditoria e revisão humana.

Essa separação é parte central da metodologia: evidência visual, evidência territorial e referência observacional candidata não são tratadas como a mesma coisa.

---

## 3. Corpus consolidado

O corpus atual possui **59 patches territoriais/contextuais** distribuídos em três regiões brasileiras.

| Região | Patches territoriais | Assets Sentinel candidatos |
|---|---:|---:|
| Recife | 18 | 37 |
| Petrópolis | 27 | 48 |
| Curitiba | 14 | 43 |
| **Total** | **59** | **128** |

Os **59 patches** representam unidades territoriais/contextuais.

Os **128 assets Sentinel** representam imagens candidatas associadas ao pipeline Sentinel-first.

Essas contagens não são equivalentes: uma descreve o corpus territorial consolidado; a outra descreve o inventário de imagens Sentinel candidatas.

---

## 4. Estrutura do repositório

```text
REV-P/
├── configs/                 # Configurações e templates de execução local
├── datasets/                # Registries CSV/JSON, schemas e tabelas auxiliares
├── docs/                    # Documentação técnica e metodológica
├── manifests/               # Manifests auditáveis de corpus, preflight e validação
├── outputs_public/          # Artefatos públicos finais da entrega
│   ├── figures/             # Figuras finais usadas no artigo/apresentação
│   ├── tables/              # Tabelas finais e auxiliares
│   ├── metrics/             # Métricas finais e resumos quantitativos
│   ├── logs_summary/        # Logs resumidos de execução, QA e testes
│   ├── execution_reports/   # Relatórios finais de execução/auditoria
│   └── model/               # Declaração sobre ausência de modelo operacional treinado
├── scripts/                 # Scripts de extração, análise, QA e orquestração
├── tests/                   # Testes automatizados do pipeline
├── requirements.txt         # Dependências Python
└── README.md                # Este arquivo
```

---

## 5. Artefatos entregues

A tabela abaixo descreve os principais artefatos submetidos, sua função no projeto e os diretórios correspondentes.

| Artefato | Diretório/arquivo | Função no projeto |
|---|---|---|
| README | `README.md` | Descreve a entrega, a estrutura dos artefatos, os limites metodológicos e os parâmetros relevantes de execução. |
| Código-fonte | `scripts/` | Scripts do pipeline Sentinel-first, DINOv2, QA, auditorias, análises estruturais, exportação de figuras/tabelas e orquestração. |
| Configurações | `configs/` | Templates e checklists de execução local, incluindo variáveis de ambiente para execução DINO local controlada. |
| Dependências | `requirements.txt` | Lista de bibliotecas Python necessárias para execução do pipeline. |
| Registries públicos | `datasets/*.csv`, `datasets/*.json` | Tabelas auditáveis com corpus, fontes externas, decisões, lacunas, evidências, claims permitidos/proibidos e estados metodológicos. |
| Schemas dos registries | `datasets/schemas/` | Estrutura esperada dos campos dos registries públicos. |
| Manifests | `manifests/` | Inventários e registros auditáveis de patches, assets Sentinel, preflight e validação. |
| Documentação técnica | `docs/` | Explicação metodológica do pipeline, Protocolo C, DINO Sentinel-first, linhagem dos patches e governança dos claims. |
| Testes automatizados | `tests/` | Testes de consistência e regressão do pipeline. |
| Figuras finais | `outputs_public/figures/` | Figuras finais ou publicáveis usadas como apoio visual no artigo/apresentação. |
| Tabelas finais | `outputs_public/tables/` | Tabelas consolidadas de corpus, evidência externa, análise estrutural e apoio ao artigo. |
| Métricas finais | `outputs_public/metrics/` | Métricas estruturais, quantitativas ou de QA usadas como evidência de resultado. |
| Logs resumidos | `outputs_public/logs_summary/` | Registros resumidos de execução, testes e QA. |
| Relatórios de execução | `outputs_public/execution_reports/` | Relatórios finais de preflight, readiness, QA automation, validação externa, rastreabilidade e execução do pipeline. |
| Declaração sobre modelo | `outputs_public/model/NO_OPERATIONAL_TRAINED_MODEL.md` | Explicita que o projeto não entrega pesos de classificador supervisionado operacional. |

---

## 6. Artefatos públicos de resultado

A pasta `outputs_public/` concentra os arquivos finais leves usados para comprovar os resultados apresentados no artigo e na apresentação.

O índice principal dos artefatos está em:

```text
outputs_public/execution_reports/final_delivery_artifact_index.md
```

Esse índice lista os arquivos finais entregues, seus caminhos, sua função no projeto e sua observação metodológica.

### 6.1. Figuras finais

Diretório:

```text
outputs_public/figures/
```

Contém figuras finais ou publicáveis, incluindo:

- figura principal validada de Recife;
- figuras finais de Petrópolis e Curitiba, quando disponíveis;
- gráficos de PCA;
- gráficos de clustering;
- grafos de vizinhança;
- mapas/matrizes de similaridade;
- figuras de outliers, medoids e análise estrutural;
- figuras finais usadas no artigo e/ou apresentação.

A figura principal validada de Recife está registrada como:

```text
outputs_public/figures/fig_recife_main_publication_v15_final.png
```

Essa figura usa o patch principal `REC_00205` como entrada visual/espectral e preserva o uso contextual da evidência, sem declarar ground truth operacional, classe, label ou predição.

### 6.2. Tabelas finais

Diretório:

```text
outputs_public/tables/
```

Contém tabelas finais ou auxiliares, incluindo:

- tabela do corpus consolidado;
- tabela dos 59 patches;
- distribuição por região;
- tabelas de evidência externa;
- tabelas de resultados usadas no artigo;
- tabelas de DINOv2, PCA, vizinhos próximos, outliers e similaridade;
- tabelas de apoio à interpretação metodológica.

### 6.3. Métricas finais

Diretório:

```text
outputs_public/metrics/
```

Contém métricas finais ou resumos quantitativos, incluindo:

- métricas de QA;
- métricas de PCA;
- métricas de clustering;
- resumos de vizinhos próximos;
- resumos de outliers;
- análise de similaridade;
- robustez;
- readiness;
- sensibilidade;
- resultados quantitativos citados ou derivados do artigo.

Essas métricas têm função descritiva, diagnóstica e estrutural. Elas não medem desempenho operacional de detecção de inundação.

### 6.4. Logs resumidos

Diretório:

```text
outputs_public/logs_summary/
```

Contém logs leves e resumidos, incluindo:

- log resumido de execução;
- resultado de testes;
- resumo de QA;
- mensagens finais de validação;
- registros que documentam que o pipeline foi executado.

Logs brutos completos e diretórios locais pesados permanecem fora do repositório público.

### 6.5. Relatórios de execução e auditoria

Diretório:

```text
outputs_public/execution_reports/
```

Contém relatórios finais de execução/auditoria, incluindo:

- report de preflight;
- report de readiness;
- report de QA automation;
- report de validação externa;
- report de rastreabilidade;
- report de guardrails;
- report final do pipeline;
- índice final de artefatos da entrega.

### 6.6. Modelo

Diretório:

```text
outputs_public/model/
```

Este projeto **não entrega modelo supervisionado operacional treinado**.

O arquivo abaixo explicita essa decisão metodológica:

```text
outputs_public/model/NO_OPERATIONAL_TRAINED_MODEL.md
```

O REV-P usa DINOv2 como encoder visual congelado para extração de embeddings e análise estrutural. Não há pesos finais de classificador supervisionado a serem submetidos nesta etapa.

---

## 7. DINOv2 Sentinel-first

A trilha DINOv2 Sentinel-first usa imagens Sentinel como entrada visual e um encoder DINOv2 congelado para extração de embeddings.

No estado consolidado da entrega:

- há 12 embeddings reais;
- são 4 embeddings por região;
- cada vetor possui 768 dimensões;
- as dimensões individuais não são interpretadas como variáveis físicas diretas;
- o vetor completo é usado para análise de similaridade, vizinhança, PCA, medoids, outliers e revisão estrutural.

O DINOv2 não é usado como detector, classificador supervisionado ou preditor operacional de inundação.

---

## 8. Protocolo C e referência observacional

O **Protocolo C** organiza o uso de evidências externas e referências observacionais candidatas.

Ele existe para separar:

- evidência contextual;
- proxy físico-ambiental;
- suporte territorial externo;
- candidato de referência observacional;
- ground truth operacional.

No estado atual, o REV-P não promove automaticamente fontes externas a ground truth operacional. A ausência de ground truth operacional patch-level é tratada como uma limitação metodológica explícita e auditável.

Documentos relacionados estão em:

```text
docs/metodologia_cientifica/
```

---

## 9. Linhagem dos patches

Os patches são recortes territoriais pré-existentes sobre áreas urbanas de Recife, Petrópolis e Curitiba.

O pipeline DINOv2 opera sobre imagens Sentinel associadas a esses patches, mas não redefine automaticamente seus limites territoriais.

A documentação de linhagem e grounding está em:

```text
docs/patch_lineage_and_grounding.md
```

---

## 10. O que não está versionado

Por limitação de tamanho, licença, reprodutibilidade local ou política de entrega pública, os seguintes arquivos não são versionados no GitHub:

- GeoTIFFs brutos;
- shapefiles brutos;
- GeoJSONs locais convertidos;
- PE3D/MDE bruto;
- embeddings `.npz` completos;
- modelo DINO original;
- caches;
- ambientes virtuais;
- diretórios `local_runs/` completos;
- logs brutos extensos;
- arquivos temporários de desenvolvimento;
- dados privados do workspace original.

Esses itens permanecem locais quando necessário. A entrega pública disponibiliza manifests, registries, relatórios, tabelas resumidas, figuras derivadas e logs leves.

---

## 11. Execução local

### 11.1. Preparação do ambiente

Exemplo de preparação em ambiente Python:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 11.2. Execução dos testes

```bash
python -m pytest tests
```

### 11.3. Geração/validação dos artefatos públicos

A pasta `outputs_public/` pode ser gerada, finalizada ou validada pelos scripts de organização da entrega.

Execução base:

```bash
python scripts/repository/build_outputs_public_delivery.py
```

Finalização:

```bash
python scripts/repository/build_outputs_public_delivery.py --finalize
```

Validação:

```bash
python scripts/repository/build_outputs_public_delivery.py --validate-only
```

Esses comandos produzem ou verificam os artefatos públicos finais sem exigir que dados brutos pesados sejam versionados no GitHub.

---

## 12. Parâmetros e decisões metodológicas relevantes

Parâmetros e decisões fixadas nesta entrega:

| Item | Estado adotado |
|---|---|
| Modo do projeto | `review-only` |
| Fonte visual principal | Sentinel-first |
| Encoder visual | DINOv2 congelado |
| Dimensão dos embeddings | 768D |
| Número de embeddings reais | 12 |
| Regiões | Recife, Petrópolis e Curitiba |
| Corpus territorial/contextual | 59 patches |
| Assets Sentinel candidatos | 128 |
| Modelo operacional treinado | Não entregue |
| Ground truth operacional patch-level | Não declarado |
| Labels binários | Não criados |
| Classificação supervisionada | Não realizada como entrega operacional |
| Uso das evidências externas | Contextual, auditável e metodológico |
| Uso das figuras | Interpretação visual/espectral e suporte territorial externo |
| Uso das métricas | Análise estrutural, QA, similaridade e revisão |

---

## 13. Interpretação correta dos resultados

Os resultados do REV-P devem ser lidos como evidência de organização, auditoria e análise estrutural do corpus.

Interpretação permitida:

- análise de coerência externa de suscetibilidade;
- suporte territorial externo;
- leitura visual/espectral Sentinel;
- análise estrutural por embeddings;
- identificação de vizinhos próximos;
- identificação de outliers e medoids;
- PCA e clustering exploratórios;
- QA e rastreabilidade;
- revisão humana orientada por evidências.

Interpretação não permitida:

- detecção operacional de inundação;
- predição de enchente;
- validação automática de evento observado;
- classe supervisionada;
- label binário;
- negativo por ausência de evidência;
- ground truth patch-level;
- desempenho operacional de classificador.

---

## 14. Relação com a entrega acadêmica

Este repositório foi organizado para atender à exigência de disponibilizar uma pasta pública com todos os artefatos relevantes do projeto.

A entrega inclui:

- código-fonte;
- documentação técnica;
- README objetivo;
- parâmetros de execução;
- registries e manifests;
- tabelas auxiliares;
- figuras finais;
- métricas finais;
- logs resumidos;
- relatórios de execução;
- declaração explícita sobre ausência de modelo operacional treinado.

Os arquivos estão organizados por função e identificados por diretório, com índice final em:

```text
outputs_public/execution_reports/final_delivery_artifact_index.md
```

---

## 15. Como citar ou navegar pelo repositório

Para uma revisão rápida, recomenda-se seguir esta ordem:

1. `README.md`
2. `outputs_public/README.md`
3. `outputs_public/execution_reports/final_delivery_artifact_index.md`
4. `outputs_public/figures/`
5. `outputs_public/tables/`
6. `outputs_public/metrics/`
7. `outputs_public/execution_reports/`
8. `docs/`
9. `datasets/`
10. `manifests/`
11. `scripts/`
12. `tests/`

---

## 16. Observação final

O REV-P deve ser entendido como um pipeline auditável, Sentinel-first e review-only para organizar evidências físico-ambientais urbanas, suporte territorial externo e análise visual-estrutural com DINOv2.

A contribuição principal do projeto não é entregar um detector operacional, mas consolidar uma base rastreável, validável e metodologicamente segura para futuras etapas de referência observacional, revisão supervisionada e eventual modelagem operacional.
