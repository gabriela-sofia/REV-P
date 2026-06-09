# REV-P — Repositório de Entrega dos Artefatos

Este repositório reúne os artefatos públicos do projeto **REV-P**, desenvolvido como pipeline auditável para organização, validação e inspeção de evidências territoriais, geoespaciais e visuais associadas à suscetibilidade urbana a inundações e alagamentos.

A entrega disponibiliza código-fonte, configurações, documentação metodológica, registries, manifests, testes automatizados, tabelas auxiliares e artefatos públicos de resultado. Arquivos brutos pesados e saídas locais completas não são versionados no GitHub; quando aplicável, são referenciados por manifests, registries e relatórios públicos.

---

## 1. Objetivo do projeto

O **REV-P** tem como objetivo estruturar uma trilha auditável de evidências para apoiar análise físico-ambiental urbana com imagens Sentinel e embeddings DINOv2.

O projeto não parte diretamente para classificação operacional. Antes disso, organiza:

- patches territoriais de Recife, Petrópolis e Curitiba;
- assets Sentinel candidatos;
- evidências GIS e territoriais externas;
- registries de fontes, lacunas e decisões metodológicas;
- embeddings visuais extraídos com encoder DINOv2 congelado;
- auditorias de QA, rastreabilidade e governança de claims.

O foco principal é permitir revisão humana, rastreabilidade científica e controle metodológico antes de qualquer etapa supervisionada ou operacional.

---

## 2. Escopo científico e limites metodológicos

O REV-P opera em modo **review-only**.

No estado atual, o projeto:

- não entrega um detector operacional de inundação;
- não declara *ground truth* operacional em nível de patch;
- não cria labels binários de treinamento;
- não treina classificador supervisionado operacional;
- não usa DINOv2 como classificador físico-ambiental;
- não transforma evidência contextual em validação automática de evento observado.

As imagens Sentinel, embeddings DINOv2, evidências GIS e fontes externas são usadas como suporte para análise estrutural, inspeção contextual, auditoria e revisão humana.

Essa separação é parte central da metodologia: evidência visual, evidência territorial e referência observacional candidata não são tratadas como a mesma coisa.

---

## 3. Corpus consolidado

O corpus atual possui **59 patches territoriais/contextuais** distribuídos em três regiões brasileiras:

| Região | Patches territoriais | Assets Sentinel candidatos |
|---|---:|---:|
| Recife | 18 | 37 |
| Petrópolis | 27 | 48 |
| Curitiba | 14 | 43 |
| **Total** | **59** | **128** |

Os 59 patches representam unidades territoriais/contextuais.  
Os 128 assets Sentinel representam imagens candidatas associadas ao pipeline Sentinel-first.

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
│   ├── metrics/             # Métricas finais ou resumos quantitativos
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
| README | `README.md` | Descreve a entrega, a estrutura dos artefatos, os limites metodológicos e os parâmetros relevantes de execução |
| Código-fonte | `scripts/` | Scripts do pipeline Sentinel-first, DINO, QA, auditorias, análises estruturais, exportação de figuras/tabelas e orquestração |
| Configurações | `configs/` | Templates e checklists de execução local, incluindo variáveis de ambiente para execução DINO local controlada |
| Dependências | `requirements.txt` | Lista de bibliotecas Python necessárias para o pipeline |
| Registries públicos | `datasets/*.csv`, `datasets/*.json` | Tabelas auditáveis com corpus, fontes externas, decisões, lacunas, evidências, claims permitidos/proibidos e estados metodológicos |
| Schemas dos registries | `datasets/schemas/` | Estrutura esperada dos campos dos registries públicos |
| Manifests | `manifests/` | Inventários e registros auditáveis de patches, assets, preflight e validação |
| Documentação técnica | `docs/` | Explicação metodológica do pipeline, Protocolo C, DINO Sentinel-first, linhagem dos patches e governança dos claims |
| Testes automatizados | `tests/` | Testes de consistência e regressão do pipeline |
| Figuras finais | `outputs_public/figures/` | Figuras finais ou publicáveis usadas como apoio visual no artigo/apresentação |
| Tabelas finais | `outputs_public/tables/` | Tabelas consolidadas de resultados, corpus ou apoio ao artigo |
| Métricas finais | `outputs_public/metrics/` | Métricas estruturais, quantitativas ou de QA usadas como evidência de resultado |
| Logs resumidos | `outputs_public/logs_summary/` | Registros resumidos de execução, testes e QA |
| Relatórios de execução | `outputs_public/execution_reports/` | Relatórios finais de preflight, readiness, auditorias, validação externa ou execução do pipeline |
| Declaração sobre modelo | `outputs_public/model/NO_OPERATIONAL_TRAINED_MODEL.md` | Explicita que o projeto não entrega pesos de classificador supervisionado operacional |

---

## 6. Artefatos públicos de resultado

A pasta `outputs_public/` concentra os arquivos finais leves usados para comprovar os resultados apresentados no artigo e na apresentação.

### `outputs_public/figures/`

Contém figuras finais ou publicáveis, como:

- figura visual/espectral de Recife;
- figuras finais de Petrópolis e Curitiba, quando disponíveis;
- gráficos de PCA, clustering, vizinhança, outliers ou análise estrutural;
- imagens finais usadas no artigo ou apresentação.

### `outputs_public/tables/`

Contém tabelas finais ou auxiliares, como:

- tabela do corpus consolidado;
- tabela dos 59 patches;
- tabela por região;
- tabelas de evidência externa;
- tabelas de resultados ou apoio ao artigo.

### `outputs_public/metrics/`

Contém métricas finais ou resumos quantitativos, como:

- métricas de QA;
- métricas de PCA ou clustering;
- resumos de vizinhos próximos;
- resumos de outliers;
- resultados quantitativos citados no artigo.

### `outputs_public/logs_summary/`

Contém logs leves e resumidos, como:

- log resumido de execução;
- resultado de testes;
- resumo de QA;
- mensagens finais de validação.

### `outputs_public/execution_reports/`

Contém relatórios finais de execução/auditoria, como:

- preflight;
- readiness;
- QA automation;
- validação externa;
- relatórios finais do pipeline.

### `outputs_public/model/`

O REV-P não entrega modelo supervisionado operacional treinado. Por isso, esta pasta contém uma declaração metodológica, e não pesos finais de classificador.

O arquivo esperado é:

```text
outputs_public/model/NO_OPERATIONAL_TRAINED_MODEL.md
```

Esse arquivo documenta que o uso de DINOv2 ocorre como encoder visual congelado para extração de embeddings, não como modelo operacional treinado para detecção ou predição.

---

## 7. Dados e arquivos não versionados

Por limite de tamanho, rastreabilidade operacional, licenciamento e organização científica, o repositório público não versiona arquivos brutos ou pesados.

Não são versionados:

- GeoTIFFs Sentinel brutos;
- shapefiles e GeoJSONs pesados;
- embeddings `.npz`;
- caches locais;
- arquivos intermediários completos;
- modelos pesados;
- outputs integrais de execução local;
- pastas `local_runs/`;
- ambientes virtuais Python;
- arquivos temporários.

Esses artefatos são tratados como dados locais ou reprodutíveis a partir dos scripts, configurações e manifests. O que é versionado publicamente é a camada auditável: código, registries, manifests, documentação, testes, métricas resumidas, figuras finais, tabelas finais e relatórios públicos.

Essa decisão evita que o repositório se torne impraticável para avaliação e preserva a separação entre:

- dado bruto local;
- evidência auditável pública;
- resultado leve de entrega;
- documentação metodológica.

---

## 8. Parâmetros relevantes de execução

O projeto não segue o formato simples `train.py` / `test.py`, pois não há treinamento supervisionado operacional nesta etapa. A execução é organizada por scripts numerados do pipeline, principalmente em `scripts/dino/`.

Os parâmetros de execução local são configurados por variáveis de ambiente e arquivos em `configs/`.

Principais variáveis usadas na execução DINO local controlada:

| Variável | Função |
|---|---|
| `REVP_DINO_MODEL_PATH` | Caminho local para o modelo DINOv2. Não deve ser commitado com path absoluto real |
| `REVP_SENTINEL_LOCAL_ROOT` | Diretório local contendo os arquivos Sentinel |
| `REVP_DINO_VISUAL_ROOT` | Diretório local de assets visuais usados na execução |
| `REVP_DINO_ASSET_ROOT` | Diretório local de assets auxiliares do pipeline |
| `REVP_V1PQ_QUEUE_PATH` | Caminho relativo para a fila de execução DINO |
| `REVP_DINO_ALLOW_DOWNLOAD` | Controla download automático de modelo. Deve permanecer `false` por segurança/reprodutibilidade |
| `REVP_DINO_DRY_RUN` | Controla execução simulada. Valor padrão seguro: `true` |
| `REVP_DINO_PIXEL_READ_ALLOWED` | Controla permissão de leitura real de pixels Sentinel |
| `HF_HUB_OFFLINE` | Força modo offline do HuggingFace Hub |
| `REVP_DINO_BATCH_SIZE` | Tamanho de lote para execução local |
| `REVP_DINO_MAX_EXECUTE` | Número máximo de itens processados em execução controlada |

O template público de configuração não contém caminhos absolutos reais. Para executar localmente, deve-se copiar o template, preencher os caminhos apenas no ambiente local e nunca versionar o arquivo preenchido.

---

## 9. Como preparar o ambiente local

### 9.1. Criar ambiente virtual

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 9.2. Instalar dependências

```bash
pip install -r requirements.txt
```

Principais bibliotecas usadas:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `torch`
- `torchvision`
- `timm`
- `transformers`
- `faiss-cpu`
- `umap-learn`
- `hdbscan`

---

## 10. Execução básica dos testes

Quando o ambiente de teste estiver disponível:

```bash
pip install pytest
python -m pytest tests
```

Os testes verificam consistência de scripts, registries, guardrails e etapas automatizadas do pipeline.

---

## 11. Execução local controlada do pipeline DINO

A reprodução integral depende dos dados locais não versionados. Para execução controlada:

1. Copiar o template de configuração local.
2. Preencher os caminhos locais no ambiente da máquina.
3. Executar os scripts de auditoria antes de qualquer leitura real de pixels.
4. Manter `REVP_DINO_DRY_RUN=true` até que todos os gates passem.
5. Habilitar leitura de pixels apenas em execução controlada e revisada.

Exemplo de variáveis em PowerShell:

```powershell
$env:REVP_DINO_MODEL_PATH = "<CAMINHO_LOCAL_DO_MODELO_DINO>"
$env:REVP_SENTINEL_LOCAL_ROOT = "<CAMINHO_LOCAL_DOS_SENTINEL>"
$env:REVP_DINO_ALLOW_DOWNLOAD = "false"
$env:HF_HUB_OFFLINE = "1"
$env:REVP_DINO_DRY_RUN = "true"
$env:REVP_DINO_PIXEL_READ_ALLOWED = "false"
```

Scripts de auditoria e execução local controlada ficam em:

```text
scripts/dino/
```

Exemplos de etapas documentadas no pipeline:

```bash
python scripts/dino/revp_v1qn_local_root_environment_audit.py
python scripts/dino/revp_v1qo_smoke_asset_local_reconciliation.py
python scripts/dino/revp_v1qg_local_dino_model_offline_audit.py
python scripts/dino/revp_v1qi_local_asset_preprocessing_audit.py
python scripts/dino/revp_v1qj_controlled_real_smoke_embedding_executor.py
```

A execução real só deve ocorrer após validação dos gates de auditoria e com os dados locais disponíveis.

---

## 12. Principais scripts do projeto

A pasta `scripts/dino/` contém os scripts centrais da trilha Sentinel-first e DINO.

| Script | Função |
|---|---|
| `revp_v1fu_dino_sentinel_input_manifest.py` | Gera/organiza o manifesto Sentinel-first de assets candidatos |
| `revp_v1fv_dino_local_asset_preflight.py` | Executa preflight local de assets |
| `revp_v1fx_dino_smoke_embedding_execution.py` | Executa etapa smoke de embeddings |
| `revp_v1fy_dino_embedding_corpus_analysis.py` | Analisa o corpus de embeddings |
| `revp_v1fz_dino_balanced_embedding_corpus.py` | Organiza corpus balanceado de embeddings |
| `revp_v1ga_dino_embedding_structural_consistency_analysis.py` | Avalia consistência estrutural dos embeddings |
| `revp_v1gd_dino_embedding_perturbation_robustness_diagnostics.py` | Executa diagnóstico de robustez por perturbação |
| `revp_v1ge_dino_expanded_sentinel_embedding_corpus.py` | Expande o corpus Sentinel para embeddings |
| `revp_v1gk_dino_pipeline_reproducibility_audit.py` | Audita reprodutibilidade do pipeline |
| `revp_v1go_dino_pipeline_orchestrator.py` | Orquestra etapas do pipeline |
| `revp_v1gq_gis_multicriteria_vulnerability_baseline.py` | Constrói baseline multicritério GIS |
| `revp_v1gx_tcc_figures_and_tables_export_plan.py` | Planeja exportação de figuras e tabelas do TCC |
| `revp_v1gy_tcc_visual_evidence_export_package.py` | Organiza pacote de evidência visual |
| `revp_v1hc_sentinel_visual_review_preview_package.py` | Gera pacote de previews visuais Sentinel para revisão |

Essa lista não esgota todos os scripts, mas indica os principais pontos de entrada da trilha metodológica.

---

## 13. Como os artefatos comprovam os resultados do artigo

Os resultados apresentados no artigo são comprovados por uma combinação de artefatos:

| Resultado/afirmação | Artefatos correspondentes |
|---|---|
| Existência do corpus territorial de 59 patches | `datasets/`, `manifests/`, documentação em `docs/` |
| Inventário Sentinel-first de 128 assets candidatos | manifests e scripts `revp_v1fu*` |
| Separação entre corpus territorial e assets Sentinel | registries em `datasets/` e documentação metodológica |
| Uso de DINOv2 como encoder congelado | scripts DINO, configs e documentação técnica |
| Auditoria de evidências externas | registries de evidência externa em `datasets/` |
| Governança de claims | registries e documentação sobre Protocolo C e limites metodológicos |
| Figuras finais do artigo/apresentação | `outputs_public/figures/` |
| Tabelas finais do artigo/apresentação | `outputs_public/tables/` |
| Métricas e resultados quantitativos | `outputs_public/metrics/` |
| Logs e relatórios de execução | `outputs_public/logs_summary/` e `outputs_public/execution_reports/` |
| Ausência de modelo supervisionado operacional | `outputs_public/model/NO_OPERATIONAL_TRAINED_MODEL.md` |

---

## 14. Protocolo C e governança de claims

O **Protocolo C** é a camada de governança que organiza evidências externas e referências observacionais candidatas.

Ele impede que evidências contextuais sejam promovidas automaticamente a:

- ground truth operacional;
- label supervisionado;
- classe binária;
- validação de evento observado;
- positivo/negativo de treinamento.

A função do Protocolo C é separar:

- evidência contextual;
- suporte territorial;
- referência observacional candidata;
- lacuna metodológica;
- uso permitido;
- uso proibido.

Essa camada é fundamental para manter a validade científica da entrega.

---

## 15. Observação sobre modelo treinado

Esta entrega não inclui pesos de modelo supervisionado operacional porque o projeto ainda não possui esse tipo de modelo.

O componente DINOv2 é usado como **encoder visual congelado**, responsável por gerar representações vetoriais de imagens Sentinel. Esses embeddings apoiam análise estrutural, vizinhança, clusterização, outliers e revisão humana.

Portanto, não há arquivo equivalente a:

```text
output/model/final_model.pt
```

ou

```text
output/model/weights.pth
```

A ausência desses arquivos é intencional e está documentada em:

```text
outputs_public/model/NO_OPERATIONAL_TRAINED_MODEL.md
```

---

## 16. Observação sobre cronograma e integridade da entrega

Este repositório deve ser avaliado conforme o estado existente até a data de entrega definida na tarefa.

Após a data de entrega, não devem ser considerados commits, arquivos ou atualizações posteriores.

Para garantir reprodutibilidade da avaliação, recomenda-se que o avaliador considere:

- o link público do repositório;
- o histórico de commits até a data de entrega;
- os artefatos públicos presentes em `outputs_public/`;
- os registries e manifests públicos;
- a documentação metodológica em `docs/`;
- os testes e scripts versionados.

---

## 17. Resumo final da entrega

Esta entrega contém:

- README objetivo com descrição dos artefatos;
- código-fonte do pipeline;
- dependências Python;
- configurações e templates de execução;
- registries e schemas públicos;
- manifests auditáveis;
- documentação técnica;
- testes automatizados;
- figuras finais;
- tabelas finais;
- métricas e logs resumidos;
- relatórios públicos de execução;
- declaração formal de ausência de modelo supervisionado operacional.

A estrutura submetida comprova a evolução metodológica do projeto, os resultados apresentados no artigo e a organização dos artefatos relevantes para avaliação.
