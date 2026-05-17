# Datasets auditáveis do REV-P

## O que este diretório documenta

Este diretório registra os datasets e corpora produzidos ou utilizados pelo REV-P como
evidência científica auditável. Ele não contém dados brutos — contém registros
estruturados que descrevem o que existe, onde está, como foi produzido e quais são as
suas limitações.

## Quatro categorias de material

**Dataset público:** manifest ou registro commitado neste repositório. Acessível a
qualquer leitura, sem dependência de ambiente local.

**Registro auditável:** tabela que descreve um corpus local sem replicar os arquivos
pesados. Prova que o corpus existe e como foi construído, sem exigir que o repositório
hospede os rasters.

**Dado local:** arquivo que existe apenas no workspace privado (rasters Sentinel,
embeddings `.npz`, shapefiles brutos). Referenciado pelos manifests públicos, mas não
versionado.

**Artefato pesado:** dado que não pode ou não deve ser versionado por tamanho, por
conteúdo sensível ou por ser reproduzível a partir dos scripts e manifests públicos.

## Por que o GitHub publica rastreabilidade, não rasters

Os GeoTIFFs Sentinel originais têm entre 10 MB e 200 MB por arquivo. O corpus de 128
patches totaliza múltiplos gigabytes. Versionar esses arquivos incharia o repositório
sem benefício científico: os patches são gerados a partir de imagens Sentinel-2 Level-2A
de acesso público, e a metodologia de derivação está documentada nos manifests.

O que prova a legitimidade científica do corpus não é a presença dos rasters — é a
rastreabilidade da cadeia: qual imagem Sentinel originou cada patch, qual preflight foi
executado, qual QA foi aprovado antes da extração de embeddings.

## Arquivos neste diretório

| Arquivo | Conteúdo |
|---|---|
| `dataset_registry.csv` | Registro geral de datasets e corpora do projeto |
| `patch_corpus_registry.csv` | Registro dos corpora de patches Sentinel por estágio |
| `external_evidence_registry.csv` | Registro das evidências GIS externas por região |
| `schemas/dataset_registry_schema.csv` | Schema de campos de dataset_registry.csv |
| `schemas/patch_corpus_schema.csv` | Schema de campos de patch_corpus_registry.csv |
| `schemas/external_evidence_schema.csv` | Schema de campos de external_evidence_registry.csv |

## O que não está aqui

- GeoTIFFs, rasters, GeoJSONs brutos, shapefiles, geodatabases
- Embeddings `.npz` ou `.npy`
- Outputs de execução local (`local_runs/`)
- Dados de validação externa pesados (PE3D/MDE, SGB/RIGeo, GeoCuritiba)
- Qualquer arquivo que contenha paths absolutos de máquina local

Os registros descrevem esses materiais. Os materiais ficam locais.

## Navegação relacionada

- [`docs/metodologia_cientifica/research_datasets_and_artifacts.md`](../docs/metodologia_cientifica/research_datasets_and_artifacts.md) — narrativa metodológica completa
- [`docs/metodologia_cientifica/patch_lineage_and_grounding.md`](../docs/metodologia_cientifica/patch_lineage_and_grounding.md) — linhagem dos patches
- [`manifests/`](../manifests/) — manifests CSV/JSON por estágio do pipeline
