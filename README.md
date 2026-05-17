# REV-P

## Visão geral

O REV-P é um protocolo auditável para organizar e inspecionar evidências físico-ambientais, geoespaciais e visuais sobre patches urbanos associados a suscetibilidade a inundação e alagamento. O repositório concentra manifests, scripts, testes e documentação técnica do pipeline DINO Sentinel-first.

## Escopo científico

O projeto está em estágio de revisão e auditoria estrutural. Não há classificação supervisionada de suscetibilidade, rótulos binários de enchente observada, alvos de treinamento ou afirmações preditivas.

O DINO é usado exclusivamente como encoder visual congelado para extração de características estruturais de patches Sentinel. O índice GIS (v1gq–v1gt) é um proxy interpretável para comparação e triagem — não é verdade de campo nem alvo supervisionado.

## Estrutura do repositório

```
configs/          Configurações de exemplo (parâmetros de extração DINO)
manifests/        Manifests CSV/JSON auditáveis de patches, preflight e validação
scripts/          Scripts do pipeline (trilha DINO e preparação de treinamento)
tests/            Testes automatizados de cada estágio do pipeline
docs/             Protocolo técnico, registro de comandos e estado metodológico
requirements.txt  Dependências Python do projeto
```

## O que não está versionado

Dados brutos, GeoTIFFs, shapefiles, GeoJSONs convertidos, embeddings `.npz`, outputs locais em `local_runs/`, caches, modelos pesados e arquivos locais de desenvolvimento não são versionados nem enviados ao repositório público.

## Trilha DINO Sentinel-first

O pipeline segue a ordem:

1. Manifesto Sentinel (v1fu) — inventário canônico de patches elegíveis
2. Preflight local (v1fv) — verificação de assets antes da extração
3. Execução smoke de embeddings (v1fx) — leitura real de pixels, extração local
4. Análise estrutural (v1fy–v1gi) — PCA, clustering, vizinhos, outliers, proveniência
5. Auditorias operacionais (v1gn–v1gp) — saúde, orquestração, prontidão para release
6. Auditorias GIS (v1gq–v1gt) — baseline multicritério, uso do solo, cobertura de fontes

Todos os outputs de execução ficam exclusivamente em `local_runs/`.

## Travas metodológicas

- Sem labels ou targets supervisionados
- Sem treinamento supervisionado
- Sem afirmações preditivas de vulnerabilidade
- Sem ativação multimodal (em espera)
- Índice GIS não é ground truth
- DINO não prediz vulnerabilidade
- `review_only=true`

## Documentação técnica

- [docs/dino_sentinel_embedding_protocol.md](docs/dino_sentinel_embedding_protocol.md) — protocolo completo do pipeline DINO
- [docs/dino_command_registry.md](docs/dino_command_registry.md) — registro de comandos para reprodução local
- [docs/dino_sentinel_scientific_evidence_summary.md](docs/dino_sentinel_scientific_evidence_summary.md) — resumo de evidências científicas
- [docs/estado_metodologico_revp.md](docs/estado_metodologico_revp.md) — estado e limitações metodológicas atuais
