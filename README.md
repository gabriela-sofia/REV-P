# REV-P

Repositório consolidado do projeto REV-P a partir do ponto em que o trabalho passa para a etapa DINO / self-supervised review-only.

Este repositório contém apenas o que é essencial para continuar o projeto:
- estado metodológico consolidado;
- manifests CSV/JSON auditáveis;
- scripts de training readiness v1fp-v1ft;
- configuração inicial do protocolo DINO review-only.

Este repositório não contém dados brutos, GeoTIFFs, ZIPs, NPY/NPZ, checkpoints, cache, arquivos locais de agentes, documentos antigos de reunião ou material legado.

## Estado científico atual

O projeto está em estágio review-only.

Permitido:
- organização auditável dos patches;
- extração futura de embeddings DINO;
- PCA/UMAP;
- FAISS/kNN;
- clustering;
- revisão visual e análise exploratória.

Bloqueado:
- classificação supervisionada de suscetibilidade;
- rótulo binário de enchente observada;
- weak supervision como verdade;
- métricas preditivas de modelo;
- promoção de patch_bound_validated;
- promoção de preflight_ready;
- promoção de CRS gate.

## Direção atual

A próxima etapa técnica é implementar DINO como encoder congelado, Sentinel-first, para extração de embeddings e agrupamento exploratório dos patches elegíveis.

## Protocolo DINO Sentinel-first

A documentação técnica auditável do fluxo DINO Sentinel-first está em `docs/dino_sentinel_embedding_protocol.md`.
