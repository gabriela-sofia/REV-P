# v1pg — DINO Artifact Discovery (metadata-only)

## Objetivo

Escanear o repositório por artefatos relacionados a DINO/embeddings usando correspondência de termos em nomes de arquivo e cabeçalhos CSV. Leitura apenas de metadados — nunca de pixels ou vetores brutos.

## Termos detectados

`dino`, `embedding`, `768`, `pca`, `neighbors`, `neighbor`, `similarity`, `cluster`, `patch_id`, `alias`.

## Guardrails

Saídas usam apenas caminhos relativos POSIX e hash de caminho. Blobs binários (.npy/.npz) são listados como metadados bloqueados. Fixtures/sintéticos são marcados e bloqueados. Nenhum label, target ou ground truth é criado.

## Resultado

Artefatos escaneados: 510.
