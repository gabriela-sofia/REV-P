# v1qu — Smoke Sample Relative-Path Linker

## Objetivo

Resolver o `relative_path` esperado (convenção `patch_<regiao>_<numero>.tif`) para as 32 linhas de v1qh, sem reescrever o artefato histórico v1qh e sem ler nenhum pixel.

## Limite metodológico

Apenas `Path.exists()` (stat de arquivo) é usado para reportar `expected_file_found`. Nenhum arquivo é aberto; nenhum pixel é lido. `pixel_read_allowed`/`pixel_read_performed` continuam sendo responsabilidade exclusiva de v1qi, controlada por REVP_DINO_PIXEL_READ_ALLOWED.

## Resultado

Linhas: 32. Linkadas: 32. Região não mapeada: 0. Arquivo esperado ausente: 0.
