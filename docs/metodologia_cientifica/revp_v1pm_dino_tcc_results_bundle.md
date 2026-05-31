# v1pm — DINO TCC Results Bundle

## Objetivo

Consolidar v1pg-v1pl em tabelas TCC-ready, manifest e summary científico. Reframe para escrita, sem recalcular análises.

## Interpretação metodológica (texto para o TCC)

Os embeddings DINOv2 foram tratados como representação vetorial auto-supervisionada dos patches Sentinel, não como rótulo supervisionado. As análises de similaridade, vizinhança, PCA e agrupamento exploratório foram usadas para avaliar coerência visual/semântica entre patches e regiões, sem validar evento observado, sem criar ground truth operacional e sem treinar classificador de inundação.

## Papel do DINOv2

DINOv2 with registers é representação visual/semântica review-only — não validador de evento, não criador de rótulo, não treinador de classificador.

## Status final

Status final da camada DINO: **DINO_EMBEDDINGS_NOT_FOUND_FAIL_CLOSED**.
