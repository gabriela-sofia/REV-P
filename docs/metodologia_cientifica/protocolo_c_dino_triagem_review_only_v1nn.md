# Protocolo C - DINO triagem review-only v1nn

Triagem criada para 9 anchors C3 e 4 candidatos controle/pseudo-ausencia.

Embeddings DINO permanecem congelados e metadata-only nos outputs versionaveis. Valores vetoriais brutos nao sao carregados nem versionados.

A fila serve para priorizar revisao visual, vizinhos, outliers e busca oficial de evidencia negativa. Nao cria label, nao valida evento e nao treina classificador.
