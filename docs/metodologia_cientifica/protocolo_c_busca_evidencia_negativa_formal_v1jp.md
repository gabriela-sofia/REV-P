# Protocolo C - busca de evidencia negativa formal v1jp

## Objetivo

v1jp procura evidencia negativa formal em documentos, registries e metadados locais ja disponiveis. Evidencia negativa formal exige fonte oficial/auditavel, localidade ou coordenada, janela temporal, fenomeno compativel e declaracao explicita de ausencia, estabilidade ou vistoria sem ocorrencia.

## O que nao vale como negativo

Ausencia de registro, distancia de anchor positivo, baixo risco generico, suscetibilidade baixa, pseudo-ausencia, background e contexto pre-evento nao bastam. Esses itens podem orientar revisao, mas nao criam negativo formal.

## Padroes buscados

Foram buscadas expressoes como: sem indicios, sem evidencias, nao foi observado deslizamento, ausencia de instabilidade, area estavel, sem movimentacao, sem ocorrencia, vistoria sem ocorrencia, ponto controle, area controle, estabilidade observada e sem feicoes.

## Escopo escaneado

Arquivos de texto/CSV/JSON/Markdown escaneados: 870. Rasters, PDFs, NPY, NPZ e dados brutos nao foram versionados nem processados.

## Classificacao

- FORMAL_NEGATIVE_READY: 0
- FORMAL_NEGATIVE_CANDIDATE_REVIEW: 5
- INVALID_NEGATIVE_ABSENCE_ASSUMPTION: 175814
- INSUFFICIENT_EVIDENCE: 5

Pseudo-ausencia permanece apenas PU/sandbox local-only. v1jp nao altera v1jn/v1jo e nao cria label.
