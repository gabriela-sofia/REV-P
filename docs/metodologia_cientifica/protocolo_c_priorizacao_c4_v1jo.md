# Protocolo C - priorizacao C4 v1jo

## Escopo

v1jo analisa a camada C1-C4 ja consolidada em v1jn e transforma os bloqueios remanescentes em uma fila objetiva de evidencias. A etapa nao cria label operacional, nao treina modelo, nao descongela DINO e nao promove pseudo-ausencia a negativo.

## Estado de entrada

- Eventos C3 confirmados: 9
- Eventos C4 prontos: 0
- Negativos formais: 0
- Pseudo-ausencias review-only: 4
- Pares S1 completos: 1
- Pares S1 parciais: 8

## Interpretacao metodologica

C3 e um avanco real porque liga evento oficial CPRM, coordenada explicita e patch multimodal com S2/DEM/DINO em QA. Isso cria referencia cientifica para revisao, nao label operacional.

C4 nao esta liberado. O gargalo primario e a ausencia de negativos formais: sem evidencia explicita de ausencia, estabilidade ou vistoria sem ocorrencia, nao existe classe negativa defensavel. Pseudo-ausencia permanece apenas PU/sandbox local-only.

S1 parcial e um bloqueio secundario de robustez multimodal. Completar S1 fortalece C3 e melhora a auditoria dos patches, mas nao cria negativo, nao cria positivo operacional e nao autoriza treino.

## Prioridade de blockers

- 1. FORMAL_NEGATIVES_ZERO: afeta 9 eventos; bloqueia C4=true.
- 2. POSITIVE_LABEL_GATE_NOT_OPERATIONAL: afeta 9 eventos; bloqueia C4=true.
- 3. SPLIT_LEAKAGE_NOT_READY: afeta 9 eventos; bloqueia C4=true.
- 4. S1_PARTIAL_COVERAGE: afeta 8 eventos; bloqueia C4=false.
- 5. EXTERNAL_VALIDATION_OPTIONAL: afeta 9 eventos; bloqueia C4=false.

## Proxima acao

Acao programatica de maior valor: executar a fila metadata-only de busca de negativos formais e preparar intake para fontes oficiais com ausencia/estabilidade explicita. Em paralelo, completar S1 para os 8 anchors incompletos fortalece C3, mas nao altera C4.

Acao cientifica de maior valor: obter evidencia oficial explicita de ausencia, estabilidade ou vistoria sem ocorrencia em janela temporal compativel, com localizacao auditavel e regra de vazamento posterior.
