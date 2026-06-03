# Protocolo C v1jl - Negativos e controles formais

A v1jl verifica se algum controle candidato pode ser promovido a negativo formal. A resposta metodologica e conservadora: nenhum candidato atual possui evidencia oficial explicita de ausencia ou estabilidade para area, janela temporal e fenomeno compativeis.

## Criterio para negativo formal

Um negativo formal exigiria, no minimo:

- fonte oficial;
- area ou coordenada explicita;
- data ou janela temporal compativel;
- fenomeno definido;
- evidencia explicita de ausencia ou estabilidade;
- patch QA;
- decisao de revisao supervisora;
- protocolo split/leakage fechado.

Distancia de anchor, ausencia de registro, patch pre-evento do mesmo anchor e material de outra regiao nao satisfazem esses gates.

## Classificacao dos controles

A v1jl separa:

- `STRONG_CONTROL_CANDIDATE`: baseline temporal do mesmo anchor, util para revisao, mas nao negativo independente;
- `REVIEW_CONTROL_ONLY`: camadas contextuais ou material cross-region para apoio interpretativo;
- `INSUFFICIENT_EVIDENCE`: patches/contextos sem evidencia explicita de ausencia ou estabilidade;
- `INVALID_NEGATIVE_ABSENCE_ASSUMPTION`: tentativa bloqueada de tratar ausencia de registro como ausencia de evento.

## Boundary

O projeto passa a ter controles de revisao mais bem classificados, mas continua sem negativo formal. O status supervisionado permanece `SUPERVISED_TRAINING_BLOCKED_NO_NEGATIVES`.

DINO, PCA, sandbox e prototipos continuam como recursos review-only. Eles nao criam classe, label ou negativo.
