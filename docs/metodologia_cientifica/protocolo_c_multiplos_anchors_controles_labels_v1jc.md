# Protocolo C v1jc - Multiplos anchors, controles e prontidao de labels

A v1jc cria a camada de governanca necessaria antes de qualquer treino. A etapa parte do anchor oficial CPRM ja consolidado em Moinho Preto e reavalia todas as unidades documentais oficiais disponiveis para saber quantas podem virar anchors espaciais, quantas permanecem apenas documentais e quais controles podem ser tratados apenas como candidatos de revisao.

Nao houve geocodificacao de bairro, centroide, aproximacao por localidade ou coordenada inventada.

## Por que um anchor nao basta

Um unico anchor forte e suficiente para construir uma referencia de revisao, mas nao e suficiente para treino. Treino exigiria multiplos casos positivos independentes, controles formais, criterios de split e uma regra de vazamento. Sem isso, qualquer ajuste de modelo ficaria preso ao caso unico e nao teria base para generalizacao.

Por isso, a v1jc separa quatro niveis:

- anchor positivo: evento oficial com coordenada, data, fenomeno e localidade;
- referencia positiva candidata: anchor que pode orientar revisao multimodal;
- label positivo: status supervisionado formal, ainda nao criado;
- controle candidato: contexto de comparacao, sem claim de ausencia.

## Anchors recuperados

A leitura dos registries oficiais encontrou 11 unidades documentais CPRM.

Classificacao v1jc:

- 1 anchor confirmado: `ANCHOR_PET2022_CPRM_ANEXOII_19022022`;
- 1 coordenada explicita recuperada;
- 9 unidades documentais com data/fenomeno/localidade, mas sem coordenada explicita;
- 1 unidade com evidencia insuficiente para anchor espacial estruturado.

O anchor confirmado e o ANEXO-II, Bairro Moinho Preto, vistoria de 19/02/2022, movimento de massa, com coordenada CPRM documentada em graus decimais: latitude -22.484251 e longitude -43.211257.

As demais unidades permanecem como evidencia documental. Elas nao viram anchor espacial porque nao possuem coordenada textual explicita no registro auditado.

## Controles candidatos

A v1jc criou controles candidatos apenas como objetos de revisao:

- `TEMPORAL_SELF_CONTROL`: patch pre-evento do mesmo anchor;
- `SPATIAL_CONTEXT_CONTROL_CANDIDATE`: classe futura de patch regional, ainda dependente de regra de buffer e auditoria;
- `EXISTING_PATCH_BACKGROUND_CANDIDATE`: patches PET existentes, sanitizados por id, sem vinculo oficial ao anchor;
- `INVALID_NEGATIVE_LABEL`: linha de governanca para impedir que ausencia de registro seja tratada como negativo formal.

Controle candidato nao e negativo. Ausencia de registro nao e ausencia de evento. Um negativo formal exigiria protocolo posterior especifico, com fonte, area, periodo, regra de exclusao e revisao supervisora.

## Matriz de labels

A matriz v1jc mantem:

- `POSITIVE_REFERENCE_CANDIDATE`: permitido apenas como referencia de revisao;
- `POSITIVE_LABEL_READY`: false nesta etapa;
- `NEGATIVE_LABEL_READY`: false nesta etapa;
- `TRAINING_READY`: false nesta etapa;
- `REVIEW_ONLY_READY`: true, porque existe anchor, par Sentinel, QA local, embedding DINO frozen e probe multimodal.

## Boundary de treino

Status: `TRAINING_BLOCKED_INSUFFICIENT_LABELS`.

Ainda faltam:

- multiplos anchors oficiais com coordenadas explicitas;
- controles candidatos auditados por regra espacial e temporal;
- negativos formais aprovados;
- labels formalizados;
- split por evento e localidade;
- protocolo de vazamento;
- metricas supervisionadas;
- regra de promocao de evidencia.

O DINO continua frozen e review-only. Embedding nao cria label, controle nem permissao de treino.
