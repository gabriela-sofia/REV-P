# Relatorio v1jj - Boundary de controles, split e sandbox

## Resultado principal

A v1jj confirma que o batch v1ji e suficiente para revisao multimodal, mas ainda nao atende aos requisitos de treino supervisionado. A etapa nao cria labels, nao cria negativos formais e nao executa treino.

## Contagem DINO normalizada

O registro DINO v1ji tem uma linha por par pre/pos. Assim, a contagem correta deve ser lida como:

- embeddings pre: 9;
- embeddings pos: 9;
- diagnosticos de par: 9;
- dimensao: 768;
- QA: PASS.

Portanto, "DINO frozen=9" significa 9 diagnosticos pareados, nao apenas 9 vetores soltos.

## Controles

Foram formalizados controles candidatos:

- controles temporais do mesmo anchor;
- candidatos de fundo PET ja existentes;
- contexto cross-region Recife/Curitiba;
- uma linha de bloqueio para uso indevido de ausencia de registro como negativo.

Todos permanecem com `can_be_negative_label=false`.

## Split e leakage

O protocolo define que a unidade pareada do mesmo anchor nao pode ser quebrada entre lados opostos de um split. Tambem exige separacao por unidade documental, localidade e evento/data, alem de buffer espacial minimo.

O status continua `LEAKAGE_PROTOCOL_REQUIRED`, porque ainda nao ha labels e negativos formais para construir um split supervisionado.

## Sandbox

Sandbox fraco local fica permitido apenas como engenharia, com status `INVALID_FOR_SCIENTIFIC_CLAIM`. Ele nao salva pesos, nao vira resultado cientifico e nao altera o gate.

## Boundary

O estado final permanece:

- review-only batch: pronto;
- weak sandbox local: permitido com guardrail;
- treino supervisionado: bloqueado;
- DINO unfreeze para claim: bloqueado;
- label operacional: bloqueado.

O proximo avanco cientifico exigiria protocolo formal de labels, negativos/controles com evidencia de ausencia e split auditavel antes de qualquer treinamento.
