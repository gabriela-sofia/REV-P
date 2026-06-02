# Relatorio v1jd - Auditoria direcionada de coordenadas em anexos CPRM

## Escopo

A v1jd executou uma auditoria direcionada nos anexos CPRM/DIGEAP locais para tentar recuperar coordenadas explicitas adicionais. A etapa foi desenhada para ampliar o conjunto de anchors oficiais sem introduzir coordenadas aproximadas.

Nao houve download de novos documentos, treino, descongelamento de DINO, criacao de label operacional ou reabertura do Protocolo B.

## Por que a etapa foi necessaria

A v1jc mostrou que o projeto possui:

- 11 unidades documentais oficiais;
- 1 coordenada explicita recuperada;
- 1 anchor confirmado;
- 9 eventos documentais sem coordenada;
- 0 negativos formais;
- treino bloqueado por insuficiencia de labels.

Com apenas um anchor forte, o projeto pode seguir em revisao multimodal, mas nao pode treinar modelo. Para treino futuro seriam necessarios multiplos anchors independentes, controles formais, labels aprovados, split e protocolo de vazamento.

## Execucao

Foram auditados 10 PDFs/anexos CPRM locais. A extracao tentou usar texto nativo e tabelas quando as dependencias estivessem disponiveis. No ambiente atual:

- PyMuPDF: indisponivel;
- pdfplumber: indisponivel;
- pytesseract/pdf2image: indisponiveis;
- OCR: `OCR_NOT_AVAILABLE`;
- varredura textual crua local: executada como fallback limitado.

A ausencia de OCR nao quebrou a etapa; ela foi registrada como limitacao operacional.

## Resultado da recuperacao

Resultado consolidado:

- PDFs auditados: 10;
- coordenadas candidatas encontradas: 1;
- coordenadas validas: 1;
- anchors oficiais novos: 0;
- eventos restantes sem coordenada: 9.

A coordenada valida corresponde ao ANEXO-II, ja conhecido em v1ir/v1is:

- localidade: Bairro Moinho Preto;
- data: 19/02/2022;
- fenomeno: movimento de massa;
- latitude: -22.484251;
- longitude: -43.211257;
- confianca: `EXPLICIT_COORDINATE_HIGH`.

Nenhum dos outros anexos gerou coordenada explicita adicional com os recursos locais disponiveis.

## Efeito na matriz de labels

A matriz de labels nao mudou em termos de prontidao. Como nao houve novo anchor oficial adicional, a contagem efetiva continua insuficiente para treino.

Status preservado:

- `TRAINING_BLOCKED_INSUFFICIENT_LABELS`;
- `REVIEW_ONLY_READY`;
- `can_create_training_label=false`;
- `can_train_model=false`;
- `can_unfreeze_dino_for_scientific_claim=false`.

## Proximo passo

Se a meta for aumentar anchors, ha duas rotas defensaveis:

- habilitar extracao/OCR local e repetir a v1jd sobre os mesmos PDFs;
- obter fonte oficial complementar que contenha coordenadas explicitas.

Somente depois de recuperar novos anchors oficiais faz sentido preparar batch Sentinel patch generation para cada novo ponto.
