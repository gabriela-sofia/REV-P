# Relatorio v1jm - Escada de evidencia negativa e pseudo-ausencia

## Resultado principal

A v1jm nao encontrou negativos formais prontos. Tambem nao promoveu ausencia de registro, distancia de anchor, controle temporal ou background a negativo.

O ganho real foi separar o que antes ficava misturado:

- `FORMAL_NEGATIVE_READY`: 0;
- candidatos formais em revisao: mantidos somente se a evidencia textual cumprir os gates;
- `PSEUDO_ABSENCE_REVIEW_ONLY`: candidatos auditados, sem label;
- `BACKGROUND_UNLABELED`: material de fundo para sandbox e balanceamento;
- `PU_SANDBOX_LOCAL_ONLY_READY`: liberado apenas como uso local exploratorio;
- treino supervisionado: `SUPERVISED_TRAINING_BLOCKED_NO_FORMAL_NEGATIVES`.

## Leitura cientifica

A ausencia de um registro nao prova ausencia de evento. Um patch distante tambem nao prova estabilidade. O pre-evento do mesmo anchor e um controle forte para revisao temporal, mas nao e amostra negativa independente. Material externo ou cross-region pode ajudar transferencia e comparacao, mas nao vira ground truth local.

## Opcao externa

Landslide4Sense fica registrado apenas como `EXTERNAL_SUPERVISED_PRETRAINING_OPTION`. Ele nao foi baixado, nao fornece negativo local e exigiria auditoria separada de licenca, sensores, dominio e semantica de classe antes de qualquer uso.

## Proxima decisao

Ha dois caminhos defensaveis:

1. buscar fonte oficial com area ou coordenada, janela temporal e declaracao explicita de ausencia/estabilidade;
2. usar PU sandbox local-only, mantendo unlabeled como unlabeled e sem declarar desempenho supervisionado.

Enquanto isso nao ocorrer, `can_create_training_label=false` e `can_train_supervised_model=false`.
