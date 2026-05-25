# Protocolo C v1jm - Negativos, pseudo-ausencia e background

A v1jm resolve o gargalo dos negativos por hierarquia metodologica, sem forcar label falso. A ordem adotada e:

1. negativo formal;
2. pseudo-ausencia auditada;
3. background ou controle;
4. positive-unlabeled local-only;
5. one-class ou prototipo review-only.

## Negativo formal

Negativo formal exige evidencia explicita de ausencia ou estabilidade para area, janela temporal e fenomeno compativeis. A fonte precisa ser oficial ou auditavel, com area ou coordenada explicita, patch multimodal QA, regra de buffer/leakage e split possivel.

Baixo risco, fora de area de risco, ausencia de registro, distancia do anchor, pre-evento do mesmo anchor e material cross-region nao bastam. Esses casos podem informar revisao, mas nao criam negativo formal.

## Pseudo-ausencia

Pseudo-ausencia e um candidato unlabeled auditado. Ela pode ser util em sandbox ou em desenho PU, desde que mantenha as restricoes:

- nao afirma ausencia do fenomeno;
- nao vira ground truth;
- nao cria label de treino;
- exige distancia/buffer, mesma regiao quando possivel, estrato ambiental e QA de patch antes de qualquer uso local.

Na v1jm, os candidatos derivados dos controles existentes ficam como `PSEUDO_ABSENCE_REVIEW_ONLY`. Onde faltam ponto explicito, distancia ou auditoria de buffer, o bloqueio fica registrado.

## Background

Background e material de fundo unlabeled. Ele serve para representar gradientes ambientais e balanceamento espacial em revisao, mas continua sem classe. Background nunca e negativo formal.

## PU sandbox

O limite permitido e `PU_SANDBOX_LOCAL_ONLY_READY`: positivos sao os 9 anchors oficiais e unlabeled vem de pseudo-ausencia, background e controles review-only. As metricas seriam exploratorias, nenhum peso deve ser salvo e nenhum resultado pode ser usado como desempenho cientifico supervisionado.

## Status de treino

O status supervisionado continua bloqueado: `SUPERVISED_TRAINING_BLOCKED_NO_FORMAL_NEGATIVES`. O caminho real para evoluir e obter evidencia oficial explicita de ausencia/estabilidade para area, tempo e fenomeno, ou manter PU como sandbox local sem claim supervisionado.
