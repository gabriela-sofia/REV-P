# Protocolo C v1jn - Camada C1-C4 de referencia terrestre

## Linha metodologica

A v1jn organiza a evidencia nesta ordem: evento observado, fonte e confiabilidade, data e precisao temporal, localizacao e precisao espacial, patch Sentinel/multimodal com janela temporal, decisao C1/C2/C3/C4 e uso permitido.

## Niveis C

- C1_EVENT_DOCUMENTED: evento em fonte oficial ou auditavel, com fenomeno, data e localidade documental.
- C2_EVENT_GEOREFERENCED: C1 acrescido de coordenada explicita ou geometria auditavel.
- C3_EVENT_PATCH_LINKED: C2 acrescido de patch Sentinel/multimodal com QA e janela temporal documentada.
- C4_OPERATIONAL_LABEL_CANDIDATE: C3 acrescido de gates completos de label, negativos formais, split, vazamento e revisao.

## Resultado consolidado

- C1 documentado: 9
- C2 georreferenciado: 9
- C3 ligado a patch: 9
- C4 candidato operacional: 0

C3 e um avanco real porque liga a unidade documental oficial a uma coordenada explicita e a um conjunto S2/DEM/DINO com QA, mantendo S1 como limitacao quando parcial. Isso permite revisao cientifica rastreavel do evento e do patch, sem transformar a referencia em label.

## Limites

C4 e treino seguem bloqueados porque os negativos formais continuam em 0, os labels positivos formais continuam bloqueados e o split/leakage ainda nao esta fechado. Pseudo-ausencia continua como material unlabeled de auditoria ou sandbox local, nunca como negativo formal. DINO permanece congelado e serve apenas como diagnostico estrutural de revisao.

## Referencia, label, pseudo-ausencia e negativo formal

Referencia e evidencia oficial organizada para revisao. Label e uma decisao operacional posterior, dependente de gates completos. Pseudo-ausencia e unlabeled auditado, sem afirmacao de ausencia. Negativo formal exige evidencia explicita de ausencia ou estabilidade para area, tempo e fenomeno, alem de QA e protocolo de split/leakage.

## Usos permitidos

Uso permitido: referencia review-only, candidato multimodal de referencia e PU sandbox local-only quando registrado como unlabeled/positivo de referencia sem pesos, metricas operacionais ou claim supervisionado.

## Usos proibidos

Uso proibido: label operacional, negativo formal por pseudo-ausencia, treino supervisionado, descongelamento de DINO, claim cientifico de modelo e promocao automatica de PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA para ground truth operacional.
