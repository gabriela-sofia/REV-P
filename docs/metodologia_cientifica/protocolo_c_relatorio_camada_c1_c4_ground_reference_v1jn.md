# Relatorio v1jn - Camada C1-C4 de ground reference

## Escopo executado

A v1jn consolidou os registries canonicos `ground_reference_event_registry.csv`, `event_patch_linkage_registry.csv` e `ground_truth_candidate_decision_audit.csv`. A etapa apenas reuniu evidencia ja registrada: anchors oficiais CPRM, QA S2/DEM/DINO, S1 parcial, negativos formais, pseudo-ausencias review-only e gates de treino.

## Contagens

- Eventos oficiais consolidados: 9
- C1 documentado: 9
- C2 georreferenciado: 9
- C3 ligado a patch: 9
- C4 candidato operacional: 0

## Linkage evento-patch

O linkage ficou com 1 forte, 8 moderado, 0 fraco e 0 bloqueado. Todos os 9 anchors oficiais chegaram a C3 porque tinham coordenada explicita, S2 pre/pos QA_PASS, DEM QA_PASS, DINO_QA_PASS e janela temporal registrada. A limitacao principal e S1 parcial.

## Bloqueio C4

C4 permanece bloqueado por tres motivos: negativos formais iguais a 0, labels positivos formais ainda nao liberados e split/leakage incompleto. Pseudo-ausencia nao altera esse bloqueio e nao vira negativo. DINO nao cria classe nem label.

## Usos permitidos e proibidos

Permitido: revisao cientifica, referencia multimodal candidata e sandbox PU local-only sob as restricoes registradas. Proibido: label operacional, treino supervisionado, negativo formal por ausencia de registro, claim cientifico de modelo, descongelamento de DINO e promocao automatica de PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA para ground truth operacional.
