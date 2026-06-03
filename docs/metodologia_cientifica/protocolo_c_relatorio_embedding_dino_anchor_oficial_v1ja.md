# Relatorio v1ja - Embedding DINOv2 do par Sentinel oficial

## Escopo

A v1ja extraiu embeddings DINOv2 frozen para o par Sentinel final definido na v1iz. A etapa e review-only e produz diagnostico estrutural pre/pos para o anchor oficial CPRM.

Nao houve label, target, treinamento, nem reabertura de Protocolo B. Os vetores brutos nao foram versionados.

## Par de entrada

O par usado foi:

- pre: 2022-01-18, `20220118T130239_20220118T130322_T23KPR`;
- pos: 2022-03-04, `20220304T130251_20220304T130743_T23KPR`.

O par ja vinha de v1iz com nuvem local resolvida por SCL/QA60 e status `PATCH_PAIR_USABLE_FOR_REVIEW`.

## Embeddings

Modelo: `facebook/dinov2-with-registers-base`.

Modo: frozen encoder, CPU, sem ajuste de pesos.

Entrada visual: composicao Sentinel B04/B03/B02 com normalizacao documentada.

Resultados:

- dimensao pre: 768;
- dimensao pos: 768;
- norma pre: 21.89013100;
- norma pos: 22.62345695;
- cosine_similarity: 0.83855718;
- euclidean_distance: 12.66650963;
- QA: PASS.

## Interpretacao

O status estrutural foi `MODERATE_STRUCTURAL_DIFFERENCE_REVIEW_ONLY`. Isso indica diferenca representacional entre as duas entradas visuais, dentro de uma analise estrutural. A interpretacao continua dependendo de revisao supervisora e de outros registros do Protocolo C.

O embedding nao e label, nao e alvo supervisionado e nao prova evento observado. Ele apenas adiciona uma camada de representacao visual congelada para revisao multimodal.

## Readiness

O anchor fica como `MULTIMODAL_REFERENCE_CANDIDATE_REVIEW_ONLY`.

Isso e um avanco metodologico real porque conecta documento oficial, coordenada, par Sentinel com qualidade local e embedding DINOv2 valido. O treino continua bloqueado porque nao ha ground truth operacional, nao ha label aprovado e nao ha alvo supervisionado.
