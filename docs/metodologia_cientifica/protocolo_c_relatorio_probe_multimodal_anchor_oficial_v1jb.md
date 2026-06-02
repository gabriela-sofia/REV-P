# Relatorio v1jb - Probe multimodal e boundary de treino

## Escopo

A v1jb consolida o estado atual do anchor oficial CPRM de Moinho Preto em um probe multimodal de revisao. A etapa combina:

- par Sentinel final da v1iz;
- qualidade local ja resolvida por mascara SCL/QA60;
- embeddings DINOv2 frozen da v1ja;
- deltas espectrais locais;
- decisao formal sobre treino e descongelamento.

Nao houve criacao de label, target, classe, treino, split, negativo ou ground truth operacional.

## Resultado real ja obtido

O projeto agora possui um anchor oficial com cadeia multimodal minima:

- unidade documental oficial CPRM;
- coordenada explicita;
- patch Sentinel pre-evento de 2022-01-18;
- patch Sentinel pos-evento de 2022-03-04;
- QA local do par com status `PATCH_PAIR_USABLE_FOR_REVIEW`;
- embedding DINOv2 frozen pre/pos com dimensao 768;
- probe espectral e estrutural consolidado.

Esse conjunto e um avanco real para revisao multimodal. Ele ainda nao e base supervisionada.

## Resultados espectrais

A v1jb calculou os deltas locais por banda e por indices aproximados.

Resumo dos indices:

- NDWI pre: -0.689760;
- NDWI pos: -0.700027;
- delta NDWI: -0.010267;
- NDBI pre: -0.241666;
- NDBI pos: -0.262730;
- delta NDBI: -0.021064.

Resumo dos deltas medios por banda:

- B02: -19.088216;
- B03: -18.024848;
- B04: 15.759332;
- B08: 4.751845;
- B11: -73.627604;
- B12: -54.797852.

O status espectral foi `LOW_SPECTRAL_DIFFERENCE_REVIEW_ONLY`.

## Resultado DINO

O diagnostico DINO usado e o da v1ja, em modo frozen:

- cosine_similarity: 0.83855718;
- euclidean_distance: 12.66650963;
- norma pre: 21.89013100;
- norma pos: 22.62345695;
- status estrutural: `MODERATE_STRUCTURAL_DIFFERENCE_REVIEW_ONLY`.

O DINO segue congelado porque nao existe desenho supervisionado valido para ajustar pesos. O embedding e um vetor de revisao, nao um label.

## Comparacao de referencia

A comparacao encontrou uma referencia DINO local review-only com 132 valores de similaridade e uma referencia espectral limitada baseada em alternativas pre-evento. O status consolidado foi `DINO_REFERENCE_AVAILABLE+SPECTRAL_REFERENCE_LIMITED`.

Essa referencia e util para contexto, mas nao fecha uma distribuicao supervisionada. Ela nao possui labels formais, negativos aprovados, split, protocolo de vazamento ou metricas de generalizacao.

## Boundary de treino

A decisao formal e `TRAINING_BLOCKED_INSUFFICIENT_LABELS`.

O descongelamento do DINO permanece bloqueado para qualquer claim cientifico nesta etapa. O sandbox nao-congelado foi solicitado, mas nao rodou; o status registrado foi `SANDBOX_SKIPPED_BY_SCIENTIFIC_GUARDRAIL`.

Para treino real ainda faltam:

- multiplos anchors oficiais;
- controles ou negativos aprovados;
- labels formalizados;
- split de treino, validacao e teste;
- protocolo de vazamento;
- metricas supervisionadas;
- regra explicita de promocao de evidencia.

## Conclusao

A v1jb deixa o anchor como `MULTIMODAL_CHANGE_PROBE_READY` e candidato multimodal de referencia para revisao. O resultado cientifico continua limitado a DINO frozen, QA Sentinel e deltas espectrais auditaveis.

A etapa nao autoriza treino, nao cria label e nao transforma o anchor em ground truth operacional.
