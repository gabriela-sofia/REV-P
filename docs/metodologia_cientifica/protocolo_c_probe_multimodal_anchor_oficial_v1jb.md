# Protocolo C v1jb - Probe multimodal do anchor oficial

Esta etapa consolida o par Sentinel final da v1iz, a auditoria local de qualidade e os embeddings DINOv2 frozen da v1ja para o anchor oficial CPRM de Moinho Preto, Petropolis.

O objetivo da v1jb nao e treinar modelo. A etapa organiza evidencias ja produzidas em um probe multimodal controlado, com deltas espectrais e comparacao estrutural DINO pre/pos, mantendo a fronteira metodologica explicita.

## Entradas

O par Sentinel usado foi:

- pre-evento: 2022-01-18, cena `20220118T130239_20220118T130322_T23KPR`;
- pos-evento: 2022-03-04, cena `20220304T130251_20220304T130743_T23KPR`.

A v1iz classificou esse par como `PATCH_PAIR_USABLE_FOR_REVIEW`, com nuvem local baixa no recorte do anchor. A v1ja gerou embeddings DINOv2 frozen validos para o mesmo par.

## Deltas espectrais

A v1jb recalculou as medias e desvios das bandas Sentinel B02, B03, B04, B08, B11 e B12, alem dos indices aproximados NDWI e NDBI.

Resultados principais:

- NDWI pre: -0.689760;
- NDWI pos: -0.700027;
- delta NDWI: -0.010267;
- NDBI pre: -0.241666;
- NDBI pos: -0.262730;
- delta NDBI: -0.021064.

O status espectral emitido foi `LOW_SPECTRAL_DIFFERENCE_REVIEW_ONLY`. Isso descreve a magnitude dos deltas calculados no recorte local; nao cria rotulo e nao define classe.

## DINO frozen

O resultado DINO usado continua sendo o da v1ja:

- modelo: `facebook/dinov2-with-registers-base`;
- modo: frozen encoder;
- dimensao dos embeddings: 768;
- cosine_similarity: 0.83855718;
- euclidean_distance: 12.66650963;
- status estrutural: `MODERATE_STRUCTURAL_DIFFERENCE_REVIEW_ONLY`.

O DINO permanece congelado como resultado cientifico. A comparacao pre/pos e um diagnostico estrutural de revisao, nao uma inferencia supervisionada.

## Referencia comparativa

Quando possivel, a v1jb comparou o anchor com distribuicoes locais ja existentes. Foi encontrada uma distribuicao operacional DINO review-only com 132 comparacoes de similaridade e uma referencia espectral limitada baseada em cenas pre-evento alternativas.

Status consolidado: `DINO_REFERENCE_AVAILABLE+SPECTRAL_REFERENCE_LIMITED`.

Essa comparacao ajuda a posicionar o anchor em relacao a artefatos locais de revisao, mas nao substitui validacao supervisionada, labels formais, controles ou splits.

## Fronteira de treino

A decisao formal da v1jb e `TRAINING_BLOCKED_INSUFFICIENT_LABELS`.

O treino continua bloqueado porque a etapa possui:

- um anchor oficial;
- zero labels formais;
- zero negativos ou controles aprovados;
- nenhum split;
- nenhum protocolo de vazamento;
- nenhuma metrica supervisionada.

Qualquer experimento nao-congelado fica fora do resultado cientifico. Nesta execucao, o sandbox nao-congelado foi solicitado pela CLI, mas foi pulado pela guardrail metodologica com status `SANDBOX_SKIPPED_BY_SCIENTIFIC_GUARDRAIL`.

## Status

A v1jb torna o anchor um candidato multimodal de referencia para revisao:

- documento oficial presente;
- coordenada explicita;
- par Sentinel selecionado e auditado;
- deltas espectrais calculados;
- embeddings DINOv2 frozen validos;
- fronteira de treino registrada.

Mesmo assim, a etapa nao cria label, target, classe, treino, ground truth operacional ou reabertura do Protocolo B.
