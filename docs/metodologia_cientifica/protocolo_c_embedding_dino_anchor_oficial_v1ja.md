# Protocolo C v1ja - Embedding DINO do anchor oficial

Esta etapa usa o par Sentinel final selecionado na v1iz para o anchor oficial CPRM de Moinho Preto, Petropolis. O par usado foi:

- pre-evento: Sentinel-2 de 2022-01-18, cena `20220118T130239_20220118T130322_T23KPR`;
- pos-evento: Sentinel-2 de 2022-03-04, cena `20220304T130251_20220304T130743_T23KPR`.

A v1iz resolveu a ressalva de nuvem local e classificou o par como `PATCH_PAIR_USABLE_FOR_REVIEW`. A v1ja avanca a partir desse par, sem buscar novo dado Sentinel e sem alterar o status cientifico do projeto.

## Entrada visual

O DINOv2 recebe uma entrada visual RGB. Como os patches Sentinel sao multibanda, a v1ja usou a composicao B04, B03 e B02. A normalizacao foi feita por percentis locais 2-98 por banda, seguida da normalizacao visual esperada pelo encoder.

Essa conversao e uma representacao visual reduzida do patch Sentinel. Ela nao usa todas as bandas como entrada do encoder e nao transforma o patch em rotulo.

## Modelo

O modelo usado foi `facebook/dinov2-with-registers-base`, em modo frozen encoder. Os pesos nao foram ajustados, nao houve otimizador, nao houve treinamento e nao foi criado alvo supervisionado.

## Resultado

Foram gerados dois embeddings validos:

- embedding pre: 768 dimensoes, sem NaN/inf;
- embedding pos: 768 dimensoes, sem NaN/inf.

A comparacao estrutural pre/pos gerou:

- cosine_similarity: 0.83855718;
- euclidean_distance: 12.66650963;
- status estrutural: `MODERATE_STRUCTURAL_DIFFERENCE_REVIEW_ONLY`.

Esse resultado e um diagnostico estrutural para revisao. Ele nao afirma evento, nao cria label e nao libera treino.

## Status metodologico

O anchor passa a ser candidato multimodal de referencia para revisao, porque agora possui:

- unidade documental oficial;
- coordenada explicita;
- par Sentinel selecionado com qualidade local;
- embeddings DINOv2 frozen validos.

Mesmo assim, o status permanece restrito:

- nao e ground truth operacional;
- nao cria label;
- nao cria target;
- nao autoriza treinamento;
- nao reabre Protocolo B.
