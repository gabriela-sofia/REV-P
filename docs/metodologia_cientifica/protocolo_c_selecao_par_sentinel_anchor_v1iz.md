# Protocolo C v1iz - Selecao do par Sentinel para o anchor oficial

Esta etapa resolve a ressalva aberta na v1iy sobre nuvem local no patch pre-evento do anchor oficial CPRM de Moinho Preto, Petropolis.

A v1ix gerou patches Sentinel-2 reais, pre e pos-evento, centrados no anchor oficial. A v1iy confirmou boa qualidade local de bandas, geometria e pixels validos, mas manteve o status `PRE_PATCH_CLOUD_RISK_HIGH` porque o metadado global de nuvem da cena pre era alto e o patch local nao incluia SCL/QA60.

Na v1iz, foram baixadas mascaras locais SCL/QA60 para as cenas v1ix e tambem foram avaliadas cenas pre-evento alternativas antes de 2022-02-15.

## Resultado da mascara local

A mascara local confirmou que a cena pre da v1ix tinha nuvem local alta no recorte do anchor:

- cena pre v1ix: `20220202T130251_20220202T130247_T23KPR`;
- data: 2022-02-02;
- cloud metadata global: 90.359181;
- cloud local SCL/QA60: 0.815104.

A cena pos da v1ix permaneceu adequada:

- cena pos v1ix: `20220304T130251_20220304T130743_T23KPR`;
- data: 2022-03-04;
- cloud metadata global: 2.394584;
- cloud local SCL/QA60: 0.000000.

## Alternativa pre-evento selecionada

A busca expandida encontrou seis cenas pre-evento antes de 2022-02-15. A melhor alternativa foi:

- cena: `20220118T130239_20220118T130322_T23KPR`;
- data: 2022-01-18;
- cloud metadata global: 7.460087;
- cloud local SCL/QA60: 0.002387;
- valid_pixel_fraction: 1.000000;
- bandas: B02, B03, B04, B08, B11, B12;
- indices espectrais aproximados: OK.

## Decisao final

O par final selecionado e:

- pre: `20220118T130239_20220118T130322_T23KPR`;
- pos: `20220304T130251_20220304T130743_T23KPR`.

Status final: `PATCH_PAIR_USABLE_FOR_REVIEW`.

Esse status significa que o par e utilizavel como candidato de referencia para revisao multimodal. Ele nao cria label, target, treino, nem ground truth operacional.

Qualquer uso mais forte ainda depende de interpretacao independente e de uma regra metodologica posterior explicitamente aprovada.
