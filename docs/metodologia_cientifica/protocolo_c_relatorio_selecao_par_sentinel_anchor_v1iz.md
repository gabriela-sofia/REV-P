# Relatorio v1iz - Mascara de nuvem e selecao final do par Sentinel

## Escopo

A v1iz usou GEE autenticado apenas para baixar patches pequenos de mascara SCL/QA60 e cenas pre-evento alternativas para o mesmo anchor oficial. Todos os rasters ficaram em `local_runs/`. Os outputs publicos contem apenas metadados e decisoes auditaveis.

Nao houve label, target, treino ou reabertura de Protocolo B.

## Achados

A mascara SCL/QA60 local foi obtida para o par v1ix:

- pre v1ix: cloud local 0.815104;
- pos v1ix: cloud local 0.000000.

Isso confirmou que o risco da cena pre v1ix nao era apenas um problema de metadado global. O recorte local tambem tinha nuvem alta.

A busca por alternativa pre-evento avaliou seis cenas antes de 2022-02-15. A melhor cena foi `20220118T130239_20220118T130322_T23KPR`, de 2022-01-18, com cloud local 0.002387, pixels validos completos, bandas completas e indices espectrais computaveis.

## Decisao

O par Sentinel final selecionado para revisao e:

- pre: 2022-01-18, `20220118T130239_20220118T130322_T23KPR`;
- pos: 2022-03-04, `20220304T130251_20220304T130743_T23KPR`.

Status: `PATCH_PAIR_USABLE_FOR_REVIEW`.

O par permanece candidato de referencia e candidato de revisao multimodal. Ele nao e ground truth operacional, nao cria label e nao autoriza treino.

## Uso recomendado

Em texto metodologico, a formulacao correta e: a v1iz substituiu a cena pre inicialmente selecionada por uma cena pre-evento alternativa com mascara local SCL/QA60 favoravel, mantendo o par Sentinel como material auditavel para revisao. O resultado melhora a confiabilidade visual/espectral do par, mas nao muda o status supervisionado do projeto.
