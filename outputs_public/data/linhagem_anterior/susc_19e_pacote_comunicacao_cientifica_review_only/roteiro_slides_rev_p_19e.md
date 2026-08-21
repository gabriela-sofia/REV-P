# Roteiro de slides - REV-P (comunicacao review-only)

Formato por slide: titulo, mensagem principal, bullets, figura sugerida, tabela sugerida e fala do apresentador.

## Slide 1 - Problema

- **Mensagem principal**: Enchentes urbanas exigem leitura de suscetibilidade, nao previsao
- **Bullets**:
  - impacto urbano de enchentes
  - Recife/Curitiba/Petropolis
  - lacuna de dados observacionais
- **Figura sugerida**: FIG_01
- **Tabela sugerida**: tabela_estado_por_regiao_19e
- **Fala do apresentador**: Abrir com o problema urbano e o recorte review-only.

## Slide 2 - O que o REV-P faz

- **Mensagem principal**: Framework multimodal auditavel de suscetibilidade urbana review-only
- **Bullets**:
  - features multimodais por patch
  - score_v6 candidato
  - avaliacao observacional review-only
- **Figura sugerida**: FIG_01
- **Tabela sugerida**: tabela_cobertura_multimodal_19e
- **Fala do apresentador**: Enfatizar auditavel e review-only.

## Slide 3 - O que o REV-P nao promete

- **Mensagem principal**: Nao e previsao, nao e ground truth, nao e modelo treinado
- **Bullets**:
  - sem previsao operacional
  - sem ground truth patch-level
  - sem benchmark
  - sem score_v7
  - sem alerta
- **Figura sugerida**: FIG_08
- **Tabela sugerida**: politica_claims_rev_p_19e
- **Fala do apresentador**: Deixar claros os limites antes dos resultados.

## Slide 4 - Pipeline multimodal

- **Mensagem principal**: Sentinel-first e 6 familias de features por patch
- **Bullets**:
  - fisico
  - hidrologico
  - espectral
  - territorial
  - chuva
  - documental
- **Figura sugerida**: FIG_02
- **Tabela sugerida**: tabela_cobertura_multimodal_19e
- **Fala do apresentador**: Mostrar o pipeline e a separacao de camadas.

## Slide 5 - Features por patch

- **Mensagem principal**: 300 patches com fisico/espectral/chuva completos e territorial parcial
- **Bullets**:
  - 300 patches
  - cobertura territorial 33.3%
  - missingness MapBiomas/solo/agua/impervious
- **Figura sugerida**: FIG_05
- **Tabela sugerida**: tabela_cobertura_multimodal_19e
- **Fala do apresentador**: Explicitar a lacuna territorial.

## Slide 6 - Evidencia observacional review-only

- **Mensagem principal**: 7 patches com evidencia review-only, nunca label
- **Bullets**:
  - event_record
  - source_footprint
  - derived_patch_link
  - feature_evidence
  - score_evaluation
- **Figura sugerida**: FIG_03
- **Tabela sugerida**: tabela_resultados_observacionais_19e
- **Fala do apresentador**: Explicar a separacao das 5 camadas.

## Slide 7 - Recife

- **Mensagem principal**: Referencia forte review-only de uma regiao e um evento
- **Bullets**:
  - 5 canarios
  - coerencia urbana/topografica
  - amostra pequena
- **Figura sugerida**: FIG_04
- **Tabela sugerida**: tabela_estado_por_regiao_19e
- **Fala do apresentador**: Recife e o caso mais solido, ainda review-only.

## Slide 8 - Curitiba SAR

- **Mensagem principal**: Segunda regiao tecnica SAR sem geometria oficial
- **Bullets**:
  - 2 overlays SAR
  - SAR e pos-evento
  - nao e geometria oficial
- **Figura sugerida**: FIG_04
- **Tabela sugerida**: tabela_estado_por_regiao_19e
- **Fala do apresentador**: SAR nao e geometria oficial.

## Slide 9 - Petropolis bloqueado

- **Mensagem principal**: Fenomeno misto sem separacao permanece bloqueado
- **Bullets**:
  - deslizamento e inundacao misturados
  - fora de observado
  - 29 patches bloqueados
- **Figura sugerida**: FIG_04
- **Tabela sugerida**: tabela_estado_por_regiao_19e
- **Fala do apresentador**: Petropolis nao e promovido.

## Slide 10 - Matriz multimodal 300 patches

- **Mensagem principal**: Consolidacao escalavel por patch
- **Bullets**:
  - 1 linha por patch
  - 6 familias
  - cobertura nao e score
- **Figura sugerida**: FIG_02
- **Tabela sugerida**: tabela_cobertura_multimodal_19e
- **Fala do apresentador**: Coverage nao e suscetibilidade.

## Slide 11 - Avaliacao 19C

- **Mensagem principal**: Score observado maior que background, mas 0/7 no top-30 global
- **Bullets**:
  - score obs 0.596 vs background 0.521
  - 0/7 top-30 global
  - 3/7 top-30 regional
- **Figura sugerida**: FIG_06
- **Tabela sugerida**: tabela_resultados_observacionais_19e
- **Fala do apresentador**: hit-rate e exploratorio, nao benchmark.

## Slide 12 - Diagnostico 19D

- **Mensagem principal**: Aderencia urbana/topografica e divergencia hidrologica/chuva
- **Bullets**:
  - familias aderentes
  - familias divergentes
  - curitiba_01101 divergencia forte
- **Figura sugerida**: FIG_07
- **Tabela sugerida**: tabela_resultados_observacionais_19e
- **Fala do apresentador**: Explicar por que os observados nao vao ao extremo.

## Slide 13 - Limitacoes

- **Mensagem principal**: Amostra minima, missingness territorial, sem benchmark, sem geometria oficial
- **Bullets**:
  - n=7 sem conclusao forte
  - territorial incompleto
  - score_v7 bloqueado
  - background nunca negativo
- **Figura sugerida**: FIG_08
- **Tabela sugerida**: tabela_bloqueios_score_v7_19e
- **Fala do apresentador**: Ser explicito nas limitacoes.

## Slide 14 - Proximos passos

- **Mensagem principal**: Ampliar amostra, preencher territorial, separar fenomeno
- **Bullets**:
  - mais eventos com geometria oficial
  - pacote MapBiomas/GEE
  - separar Petropolis
- **Figura sugerida**: FIG_08
- **Tabela sugerida**: tabela_proximos_passos_19e
- **Fala do apresentador**: Roadmap sem prometer score_v7.

## Slide 15 - Conclusao

- **Mensagem principal**: Contribuicao: framework multimodal auditavel review-only, com diagnostico honesto
- **Bullets**:
  - suscetibilidade review-only
  - divergencias documentadas
  - score_v6 intacto
- **Figura sugerida**: FIG_01
- **Tabela sugerida**: tabela_resultados_observacionais_19e
- **Fala do apresentador**: Fechar com a contribuicao e a honestidade metodologica.

> Todos os slides mantem o enquadramento review-only: sem previsao operacional, sem ground truth patch-level, sem modelo treinado, sem benchmark e sem score_v7. O background nunca e negativo e o SAR nao e geometria oficial.
