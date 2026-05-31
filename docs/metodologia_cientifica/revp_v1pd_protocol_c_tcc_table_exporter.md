# v1pd - Protocol C TCC-Ready Table Exporter

## Objetivo

Gerar tabelas pequenas e diretamente reutilizaveis no TCC a partir dos
summaries v1ot e v1pa. Nao inventa dado — apenas reformata metricas
existentes com frases em portugues tecnico prontas para copia.

## Tabelas geradas

1. **Temporal Recovery** — metricas da recuperacao temporal Sentinel
2. **Observed Evidence** — metricas da camada observacional
3. **Guardrails** — evidencias de conformidade anti-overclaim
4. **Decision Levels** — definicao e contagem dos niveis C1-C4

## Uso no TCC

As colunas `tcc_sentence` contem frases em portugues tecnico que podem
ser copiadas diretamente para secoes de Metodos, Resultados ou Discussao.
Nenhuma frase faz overclaim ou promove evidencia fraca.
