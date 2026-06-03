# v1sy — Hydrometeorological Context Runbook

## Declaracao obrigatoria

A camada v1sr–v1sz usa dados hidrometeorológicos oficiais apenas como contexto temporal e regional review-only. Precipitação observada, proximidade de estação ou janela temporal compatível não validam automaticamente evento, não criam ground truth operacional, não criam negativo formal e não substituem revisão supervisora.

## Como usar dados INMET/ANA como contexto

1. **Identificar estações próximas** (v1sr): usar somente estações dentro de 100 km da região de interesse como contexto espacial plausível.

2. **Construir janelas temporais** (v1ss): T-7 a T+1 relativas à data documentada do evento. Janela não implica causalidade.

3. **Consultar precipitação no período** (v1st): valores de precipitação diária como descrição do contexto meteorológico. Ausência de precipitação não é evidência negativa de evento.

4. **Calcular features contextuais** (v1su): acumulados 1d/3d/7d são descritivos; não podem ser usados como target ou label supervisionado.

5. **Crosswalk de intake** (v1sv): para cada janela, criar entrada de intake manual no v1rb com todos os campos review-only.

6. **Revisão supervisora obrigatória**: qualquer interpretação de causalidade entre precipitação e evento requer revisão por especialista.

## O que não fazer

- Não tratar compatibilidade de precipitação como validação de evento.

- Não tratar ausência de precipitação como prova de que o evento não ocorreu.

- Não usar features de precipitação como target supervisionado.

- Não abrir C4 (análise exploratória supervisionada) com estes dados.

- Não citar DINO como validação de contexto.

## Status do pipeline

- Estações INMET dentro de 100km: 25

- Janelas de evento construídas: 9

- Linhas de contexto de precipitação: 810

- Guardrail audit status: GUARDRAIL_PASS_ALL

## Fontes

INMET — Instituto Nacional de Meteorologia. Dados históricos automáticos. Fonte pública oficial. Verificar licença antes de publicação.
