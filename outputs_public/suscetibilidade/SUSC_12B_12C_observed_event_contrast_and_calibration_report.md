# SUSC-12B/12C - Observed Event Contrast and Proxy Calibration

Status: **review-only** | `score_v7_created=false`

O SUSC-12B/12C compara métricas de suscetibilidade com evidências observacionais de alagamento/inundação para calibração review-only do proxy. A análise não cria ground truth, não treina modelo supervisionado e não transforma ausência de registro em negativo verdadeiro.

## 1. Objetivo
Comparar patches com evidencia observacional moderada a controles sem evento
documentado e gerar recomendacoes conservadoras para calibracao do proxy/score.

## 2. Diferenca entre contraste direto e avaliacao temporal
O SUSC-12A avaliou validade pre-evento e encontrou bloqueios temporais. O SUSC-12B
faz contraste observacional direto, sem afirmar temporalidade pre-evento.

## 3. Eventos usados
Patches evento/moderados: **9**. Patches de
contexto fraco: **5**.

## 4. Controles
Controles criados: **9**. Todos sao
`no_documented_event_control` e nao representam ausencia formal de evento.

## 5. Resultado do score v6 em patches com evento
Media evento: **0.593971**. Media controle:
**0.721077**. Diferenca evento-controle:
**-0.127106**.

## 6. Features que concordam com eventos observados
chirps_30d_mm, chirps_3d_mm, distance_to_water_mean, runoff_context_7d, urban_prop

## 7. Features que divergem
chirps_7d_mm, elevation_mean, flow_acc_log_mean, hand_mean, mndwi_mean, ndbi_mean, ndvi_mean, rain_7d_30d_ratio, rain_persistence_index, runoff_context_30d, slope_mean, tpi_250m_mean, twi_mean, vegetation_prop

## 8. Features inconclusivas
Sem features inconclusivas no contraste numerico, mas a confianca segue baixa.

## 9. Recomendacoes de calibracao
| recomendacao | n |
|---|---:|
| needs_more_observed_events | 19 |

## 10. Se score v7 foi criado ou nao
Nao foi criado. A saida oficial e `SUSC_12C_score_v7_not_ready_report.md`.

## 11. Limitacoes
- Amostra pequena de patches com evento moderado.
- Eventos fracos/contextuais nao foram usados como evidencia forte.
- Controles nao sao ausencia formal de evento.
- Nao ha treinamento, modelo persistido ou ground truth.

## 12. O que nao pode ser afirmado
- Nao se pode afirmar que o score v6 valida ocorrencia por patch.
- Nao se pode afirmar ausencia de evento nos controles.
- Nao se pode treinar modelo supervisionado com estes grupos.

## 13. O que ja pode ser afirmado
- O contraste evento/controle foi produzido de forma auditavel.
- A amostra atual nao justifica score v7.
- As recomendacoes sao conservadoras e review-only.

## 14. Proximo marco
SUSC-13A: ampliar evidencias observadas fortes/moderadas e revisar manualmente
os grupos antes de qualquer nova calibracao deterministica.

## Apêndice - score
| group | n | mean_score_v6 | median_score_v6 |
|---|---|---|---|
| event_patch | 9 | 0.593971 | 0.670423 |
| no_documented_event_control | 9 | 0.721077 | 0.689425 |
| event_minus_control | 9 | -0.127106 |  |

## Apêndice - regioes
| region | event_patch_count | control_patch_count | mean_score_event | mean_score_control | score_difference_event_minus_control |
|---|---|---|---|---|---|
| petropolis | 9 | 9 | 0.593971 | 0.721077 | -0.127106 |
| recife | 0 | 0 |  |  |  |
