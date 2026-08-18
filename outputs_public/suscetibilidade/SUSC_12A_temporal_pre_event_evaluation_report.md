# SUSC-12A - Temporal Pre-Event Susceptibility Evaluation

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

O SUSC-12A avalia se patches com evidência observada apresentavam maior suscetibilidade antes do evento, quando a temporalidade dos dados permite. A análise é review-only, não cria ground truth, não treina modelo supervisionado e não constitui predição operacional.

## 1. Objetivo
Avaliar se patches ligados a evidencias observadas/contextuais no SUSC-11C ja
apareciam com maior score v6 candidato antes do evento, sem transformar essa
aderencia em rotulo, ground truth ou predicao operacional.

## 2. O que e avaliacao temporal pre-evento
E uma checagem de coerencia temporal: uma feature so conta como pre-evento quando
sua data de referencia e comprovadamente anterior ao evento. Quando a data e
ausente, igual ao periodo do evento ou posterior, a analise fica incerta ou com
risco de vazamento.

## 3. Eventos usados
Eventos distintos avaliados: **9**. Linhas
evento-patch: **44**. Patches distintos com ligacao:
**14**.

## 4. Criterio de pre-evento
`pre_event_valid=true` exige que nenhuma feature temporal usada no score tenha
risco pos-evento ou metadado temporal ausente/incerto. Features fisicas estaticas
sao aceitas como `static_feature`.

## 5. Features estaticas
Topografia, hidrologia estrutural e distancia/forma de drenagem entram como
condicionantes fisicos estaticos ou quase estaticos, ainda review-only.

## 6. Features temporais
Chuva, runoff, indices espectrais e uso/cobertura urbana exigem metadado temporal
pre-evento. Quando a referencia disponivel e posterior ao evento, a linha e
marcada como `post_event_risk`.

## 7. Riscos de vazamento
| status | n |
|---|---:|
| high | 8 |
| temporal_uncertain | 80 |

Validade temporal agregada:

| feature_temporal_status | n |
|---|---:|
| missing_temporal_metadata | 80 |
| post_event_risk | 8 |

## 8. Comparacao evento vs controle
Media evento: **0.664892**.
Media controle: **0.696344**.
Diferenca media evento-controle: **-0.031453**.

Controles sao `no_documented_event_control`; eles nao representam ausencia formal de evento.

## 9. Hit-rate top-k
| top-k | limiar score | hit-rate evento | hit-rate controle | razao |
|---|---:|---:|---:|---:|
| top_10 | 0.801302 | 0.0 | 0.318182 | 0.0 |
| top_20 | 0.698025 | 0.386364 | 0.5 | 0.772727 |
| top_30 | 0.633684 | 0.636364 | 0.681818 | 0.933333 |

## 10. Resultados por regiao
| regiao | evento-patch | controles | media evento | media controle | diferenca |
|---|---:|---:|---:|---:|---:|
| petropolis | 30 | 30 | 0.670825 | 0.694349 | -0.023525 |
| recife | 14 | 14 | 0.652178 | 0.700619 | -0.048441 |

## 11. Limitacoes
- Amostra exploratoria e pequena para inferencia estatistica.
- As ligacoes SUSC-11C permanecem review-only.
- Varios eventos nao tem data suficiente para validar temporalidade.
- Score v6 usa componentes temporais/espectrais que podem ter referencia posterior ao evento.
- Controle sem evento documentado nao e ausencia formal de evento.

## 12. O que nao pode ser afirmado
- Nao ha ground truth por patch.
- Nao ha treino supervisionado.
- Nao ha predicao operacional.
- Controle sem evento documentado nao representa ausencia formal de evento.
- Nao se pode afirmar causalidade ou confirmacao automatica por patch.

## 13. O que ja pode ser afirmado
- A comparacao evento vs controle foi gerada de modo reproduzivel e auditavel.
- A auditoria temporal identifica quais linhas ficam bloqueadas por risco pos-evento
  ou metadado temporal insuficiente.
- A leitura quantitativa e exploratoria e review-only.

## 14. Proximo marco
SUSC-12B: readiness/gap report para preencher metadados temporais pre-evento e
separar features estaticas de features dinamicas em uma avaliacao sem vazamento.
