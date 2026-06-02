# Protocolo C - precheck split/leakage positivo-negativo v1jy

v1jy cria apenas um precheck. A separacao respeita evento, localidade e anchor; nao ha split aleatorio por patch. Controles temporais do mesmo anchor nunca entram como negativos.

`can_train_model` permanece falso em todos os cenarios desta etapa.
