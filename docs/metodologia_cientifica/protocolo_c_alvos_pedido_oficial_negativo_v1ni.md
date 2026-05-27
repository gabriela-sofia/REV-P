# Protocolo C - alvos de pedido oficial negativo v1ni

Esta etapa criou 9 alvos objetivos de pedido oficial para a lacuna C4 FORMAL_NEGATIVES_ZERO.

Negativo formal exige declaracao oficial explicita de ausencia, estabilidade, sem ocorrencia, sem instabilidade ou sem dano geologico, com local, periodo e fenomeno compativeis.

Ausencia de registro, Diario Oficial sem ato tecnico, background patch, distancia de anchor positivo e pseudo-ausencia nao sao negativos formais.

Nenhum alvo abre C4 sozinho. Todos mantem can_unlock_c4_alone=false e dependem de intake, adjudicacao estrita e checagem de leakage.
