# v1r4 — Extended DINO x SEDEC Analysis (n=81, all 22 real negatives covered)

## Objetivo

Após baixar 13 patches novos centrados em evidência (negativos reais) e 2 patches oficiais adicionais nesta sessão, 78 dos 163 registros SEDEC primary (real-vs-real) agora têm embedding DINO real: 56 positivos, 22 negativos — TODOS os 22 negativos reais cobertos.

## Disciplina metodológica

Duas análises separadas, nunca misturadas: (A) screen univariado Mann-Whitney nas 6 variáveis físicas + 2 componentes PCA do DINO, cada uma isolada; (B) modelo Firth multivariado usando SOMENTE os 2 componentes PCA do DINO (EPV=11, atende a heurística >=10) — nunca combinado com as 6 variáveis físicas no mesmo modelo (isso daria EPV=2.75, abaixo de qualquer heurística usável).

## Resultado

LOO AUC do modelo Firth somente-DINO: 0.4903.
