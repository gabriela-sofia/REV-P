# REV-P v2cm - replay patch-evento bloqueavel

Este marco prepara replay patch-evento somente quando todos os pre-requisitos
existem: patch boundary, geometria observada candidata validada, CRS, reprojecao
possivel, proveniencia, hash e permissao do contrato `v2cl`.

Quando qualquer requisito falta, o replay bloqueia com motivo explicito. Campos de
area e razao de intersecao permanecem vazios em replays bloqueados.

Mesmo se um replay futuro for executado, a saida deve continuar como intersecao
candidata computada, sem ground truth operacional, sem label e sem claim de modelo.

