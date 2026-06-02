# v1tb — INMET Coordinate Parse Discrepancy Audit

## Objetivo

Documentar discrepância entre coordenadas v1si (parse quebrado) e v1ta (parse correto com decimal-vírgula). v1si não é modificado.

## Causa do bug em v1si

O extrator v1si interpretou a vírgula decimal dos CSVs INMET como separador de milhar (ex: -22,75 lido como -2275.0 ou '-22,' + '75'). v1sr e v1ta corrigem substituindo vírgula por ponto antes do parse.

## Resultado
Estações comparadas: 668. Corrigidas em v1ta: 668. Afeta matching de região: 668.
