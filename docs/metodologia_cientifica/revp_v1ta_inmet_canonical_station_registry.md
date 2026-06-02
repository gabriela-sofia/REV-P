# v1ta — INMET Canonical Station Registry

## Objetivo

Registry canônico de estações INMET com coordenadas corrigidas (decimal-vírgula → ponto). Preserva provenance de v1si, v1sr e ZIPs brutos.

## Resultado
Estações: 668. OK coords: 668. Dentro de 100km: 25.

## Nota metodológica

v1si tinha bug de parse: vírgula decimal interpretada como milhar. v1ta corrige usando os ZIPs originais. v1si não é modificado.
