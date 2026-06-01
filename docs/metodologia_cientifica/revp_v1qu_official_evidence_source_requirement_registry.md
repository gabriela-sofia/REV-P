# v1qu — Official Evidence Source Requirement Registry

## Objetivo

Definir requisitos de fonte externa por regiao, tipo de ameaca e necessidade de evidencia. Fontes nao presentes localmente sao marcadas SOURCE_REQUIRED_NOT_LOCAL. Nao inventa que uma fonte existe; nao usa internet; nao baixa dados.

## Familias de fonte priorizadas

CEMADEN, ANA/HidroWeb, INMET/BDMEP (hidrometeorologicas), SGB/CPRM (geologica), Defesa Civil municipal/estadual, Diario Oficial (publicacao governamental), IBGE/MapBiomas (contexto territorial, nunca label de evento).

## Resultado

Total de requisitos: 18. SOURCE_REQUIRED_NOT_LOCAL: 8. Parcialmente local: 10. Requisitos que bloqueiam C3: 12.

## Guardrails

review_only=true em todas as linhas. Nenhuma fonte midiatica ou social fecha o gate C3 sozinha. Nenhum requisito cria label, target ou ground truth operacional.
