# Protocolo C - descoberta de fontes negativas oficiais abertas v1jq

## Objetivo

v1jq encerra a repeticao do scan interno de v1jp e abre uma fila dirigida de fontes oficiais abertas. A etapa procura fontes que possam conter evidencia explicita de ausencia, estabilidade ou vistoria sem ocorrencia para Petropolis 2022, mas nao cria negativo formal, label operacional ou treino.

## Por que v1jp nao foi repetido

v1jp escaneou documentos e metadados locais e nao encontrou evidencia negativa formal pronta nem candidato em revisao. O gargalo real passou a ser fonte oficial externa aberta, nao mais varredura local ampla.

## Fontes tentadas

Foram inventariadas 12 fontes abertas: SGB/CPRM RIGeo, Prefeitura/Defesa Civil de Petropolis, DRM-RJ, Dados Abertos RJ, S2ID/Atlas Digital, dados.gov.br, Cemaden, IBGE/BDIA, MapBiomas e contexto INPE/Charter.

## Regra de uso

Contexto, carta de risco, classe de baixo risco, catalogo sem registro, area fora de risco, monitoramento, background e pseudo-ausencia nao valem como negativo formal. Uma fonte so avanca se trouxer declaracao explicita de ausencia/estabilidade ou vistoria sem ocorrencia com local, data, fenomeno compativel e possibilidade de linkage a patch.

## Classificacao

- HIGH_POTENTIAL_FORMAL_NEGATIVE_SOURCE: 1
- MODERATE_REVIEW_SOURCE: 3
- CONTEXT_ONLY_SOURCE: 8
- INACCESSIBLE_OR_NOT_FOUND: 0

## Efeito em C4

summary_decision = EXTERNAL_DISCOVERY_REVIEW_SOURCES_FOUND;C4_STILL_BLOCKED

C4 segue bloqueado. Completar descoberta de fonte nao basta; e necessario extrair evidencia formal, revisar leakage e manter pseudo-ausencia apenas como PU/sandbox.
