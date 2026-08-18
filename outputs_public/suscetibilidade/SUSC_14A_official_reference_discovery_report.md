# SUSC-14A - Descoberta de referencias espaciais oficiais

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

## 1. Objetivo
Localizar fontes oficiais/rastreaveis capazes de fornecer geometria de referencia
(logradouros, eixos de via, bairros, enderecos, pontos criticos de alagamento,
drenagem, mapas de risco/relatorios) para tentar resgatar o vinculo espacial de
ocorrencias oficiais de cheia registradas sem lat/lon (gargalo do SUSC-13C).

## 2. Politica de rede
- Ativacao live: `SUSC_13B_NETWORK=1`; rede nesta execucao: **Nao**.
- Sem Google Maps, sem geocoding generico, sem chave de API, sem raster.
- OSM/Nominatim nao e consultado; apenas camadas oficiais/rastreaveis.

## 3. Referencias registradas
Total: **13** (endpoints institucionais + camadas locais ja adquiridas).

| regiao | referencias |
|---|---|
| curitiba | 3 |
| petropolis | 5 |
| recife | 5 |

- Com pontos de endereco: **8**
- Com eixos/logradouros: **8**
- Com pontos/poligonos de cheia: **11**
- Camadas locais reutilizadas (13C): **1**
- Candidatas a download live: **9**

## 4. Limites e governanca
- Endpoints sao raizes institucionais; nenhum URL de download direto e inventado.
- Centroide de municipio/bairro nunca vira evento patch-level.
- Mapas de risco/suscetibilidade nunca viram ocorrencia observada.
- Tudo permanece review-only; nada cria ground truth, treino ou score v7.
