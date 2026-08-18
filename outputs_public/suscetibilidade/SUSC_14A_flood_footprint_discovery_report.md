# SUSC-14A - Descoberta de footprints publicos de cheia

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

## 1. Objetivo
Localizar footprints publicos de area alagada/inundada em formato vetorial
(GeoJSON/SHP/KML/GPKG/WKT ou PDF com vetor/tabela) para Recife, Petropolis e
Curitiba, priorizando a ativacao Copernicus EMS de Petropolis 2022. **Nenhum
raster pesado e baixado**: se so houver raster/preview, registra-se
`footprint_unavailable_vector_required`.

## 2. Politica de rede
- Ativacao live: `SUSC_13B_NETWORK=1`; rede nesta execucao: **Nao**.
- Sem Google Maps, sem geocoding, sem chave de API, sem raster Sentinel.

## 3. Fontes registradas
Total: **10** | com expectativa de vetor: **9** | com risco de so-raster: **5**.

| regiao | fontes |
|---|---|
| curitiba | 2 |
| petropolis | 5 |
| recife | 3 |

## 4. Petropolis 2022
Copernicus EMS Rapid Mapping e a fonte de maior prioridade (ativacao com periodo
explicito 2022-02). INEA-RJ, Defesa Civil RJ e SGB/CPRM complementam com setores
afetados/areas inundaveis.

## 5. Governanca
Fontes sao raizes/catalogos institucionais; nenhum URL de download direto e
inventado. Tudo review-only; nada vira ground truth, treino ou score v7.
