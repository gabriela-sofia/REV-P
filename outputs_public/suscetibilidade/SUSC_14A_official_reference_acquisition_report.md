# SUSC-14A - Aquisicao de referencias espaciais oficiais

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

## 1. Politica
- Rede habilitada: **Nao** (ativacao `SUSC_13B_NETWORK=1`).
- Aceitos: CSV, XLSX, GeoJSON, JSON, KML/KMZ, SHP ZIP, GPKG, WFS/ArcGIS GeoJSON, PDF pequeno, TXT.
- Bloqueados: raster, Sentinel bruto, executavel, HTML sem dado, dados privados, arquivos > 250MB.
- Sem copia de dado pesado: camadas locais ja adquiridas (13C) sao registradas por SHA256/caminho.

## 2. Resultado
Total de registros no manifesto: **13**.

| download_status | n |
|---|---|
| endpoint_only | 3 |
| network_disabled | 9 |
| reused_local_official | 1 |

- Camadas oficiais locais reutilizadas (sem copia): **1**.
- Endpoints institucionais registrados (sem recurso direto inventado): **3**.

## 3. Governanca
Nenhum arquivo bruto pesado e versionado. A reproducao usa este manifesto
(URL/SHA256/tamanho). Tudo review-only; nada vira ground truth, treino ou score v7.
