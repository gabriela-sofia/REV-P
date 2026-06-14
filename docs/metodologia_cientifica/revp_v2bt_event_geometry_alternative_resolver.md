# v2bt — Reconstrução de geometria alternativa de evento e resolução de confiabilidade

Versão: `v2bt`
Modo: auditoria geométrica autônoma. Não cria label, não cria negativo, não libera treino.

## 1. Por que o v2bt existe

O v2bs bloqueou a geometria do evento `REC_2022_05_24_30` como conflitante com
os pontos da Defesa Civil
(`EVENT_GEOMETRY_RELIABILITY_BLOCKED_CONFLICTS_WITH_DEFENSE_CIVIL_POINTS`). O
v2bt resolve esse problema de confiabilidade: audita os pontos, decide o destino
do polígono charter e constrói geometrias alternativas QA-only a partir dos
pontos — sem promover nenhuma delas a ground truth.

## 2. Por que o polígono charter foi rebaixado/rejeitado

Decisão: `CHARTER_POLYGON_REJECTED_FOR_EVENT_QA`. Zero dos 400 pontos Defesa
Civil caem dentro ou perto do polígono charter (mais próximo ~13 km), e o
polígono é um produto de mídia não revisado (`can_be_ground_truth=false`). Isso
**não invalida o evento histórico** — apenas rejeita essa geometria específica
para fins de QA de overlay.

## 3. Como os pontos Defesa Civil foram auditados

400 pontos oficiais de risco (CRS EPSG:4326), nuvem compacta (~3 km × 3 km) no
sul de Recife (Ibura/Lagoa Encantada). Auditoria: contagem, pontos válidos,
bbox, centroide, extensão, pontos dentro do polígono/bbox charter, distância ao
charter, clusters (DBSCAN). Os pontos servem **apenas** para reconciliação de
geometria de evento QA-only (`can_define_gt=false`, `can_define_overlay=false`).

## 4. Quais geometrias alternativas foram criadas

A partir dos pontos válidos:

1. **Convex hull** (`POINT_CONVEX_HULL_EVENT_CANDIDATE`).
2. **Buffered point union** — buffers de 250m e 500m em EPSG:32725, união,
   reprojetados para EPSG:4326 (`POINT_BUFFER_UNION_EVENT_CANDIDATE`).
3. **Envelopes de cluster** — DBSCAN (eps=1000m, min_samples=5) sobre coordenadas
   projetadas; convex hull por cluster não-ruído
   (`POINT_CLUSTER_ENVELOPE_EVENT_CANDIDATE`).

No estado atual: 5 geometrias criadas/prontas. Clusters próximos (mesma área do
evento) não geram ambiguidade; clusters distantes (> 10 km) sim.

## 5. Por que hull/buffer/cluster são QA-only

Eles derivam de pontos de risco, não de um footprint oficial revisado de
inundação. Cada um é `POINT_DERIVED_EVENT_GEOMETRY_CANDIDATE` com
`recommended_use=USE_FOR_OVERLAY_QA_ONLY`, `can_use_for_formal_gt=false`,
`can_create_label=false`. Um ponto, hull, buffer ou cluster nunca é ground
truth.

## 6. Por que isso ainda não é ground truth

`labels_created=false`, `allowed_for_training_count=0`,
`event_geometry_ready_for_formal_gt=false`. Uma geometria QA-only derivada de
pontos de risco não pode ser label de inundação nem negativo. Ela apenas permite
uma rodada de overlay mais significativa.

## 7. Como a próxima rodada de overlay deve usar essas geometrias

A fila `alternative_overlay_retry_queue_v2bt.csv` aponta a melhor geometria
QA-only (maior score) contra os 37 patches reprocessados / 36 boundaries
recuperadas. A próxima etapa reexecuta o overlay contra ela — para testar
compatibilidade geométrica, não para criar label.

## 8. Por que treino segue bloqueado

Não existe geometria de evento revisada nem protocolo formal de positivo/negativo.
Alternativas QA-only não mudam isso. Treino segue bloqueado.

## Outputs

`local_runs/ground_truth/v2bt/` (11 arquivos `.csv`/`.json`/`.md`, leves) +
`alternative_event_geometries/` (sidecars `.geojson`). Nenhum raster, shapefile
bruto ou arquivo pesado é gravado ou versionado.
