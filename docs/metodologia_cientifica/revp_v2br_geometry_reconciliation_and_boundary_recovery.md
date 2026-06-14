# v2br — Reconciliação geométrica e recuperação de boundaries

Versão: `v2br`
Modo: auditoria geométrica autônoma. Não cria label, não cria negativo, não libera treino.

## 1. Por que o v2br existe

O v2bq calculou um overlay real (`REC_00019` × `REC_2022_05_24_30`) e encontrou
interseção zero, além de bloquear 54 candidatos sem patch geometry. O v2br não
trata essa não-interseção como descarte ingênuo nem como negativo: audita a
qualidade geométrica e tenta recuperar boundaries para preparar uma nova rodada
de overlay.

## 2. O que significa auditar uma não-interseção

Significa testar se a separação calculada é um fato espacial real ou um artefato
de CRS, axis-order, transformação UTM, offset, simplificação de geometria ou
mismatch de lineage. O v2br valida patch e evento (tipo, CRS, bbox, centroide,
área, plausibilidade em Recife, fonte), mede a distância entre eles, cruza com
os pontos independentes da Defesa Civil e roda um quadro de hipóteses de erro.

Resultado para `REC_00019`: a matemática geométrica é consistente (mesmo caminho
de CRS EPSG:32725→4326, axis-order válido, ambos dentro da janela de Recife),
mas o polígono do evento é um produto de mídia **não revisado**
(`can_be_ground_truth=false`) e os pontos da Defesa Civil não alinham com
nenhuma das geometrias. Decisão: `NON_INTERSECTION_HELD_EVENT_GEOMETRY_UNREVIEWED`
(segurar, não confirmar).

## 3. Por que não-interseção não vira negativo formal

Uma não-interseção apenas diz que estas duas geometrias atuais não se sobrepõem.
O polígono do evento não é ground truth, então o resultado não pode virar
`gt_patch_flood_observed=0`. O candidato fica `HELD_FOR_GEOMETRY_RECONCILIATION`,
com `gt_patch_flood_observed=NA` e `allowed_for_training=False`.

## 4. Por que o event polygon `provided_unreviewed` não vira ground truth

É um candidato digitalizado de um produto público de mídia do Charter, marcado
`provided_unreviewed` / `can_be_ground_truth=false`, e não alinha com os pontos
independentes da Defesa Civil. Uso recomendado: `USE_FOR_OVERLAY_QA_ONLY`.

## 5. Como as boundaries são recuperadas

A partir de bounds de header de raster **já gravados** na auditoria de sanidade
de assets `v1fs` (`bounds_if_header_available` + `crs_if_header_available`),
reprojetados de UTM (EPSG:32725) para WGS84 com pyproj. Os bounds são metadado
real registrado — não há leitura de raster ao vivo nem invenção de coordenada.
Cada boundary recuperada é gravada como sidecar GeoJSON leve em
`recovered_patch_boundaries/`. Resultado real: 36 recuperadas, 18 não
encontradas (short-ids sem bounds registrados).

## 6. Por que centroides/pontos não substituem polígonos

Um centroide ou nuvem de pontos (como os 400 pontos da Defesa Civil) é suporte
fraco de contexto: não define a extensão espacial de um patch nem um overlay
poligonal. O v2br nunca promove ponto/centroide a boundary; bounds degenerados
(um único ponto) são classificados como
`PATCH_BOUNDARY_NOT_RECOVERED_CENTROID_ONLY`.

## 7. Como prepara nova rodada do v2bq

As boundaries recuperadas e o caso `REC_00019` entram em
`next_overlay_candidate_queue_v2br.csv`. Prioridade HIGH = boundary recuperada +
event polygon existente (36 casos). O v2bq pode ser reexecutado apontando para
os sidecars recuperados para calcular os overlays reais.

## 8. Por que treino segue bloqueado

`labels_created=false`, `allowed_for_training_count=0`,
`promotion_to_operational_gt=false`. Recuperar geometria e segurar uma
não-interseção não criam label nem negativo formal. O protocolo formal de
positivo/negativo de ground truth ainda é obrigatório antes de qualquer treino.

## Outputs

`local_runs/ground_truth/v2br/` (12 arquivos `.csv`/`.json`/`.md`, leves) +
`recovered_patch_boundaries/` (sidecars `.geojson`). Nenhum raster, shapefile
bruto ou arquivo pesado é gravado ou versionado.
