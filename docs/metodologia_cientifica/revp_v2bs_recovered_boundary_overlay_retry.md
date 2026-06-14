# v2bs — Overlay retry nos boundaries recuperados e confiabilidade do evento

Versão: `v2bs`
Modo: auditoria geométrica autônoma. Não cria label, não cria negativo, não libera treino.

## 1. Por que o v2bs existe

O v2br recuperou 36 patch boundaries e segurou a não-interseção do `REC_00019`.
O v2bs reexecuta o overlay dessas boundaries contra o polígono disponível do
evento `REC_2022_05_24_30` e — crucialmente — classifica a confiabilidade desse
polígono antes que qualquer caso avance para um protocolo formal de ground
truth. Calcular overlay não basta: é preciso separar **o que a geometria atual
sugere** de **o que pode ser usado como label formal**.

## 2. Como usa as boundaries recuperadas

Lê a fila do v2br, carrega os sidecars GeoJSON recuperados (mais a boundary
original do `REC_00019`), carrega o polígono do evento e calcula a interseção
real (área, razões, distância de centroides e distância mínima) para cada caso.

## 3. Por que overlay positivo não vira label

Uma interseção aqui significa apenas que uma boundary recuperada se sobrepõe ao
polígono atual do evento. Não é um label positivo de inundação. Enquanto o
polígono do evento for `provided_unreviewed`/`can_be_ground_truth=false`, o caso
fica `OVERLAY_INTERSECTS_HELD_EVENT_GEOMETRY_UNREVIEWED` —
`gt_patch_flood_observed=NA`, `allowed_for_training=False`.

## 4. Por que não-interseção não vira negativo formal

Uma não-interseção diz apenas que estas duas geometrias atuais não se sobrepõem.
O polígono do evento não é ground truth, então não há base para
`gt_patch_flood_observed=0`. O caso fica
`OVERLAY_NO_INTERSECTION_HELD_EVENT_GEOMETRY_UNREVIEWED` (evidência geométrica
segura, não negativo).

## 5. Por que o polígono do evento ainda bloqueia promoção a GT

O polígono é um produto de mídia do Charter, não revisado,
`can_be_ground_truth=false`. No estado atual, os 400 pontos independentes da
Defesa Civil não caem dentro nem perto do polígono (mais próximo ~13 km), o que
resulta em `EVENT_GEOMETRY_RELIABILITY_BLOCKED_CONFLICTS_WITH_DEFENSE_CIVIL_POINTS`.
Nenhum overlay contra esse polígono pode ser promovido a GT
(`gt_promotion_allowed=false`, `recommended_use=USE_FOR_OVERLAY_QA_ONLY`).

## 6. Como os pontos Defesa Civil entram como QA contextual

Os pontos apenas auditam a plausibilidade do polígono do evento. Eles não
definem overlay, não definem GT e não viram negativo
(`can_define_overlay=false`, `can_define_gt=false`). Quando não caem no/perto do
polígono, indicam baixa confiabilidade do polígono — não um descarte do patch.

## 7. Quantos intersectam, não intersectam ou seguem bloqueados

No estado atual: **0 intersectam, 37 não intersectam, 0 bloqueados, 0
ambíguos**. Os 36 patches recuperados estão no sul/meio de Recife (lat
−8.16..−8.02) e o polígono do evento no norte (lat ~−7.99). Isso reforça a baixa
confiabilidade do georreferenciamento do polígono do evento.

## 8. Por que treino segue bloqueado

`labels_created=false`, `allowed_for_training_count=0`,
`promotion_to_operational_gt=false`. Recalcular overlays contra um polígono de
evento não revisado não altera o gate de treino. São necessários uma geometria
de evento revisada/alternativa e um protocolo formal de positivo/negativo antes
de qualquer treino.

## Outputs

`local_runs/ground_truth/v2bs/` (12 arquivos `.csv`/`.json`/`.md`, leves).
Nenhum raster, shapefile bruto ou arquivo pesado é gravado ou versionado.
