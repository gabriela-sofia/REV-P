# v2bq — Resolver de geometria de overlay patch-evento

Versão: `v2bq`
Modo: auditoria geométrica autônoma. Não cria label, não cria negativo, não libera treino.

## Objetivo

Atacar o blocker técnico que o v2bp deixou para os 55 candidate-positives:

```text
NO_PATCH_EVENT_OVERLAY_GEOMETRY
```

Em vez de pedir "revisão humana" genérica, o v2bq faz por si só tudo que é
tecnicamente decidível: descobre geometria real no repositório, normaliza CRS,
valida, e **calcula a interseção patch-evento de verdade**. Geometria nunca é
inventada — se não existir, registra o bloqueio específico.

## O que faz

1. Lê os candidate-positives do v2bp.
2. Descobre dinamicamente fontes de geometria (`rglob` por `*.geojson` em
   `datasets/`, `manifests/`, `outputs_public/`, `local_runs/`, `docs/`,
   `configs/`), excluindo exemplos/templates/placeholders/vazios.
3. Classifica cada fonte: patch polygon, event polygon, pontos de contexto
   (suporte fraco) ou vazio.
4. Normaliza CRS para EPSG:4326 (infere apenas o caso inequívoco de graus
   lon/lat; reprojeta com pyproj quando disponível; CRS desconhecido bloqueia).
5. Quando patch boundary e event polygon existem, calcula a interseção
   (shapely quando disponível; fail-closed se ausente registrando
   `GEOMETRY_BACKEND_UNAVAILABLE`).
6. Classifica cada candidato em estados auditáveis (resolvido / rejeitado /
   bloqueado / ambíguo).

## Regras metodológicas

- **Overlay não é label.** Um overlay resolvido apenas estabelece que patch e
  evento se intersectam (ou não). Pode mover o caso para
  `READY_FOR_FORMAL_GT_PROTOCOL`; nunca cria `gt_patch_flood_observed` e nunca
  libera treino.
- **Overlay não libera treino sozinho.** `allowed_for_training=False` sempre.
- **Ausência de geometria não vira negativo.** É registrada como bloqueio
  específico (`OVERLAY_BLOCKED_PATCH_GEOMETRY_MISSING` etc.), nunca como
  evidência negativa.
- **Centroides ≠ overlay poligonal.** Nuvens de pontos são suporte fraco e
  nunca são promovidas a overlay resolvido.
- **CRS desconhecido bloqueia a promoção** (`OVERLAY_BLOCKED_CRS_UNKNOWN`).
- **Decisão da usuária só em ambiguidade real** — múltiplas geometrias
  poligonais distintas reivindicando o mesmo patch/evento
  (`OVERLAY_REVIEW_AMBIGUOUS_MULTIPLE_GEOMETRIES`). Missing técnico nunca é
  `NEEDS_USER_DECISION`.

## Resultado atual (dados reais)

- 55 candidate-positives processados;
- 103 fontes GeoJSON descobertas;
- 1 patch polygon real (`REC_00019`, boundary reprojetado de header de raster);
- 1 event polygon real (`REC_2022_05_24_30`, charter758 produto público
  digitalizado, `provided_unreviewed`, `can_be_ground_truth=false`);
- 1 overlay computado → `OVERLAY_REJECT_NO_INTERSECTION` (o patch `REC_00019`,
  no sul de Recife, e o polígono do evento, no norte, **não se sobrepõem** —
  achado geométrico real, confiança ALTA por não-sobreposição de bounding box);
- 54 `OVERLAY_BLOCKED_PATCH_GEOMETRY_MISSING` (sem boundary de patch);
- 0 resolvidos, 0 `NEEDS_USER_DECISION`, 0 labels, 0 treino.

Este é um resultado válido obtido por auditoria geométrica, não por "falta de
revisão humana": a geometria de evento existe, mas só 1 patch tem boundary, e
esse patch não intersecta o polígono do evento.

## O que falta para ground truth formal

- Boundary de patch para os 54 candidatos ainda bloqueados (digitalização).
- Reconciliar o caso `REC_00019`: confirmar se a não-interseção reflete a
  realidade espacial ou se o polígono do evento (produto de mídia, não revisado)
  precisa de georreferenciamento mais preciso.
- Protocolo formal de positivo/negativo com negativos comparáveis.

## Outputs

`local_runs/ground_truth/v2bq/` (13 arquivos `.csv`/`.json`/`.md`, leves).
Sidecars GeoJSON derivados, se houver, apenas em
`local_runs/ground_truth/v2bq/geometries/`. Nenhum shapefile bruto, raster ou
arquivo pesado é gravado ou versionado.
