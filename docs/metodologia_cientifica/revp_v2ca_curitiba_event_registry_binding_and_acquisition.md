# v2ca — Binding de registry de eventos Curitiba e pipeline de aquisição de evidência

Versão: `v2ca`
Modo: pipeline programático real, offline-determinístico. Não cria label, não cria
negativo formal, não libera treino, **não inventa evento nem geometria**, não
promove área de risco a footprint de evento.

## 1. Por que o v2ca existe

A cadeia `v2bn–v2bz` mostrou que Recife foi processado até protocolo dry-run com um
único positivo robusto (`REC_00276`), e que a coorte não cresce sem mais positivos —
o que exige eventos fora de Recife com geometria/pontos reais. O `v2bz` provou que
Curitiba **não deve permanecer** como `CUR_EVENT_REGISTRY_MISSING`, porque existem
candidatos reais no registry `v1uv`. O `v2ca` é a etapa pesada que transforma
Curitiba de "registry ausente" numa **coorte flood-compatible auditável**, com
fontes, patches candidatos, readiness geométrica e fila para repetir a cadeia —
sem cruzar a fronteira de label.

## 2. Por que Curitiba é o próximo alvo

- **Recife**: já processado (urban flood); platô em 1 positivo dry-run.
- **Petrópolis**: eventos de `mass_movement` (deslizamento), resolvidos pelo `v2bz`
  como coorte separada; **não devem ser forçados como flood**.
- **Curitiba**: os candidatos são `urban_flooding` **oficiais** — o próximo alvo
  flood mais correto para crescer a coorte.

## 3. Como `CUR_EVENT_REGISTRY_MISSING` foi reparado

O registry ausente é reparado a partir do registry de candidatos real
`datasets/protocolo_c/v1uv_curitiba_candidate_event_registry.csv` — **nunca um
evento inventado**. Se o registry estiver vazio, nenhum evento é criado e o status
permanece `CURITIBA_EVENT_REGISTRY_STILL_MISSING` (verificado em teste). Eventos
reparados:

- `CUR_2022_01_05` — flood, 2022-01-05, `CURITIBA_FLOOD_EVENT_REGISTRY_REPAIRED`
- `CUR_2022_01_15` — flood, 2022-01-15, `CURITIBA_FLOOD_EVENT_REGISTRY_REPAIRED`

Ambos `is_official=true`, `can_create_training_label=false`, `event_status=`
`CURITIBA_EVENT_BLOCKED_NO_GEOMETRY_OR_POINTS`.

## 4. Quais eventos oficiais foram encontrados

2 eventos oficiais `urban_flooding` confirmados (`CUR_2022_01_15`, `CUR_2022_01_05`).
Se uma fonte trouxer outro evento não confirmado, ele entraria como
`CURITIBA_EVENT_CANDIDATE_UNVERIFIED` até auditoria — nunca promovido silenciosamente.

## 5. Qual evidência ainda falta

Foram inventariadas **119 fontes locais** de Curitiba (2 contexto oficial / 51
derivadas / 66 não-verificadas). **Nenhuma** traz geometria vetorial event-specific
nem pontos de ocorrência para os eventos candidatos: o inventário geométrico resolve
ambos os eventos como `CONTEXT_ONLY_NO_GEOMETRY`. Sem footprint validado ou pontos,
a cadeia geometria → overlay → dry-run não roda. O gargalo de Curitiba é, portanto,
**aquisição de geometria/pontos do evento** (Defesa Civil Curitiba, IPPUC, CEMADEN,
Simepar, Prefeitura, Governo do Paraná).

## 6. Como os patches Curitiba foram auditados

- **43 patches** Curitiba na feature table `v2bn` (sentinel input, embedding DINO,
  GIS, split group).
- **4 patches** com embedding DINO real (768D); os demais sem embedding local.
- **43/43 boundaries recuperadas** a partir dos bounds de header de raster gravados
  no `asset_sanity_audit_v1fs.csv` (CRS `EPSG:32722`), reprojetados para `WGS84` com
  pyproj. **Nenhum raster pesado foi aberto** — os bounds são metadado já gravado.
  Cada boundary recuperada é um sidecar GeoJSON com `can_be_ground_truth=false` e
  `review_status=auto_recovered_unreviewed`, salvo em
  `recovered_patch_boundaries/`. A reprojeção é validada contra a janela do Brasil e
  cai na janela metropolitana de Curitiba.
- Bloqueios honestos: CRS ausente → `CURITIBA_PATCH_BOUNDARY_BLOCKED_NO_CRS`; sem
  bounds gravados → `CURITIBA_PATCH_BOUNDARY_NOT_FOUND`; bounds conflitantes →
  `CURITIBA_PATCH_BOUNDARY_AMBIGUOUS`; ponto degenerado → `..._CENTROID_ONLY`.

## 7. Por que binding não é label

Os candidatos de binding patch-evento (86 = 2 eventos × 43 patches) registram
**quais patches poderiam** ser adjudicados/sobrepostos para um evento de Curitiba.
Um binding **não carrega** `gt_patch_flood_observed`, label ou flag de treino.
`can_enter_adjudication` exige apenas patch com embedding + evento com contexto (8
bindings prontos). `can_enter_overlay` só é marcado quando **patch boundary + event
geometry** existem simultaneamente — o que ainda não acontece (0 prontos para
overlay, porque os eventos não têm geometria). Ausência de evidência nunca vira
negativo.

## 8. Por que o treino segue bloqueado

`CURITIBA_GEOMETRY_OR_POINT_EVIDENCE_NOT_READY`: sem footprint de evento validado,
sem label formal, sem negativo formal. `can_train_supervised_model=false`,
`allowed_for_training_count=0`. Os 14 guardrails passam.

## 9. Quais módulos rodam em seguida se geometria/pontos forem adquiridos

A fila `curitiba_next_chain_execution_plan_v2ca.csv` registra a sequência (espelha a
cadeia de Recife):

1. `CURITIBA_BLOCKED_ACQUIRE_GEOMETRY` — passo de gating (aquisição externa).
2. `CURITIBA_QA_GEOMETRY_FROM_POINTS` — se houver pontos (geometria QA, nunca GT) /
   ou validação de footprint oficial.
3. `CURITIBA_V2BP_ADJUDICATION` — adjudicação evento-patch (não é label).
4. `CURITIBA_OVERLAY_RETRY` — interseção patch boundary × event geometry.
5. dry-run protocolar (espelhando `v2bx`).

## Outputs

`local_runs/ground_truth/v2ca/` (git-ignored):

- `curitiba_event_registry_repaired_v2ca.csv`
- `curitiba_event_source_inventory_v2ca.csv`
- `curitiba_event_geometry_inventory_v2ca.csv`
- `curitiba_patch_readiness_inventory_v2ca.csv`
- `curitiba_patch_boundary_recovery_audit_v2ca.csv`
- `curitiba_patch_event_binding_candidates_v2ca.csv`
- `curitiba_evidence_acquisition_queue_v2ca.csv`
- `curitiba_next_chain_execution_plan_v2ca.csv`
- `curitiba_registry_binding_gate_v2ca.json`
- `curitiba_acquisition_guardrails_v2ca.json`
- `curitiba_acquisition_summary_v2ca.json`
- `curitiba_acquisition_report_v2ca.md`
- `recovered_patch_boundaries/` (sidecars GeoJSON leves) e `source_sidecars/`

## Nota de guardrail

Auditoria metodológica autônoma estruturada. Esta etapa não afirma detecção
operacional de inundação, predição validada, acurácia de inundação ou modelo
operacional. Os outputs são locais e leves; nenhum evento e nenhuma geometria foram
inventados.
