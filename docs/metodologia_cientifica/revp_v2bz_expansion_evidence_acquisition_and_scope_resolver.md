# v2bz — Aquisição de evidência de expansão e resolvedor de escopo de hazard

Versão: `v2bz`
Modo: auditoria metodológica autônoma estruturada, offline-determinística. Não
cria label, não cria negativo, não libera treino, **não inventa evento nem
geometria**, não mistura hazards.

## 1. Por que o v2bz existe

O v2by provou que a coorte está travada em 1 positivo dry-run porque os eventos
não-Recife não têm geometria/pontos locais
(`GEOMETRY_OR_POINT_EVIDENCE_MISSING_FOR_NON_RECIFE_EVENTS`). O v2bz audita o que
de fato já existe localmente para os eventos LOW/BLOCKED, classifica cada fonte e
geometria, resolve o escopo de hazard e prepara o reparo do registry de Curitiba —
sem forçar treino e sem criar label.

## 2. Por que a expansão está bloqueada por dados

O gargalo não é cálculo. Foram inventariadas **193 fontes locais** ligadas a
Petrópolis/Curitiba, mas são em sua maioria **contexto oficial** (gazetas,
CEMADEN, SGB/RIGEO, GeoSGB, IPPUC) e **catálogos derivados** do próprio projeto —
**nenhuma com geometria vetorial event-specific** (polígono de footprint ou pontos
de ocorrência). Sem geometria/pontos, a cadeia geometria → overlay → dry-run não
roda.

## 3. Por que Petrópolis não entra automaticamente como flood

Os eventos de Petrópolis (`PET_2022_02_15`, `PET_2024_03_21_28`) são de
**movimento de massa (deslizamento)**, não inundação. Misturá-los como se fossem
flood corromperia o target. O v2bz resolve ambos como
`HAZARD_SCOPE_MASS_MOVEMENT_SEPARATE_COHORT`: `can_join_flood_cohort=false`,
`requires_separate_target=true`. Mass_movement nunca é forçado em flood — há
guardrail dedicado (`mass_movement_not_forced_into_flood`).

## 4. Como mass_movement pode virar coorte separada

O caminho permitido é uma **coorte separada** (ou um alvo multi-hazard separado),
com sua própria geometria de evento/pontos, binding patch-evento e definição de
target — mantida apartada do alvo de flood. O v2bz registra esse caminho como
`DEFINE_SEPARATE_HAZARD_SCOPE` na fila, **sem criá-lo**.

## 5. Como Curitiba precisa de registry reparado

Curitiba era `CUR_EVENT_REGISTRY_MISSING` no v2bp, mas existe um registry de
evento candidato prévio (v1uv). O scaffold de reparo referencia esse candidato
real — `CUR_2022_01_15` (urban_flooding, fonte pública oficial, 2 candidatos) —
**sem inventar evento**, com status `CURITIBA_EVENT_REGISTRY_REPAIR_SCAFFOLD_READY`.
Como o hazard detectado é flood, Curitiba é `HAZARD_SCOPE_FLOOD_COMPATIBLE`, mas
ainda **sem geometria/pontos locais**. Se nenhum candidato existisse, o status
seria `CURITIBA_EVENT_REGISTRY_STILL_MISSING` (nada inventado).

## 6. Que fontes foram encontradas ou não

- **Encontradas (local)**: contexto oficial e catálogos derivados (193 fontes).
- **Não encontradas**: geometria vetorial event-specific ou pontos de ocorrência
  para os eventos-alvo.
- **Web**: não executada (`EXTERNAL_WEB_SEARCH_NOT_PERFORMED`); os termos públicos
  de busca foram apenas logados. Nenhum arquivo pesado baixado.

## 7. Por que nenhuma aquisição vira label

Toda linha de geometria carrega `can_support_formal_gt=false`. Polígonos de área
de risco **não** são footprints de evento (`RISK_AREA_POLYGON`, não
`EVENT_FOOTPRINT_POLYGON`). O reparo de registry é um scaffold, não um label.
Ausência nunca é negativo.

## 8. Por que treino segue bloqueado

`COHORT_EXPANSION_DATA_NOT_READY`: 1 positivo dry-run, sem labels formais e sem
nova geometria. `can_train_supervised_model=false`, `can_train_dry_run_model=false`,
`allowed_for_training_count=0`.

## Outputs

`local_runs/ground_truth/v2bz/`:

- `expansion_evidence_acquisition_summary_v2bz.json`
- `target_event_source_inventory_v2bz.csv`
- `target_event_geometry_inventory_v2bz.csv`
- `hazard_scope_resolution_v2bz.csv`
- `petropolis_evidence_readiness_v2bz.csv`
- `curitiba_event_registry_repair_scaffold_v2bz.csv`
- `external_source_search_log_v2bz.csv`
- `expansion_event_processing_queue_v2bz.csv`
- `acquisition_gap_analysis_v2bz.csv`
- `cohort_growth_readiness_gate_v2bz.json`
- `expansion_acquisition_guardrails_v2bz.json`
- `expansion_acquisition_report_v2bz.md`

## Próxima etapa recomendada

Adquirir geometria/pontos event-specific para os eventos-alvo: para Curitiba,
vincular o candidato `CUR_2022_01_15` e buscar footprint/pontos oficiais; para
Petrópolis, definir formalmente a coorte separada de mass_movement antes de
adquirir geometria. Só então repetir a cadeia v2bp→v2bq→v2bt→v2bu→v2bx para os
novos eventos e medir o crescimento real da coorte. Enquanto não houver geometria
event-specific, o treino permanece bloqueado.
