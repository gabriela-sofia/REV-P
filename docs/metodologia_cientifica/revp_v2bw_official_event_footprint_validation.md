# v2bw — Validação de footprint oficial do evento e reconciliação de fontes

Versão: `v2bw`
Modo: assembly + reconciliação + gate, offline-determinístico. Não busca licença,
não depende de internet, não inventa geometria. Não cria label, não cria negativo
formal, não libera treino.

## 1. Por que o v2bw existe

A cadeia v2bp→v2bv esbarrou repetidamente no mesmo gargalo de ground truth: o
evento `REC_2022_05_24_30` (cheias do Recife, mai/2022) tem evidência forte de
contexto e de pontos, mas **não tem um footprint poligonal oficial revisado**.
Sem esse footprint, não há positivo formal, não há negativo formal e não há
treino.

O v2bw não tenta forçar o avanço. Ele faz o oposto: inventaria e reconcilia
formalmente todas as fontes locais já descobertas e **prova que o gargalo atual é
a ausência de footprint oficial poligonal revisado, não a falta de cálculo**.
Por isso é uma etapa de assembly + reconciliação + gate, e não de download
externo.

## 2. Que nenhum footprint oficial poligonal foi encontrado

Varrendo `datasets/`, `local_runs/`, `outputs_public/`, `manifests/`, `docs/` e
`configs/`, a busca encontra 69 fontes ligadas ao evento (31 oficiais), mas
**0 fontes com geometria de footprint oficial poligonal**. A decisão formal é:

```
OFFICIAL_FOOTPRINT_NOT_FOUND
```

com o detalhe `OFFICIAL_FOOTPRINT_NOT_FOUND_BUT_POINT_DERIVED_QA_GEOMETRY_AVAILABLE`.

A etapa é offline-determinística: nenhuma busca web é feita. A ausência de web é
registrada explicitamente como `EXTERNAL_WEB_SEARCH_NOT_PERFORMED` no inventário
de fontes e em `external_web_search` no summary — o pipeline segue sem depender
de rede.

## 3. Que existem fontes oficiais contextuais e pontuais

O evento tem evidência oficial real, só não em forma de polígono validado:

- **Charter 758 (ativação/API)** → contexto oficial (`OFFICIAL_CONTEXT_SOURCE`),
  sem footprint validado.
- **400 pontos da Defesa Civil** → evidência pontual oficial
  (`POINT_EVIDENCE_SOURCE`), forte para QA/contexto, mas não é footprint
  poligonal.
- **`recife_defesa_civil_risk_areas_geojson`** (se presente) → fonte contextual
  de áreas de risco, **não** o footprint do evento específico.

## 4. Que o charter polygon foi rejeitado/rebaixado

O polígono charter758 digitalizado é uma geometria **media-derived** não
revisada. Ele conflita com os pontos independentes da Defesa Civil e foi
rejeitado/rebaixado nas etapas anteriores. O v2bw mantém esse status:

```
CHARTER_POLYGON_REJECTED_FOR_EVENT_QA
```

Ele aparece no inventário de geometria como `MEDIA_DERIVED_GEOMETRY` com
`recommended_use=DO_NOT_USE_AS_EVENT_GEOMETRY`, `can_use_for_formal_gt_protocol=false`
e `can_create_label=false`. Não é repromovido.

## 5. Que as geometrias QA-derived continuam QA-only

As 5 geometrias alternativas do v2bt (convex hull, buffer unions, cluster
envelopes) são reconstruções **derivadas dos pontos** da Defesa Civil. São úteis
para QA de overlay, mas **não** são footprint oficial. No inventário aparecem
como `QA_DERIVED_GEOMETRY` com `can_use_for_formal_gt_protocol=false` e
`can_create_label=false`. Não são promovidas a ground truth.

## 6. Que REC_00276 segue como forte candidato QA, mas não label

REC_00276 (o patch QA-robusto consolidado no v2bv) alinha com a geometria QA-only,
mas não há footprint oficial contra o qual validá-lo. Ele permanece como
candidato QA forte mantido:

```
official_footprint_status   = OFFICIAL_FOOTPRINT_NOT_FOUND
alignment_decision          = ALIGNED_WITH_QA_ONLY_GEOMETRY_NO_OFFICIAL_FOOTPRINT
formal_positive_protocol_ready = false
gt_patch_flood_observed     = NA
allowed_for_training        = false
blocked_reason              = NO_OFFICIAL_FOOTPRINT_VALIDATED
```

Continua sendo um dossiê, não um label.

## 7. Que os negativos comparáveis seguem scaffold, mas não negativos formais

Os 14 negativos comparáveis QA-only do v2bv são reavaliados contra o footprint
oficial (ausente). Sem footprint oficial e sem protocolo de negativos aprovado,
permanecem scaffold:

```
official_footprint_status      = OFFICIAL_FOOTPRINT_NOT_FOUND
formal_negative_protocol_ready = false
formal_negative_label_created  = false
gt_patch_flood_observed        = NA
allowed_for_training           = false
blocked_reason                 = NO_OFFICIAL_FOOTPRINT_AND_NO_NEGATIVE_PROTOCOL
```

Não-interseção não é negativo; ausência não é negativo.

## 8. Que o treino segue bloqueado

Nenhum footprint oficial, nenhum label formal, nenhum negativo formal, nenhum
target de treino:

- `official_footprint_validated_for_gt_protocol=false`
- `formal_positive_protocol_ready=false`
- `formal_negative_protocol_ready=false`
- `labels_created=false`
- `allowed_for_training_count=0`
- `can_train_supervised_model=false`
- `supervised_training_enabled=false`

## Outputs

`local_runs/ground_truth/v2bw/`:

- `official_event_footprint_source_inventory_v2bw.csv`
- `official_event_footprint_geometry_inventory_v2bw.csv`
- `event_source_reconciliation_matrix_v2bw.csv`
- `official_footprint_candidate_scoring_v2bw.csv`
- `charter_vs_official_vs_qa_decision_v2bw.csv`
- `rec00276_formal_footprint_alignment_v2bw.csv`
- `comparable_negative_footprint_alignment_v2bw.csv`
- `formal_footprint_validation_gate_v2bw.json`
- `gt_protocol_readiness_after_footprint_v2bw.json`
- `footprint_validation_guardrails_v2bw.json`
- `footprint_validation_summary_v2bw.json`
- `footprint_validation_report_v2bw.md`

## Próxima etapa recomendada

Aquisição de um footprint oficial poligonal revisado do evento (Defesa Civil /
APAC / CEMADEN / CPRM / Copernicus EMS) **ou** desenho de um protocolo formal que
use explicitamente a geometria QA-derived dos pontos como referência declarada —
com a limitação documentada. Enquanto isso, o gargalo permanece a ausência de
footprint oficial, não a falta de cálculo.
