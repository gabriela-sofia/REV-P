# Protocolo C — v1ue Event-Specific Evidence Deepening and Station/Asset Binding

## Objetivo

Transformar evidência genérica de portal (v1ud) em busca dirigida por evento, data,
cidade, estação, produto e ativo observacional específico — sem criar ground truth,
label, geometria, nem inventar coordenada.

## Pipeline

```
event_candidate_registry
        ↓
Temporal Window Builder → 5 janelas/evento
        ↓
Station Candidate Builder → estações oficiais candidatas (coord MISSING até resolução)
        ↓
Official Dataset Resolver → datasets ano/cidade-específicos (não só homepage)
        ↓
Event Deepening (orchestrator) → vínculo source→event→window→station
        ↓
Observation Series Audit → análise de assets (CSV/ZIP/PDF/HTML/geodata)
        ↓
Event Evidence Scorecard → 8 dimensões + classificação (sem promoção)
        ↓
Next Actions + Relatório
```

## Janelas Temporais

| Tipo | Definição | Apoia gate temporal? |
|------|-----------|---------------------|
| event_core_window | start_date a end_date | sim |
| pre_event_window_3d | 3 dias antes | sim |
| pre_event_window_7d | 7 dias antes | sim |
| post_event_window_3d | 3 dias depois | sim |
| sentinel_link_window | -7/+7 (cruzamento futuro) | não (sem overlay) |

## Estações: Regras de Coordenada

1. Coordenada só registrada se vier da fonte oficial
2. Nunca inventar coordenada → `coordinate_status=MISSING`
3. Estação ancora tempo/plausibilidade hidrometeorológica
4. **Estação NÃO é geometria de inundação** (`can_anchor_spatial_evidence=false`)
5. Centroide de cidade (IBGE) usado só para distância, nunca como evento

## Dimensões do Scorecard

| Dimensão | O que mede |
|----------|-----------|
| temporal_evidence_score | Âncora temporal (janelas + estações + série ano-específica) |
| hydrometeorological_score | Sinal hidrometeorológico observado |
| phenomenon_typing_score | Capacidade de tipar/separar fenômeno |
| locality_score | Localidade (só de assets substantivos, não portal) |
| geometry_score | Geometria observacional disponível |
| source_authority_score | Autoridade oficial da fonte |
| independence_score | Independência (fontes distintas) |
| review_readiness_score | Prontidão para revisão humana |

## Classificações (nenhuma promove ground truth)

- `CONTEXT_ONLY`
- `TEMPORAL_ANCHOR_ONLY`
- `OBSERVATIONAL_CANDIDATE_WEAK`
- `OBSERVATIONAL_CANDIDATE_MODERATE`
- `READY_FOR_HUMAN_REVIEW`
- `BLOCKED_FORMAL_REQUEST_REQUIRED`
- `BLOCKED_PHENOMENON_SEPARATION_REQUIRED`
- `BLOCKED_GEOMETRY_MISSING`

## Princípios Metodológicos

1. **HTML de portal não fecha gate de evento** — termos de navegação (rua/avenida)
   num homepage não contam como localidade real; só assets substantivos pontuam.
2. **Estação meteorológica/hidrológica ancora tempo, não geometria de inundação.**
3. **Suscetibilidade SGB/CPRM é contexto, não ocorrência observada.**
4. **Quickview é pista, não produto validado.**
5. **Score alto define apenas próxima ação, nunca cria label ou ground truth.**

## Guardrails Permanentes

- `ground_truth_operational = false`
- `can_create_training_label = false`
- `can_reopen_protocol_b = false`
- `dino_usage = SUPPORT_ONLY`
- `no_overlay_executed = true`
- `no_coordinates_invented = true`
- `can_create_ground_reference = false` (todos os eventos)
- `supervisor_review_completed = false` (todos os eventos)
