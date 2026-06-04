# Protocolo C — v1uf Station-Resolved Official Data Acquisition and Hydrometeorological Evidence Extraction

## Objetivo

Transformar os ZIPs/datasets oficiais ano-específicos resolvidos na v1ue em
evidência hidrometeorológica **por estação e janela temporal**, sem criar ground
truth, sem criar label, sem executar overlay e sem inventar coordenada.

## Pipeline

```
v1ue resolution registry (INMET year ZIPs)
        ↓
Large Official Download Policy → download manifest (allowlist + size limit)
        ↓
Station-Resolved Acquisition → ZIPs em local_only/ (dedup por URL, SHA256)
        ↓
INMET ZIP Selective Extractor → só arquivos da estação-alvo (A610, A301)
        ↓
Official Station Catalog Resolver → coordenadas oficiais (API INMET) ou MISSING
        ↓
Hydromet Window Metrics → precipitação por janela v1ue
        ↓
Station Evidence Integrity Audit → 8 checks por asset
        ↓
Event Hydromet Scorecard → nível hidromet + gate delta + next actions + relatório
```

## Large Official Download Policy

| Regra | Enforcement |
|-------|-------------|
| Download maior só para fonte allowlisted | HARD_BLOCK |
| INMET ZIP anual até 150MB | LIMIT |
| Download streaming | MANDATORY |
| Salvar só em local_only/.../v1uf/ | HARD_BLOCK |
| SHA256 após download | MANDATORY |
| Nunca versionar ZIP bruto | HARD_BLOCK |
| Exceder limite → REJECTED_TOO_LARGE | HARD_BLOCK |
| Fonte não allowlisted → BLOCKED | HARD_BLOCK |
| Licença desconhecida → LICENSE_REVIEW_REQUIRED | MANDATORY |

## Extração Seletiva

O extractor abre o ZIP anual com segurança (bloqueia path traversal), lista os
arquivos internos e extrai **somente** os que casam com a estação-alvo (código
A610/A301, nome, ano). Nunca extrai o ano inteiro indiscriminadamente. Cada asset
recebe hash do ZIP e hash do arquivo extraído.

## Resolução de Coordenada — Regras Estritas

1. Coordenada só de catálogo oficial baixado (com hash) ou arquivo oficial versionado
2. Sem catálogo → `coordinate_status=MISSING`
3. Match no catálogo → `coordinate_status=FROM_OFFICIAL_CATALOG`
4. **NUNCA** geocoding por nome de cidade
5. **NUNCA** centroide de município como coordenada de estação
6. **NUNCA** Google Maps/manual sem fonte oficial
7. **Estação não é geometria de inundação** (`can_anchor_spatial_context=false`)

O catálogo oficial é autoritativo: se a config sugere um nome e o catálogo diz outro,
o catálogo vence (ex.: A610 = "PICO DO COUTO", não o nome sugerido na config).

## Métricas de Janela

Para cada série extraída (CSV INMET, encoding latin-1, separador `;`, decimal `,`,
data `YYYY/MM/DD`, NA `-9999`):

- precipitation_total_mm, precipitation_max_hourly_mm, precipitation_max_daily_mm
- valid_observation_count, missing_observation_count, coverage_ratio
- first/last observation datetime

Janelas: event_core, pre_event_3d, pre_event_7d, post_event_3d.
Cobertura < 0.5 → `INSUFFICIENT_COVERAGE`.

## Níveis Hidromet (nenhum promove ground truth)

- `NO_STATION_DATA`
- `OFFICIAL_YEAR_DATA_AVAILABLE`
- `OFFICIAL_WINDOW_DATA_AVAILABLE`
- `TEMPORAL_HYDROMET_ANCHOR_CONFIRMED`
- `TEMPORAL_ANCHOR_ONLY_NO_GEOMETRY`
- `BLOCKED_INSUFFICIENT_COVERAGE`
- `BLOCKED_STATION_COORDINATES_MISSING`
- `BLOCKED_EVENT_GEOMETRY_MISSING`
- `BLOCKED_PHENOMENON_SEPARATION_REQUIRED`

**Regra central:** mesmo chuva forte no evento NÃO cria ground reference. Apenas
melhora plausibilidade temporal. A estação é sensor pontual, não extensão de inundação.

## Lição Metodológica Concreta

A estação A610 (Pico do Couto, 1777m) registrou apenas ~2.8mm no dia central do
evento de Petrópolis 2022, enquanto o aguaceiro real devastou o centro da cidade.
Isso demonstra empiricamente por que **precipitação de estação ≠ ground truth
espacial**: a estação mais próxima pode não capturar o fenômeno localizado.

## Guardrails Permanentes

- `ground_truth_operational = false`
- `can_create_ground_reference = false`
- `can_create_training_label = false`
- `can_reopen_protocol_b = false`
- `dino_usage = SUPPORT_ONLY`
- `no_overlay_executed = true`
- `no_coordinates_invented = true`
- `supervisor_review_completed = false`
- estação oficial não é geometria de inundação
- precipitação ancora plausibilidade temporal, não patch-level truth
