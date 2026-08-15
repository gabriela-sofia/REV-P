# SUSC-GT05 - Pacote de Aquisição de Geometria Oficial

## 1. Escopo do marco

Este marco classifica, prioriza e planeja a **aquisição futura de geometria oficial**
(ou geometria observacional forte) para os alvos já datados pelo GT04. É um pacote
**offline e review-only** (uso restrito a revisão): não busca internet, não baixa dado
novo, não consulta API, não geocodifica, não roda SAR nem GEE, não baixa raster, não
cria footprint, **não cria geometria real**, não altera o `score_v6`, não cria
`score_v7`, não treina modelo e não promove nada a ground truth nem a `positive_strong`.

## 2. Relação com GT01, GT02, GT03 e GT04

O GT01 definiu a política; o GT02 aplicou-a; o GT03 montou a fila de alvos; o GT04
resolveu as datas e janelas. O GT05 usa as datas do GT04 e recupera do GT03/GT02 os
campos geométricos (tipo de geometria, patch, vínculo, fonte) para atacar o próximo
gargalo: a **geometria oficial**.

## 3. Por que geometria vem depois da data e antes do QA forte

Sem data não há janela; com data, o passo seguinte é ter uma geometria confiável (ponto,
polígono ou footprint) para vincular ao patch. O QA humano forte só faz sentido sobre
uma geometria já candidata; por isso a geometria vem antes do QA definitivo.

## 4. Por que o GT05 não busca internet, não geocodifica e não cria footprint

Este é um marco de **planejamento determinístico**: monta a fila e o plano a partir do
que já existe no repositório. A aquisição real (polígono oficial, ponto oficial,
geocodificação, footprint SAR) fica para marcos seguintes, com execução controlada.

## 5. Entradas usadas

Datas e janelas do GT04 e campos geométricos do GT03 (recuperados por `target_id`), lidos
dos outputs versionados sem regeração. Total de alvos: **557**.

## 6. Classes de estado geométrico

`geometria_forte_existente` (geometria oficial forte + vínculo mínimo),
`geometria_parcial_resolvivel` (ponto/endereço/vetor bruto convertível a oficial),
`geometria_contextual_insuficiente` (só cidade/bairro/área de risco),
`geometria_ausente` (sem sinal útil), `requer_footprint_tecnico_futuro` (datado sem
geometria, candidato a SAR) e `geometria_rejeitada` (incompatível ou fenômeno misto).

## 7. Tipos de geometria forte e tipos proibidos

Fortes: `official_observed_event_polygon/point`, `technical_remote_sensing_flood_footprint`,
`official_polygon/point`, `technical_remote_sensing_polygon` e `official_bbox` com
precisão. **Nunca** fortes: `city_level_record`, `municipal_context`,
`street_neighborhood_context`, `risk_area_not_event`, `alert_only`, `news_only`,
`documentary_context_only`, `susceptibility_map_only`,
`administrative_record_without_geometry` e centroides (`centroid_guess`,
`neighborhood_centroid`, `municipality_centroid`). Observação: nos outputs versionados do
GT02/GT03 a geometria aparece com o tipo bruto (Polygon, bbox, point), sem a qualificação
oficial; por isso, fail-closed, ela entra como parcial resolvível, não como forte.

## 8. Critérios de priorização geométrica

O `geometry_priority_score` (0 a 100) pontua positivamente data operacional, precisão
`exact_day`/`inferred_from_event_id`, fonte forte, geometria/footprint/patch existentes,
fenômeno de inundação, cidade/região no escopo e poucos requisitos faltantes; pontua
negativamente geometria ausente/contextual, tipos proibidos, Petrópolis misto, ausência
de patch e precisão temporal `unknown`. A priorização **não** promove estado.

## 9-13. Distribuição dos estados geométricos

- `geometria_forte_existente`: **0**.
- `geometria_parcial_resolvivel`: **263**.
- `requer_footprint_tecnico_futuro`: **277**.
- `geometria_contextual_insuficiente`: **0**.
- `geometria_ausente`: **9**.
- `geometria_rejeitada`: **8**.

| geometry_acquisition_status | alvos | score médio | podem ir a QA | podem ir a SAR |
| --- | --- | --- | --- | --- |
| geometria_parcial_resolvivel | 263 | 15.51 | 263 | 15 |
| geometria_ausente | 9 | 0.0 | 0 | 0 |
| requer_footprint_tecnico_futuro | 277 | 5.7 | 0 | 277 |
| geometria_rejeitada | 8 | 0.0 | 0 | 0 |

## 14. Principais bloqueios geométricos

Registrados em `susc_gt05_bloqueios_geometricos.csv`: geometria oficial ausente nos
alvos datados (dependem de footprint SAR futuro), tipos contextuais proibidos (só
contexto) e Petrópolis com fenômeno misto (bloqueio `phenomenon_mismatch_or_mixed_event`).

## 15. Petrópolis e eventos mistos

Alvos de Petrópolis (9 no total) com deslizamento ou fenômeno misto são
classificados como `geometria_rejeitada`, exigem a trilha `separar_fenomeno` e **nunca**
recebem prioridade geométrica alta sem essa separação.

## 16. Exemplos de alvos

- **Geometria forte existente** (`geometria_forte_existente`): nenhum alvo neste estado.
- **Geometria parcial resolvível** (`geometria_parcial_resolvivel`): alvo `GEO_0001`, event_id `S16ALOCG_00070`, geometry_type `not_available`, classe `bloqueado_por_geometria_insuficiente`, trilhas `resolver_endereco_oficial;consultar_base_setorial_oficial_futura;estimar_incerteza_geometrica`.
- **Geometria contextual insuficiente** (`geometria_contextual_insuficiente`): nenhum alvo neste estado.
- **Geometria ausente** (`geometria_ausente`): alvo `GEO_0174`, event_id `CUR_HISTORICAL`, geometry_type `not_available`, classe `bloqueado_por_geometria_insuficiente`, trilhas `buscar_ponto_oficial;resolver_endereco_oficial;consultar_base_setorial_oficial_futura`.
- **Requer footprint técnico futuro** (`requer_footprint_tecnico_futuro`): alvo `GEO_0076`, event_id `REC_2022_05_24_30`, geometry_type `not_available`, classe `bloqueado_por_geometria_insuficiente`, trilhas `preparar_footprint_sar_futuro`.
- **Geometria rejeitada** (`geometria_rejeitada`): alvo `GEO_0184`, event_id `PET_2022_02_15`, geometry_type `not_available`, classe `bloqueado_por_geometria_insuficiente`, trilhas `preparar_footprint_sar_futuro;separar_fenomeno`.

## 17. Confirmação explícita dos bloqueios

Este marco **não** usou internet, **não** geocodificou, **não** executou SAR nem GEE,
**não** baixou raster, **não** criou footprint, **não** criou geometria real
(`geometria_criada=0`), **não** treinou modelo, **não**
produziu ground truth, **não** criou `score_v7`, **não** alterou o `score_v6`
(`score_v6_changed=false`) e **não** promoveu
nenhum alvo a `positive_strong`
(`positive_strong_promovidos=0`). Contagens de
controle: `eligible_for_training=true` → 0;
`eligible_for_ground_truth=true` → 0;
`score_v7_candidate=true` → 0.

O REV-P não prevê enchentes operacionalmente: produz análise estrutural review-only com
evidência observacional auditável.

## 18. Próximo passo recomendado

**GT06 - Preparacao de Canarios SAR sem Execucao**. Com **540** alvos liberados
para QA ou footprint SAR futuro, o próximo gargalo é obter/validar geometria: os alvos
datados sem geometria oficial devem seguir para a preparação de canários SAR (sem
execução), enquanto os que já têm geometria parcial devem seguir para QA humano.
