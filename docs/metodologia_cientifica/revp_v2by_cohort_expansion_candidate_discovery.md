# v2by — Planejador de expansão de coorte e descoberta de candidatos dry-run

Versão: `v2by`
Modo: auditoria metodológica autônoma estruturada, offline-determinística. Não
cria label, não cria negativo formal, não libera treino, não inventa geometria.

## 1. Por que o v2by existe

O v2bx provou que o projeto tem um protocolo dry-run **correto**, mas com **1 só
positivo** (REC_00276). O gargalo deixou de ser metodológico ("como montar o
protocolo") e passou a ser de **massa**:
`TOO_FEW_POSITIVES_FOR_ANY_TRAINING_OR_EVALUATION`.

O v2by muda a pergunta de novo: *onde mais a cadeia
`evento/patch → evidência → geometria/pontos → QA geometry → overlay sensitivity → dry-run protocol`
pode ser repetida?* Ele escaneia todo o universo de eventos/patches já presente no
repositório, classifica readiness, prioriza e projeta o yield potencial — sem
treinar e sem criar label.

## 2. Por que 1 positivo dry-run não basta

Um único positivo não dá split treino/teste, não dá balanceamento de classe e não
permite estimar generalização. Forçar REC_00276 a virar label fabricaria ground
truth. A única jogada segura é **crescer a coorte de candidatos primeiro**.

## 3. Como a expansão de coorte é priorizada

A prioridade combina sinais reais por evento:

- **HIGH** = evidência pontual/poligonal **+** patches com boundary **+**
  embeddings → pronto para repetir a cadeia (`EXPANSION_EVENT_READY_FOR_QA_GEOMETRY`).
- **MEDIUM** = evidência pontual/poligonal mas **sem boundary** ainda
  (`HAS_POINT_EVIDENCE` / `HAS_POLYGON_GEOMETRY`).
- **LOW** = só contexto oficial, sem geometria/pontos locais (`CONTEXT_ONLY`).
- **BLOCKED** = registry ausente/rejeitado ou sem evidência operacional
  (`BLOCKED_SOURCE_INSUFFICIENT` / `BLOCKED_NO_PATCH_BINDING` /
  `BLOCKED_NO_GEOMETRY_OR_POINTS`).

Estado real: **4 eventos / 114 patches**. REC_2022_05_24_30 já processado (urban
flood, com pontos + polígono + QA + 36 boundaries). PET_2022_02_15 e
PET_2024_03_21_28 ficam **LOW** (mass_movement, contexto mas sem geometria/pontos
locais). Curitiba fica **BLOCKED** (registry ausente). 0 HIGH, 0 MEDIUM.

## 4. Por que eventos contexto-only continuam bloqueados

Eventos com contexto oficial mas **sem geometria poligonal nem pontos locais** (e
sem pontos a partir dos quais derivar QA geometry) não conseguem entrar na cadeia
geometria → overlay → dry-run. Ficam LOW, com ação clara (adquirir geometria
pontual/poligonal), **nunca promovidos** e **nunca usados como negativos**. O
sinal de ponto/polígono vem de um scan real de geojson, nunca assumido — por isso
Petrópolis (sem geojson local) fica honestamente bloqueado mesmo tendo o flag
coarse `EVENT_GEOMETRY_PRESENT` do v2bp.

## 5. Como evidência pontual/poligonal muda a prioridade

Um evento com evidência pontual ou polígono real **mais** boundaries recuperáveis
e embeddings sobe para HIGH (pronto para repetir a cadeia). Com evidência mas sem
boundary ainda, fica MEDIUM. A presença de ponto/polígono é tirada do scan real de
geojson em `datasets/`, `manifests/`, `outputs_public/`.

## 6. Como boundary, DINO e GIS entram na readiness

A readiness de patch combina: boundary recuperada (v2br), embedding DINO (feature
table v2bn) e features GIS. Só patches com boundary + embedding + binding +
evidência ficam `EXPANSION_PATCH_READY_FOR_OVERLAY`. No estado real, esses sinais
(boundaries) só existem para Recife — que já está processado.

## 7. Por que projeção de yield não é label

A projeção de yield estima, de forma conservadora, quantos positivos/negativos
dry-run adicionais uma expansão *poderia* produzir. Onde não há base, usa
`NOT_ESTIMABLE`/`UNKNOWN`. Nunca inventa números, nunca vira label nem target de
treino. No estado real, REC = `NO_CHANGE` (já contado) e os demais =
`NOT_ESTIMABLE`/`BLOCKED`.

## 8. Por que treino segue bloqueado

Sem labels formais, com 1 positivo dry-run e crescimento projetado ainda não
estimável: `can_train_supervised_model=false`, `can_train_dry_run_model=false`,
`allowed_for_training_count=0`. O gate de expansão só reabre quando a coorte
alcançar pelo menos **10 positivos** — uma heurística conservadora de planejamento
para split seguro contra leakage, **não** um limiar estatístico validado.

## Nota metodológica

Os eventos candidatos de Petrópolis são de **movimento de massa (deslizamento)**,
não inundação. Se vale dobrar um tipo de hazard diferente numa coorte orientada a
inundação é uma decisão futura de escopo. Como estão bloqueados em geometria/pontos
locais de qualquer forma, nenhuma decisão é forçada agora — fica registrada para
quando a geometria for adquirida.

## Outputs

`local_runs/ground_truth/v2by/`:

- `cohort_expansion_summary_v2by.json`
- `event_expansion_candidate_inventory_v2by.csv`
- `patch_expansion_candidate_inventory_v2by.csv`
- `evidence_source_expansion_audit_v2by.csv`
- `geometry_readiness_expansion_audit_v2by.csv`
- `dino_gis_feature_readiness_expansion_audit_v2by.csv`
- `dry_run_yield_projection_v2by.csv`
- `next_event_processing_queue_v2by.csv`
- `cohort_expansion_antileakage_plan_v2by.csv`
- `cohort_expansion_training_gate_v2by.json`
- `cohort_expansion_guardrails_v2by.json`
- `cohort_expansion_report_v2by.md`

## Próxima etapa recomendada

Atacar o gargalo de massa pela aquisição de geometria/pontos para os eventos LOW
(Petrópolis) — footprint oficial ou evidência pontual — e reparar/adquirir o
registry de Curitiba. Com geometria/pontos disponíveis, repetir a cadeia
v2bp→v2bq→v2bt→v2bu→v2bx para esses eventos e medir o crescimento real da coorte.
Enquanto a coorte não alcançar a massa mínima, o treino permanece bloqueado.
