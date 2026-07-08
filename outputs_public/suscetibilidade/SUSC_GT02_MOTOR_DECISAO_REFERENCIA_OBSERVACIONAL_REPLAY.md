# SUSC-GT02 - Motor de Decisão de Referência Observacional e Replay de Evidências

## 1. Escopo do marco

Este marco transforma a política do SUSC-GT01 em um **motor de decisão executável**
que lê os artefatos observacionais **já existentes** no repositório e classifica cada
evidência/patch-link em um dos seis estados de referência. É um replay **review-only**:
não busca internet, não baixa dado novo, não roda SAR, não altera o `score_v6`, não
cria `score_v7` e não treina modelo. Nenhum registro é promovido a ground truth
supervisionado.

## 2. Relação com o SUSC-GT01

O GT01 definiu a taxonomia (estados, campos obrigatórios, usos permitidos) e os
bloqueios metodológicos. O GT02 **aplica** essa política: reutiliza os nomes de estado
(`positive_strong`, `positive_provisional`, `unlabeled`, `hard_negative_audited`,
`no_data`, `rejected`) e os guardrails constantes. O manifesto registra se o validador
do GT01 ainda passa (`gt01_valida=true`).

## 3. Entradas descobertas

A descoberta é conservadora: varre os CSVs de `outputs_public/suscetibilidade/` que têm
ao menos uma coluna identificadora (patch/evento/footprint) e uma coluna observacional,
excluindo os próprios artefatos de política (GT01/GT02). Foram descobertos
**29** candidatos; **23** úteis
para replay e **6** bloqueados (sem linhas ou sem colunas
mínimas). O inventário completo está em `susc_gt02_inventario_entradas.csv`.

| input_id | arquivo | tipo estimado | linhas | usável |
| --- | --- | --- | --- | --- |
| IN_001 | susc_17a_reference_evidence_protocol_registry_stub.csv | patch_link | 0 | false |
| IN_002 | susc_17a_reference_evidence_registry.csv | patch_link | 75 | true |
| IN_003 | susc_17c14_accepted_ground_reference_candidates.csv | reference_evidence | 0 | false |
| IN_004 | susc_17c15_accepted_ground_reference_candidates.csv | reference_evidence | 0 | false |
| IN_005 | susc_17c16_accepted_ground_reference_candidates.csv | reference_evidence | 0 | false |
| IN_006 | susc_17c27_observed_event_candidates.csv | event_record | 1 | true |
| IN_007 | susc_17c28_specific_observed_event_candidates.csv | event_record | 4 | true |
| IN_008 | susc_17c29_local_observed_event_candidates.csv | event_record | 7 | true |
| IN_009 | susc_17c2_canary_execution_registry.csv | event_record | 5 | true |
| IN_010 | susc_17c2_candidate_footprint_registry.csv | footprint_registry | 1 | true |
| IN_011 | susc_17c2_candidate_patch_links.csv | footprint_registry | 0 | false |
| IN_012 | susc_17c30_event_record_candidate_registry.csv | event_record | 11 | true |
| IN_013 | susc_17c30_g4_subgate_evaluation.csv | event_record | 12 | true |
| IN_014 | susc_17c31_geometry_candidate_registry.csv | event_record | 2 | true |
| IN_015 | susc_17c32_g4d_patch_buffer_link_evaluation.csv | event_record | 11 | true |
| IN_016 | susc_17c32_geocoded_point_registry.csv | event_record | 11 | true |
| IN_017 | susc_17c33_event_anchored_canary_patch_registry.csv | event_record | 11 | true |
| IN_018 | susc_17c34_canary_geometry_audit.csv | event_record | 11 | true |
| IN_019 | susc_17c34_canary_pre_event_feature_matrix.csv | event_record | 11 | true |
| IN_020 | susc_17c3_official_source_acquisition_targets.csv | event_record | 9 | true |

## 4. Como as colunas foram normalizadas

Cada coluna original é mapeada para um dos quatro objetos conceituais — `event_record`,
`footprint`, `patch_link`, `score_evaluation` — por sinônimos conservadores, **sem
inferir valores**. O mapeamento auditável está em
`susc_gt02_mapeamento_colunas_entrada.csv`. Campos técnicos em inglês (mantidos por
compatibilidade com schemas e GT01) e seu significado público:

- `event_id` / `event_date`: identificador e data do evento observado.
- `geometry_type`: tipo de geometria (forte só se oficial/observada, ex.:
  `official_observed_event_polygon`).
- `patch_link_quality`: qualidade do vínculo footprint→patch (forte só em
  sobreposição alta, ex.: `high_spatial_overlap`).
- `source_authority`: autoridade/fonte rastreável.
- `uncertainty_m`: incerteza posicional em metros.
- `qa_status`: situação do controle de qualidade humano (forte só se aceito).
- `review_only`: uso restrito a revisão (sempre `true`).
- `eligible_for_evaluation` / `eligible_for_calibration`: elegibilidade review-only.
- `eligible_for_training` / `eligible_for_ground_truth` / `score_v7_candidate` /
  `trainable`: **sempre `false`** neste marco.

## 5. Como a política foi aplicada

O motor é fail-closed: só atribui `positive_strong` quando estão presentes,
simultaneamente, evento, data, fonte, **geometria forte**, patch, **patch_link forte**,
incerteza, **QA aceito** e fenômeno compatível. Sem isso, o registro cai para
`positive_provisional` (evento documentado, mas geometria/patch_link/QA insuficientes),
`no_data` (metadados insuficientes para decidir — exige `blocking_reason`), `rejected`
(incompatibilidade explícita — exige motivo) ou `unlabeled` (sem evidência suficiente;
**nunca negativo**). Fontes fracas (alerta, área de risco, notícia, contexto
municipal/de bairro, mapa de suscetibilidade, registro administrativo sem geometria)
**nunca** geram `positive_strong`.

## 6. Distribuição dos estados

| assigned_state | registros | elegíveis avaliação | elegíveis calibração | treino permitido | ground truth permitido |
| --- | --- | --- | --- | --- | --- |
| positive_provisional | 86 | 0 | 86 | 0 | 0 |
| no_data | 471 | 0 | 0 | 0 | 0 |

Leitura pública das colunas: *registros* é a contagem por estado; *elegíveis avaliação/
calibração* são usos review-only; *treino permitido* e *ground truth permitido* são
**sempre zero** por política.

## 7. Principais bloqueios encontrados

Os bloqueios estão em `susc_gt02_bloqueios.csv`. Os mais comuns neste replay são a
ausência de `event_date`, `uncertainty_m` e `qa_status` aceito nos registros de
referência (que impede `positive_strong` e rebaixa para `positive_provisional`), e a
falta de metadados de evento nas tabelas de avaliação de score (que gera `no_data`).
Evidências de Petrópolis com fenômeno de deslizamento ou misto sem separação clara da
mancha de inundação recebem o bloqueio `phenomenon_mismatch_or_mixed_event`, que impede
calibração.

## 8. Exemplos de decisões

- **Referência forte** (`positive_strong`): nenhum registro neste estado no replay atual.
- **Referência provisória** (`positive_provisional`): decisão `DEC_0001`, patch `not_available`, evento `S16ALOCG_00054`. Base: evento documentado e rastreavel, mas sem geometria/patch_link/QA suficientes para referencia forte. Requisitos ausentes: event_date;patch_id;patch_link_quality;uncertainty_m;qa_status.
- **Não rotulado** (`unlabeled`): nenhum registro neste estado no replay atual.
- **Sem dados** (`no_data`): decisão `DEC_0076`, patch `not_available`, evento `REC_2022_05_24_30`. Base: metadados insuficientes para decidir (sem data/fonte/geometria suficientes). Requisitos ausentes: event_date;source_authority;geometry_type;patch_id;patch_link_quality;uncertainty_m;qa_status;phenomenon_type.
- **Rejeitado** (`rejected`): nenhum registro neste estado no replay atual.
- **Negativo auditado** (`hard_negative_audited`): nenhum registro neste estado no replay atual.

## 9. Confirmação explícita dos bloqueios

Este marco **não** treinou modelo, **não** produziu ground truth supervisionado,
**não** criou `score_v7`, **não** alterou o `score_v6`
(`score_v6_changed=false`), **não** rodou SAR e
**não** usou internet. Contagens de controle:
`eligible_for_training=true` → 0;
`eligible_for_ground_truth=true` → 0;
`score_v7_candidate=true` → 0.

O REV-P não prevê enchentes operacionalmente: produz análise estrutural review-only com
evidência observacional auditável.

## 10. Próximo passo recomendado

**GT03 - Pacote de Alvos para Aquisicao de Evidencia Forte**. Como o replay atual não encontrou referência forte
(`positive_strong=0`) — faltam sobretudo data,
geometria oficial forte com incerteza e QA humano aceito —, o caminho recomendado é
montar o pacote de alvos para aquisição de evidência forte antes de qualquer calibração
patch-level.
