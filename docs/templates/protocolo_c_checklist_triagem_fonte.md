# Checklist de triagem de fonte observacional — Protocolo C

**Uso**: Preencher antes de usar qualquer fonte no REV-P. Copiar o template para avaliação local. O repositório público não deve conter checklists preenchidos com dados sensíveis.

**Fonte avaliada**: [NOME_DA_FONTE]
**Data de avaliação**: [DATA]
**Avaliadora**: [INICIAIS_APENAS]

---

## Bloco A — Rastreabilidade e proveniência

| Critério | Status | Observação |
|---|---|---|
| A1. Fonte rastreável (origem identificável, URL ou referência citável) | SIM / NÃO / PARCIAL | |
| A2. Instituição ou provedor identificado | SIM / NÃO / PARCIAL | |
| A3. Referência bibliográfica ou DOI disponível | SIM / NÃO / PARCIAL | |
| A4. Metodologia de produção documentada | SIM / NÃO / PARCIAL | |

**Critério de bloqueio**: se A1 ou A2 for NÃO → `blocked_reason=SOURCE_NOT_TRACEABLE`

---

## Bloco B — Licença e redistribuição

| Critério | Status | Observação |
|---|---|---|
| B1. Licença clara encontrada | SIM / NÃO | |
| B2. Tipo de licença | PUBLIC_REUSE / VIEW_ONLY / REQUEST_REQUIRED / RESTRICTED / UNKNOWN | |
| B3. Redistribuição de dado bruto permitida | SIM / NÃO / RESTRITA / UNKNOWN | |
| B4. Citação obrigatória exigida | SIM / NÃO | |
| B5. Uso em pesquisa acadêmica explicitamente permitido | SIM / NÃO / IMPLÍCITO | |

**Critério de bloqueio**: se B2 = UNKNOWN → `blocked_reason=LICENSE_UNKNOWN`
**Critério de bloqueio**: se B2 = RESTRICTED e sem autorização → `blocked_reason=LICENSE_RESTRICTED`

---

## Bloco C — Evento e temporalidade

| Critério | Status | Observação |
|---|---|---|
| C1. Evento específico confirmado | SIM / NÃO / PARCIAL | |
| C2. Data do evento conhecida | SIM / NÃO / ESTIMADA | |
| C3. Data da fonte compatível com o evento | SIM / NÃO / PARCIAL | |
| C4. Janela temporal entre evento e observação aceitável | SIM / NÃO | |

**Critério de bloqueio**: se C1 = NÃO e gate G1/G3 requerido → `blocked_reason=NO_EVENT_LINK`
**Critério de bloqueio**: se C3 = NÃO → `blocked_reason=TEMPORAL_MISMATCH`

---

## Bloco D — Espacialidade e geometria

| Critério | Status | Observação |
|---|---|---|
| D1. Geometria disponível (shapefile, GeoJSON, bounding box) | SIM / NÃO / PARCIAL | |
| D2. CRS documentado | SIM / NÃO | |
| D3. Resolução ou escala conhecida | SIM / NÃO | |
| D4. Cobre a região do patch REV-P | SIM / NÃO / PARCIAL | |
| D5. Cobertura espacial adequada para o gate requerido | SIM / NÃO / PARCIAL | |

**Critério de bloqueio**: se D1 = NÃO e gate G4 requerido → `blocked_reason=NO_GEOMETRY`
**Critério de bloqueio**: se D4 = NÃO → `blocked_reason=SPATIAL_COVERAGE_INSUFFICIENT`

---

## Bloco E — Força metodológica e independência

| Critério | Status | Observação |
|---|---|---|
| E1. Fonte independente de modelo preditivo | SIM / NÃO / PARCIAL | |
| E2. Fonte independente de produto DINO | SIM / NÃO | |
| E3. Fonte independente de clustering | SIM / NÃO | |
| E4. Anotação humana verificável presente | SIM / NÃO / FUTURA | |
| E5. Incerteza ou acurácia documentada | SIM / NÃO | |
| E6. Tipo de fonte (observacional / produto operacional / modelado / contextual / suporte) | [PREENCHER] | |

**Critério de bloqueio**: se E1 = NÃO e E2 = NÃO e E3 = NÃO → `blocked_reason=DINO_ONLY_INSUFFICIENT`

---

## Bloco F — Gates do Protocolo C

| Gate | Pode fechar? | Condição |
|---|---|---|
| G1_EVENT_CONFIRMATION | SIM / NÃO / PARCIAL | |
| G2_SOURCE_AVAILABILITY | SIM / NÃO / PARCIAL | |
| G3_TEMPORAL_ALIGNMENT | SIM / NÃO / PARCIAL | |
| G4_SPATIAL_ALIGNMENT | SIM / NÃO / PARCIAL | |
| G5_SOURCE_STRENGTH | SIM / NÃO / PARCIAL | |
| G6_UNCERTAINTY_AND_LIMITATIONS | SIM / NÃO / PARCIAL | |
| G7_REVIEW_GATE | SIM / NÃO (nunca automático) | |
| G8_INDEPENDENT_CORROBORATION | SIM / NÃO / PARCIAL | |
| G9_PROMOTION_DECISION | NÃO (requer todos os gates anteriores) | |

**O que esta fonte não pode afirmar (mesmo que seja forte):**
- [ ] Ground truth operacional declarado
- [ ] Detecção de inundação por algoritmo
- [ ] Predição de inundação futura
- [ ] Label supervisionado de treino
- [ ] Suficiência como única fonte

---

## Bloco G — Decisão de intake

| Campo | Valor |
|---|---|
| Decisão | ACCEPT_METADATA_ONLY / ACCEPT_LOCAL_ONLY / REQUEST_MORE_INFORMATION / BLOCK_USE |
| Razão de bloqueio (se aplicável) | [blocked_reason] |
| Usos permitidos | [allowed_use] |
| Usos proibidos | [forbidden_use] |
| Requer revisão supervisora? | SIM / NÃO |
| Próxima ação | [next_action] |

---

## Bloqueios automáticos invioláveis

As seguintes decisões nunca podem ser revertidas por julgamento individual:

- [ ] **DINO sozinho não fecha nenhum gate de evidência observacional**
- [ ] **Produto operacional (GFM, CEMS) sozinho não declara ground truth**
- [ ] **Fonte com licença UNKNOWN não pode avançar para promoção**
- [ ] **Dado bruto com redistribuição proibida não vai para o GitHub**
- [ ] **Nenhuma fonte cria label supervisionado**
- [ ] **Protocolo B permanece bloqueado**
- [ ] **Multimodal permanece em hold**
