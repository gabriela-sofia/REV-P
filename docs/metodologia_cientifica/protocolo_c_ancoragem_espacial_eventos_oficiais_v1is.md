# Protocolo C — Ancoragem Espacial de Eventos Documentados Oficiais (v1is)

## Objetivo

Converter as unidades documentais oficiais construídas na v1ir em **spatial anchors auditáveis** —
pontos geoespaciais com coordenada explícita e rastreável, vinculados a um evento específico,
uma data e um fenômeno documentado por geólogos CPRM/DIGEAP.

---

## Regra Fundamental: Coordenada Explícita ou Nada

A regra central da v1is é estrita:

> **Apenas unidades com coordenada GPS explicitamente documentada no relatório oficial
> podem virar spatial anchor candidate.**

Consequências diretas:

- Não geocodificar bairro/rua sem coordenada explícita.
- Não usar centroid de bairro como coordenada.
- Não inventar coordenada com base em nome de localidade.
- Unidades sem coordenada **permanecem como DOCUMENTARY_REFERENCE_ONLY** — sem âncora geoespacial.

---

## Diferença: Documentary Reference vs. Spatial Anchor

| Aspecto | Documentary Reference | Spatial Anchor |
|---------|----------------------|----------------|
| **Base** | Relatório com data + localidade | Relatório com coordenada GPS explícita |
| **Localização** | Bairro/rua nomeada | Ponto GPS documentado |
| **Geocodificação** | NÃO — sem coordenada inferida | GPS do relatório, sem inferência |
| **Usabilidade** | Referência documental auditável | Candidato para cruzamento com imagem |
| **Exemplo (v1is)** | ANEXO-III a XI (9 unidades) | ANEXO-II (Moinho Preto, 19/02/2022) |

---

## Fonte das Coordenadas

As únicas coordenadas aceitas na v1is são aquelas **explicitamente documentadas no texto
do relatório CPRM/DIGEAP** como pontos de campo GPS:

```
ANEXO-II — Ponto 1 (Rua Stephan Zweg):
  Coordenadas: -22.484251, -43.211257
```

Esta coordenada foi documentada por geólogo CPRM durante vistoria de campo em 19/02/2022.
É a única aceita como base para spatial anchor na v1is.

---

## Resultado da v1is

### Spatial Anchor Candidates

| Anchor ID | Unidade Origem | Localidade | Data | Coordenada | Fenômeno |
|-----------|---------------|------------|------|-----------|---------|
| ANCHOR_PET2022_CPRM_ANEXOII_19022022 | PET2022_CPRM_ANEXOII_19022022 | Moinho Preto | 19/02/2022 | -22.484251, -43.211257 | MOVEMENT_OF_MASS |

**Total:** 1 spatial anchor candidate

### Documentary Reference Only (sem coordenada)

| Anexo | Data | Localidade | Razão do Bloqueio |
|-------|------|-----------|------------------|
| III | 20/02/2022 | Serra Velha, Pontilhão | NO_EXPLICIT_COORDINATE |
| IV | 22/02/2022 | Valparaíso, Rua Eugênio Barcelos | NO_EXPLICIT_COORDINATE |
| V | 23/02/2022 | Rua Teresa e imediações | NO_EXPLICIT_COORDINATE |
| VI | 24/02/2022 | Moinho Preto (revisita) | NO_EXPLICIT_COORDINATE |
| VII | 24/02/2022 | Mosella | NO_EXPLICIT_COORDINATE |
| VIII | 25–26/02/2022 | Estrada Velha, Vila Felipe | NO_EXPLICIT_COORDINATE |
| IX | 28/02/2022 | Sargento Boening | NO_EXPLICIT_COORDINATE |
| X | 01/03/2022 | Alto da Serra | NO_EXPLICIT_COORDINATE |
| XI | 02/03/2022 | Quitandinha | NO_EXPLICIT_COORDINATE |

**Total:** 9 documentary reference only

### Evidência Insuficiente

| Documento | Motivo |
|-----------|--------|
| Relatorio_Tecnico_Petropolis | Sem data de vistoria estruturada |

---

## Prontidão Sentinel (v1is)

### Janela Temporal para Anchor ANEXO-II

```
Data de vistoria:       19/02/2022
Janela de busca:        04/02/2022 a 06/03/2022 (±15 dias)
Revisita Sentinel-2:    ~5 dias (S2A + S2B combinados)
Resolução espacial:     10 m (bandas 2, 3, 4, 8)
```

### Status de Prontidão

```
sentinel_patch_candidate_available: PATCH_FOUND_COORD_UNVERIFIED
sentinel_readiness_status:          PARTIAL_PATCH_FOUND_COORD_UNVERIFIED
```

**Explicação:** 4 patches Petrópolis foram encontrados no manifest DINO (v1ge),
mas o campo `centroid_lat_lon` está vazio em todos os registros de extensão de patch (v1gt).
Não é possível verificar se algum patch cobre o anchor (-22.484251, -43.211257).

**Próximo passo para resolver o bloqueio:**
Popular `centroid_lat_lon` no registro v1gt para patches PET e verificar proximidade
ao anchor. Não baixar dados sem autorização explícita.

---

## Prontidão Multimodal (v1is)

```
multimodal_readiness_status: PARTIAL_COORD_VERIFICATION_NEEDED
can_extract_embedding:       CONDITIONAL
can_create_label:            NO  (invariante)
can_train_model:             NO  (invariante)
```

**Bloqueio:** `PET_PATCH_CENTROID_EMPTY; COORDINATE_PROXIMITY_TO_ANCHOR_UNVERIFIED`

---

## O Que Spatial Anchor NÃO É

Um spatial anchor candidate documental:

- NÃO é ground truth operacional
- NÃO é label de treino
- NÃO libera supervisão ou rotulagem
- NÃO reabre Protocolo B
- NÃO representa polígono de área atingida

**Distância até label:**

```
Spatial anchor (v1is)
└─ Verificar cobertura de patch Sentinel para o anchor
   └─ Popular centroid_lat_lon em v1gt
      └─ Confirmar patch cobre (-22.484251, -43.211257)
         └─ Análise visual de evidência estrutural (read-only)
            └─ Validação de campo (Protocolo B — não iniciado)
               └─ Ground truth operacional (hipotético)
                  └─ Label de treino (bloqueado)
```

---

## Invariantes

```
can_be_operational_ground_truth  = NO  (invariante absoluto)
can_create_training_label        = NO  (invariante absoluto)
can_train_model                  = NO  (invariante absoluto)
can_reopen_protocol_b            = NO  (invariante absoluto)
```

- [x] Nenhuma coordenada inventada
- [x] Nenhuma geocodificação de bairro/rua sem GPS explícito
- [x] Nenhum centroid de bairro usado
- [x] Nenhum label criado
- [x] Protocolo B não reaberto
- [x] Sem path privado em arquivos públicos
- [x] local_runs/ não versionado

---

**Versão:** v1is — Official Event Unit Spatial Anchor & Sentinel Readiness  
**Spatial anchors criados:** 1 (ANEXO-II, Moinho Preto, 19/02/2022)  
**Documentary reference only:** 9  
**Sentinel readiness:** PARTIAL_PATCH_FOUND_COORD_UNVERIFIED  
**Multimodal readiness:** PARTIAL_COORD_VERIFICATION_NEEDED  
**Markdown público:** Português  
**Sem claims preditivos, sem labels, sem supervisão — rigor máximo.**
