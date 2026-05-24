# Relatório Científico: Eventos Documentados Oficiais (v1ir)

## Resumo Executivo

**RESULTADO PRINCIPAL: 10 ground reference candidates documentais identificados**

v1ir extraiu unidades documentais de evento de 11 relatórios técnicos CPRM/DIGEAP
(Anexos I-XI da avaliação pós-desastre de Petrópolis 2022).

**Status:** AUDITORIA COMPLETA (v1ir)  
**Documentos:** 11 PDFs CPRM/DIGEAP  
**Candidatos:** 10 (9 documentary-only + 1 com coordenada GPS documentada)  
**Não-candidato:** 1 (relatório técnico principal sem data estruturada)  
**Temporal:** EXACT_DATE em 10 unidades (19/02/2022 a 02/03/2022)  
**Espacial melhor:** EXACT_COORDINATE (1 unidade, Ponto Moinho Preto)  
**Fenômenos:** MOVEMENT_OF_MASS (9), RISK_AREA_MIXED (1)

---

## 1. Por Que v1ir Muda a Abordagem

### O Problema Estrutural de Cicatriz_Area_A

Três camadas independentes de auditoria confirmaram o mesmo bloqueio:

```
v1ij: cicatrizes cumulativas sem data específica
v1iq-R2: 444 registros, OBS vazio, sem campo DATA, has_event_or_survey_date=NO
v1ir (fotointerpretação): SIG de 2013-2015, "2022" ausente em todos os XMLs
→ STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK (definitivo)
```

### A Nova Unidade Documental

Abandona-se a busca por vetor que vincule ao evento 2022. A unidade correta é:

> **relatório oficial com data, localidade, fenômeno e fonte rastreável**

Os relatórios CPRM/DIGEAP (Anexos I-XI) atendcem todos os 5 critérios para
ground reference candidate documental.

---

## 2. Documentos Processados

### Estrutura dos Relatórios CPRM/DIGEAP

```
Título: RELATÓRIO TÉCNICO PARA IDENTIFICAÇÃO DE ÁREAS COM RISCO EM CARÁTER EMERGENCIAL
Instituição: DHT / DEGET / DIGEAP (CPRM)
Conteúdo: vistoria de campo pós-evento, localidade, pontos de campo com coordenadas,
          descrição de fenômenos, classificação de risco, contagem de danos
```

### Inventário de Documentos

| Arquivo | Data | Localidade | Fenômeno | Status |
|---------|------|------------|----------|--------|
| ANEXO-II (Moinho Preto) | 19/02/2022 | Moinho Preto | MM | CANDIDATE_WITH_COORD |
| ANEXO-III (Serra Velha) | 20/02/2022 | Serra Velha, Pontilhão | MM | CANDIDATE_DOC |
| ANEXO-IV (Valparaíso) | 22/02/2022 | Valparaíso, Rua Eugênio Barcelos | MM | CANDIDATE_DOC |
| ANEXO-V (Rua Teresa) | 23/02/2022 | Rua Teresa e imediações | MM | CANDIDATE_DOC |
| ANEXO-VI (Moinho Preto revisita) | 24/02/2022 | Moinho Preto | RISK | CANDIDATE_DOC |
| ANEXO-VII (Mosella revisita) | 24/02/2022 | Mosella | MM | CANDIDATE_DOC |
| ANEXO-VIII (Estrada Velha) | 25–26/02/2022 | Estrada Velha, Vila Felipe | MM | CANDIDATE_DOC |
| ANEXO-IX (Sargento Boening) | 28/02/2022 | Sargento Boening | MM | CANDIDATE_DOC |
| ANEXO-X (Alto da Serra) | 01/03/2022 | Alto da Serra | MM | CANDIDATE_DOC |
| ANEXO-XI (Quitandinha) | 02/03/2022 | Quitandinha | MM | CANDIDATE_DOC |
| Relatorio_Tecnico_Petropolis | — | Mosella | MM | INSUFFICIENT |

*MM = MOVEMENT_OF_MASS*

---

## 3. Evidências Extraídas

### 3.1 Evidência com Coordenada GPS

**Unidade mais forte encontrada:**

```
Unit ID:           PET2022_CPRM_ANEXOII_19022022
Instituição:       CPRM / DIGEAP
Data de vistoria:  19/02/2022
Localidade:        Moinho Preto (Petrópolis, RJ)
Coordenadas:       -22.484251, -43.211257
Ponto:             Rua Stephan Zweg
Fenômeno:          Solapamento de margem; risco de enchente/enxurrada
Spatial precision: EXACT_COORDINATE
Temporal prec.:    EXACT_DATE
Candidato:         CANDIDATE_WITH_DOCUMENTED_COORDINATE
```

**Descrição documentada no relatório:**
"Solapamento da margem direita do afluente do Rio Piabanha derrubou parte do acostamento,
um poste de iluminação e mobilizou blocos de construção. Residências próximas, com uma casa
de alvenaria e madeira parcialmente danificadas por enxurrada."

### 3.2 Coordenadas Documentadas (Total)

3 coordenadas GPS extraídas de textos de relatório (todas no ANEXO-II).

### 3.3 Localidades Documentadas

10 localidades distintas (bairro ou rua):

- Moinho Preto (2 vistorias: 19/02 e 24/02)
- Serra Velha, Localidade Pontilhão
- Valparaíso, Rua Eugênio Barcelos e imediações
- Rua Teresa e imediações
- Mosella (2 vistorias: 24/02 revisita + relatório principal)
- Estrada Velha e Vila Felipe
- Sargento Boening
- Alto da Serra (via Servidão Alápio Gomes da Costa)
- Quitandinha

---

## 4. Gates de Avaliação

Todos os 10 candidatos passam em todos os gates críticos:

| Gate | Resultado |
|------|-----------|
| `source_official` | PASS — CPRM/DIGEAP (autoridade federal) |
| `event_date_or_window` | PASS — datas de vistoria pós-evento documentadas |
| `phenomenon_explicit` | PASS — deslizamento/risco descritos nos relatórios |
| `location_explicit` | PASS — bairro/rua identificada |
| `spatial_precision_sufficient` | PASS — NEIGHBORHOOD ou EXACT_COORDINATE |
| `document_excerpt_available` | PASS — texto verificável no PDF |
| `coordinate_documented` | PASS (1 unit) / ABSENT (9 units) |

---

## 5. Precisão Obtida

### Temporal

```
Melhor:   EXACT_DATE (10/11 unidades)
Janela:   15 dias (19/02/2022 a 02/03/2022)
Evento:   2022-02-15 (4 dias antes da primeira vistoria)
```

### Espacial

```
EXACT_COORDINATE: 1 unidade (Rua Stephan Zweg, Moinho Preto)
NEIGHBORHOOD:     9 unidades (bairro nomeado no relatório)
MUNICIPAL_ONLY:   0
UNKNOWN:          0
```

---

## 6. O Que São e O Que Não São

### O Que São

- Evidências documentais rastreáveis de eventos pós-desastre
- Relatórios de campo oficiais CPRM/DIGEAP
- Candidatos documentais para cruzamento com imagens da data
- Base para verificação de evidência estrutural (não label)

### O Que NÃO São

- Vetores observados de cicatrizes
- Labels de treino
- Ground truth operacional
- Liberação de Protocolo B
- Confirmação de polígono de área atingida

---

## 7. Base para Próxima Etapa

### Há candidatos documentais suficientes?

**SIM.** 10 candidatos com data exata, localidade, fenômeno e fonte oficial.
1 com coordenada GPS explicitamente documentada.

### Próximo Passo Se Autorizado

```
CRUZAMENTO CONTROLADO:
- Coordenadas/localidades documentadas ← relatórios CPRM
- Patches Sentinel das datas de vistoria ← pipeline existente
- Verificação de evidência estrutural ← análise visual/embedding
- NÃO cria label automático
- NÃO abre Protocolo B
```

### O Que Ainda Falta

Para ground truth operacional (hipotético futuro):
- Validação de campo presencial (Protocolo B — não iniciado)
- Correlação com imagem verificada
- Polígono de área atingida com precisão métrica

---

## 8. Invariantes

```
can_be_operational_ground_truth  = NO  (invariante absoluto)
can_create_training_label        = NO  (invariante absoluto)
can_train_model                  = NO  (invariante absoluto)
can_reopen_protocol_b            = NO  (invariante absoluto)
can_be_ground_reference_candidate = YES_DOCUMENTARY (10 unidades)
```

- [x] Nenhum label criado
- [x] Nenhum modelo treinado
- [x] Protocolo B não reaberto
- [x] Sem e-mail, solicitação ou vistoria
- [x] Sem inventar coordenada
- [x] Sem inferir data
- [x] Sem usar data de sistema
- [x] Relatório NÃO virou vetor observado
- [x] Sem path privado em arquivos públicos
- [x] local_runs/ não versionado

---

## 9. Comparação com Abordagem Anterior

| Aspecto | Cicatriz_Area_A (v1iq) | Relatórios CPRM (v1ir) |
|---------|------------------------|------------------------|
| **Unidade** | Polígono shapefile | Documento textual |
| **Data** | SIG 2013-2015 | 19/02/2022 a 02/03/2022 |
| **Vínculo com 2022** | NÃO | SIM (vistoria pós-evento) |
| **Localidade** | Petrópolis genérico | Bairro/rua específica |
| **Coordenada** | Não disponível | 1 ponto GPS documentado |
| **Decisão** | STRONG_COMPOSITE_BUT_WEAK_TEMPORAL | CANDIDATE_DOCUMENTARY |

---

**Data de Execução:** 2026-05-24  
**Etapa:** v1ir — Official Documented Event Unit Ground Reference Builder  
**Documentos processados:** 11 PDFs CPRM/DIGEAP  
**Ground reference candidates:** 10 (9 documentary + 1 com GPS)  
**Temporal:** EXACT_DATE (datas de vistoria pós-evento)  
**Spatial melhor:** EXACT_COORDINATE (Moinho Preto, 19/02/2022)  
**Fenômenos:** MOVEMENT_OF_MASS, RISK_AREA_MIXED, FLOODING co-ocorrente  
**Próxima etapa:** Cruzamento controlado com Sentinel (não label, não Protocolo B)  
**Markdown público:** Português  
**Sem claims preditivos, sem labels, sem supervisão — rigor máximo.**
