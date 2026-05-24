# Protocolo C — Eventos Documentados Oficiais (v1ir)

## Contexto: Por que saímos de Cicatriz_Area_A

### O Problema de Cicatriz_Area_A.shp

`Cicatriz_Area_A.shp` foi auditada em três camadas independentes (v1ij → v1iq-R2 → v1ir):

| Camada | Bloqueio Confirmado |
|--------|---------------------|
| Registry v1ij | Cicatrizes cumulativas sem data específica |
| DBF atributos v1iq-R2 | 444 registros × 8 campos — sem DATA, OBS vazio |
| XML sidecars v1ir | Fotointerpretação de 2013; "2022" ausente |

**Decisão:** Usar `Cicatriz_Area_A.shp` apenas como evidência contextual/geomorfológica.
Não insistir em promovê-la como ground reference para o evento 2022.

### A Nova Abordagem

Abandonar a busca por vetor que "prove" 2022. Construir ground reference candidates
a partir da unidade documental correta:

> **evento / localidade / fenômeno documentado em relatório oficial,
> com data e fonte rastreável.**

---

## Diferença: Vetor Histórico vs. Evento Documentado

| Aspecto | Cicatriz_Area_A.shp | Relatório CPRM (v1ir) |
|---------|--------------------|-----------------------|
| **Tipo** | Vetor geoespacial | Documento textual |
| **Data** | SIG histórico 2013-2015 | Data exata de vistoria (pós-2022-02-15) |
| **Localidade** | Petrópolis genérico | Bairro/rua específica |
| **Fenômeno** | TIPO=Deslizamento (cumulativo) | Ocorrência específica documentada |
| **Coordenada** | Polígono de cicatriz histórica | Ponto de campo GPS (em alguns docs) |
| **Vínculo com 2022** | NÃO verificável | SIM — data de vistoria pós-evento |
| **Usabilidade como referência** | Evidência contextual | Ground reference candidate documental |

---

## Por Que Relatório Oficial Pode Formar Ground Reference Candidate

Um relatório oficial forma ground reference candidate documental quando:

1. **Fonte oficial/rastreável** — CPRM/SGB/DRM (instituição pública)
2. **Data explícita** — data da vistoria de campo documentada
3. **Fenômeno explícito** — deslizamento, escorregamento, enxurrada, etc.
4. **Localidade explícita** — bairro, rua, nome de local
5. **Trecho verificável** — excerto de texto rastreável no documento

Todos os 5 critérios são atendidos pelos Anexos CPRM/DIGEAP (v1ir).

---

## Por Que Isso Ainda NÃO É Label

Mesmo sendo ground reference candidate documental, o relatório:

- NÃO fornece polígono de área atingida com precisão métrica
- NÃO foi validado via imagem para confirmar presença de cicatriz
- NÃO representa vetor observado independente
- NÃO libera supervisão, treino ou rotulagem

**Distância até label:**

```
Relatório CPRM (v1ir)
└─ Ground reference candidate documental ✓
   └─ Cruzar com patches Sentinel da data (futuro, não feito)
      └─ Verificar evidência estrutural
         └─ Validação de campo (Protocolo B — não iniciado)
            └─ Ground truth operacional (hipotético)
               └─ Label de treino (bloqueado)
```

---

## Documentos Auditados

### Fonte Principal

**CPRM — Diretoria de Hidrologia e Gestão Territorial (DHT)**  
**Departamento de Gestão Territorial (DEGET)**  
**Divisão de Geologia Aplicada (DIGEAP)**

Relatórios Técnicos para Identificação de Áreas com Risco em Caráter Emergencial.  
Petrópolis, RJ — Fevereiro-Março de 2022.

### Anexos Processados

| Anexo | Data Vistoria | Localidade | Fenômeno | Coord. GPS |
|-------|--------------|------------|----------|------------|
| II | 19/02/2022 | Moinho Preto | MOVEMENT_OF_MASS | ✓ (-22.484251, -43.211257) |
| III | 20/02/2022 | Serra Velha, Pontilhão | MOVEMENT_OF_MASS | — |
| IV | 22/02/2022 | Valparaíso, Rua Eugênio Barcelos | MOVEMENT_OF_MASS | — |
| V | 23/02/2022 | Rua Teresa e imediações | MOVEMENT_OF_MASS | — |
| VI | 24/02/2022 | Moinho Preto (revisita) | RISK_AREA | — |
| VII | 24/02/2022 | Mosella (revisita) | MOVEMENT_OF_MASS | — |
| VIII | 25–26/02/2022 | Estrada Velha, Vila Felipe | MOVEMENT_OF_MASS | — |
| IX | 28/02/2022 | Sargento Boening | MOVEMENT_OF_MASS | — |
| X | 01/03/2022 | Alto da Serra | MOVEMENT_OF_MASS | — |
| XI | 02/03/2022 | Quitandinha | MOVEMENT_OF_MASS | — |

---

## Precisão Obtida

### Precisão Temporal

- **Todos os anexos:** `EXACT_DATE` — vistoria realizada em data documentada pós-evento
- Cobertura: 19/02/2022 a 02/03/2022 (15 dias após o evento)

### Precisão Espacial

| Nível | Unidades |
|-------|----------|
| EXACT_COORDINATE | 1 (Moinho Preto — campo GPS documentado no relatório) |
| NEIGHBORHOOD | 9 (bairro/localidade descritos) |
| MUNICIPAL_ONLY | 0 |

### Fenômenos Encontrados

- `MOVEMENT_OF_MASS` — 9 anexos
- `RISK_AREA_MIXED` — 1 (Anexo VI — delimitação de polígonos críticos)
- `FLOODING` e `EROSION` — presentes no texto como fenômenos co-ocorrentes

---

## Candidatos Ground Reference

### Resultado Final

```
Documentos encontrados:     11
Documentos extraídos:       11
Unidades de evento criadas: 11
Ground reference candidates: 10 (9 documentary + 1 with coordinate)
Insufficient evidence:        1 (relatório técnico principal sem data estruturada)
```

### Candidato com Coordenada GPS

```
Unit ID:     PET2022_CPRM_ANEXOII_19022022
Fonte:       CPRM DIGEAP (Relatório Técnico Emergencial)
Data:        19/02/2022
Localidade:  Moinho Preto
Coordenada:  -22.484251, -43.211257 (Ponto 1: Rua Stephan Zweg)
Fenômeno:    Solapamento + enxurrada + risco para edificações
Precisão:    EXACT_COORDINATE
Status:      CANDIDATE_WITH_DOCUMENTED_COORDINATE
```

---

## Base para Próxima Etapa

### Há base real para georreferenciamento controlado?

**SIM.** As evidências documentais fornecidas por v1ir são:

- Datas exatas de vistorias de campo (pós-evento)
- Localidades documentadas (bairros e ruas)
- Fenômenos descritos por geólogos CPRM
- 3 coordenadas GPS explicitamente documentadas em textos de relatório
- Fonte oficial rastreável (CPRM/DIGEAP)

### Próximo Passo (se autorizado)

Cruzar as coordenadas e localidades documentadas com patches Sentinel das datas
de vistoria para verificação de evidência estrutural.

**Isso NÃO cria label automático.** É análise visual de correspondência
entre o que o geólogo documentou e o que é visível na imagem.

---

## Invariantes

```
can_be_operational_ground_truth  = NO  (sempre — requer Protocolo B)
can_create_training_label        = NO  (sempre — invariante absoluto)
can_train_model                  = NO  (sempre — invariante absoluto)
can_reopen_protocol_b            = NO  (sempre — invariante absoluto)
```

---

**Versão:** v1ir — Official Documented Event Unit Ground Reference Builder  
**Documentos:** 11 PDFs CPRM/DIGEAP  
**Candidatos:** 10 (9 documentary + 1 com coordenada GPS)  
**Temporal:** EXACT_DATE (19/02/2022 a 02/03/2022)  
**Espacial:** EXACT_COORDINATE (1) + NEIGHBORHOOD (9)  
**Markdown público:** Português  
**Sem claims preditivos, sem labels, sem supervisão — rigor máximo.**
