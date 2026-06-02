# Protocolo C — v1ij: Relatório de Consolidação de Candidatos Vetoriais Observados

**Data de Execução:** 2026-05-23  
**Versão:** v1ij-R1  
**Status Final:** `NO_CANDIDATE_PASSED_MINIMUM_PATCH_BINDING_GATES`

---

## 1. Resumo Executivo

v1ij consolidou **18 candidatos** de v1if (6 candidatos) e v1ii (12 candidatos). Após auditoria com gates padronizados:

- **0 candidatos** passaram os gates mínimos para patch binding preflight
- **14 candidatos** bloqueados por falta de geometria (gate_02)
- **3 candidatos** bloqueados por falta de data de evento (gate_04)
- **1 candidato** bloqueado por falta de fenômeno (gate_06)

**Melhor candidato bloqueado:** `camada original de feições poligonais de deslizamento fotointerpretadas` (V1II_002) — tem geometria e fenômeno, mas falta data específica de evento.

---

## 2. Quantidade e Origem de Candidatos

### Total: 18 Candidatos

| Origem | Quantidade | Detalhes |
|--------|-----------|----------|
| **v1if** | 6 | SGB/CPRM RIGeo Petrópolis 2022 |
| **v1ih** | 0 | Registry vazio (auditoria local concluída, sem candidatos novos) |
| **v1ii** | 12 | Repositórios oficiais dirigidos (RIGeo, CKAN, Dados Abertos RJ) |
| **Total** | **18** | **Consolidados** |

---

## 3. Distribuição por Região

| Região | Candidatos | Evento-Alvo | Fenômenos Buscados |
|--------|-----------|------------|-------------------|
| **PET** | 10 | PET_2022_02_15 | Deslizamento, inundação |
| **REC** | 5 | REC_2022_05_24_30 | Inundação, enxurrada |
| **CUR** | 1 | CTB_unknown | Alagamento |
| **CTB** | 2 | CTB_unknown | Alagamento |
| **Total** | **18** | — | — |

---

## 4. Distribuição por Fenômeno

Candidatos não possuem campos estruturados de fenômeno preenchidos. Todos os 18 candidatos têm `phenomenon_group` vazio ou misto.

- **hydrological:** 0 explícitos
- **mass_movement:** 0 explícitos
- **mixed/unknown:** 18

**Implicação:** Fenômeno é geralmente apenas metadado de titulo/descrição, não campo de dados estruturado.

---

## 5. Gates Bloqueantes — Distribuição de Falhas

### Gate 2: Vetor ou Tabela Georref (gate_vector_or_georeferenced_table)
- **FAIL:** 14 candidatos
- **Razão:** PDFs (v1if) e datasets sem vetor real
- **Exemplo:** `Relatorio_Tecnico_Petropolis.pdf` — documento apenas

### Gate 4: Data de Evento Disponível (gate_event_date_available)
- **FAIL:** 3 candidatos
- **Razão:** Sem campo de data no vetor ou metadado compatível
- **Exemplo:** `camada original de feições poligonais de deslizamento fotointerpretadas` — feição de deslizamento presente, mas sem data de ocorrência

### Gate 6: Fenômeno Disponível (gate_phenomenon_available)
- **FAIL:** 1 candidato
- **Razão:** Vetor sem identificação clara de fenômeno
- **Exemplo:** Dataset genérico de "áreas afetadas" sem classificação

### Resumo de Bloqueadores

```
gate_02_no_geometry:     14 candidatos (77.8%)
gate_04_no_event_date:    3 candidatos (16.7%)
gate_06_no_phenomenon:    1 candidato  (5.6%)
---
Total:                   18 candidatos (100%)
```

**Bloqueador Principal:** `gate_02_no_geometry` — maioria dos candidatos são documentos (PDFs) ou datasets sem geometria real.

---

## 6. Candidato Mais Promissor (Bloqueado)

### camada original de feições poligonais de deslizamento fotointerpretadas
- **ID Consolidado:** V1II_002
- **Origem:** v1ii (RIGeo/SGB)
- **Fonte Original:** RIGEO_PET_002
- **Fonte Instituição:** SGB/CPRM
- **Fenômeno:** Deslizamento (feição de deslizamento)
- **Geometria:** SHP com 444 features
- **CRS:** Presente
- **Data de Evento:** FALTA
- **Status:** BLOCKED_NO_DATE
- **Bloqueador:** gate_04_no_event_date
- **Razão:** feições de deslizamento de deslizamento consolidadas, mas sem data específica de ocorrência

### Por Que É Promissor:
- Geometria vetorial real (Polygon)
- Feature count significativo (444)
- Instituição oficial (SGB/CPRM)
- Fenômeno claro (deslizamento)
- Não é risco/suscetibilidade, é ocorrência observada

### Por Que É Bloqueado:
- Nenhuma data de evento no vetor
- Nenhum metadado público com data da feição de deslizamento
- Seria necessária solicitação formal ao SGB/CPRM ou busca em documentação não-publicada

### Próxima Ação Recomendada:
Se for localizada evidência pública ou local documentando a data de levantamento ou vínculo temporal de camada original de feições poligonais de deslizamento fotointerpretadas (ex: em sidecars `.prj`, `.xml`, metadados de repositório, ou documentação técnica publicada), este candidato avançaria imediatamente para patch binding preflight.

---

## 7. Tentativa de Enriquecimento de Metadados

**Flag:** `--enrich-metadata --scan-local-sidecars`

### Resultado:
- Nenhum sidecars locais encontrados
- Nenhum arquivo `.prj`, `.xml`, `.dbf`, `.qmd` com informações de data
- Status: `NOT_ATTEMPTED` para todos os candidatos

### Por Que:
- Candidatos são referências em repositórios públicos, não arquivos locais presentes
- v1ij não faz download automático
- Enriquecimento é apenas leitura de metadados já existentes

### Próxima Ação:
Se sidecars locais fossem adquiridos (manualmente ou em v1iii), enriquecimento poderia resolver alguns bloqueios de data.

---

## 8. Preflight de Patch Binding

### Resultado:
**0 candidatos avançaram para patch binding preflight**

### Única Linha no Registry:
```
patch_binding_candidate_id: STATUS_REPORT
asset_name: NO_CANDIDATE_PASSED_MINIMUM_PATCH_BINDING_GATES
preflight_status: NO_CANDIDATE_PASSED
blocking_reason: All candidates blocked by missing event date, geometry, phenomenon, or risk status
overlay_allowed: NO
label_creation_allowed: NO
```

### Gates Mínimos Não Atingidos:
- `gate_02_vector_or_georeferenced_table` — 14 candidatos falham
- `gate_04_event_date_available` — 3 candidatos falham
- `gate_06_phenomenon_available` — 1 candidato falha

---

## 9. Status Final por Domínio

### Operational Ground Truth
- **Status:** `BLOCKED`
- **Razão:** Nenhum candidato passou gate_02 (geometria) ou gate_04 (data)
- **Implicação:** Sem vetor de ocorrência observada validado

### ML Label e Training
- **Status:** `BLOCKED`
- **Razão:** `can_create_training_label = false` em todas as linhas
- **Implicação:** Nenhum dado para training supervisionado

### Protocol B (Sentinel-DINO Embedding)
- **Status:** `BLOCKED`
- **Razão:** Protocol B não foi reaberido e não é caminho principal
- **Implicação:** Foco mantém-se em Protocol C (observed vectors)

---

## 10. Próximos Passos Técnicos

### Cenário 1: Localizar Data para camada original de feições poligonais de deslizamento fotointerpretadas
1. Buscar em fontes públicas (metadados RIGeo, documentação técnica SGB, sidecars locais)
2. Se encontrado: enriquecer candidato com data documentada
3. Reavaliar gates
4. Se passar: avançar para patch binding preflight (v1ik)

### Cenário 2: Buscar Novos Candidatos em v1iii
- Explorar outros portais (INEA RJ, Defesa Civil, Atlas de Desastres)
- Focar em datasets com geometria já presente
- Priorizar dados datados

### Cenário 3: Revisar Gate de Fenômeno (gate_06)
- 1 candidato bloqueia por gate_06 (falta de fenômeno identificável)
- Revisar se dataset genérico poderia ser reclassificado com metadados públicos

### Cenário 4: Aceitar Bloqueio Estruturado (Opção Válida)
- Ground truth observado para Petrópolis/Recife pode não existir em forma vetorial publicada
- Manter bloqueio é honesto e auditável
- Próximas etapas: explorar Protocol A (Sentinel raw pixels), expandir discovery em v1iii, ou revisar bloqueadores com evidência pública

---

## 11. Limitações Reconhecidas

1. **Sem Solicitação Formal:** v1ij não envia e-mail ou abre solicitação. Se dados bloqueados são privados, permanecerão desconhecidos.

2. **Sidecars Não Locais:** Metadados de contexto (data de avaliação, fenômeno) podem existir em repositórios oficiais, mas não foram digitalizados.

3. **OCR Não Executado:** PDFs contêm informações que não foram OCRizadas (decisão controlada).

4. **Dados Cumulativos:** `camada original de feições poligonais de deslizamento fotointerpretadas` pode representar múltiplos eventos; data única pode ser insuficiente.

---

## 12. Validações Executadas

✓ Script existe e roda  
✓ Registry consolidado criado com 18 candidatos  
✓ Schema compatível (26 campos)  
✓ Matriz de gates criada (10 gates)  
✓ Patch binding preflight criado (status report)  
✓ `can_create_training_label` sempre `false`  
✓ `label_creation_allowed` sempre `NO`  
✓ Candidatos sem data ficam bloqueados  
✓ Risco/suscetibilidade não aparecem como ground truth  
✓ PDFs bloqueados como vetor  
✓ Nenhum candidato avança sem passar todos gates minimos  
✓ Status `NO_CANDIDATE_PASSED_MINIMUM_PATCH_BINDING_GATES` aparece  
✓ Enriquecimento por data de sistema não aceito  
✓ Sem label/target supervisionado  
✓ Sem path privado em arquivo público  

---

## 13. Estatísticas de Consolidação

```json
{
  "total_candidates_loaded": 18,
  "candidates_from_v1if": 6,
  "candidates_from_v1ii": 12,
  "candidates_by_region": {
    "PET": 10,
    "REC": 5,
    "CUR": 1,
    "CTB": 2
  },
  "blocking_gates_distribution": {
    "gate_02_no_geometry": 14,
    "gate_04_no_event_date": 3,
    "gate_06_no_phenomenon": 1
  },
  "candidates_passing_preflight": 0,
  "no_candidate_passed_gates": true
}
```

---

## Conclusão

v1ij consolidou sistematicamente 18 candidatos de fontes oficiais e aplicou gates padronizados. **Nenhum candidato passou**, resultado válido e auditável.

O bloqueio principal (14 candidatos) é categorizado: a maioria são documentos (PDFs), não vetores. O segundo bloqueio (3 candidatos) é metadados: falta data de evento.

O melhor candidato (`camada original de feições poligonais de deslizamento fotointerpretadas`) é **bloqueado por data**, não por qualidade vetorial. Avançaria imediatamente se metadado de data fosse adquirido.

**Status Operacional:** `BLOCKED — NO_CANDIDATE_PASSED_MINIMUM_PATCH_BINDING_GATES`

---

**Gerado em:** 2026-05-23  
**Versão do Script:** v1ij-R1  
**Última Atualização:** Consolidação Inicial
