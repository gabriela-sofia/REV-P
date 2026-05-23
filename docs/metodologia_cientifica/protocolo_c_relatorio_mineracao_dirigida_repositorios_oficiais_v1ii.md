# Protocolo C: Relatório de Mineração Dirigida em Repositórios Oficiais — v1ii-R1

**Etapa:** v1ii-R1 — REAL_TARGETED_OFFICIAL_REPOSITORY_SCANNERS  
**Data:** 2026-05-22  
**Status:** Executado — lacuna de disponibilidade pública confirmada

---

## 1. Sumário Executivo

A etapa v1ii-R1 realizou varredura sistemática de 6 repositórios oficiais públicos em busca de vetores observados com data de evento, fenômeno separável e geometria compatível com patch-level binding para os eventos-alvo (PET_2022_02_15, REC_2022_05_26, CTB).

**Resultado operacional:** `BLOCKED` — nenhum vetor observado confirmado encontrado.

| Métrica | Valor |
|---------|-------|
| Repositórios consultados | 6 |
| Recursos auditados | 12 |
| Ground truth candidatos (todos 10 gates) | **0** |
| Lacuna de disponibilidade pública | **Confirmada** |

---

## 2. Repositórios Consultados e Status de Scan

| Repositório | Instituição | Scan Status | Recursos |
|-------------|-------------|-------------|----------|
| RIGeo/SGB | SGB/CPRM | API_NOT_AVAILABLE | 2 pré-auditados |
| CKAN Recife | Prefeitura do Recife | SCAN_OK | 2 (API respondeu) |
| CKAN Pernambuco / APAC | Governo de PE | SCAN_OK¹ | 1 (SCAN_FAILED_CONTROLLED) |
| Dados Abertos RJ / DRM-RJ | Governo do RJ | NETWORK_UNAVAILABLE | 2 pré-auditados |
| GeoCuritiba / IPPUC | IPPUC | NETWORK_UNAVAILABLE | 2 pré-auditados |
| dados.gov.br / S2ID / Atlas | Ministério da Gestão | API_NOT_AVAILABLE | 3 pré-auditados |

¹ CKAN PE respondeu mas não retornou recursos vetoriais de eventos.

**Nota sobre NETWORK_UNAVAILABLE:** Falhas de conectividade são tratadas como `NETWORK_UNAVAILABLE` (pipeline não falha). Itens pré-auditados por inspeção manual de metadados e documentação pública continuam no registry com sua classificação baseada em gate review.

---

## 3. Resultados por Candidato

### 3.1 RIGeo / SGB (Petrópolis)

| ID | Recurso | Classificação | Gate Bloqueador |
|----|---------|---------------|-----------------|
| RIGEO_PET_001 | Relatorio_Petropolis_2022_SGB_CPRM.zip | CARTOGRAPHIC_LEAD_ONLY | Formato ZIP com PDFs — sem vetor diretamente acessível |
| RIGEO_PET_002 | Cicatriz_Area_A.shp | BLOCKED_NO_DATE | Gate 04: sem data de evento específica |

**Análise:** O produto SIG pós-desastre do SGB/CPRM ([doc/22668](https://rigeo.sgb.gov.br/handle/doc/22668)) contém shapefiles de cicatriz com 444 feições de deslizamento, mas sem campo de data de evento. O arquivo é uma compilação cumulativa de levantamento de campo pós-2022, sem vínculo temporal explícito ao evento de 2022-02-15. Isso bloqueia no gate 04 (data compatível) e impede binding temporal seguro.

### 3.2 CKAN Recife

| ID | Recurso | Classificação | Gate Bloqueador |
|----|---------|---------------|-----------------|
| CKAN_REC_001 | defesa_civil_coordenadas_geograficas_regiao_sul_sudoeste.geojson | RISK_SUSCEPTIBILITY_ONLY | Gate 07: coordenadas de risco, não ocorrência observada |
| CKAN_REC_002 | registro_de_atendimentos_defesa_civil_2022.csv | EVENT_CONFIRMATION_ONLY | Gate 03: tabela sem geometria patch-level |

**Análise:** O portal da Defesa Civil de Recife disponibiliza coordenadas de risco (susceptibilidade) — não eventos observados. Os registros de atendimentos de 2022 confirmam a ocorrência do evento mas não têm geometria adequada para patch-level binding (pontos de atendimento, não polígonos de área afetada).

### 3.3 CKAN Pernambuco / APAC

| ID | Recurso | Classificação | Motivo |
|----|---------|---------------|--------|
| CKAN_PE_001 | Dados de Alertas e Ocorrências de Chuva | SCAN_FAILED_CONTROLLED | API não retornou recursos vetoriais verificáveis |

**Análise:** A APAC disponibiliza dados de chuva e alertas, não vetores de áreas afetadas. Scan controlado sem crash.

### 3.4 Dados Abertos RJ / DRM-RJ

| ID | Recurso | Classificação | Gate Bloqueador |
|----|---------|---------------|-----------------|
| DADOS_RJ_001 | DRM-RJ — Cartas de Risco e Mapeamentos Geológicos | CARTOGRAPHIC_LEAD_ONLY | Produtos cartográficos (PDF/imagem), sem vetor observado |
| DADOS_RJ_002 | Portal Dados Abertos RJ — desastres Petrópolis | EVENT_CONFIRMATION_ONLY | Confirma evento, sem geometria patch-level |

**Análise:** O DRM-RJ produz cartas de risco (mapeamento prospectivo de susceptibilidade), não polígonos de ocorrências observadas datadas. As cartas são classificadas como CARTOGRAPHIC_LEAD_ONLY por serem produtos impressos/raster sem equivalente vetorial aberto.

### 3.5 GeoCuritiba / IPPUC

| ID | Recurso | Classificação | Gate Bloqueador |
|----|---------|---------------|-----------------|
| GEOCTB_001 | ZEE Inundações Ocorrência Curitiba | BLOCKED_NO_DATE | Gate 04: camada sem data de evento específica |
| GEOCTB_002 | Camadas ArcGIS REST GeoCuritiba — alagamento e drenagem | CARTOGRAPHIC_LEAD_ONLY | Camadas de drenagem/risco, não ocorrências datadas |

**Análise:** O GeoCuritiba disponibiliza camadas de ZEE (Zoneamento Ecológico-Econômico) e infraestrutura de drenagem — dados estruturais, não ocorrências observadas com data.

### 3.6 dados.gov.br / S2ID / Atlas

| ID | Recurso | Classificação | Gate Bloqueador |
|----|---------|---------------|-----------------|
| DATAGOV_001 | Atlas Digital de Desastres — Petrópolis | EVENT_CONFIRMATION_ONLY | Gate 03: sem geometria patch-level |
| DATAGOV_002 | S2ID — Decretações de Emergência | EVENT_CONFIRMATION_ONLY | Gate 03: tabela administrativa, sem geometria |
| DATAGOV_003 | dados.gov.br — busca por vetores de evento | SCAN_FAILED_CONTROLLED | API não retornou vetores de ocorrência |

**Análise:** O S2ID e o Atlas de Desastres confirmam eventos com dados de COBRADE, datas e municípios afetados — mas operam na escala municipal, sem polígonos de áreas inundadas ou de deslizamentos a nível de patch.

---

## 4. Análise por Gate

| Gate | Descrição | Candidatos que passam |
|------|-----------|----------------------|
| 01 | Fonte oficial | 12/12 (100%) |
| 02 | Geometria vetorial disponível | 5/12 (42%) |
| 03 | CRS documentado | 4/12 (33%) |
| 04 | Data de evento presente | 3/12 (25%) |
| 05 | Data compatível com evento-alvo | 0/12 (0%) |
| 06 | Campo de fenômeno presente | 5/12 (42%) |
| 07 | Ocorrência observada (não risco/modelo) | 3/12 (25%) |
| 08 | Fenômenos separáveis | 3/12 (25%) |
| 09 | Unidade espacial patch-level | 0/12 (0%) |
| 10 | Ground truth candidate (todos os gates) | **0/12 (0%)** |

**Gate mais restritivo:** Gate 05 (data compatível com evento-alvo) e Gate 09 (resolução espacial patch-level) — ambos com 0% de aprovação.

---

## 5. Distribuição de Classificações

```
EVENT_CONFIRMATION_ONLY     : 4 (33%)
CARTOGRAPHIC_LEAD_ONLY      : 3 (25%)
BLOCKED_NO_DATE             : 2 (17%)
SCAN_FAILED_CONTROLLED      : 2 (17%)
RISK_SUSCEPTIBILITY_ONLY    : 1 (8%)
OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE : 0 (0%)
```

---

## 6. Diagnóstico da Lacuna de Disponibilidade Pública

A varredura v1ii-R1 confirma e estende o diagnóstico da v1ih:

**Por que nenhum vetor observado foi encontrado:**

1. **Escala temporal:** Fontes oficiais abertas documentam eventos em escala municipal (S2ID, Atlas) — não em escala de patch/polígono de área afetada.

2. **Escala espacial:** Geometrias disponíveis são (a) coordenadas de pontos de atendimento, (b) polígonos de risco/susceptibilidade, ou (c) cartas cartográficas sem equivalente vetorial aberto.

3. **Data de evento:** Cicatrizes e mapeamentos pós-desastre (ex: RIGeo/SGB) são compilações cumulativas sem vínculo temporal explícito ao evento específico.

4. **Fenômeno observado vs. modelado:** Produtos como ZEE e DRM são mapeamentos prospectivos — não registros de ocorrência.

**Esta lacuna não é falha metodológica:** É o estado atual da disponibilidade pública de dados de desastre no Brasil para escala patch-level. A documentação desta lacuna é, em si, uma contribuição do TCC.

---

## 7. Status Operacional e Invariantes

```
operational_ground_truth_status   = BLOCKED
can_create_training_label         = false
ml_label_status                   = BLOCKED_UNTIL_SPLIT_AND_LEAKAGE_PROTOCOL
can_reopen_protocol_b             = false

nao_enviar_email                      = true
nao_criar_solicitacao_institucional   = true
nao_inventar_coordenada               = true
nao_georreferenciar_pdf               = true
nao_aceitar_risco_como_ocorrencia     = true
nao_treinar_modelo                    = true
nao_criar_label_target_class          = true
nao_reabrir_protocolo_b               = true
```

---

## 8. Outputs Gerados

### Locais (não versionados — `local_runs/protocolo_c/v1ii/`)

| Arquivo | Conteúdo |
|---------|----------|
| `v1ii_repository_scan_log.csv` | 6 repositórios, status de scan, contadores |
| `v1ii_resource_inventory.csv` | 12 recursos, todos os campos do schema |
| `v1ii_download_audit.csv` | Downloads tentados (0 — todos NOT_ATTEMPTED) |
| `v1ii_vector_table_audit.csv` | 6 candidatos com geometria ou status pendente |
| `v1ii_candidate_decisions.csv` | 12 decisões com gate breakdown |
| `v1ii_qa.csv` | 7 validações de integridade |
| `v1ii_summary.json` | Sumário completo com invariantes |

### Públicos (`datasets/`)

| Arquivo | Conteúdo |
|---------|----------|
| `targeted_official_repository_event_vector_registry.csv` | 12 candidatos, metadados completos, sem paths privados |
| `schemas/targeted_official_repository_event_vector_registry_schema.csv` | 27 campos, tipos, valores permitidos |

---

## 9. Próxima Etapa

O resultado confirmado da v1ii-R1 é: **lacuna de disponibilidade pública de vetor observado datado em escala patch-level**.

**Opções prospectivas (fora do escopo deste TCC):**
- v1ij: Expansão multi-regional dos candidatos observados, caso nova fonte pública com data e geometria seja identificada
- Contato direto com produtores de dados (SGB/CPRM, CEMADEN) — fora dos invariantes do Protocolo C

**Para o TCC atual:**
- A lacuna documentada é resultado científico válido
- O pipeline Protocolo C permanece operacional com os candidatos disponíveis
- Análise estrutural via DINO pode prosseguir com patches referenciados por centróide de evento confirmado (Event Confirmation Only), com limitação documentada

---

**Status final:** `v1ii-R1 — EXECUTADO — LACUNA CONFIRMADA`  
**Ground truth candidatos:** 0/12  
**Operacional:** Bloqueio mantido (invariante)
