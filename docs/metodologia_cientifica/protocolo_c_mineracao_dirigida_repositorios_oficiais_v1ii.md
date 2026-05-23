# Protocolo C: Mineração Dirigida em Repositórios Oficiais — v1ii-R1

**Etapa:** v1ii-R1 — REAL_TARGETED_OFFICIAL_REPOSITORY_SCANNERS  
**Data:** 2026-05-22  
**Status:** Executado — scanners reais implementados e validados

---

## 1. Objetivos

A etapa v1ii busca **minerar recursos vetoriais e tabelares de eventos observados** em repositórios oficiais específicos, usando:

- APIs públicas (CKAN v3, ArcGIS REST, dados.gov.br)
- Catálogos de metadados (RIGeo, portais municipais/estaduais)
- Termos de busca controlados (inundação, deslizamento, ocorrência)
- Itens pré-auditados por inspeção manual de metadados e documentação pública
- Auditoria local de recursos encontrados

**Diferença com v1ih:**
- v1ih: descoberta ampla de candidatos locais conhecidos por inspeção de ativos em disco
- v1ii: mineração dirigida em fontes específicas com APIs/URLs configuradas + pré-registro de itens conhecidos

**Diferença entre scaffold e R1:**
- Scaffold (inicial): estrutura validada, sem scanners reais
- R1 (atual): 6 scanners reais implementados, pré-registro de 12 itens conhecidos, 7 outputs locais, registry público com linhas reais

---

## 2. Repositórios-Alvo

### 2.1 RIGeo/SGB

- **URL:** https://rigeo.sgb.gov.br
- **Termos:** Petrópolis, 2022, pós-desastre, inundação, cicatriz, avaliação
- **Formato esperado:** Metadata + pequenos anexos
- **Scanner:** Inspeção de metadados via handle público (`doc/22668`)
- **Status v1ii-R1:** API_NOT_AVAILABLE (DSpace sem endpoint CKAN) — 2 itens pré-auditados

### 2.2 CKAN Recife

- **API:** https://dados.recife.pe.gov.br/api/3/
- **Termos:** alagamento, inundação, enchente, defesa civil, ocorrência
- **Formatos esperados:** CSV, GeoJSON, SHP, API
- **Scanner:** `package_search` CKAN v3 — 3 termos × até 3 resultados
- **Status v1ii-R1:** SCAN_OK — 2 recursos retornados + pré-auditados

### 2.3 CKAN Pernambuco / APAC

- **Bases:** chuva, ocorrências, alertas, áreas afetadas
- **Scanner:** `package_search` CKAN v3
- **Status v1ii-R1:** SCAN_OK (resposta) + SCAN_FAILED_CONTROLLED (sem vetores de evento)

### 2.4 Dados Abertos RJ / DRM-RJ

- **Termos:** Petrópolis, Carta de Risco, cicatriz, inundação, 2022
- **Scanner:** API REST + inspeção de catálogo público DRM-RJ
- **Status v1ii-R1:** NETWORK_UNAVAILABLE — 2 itens pré-auditados por metadados

### 2.5 GeoCuritiba / IPPUC

- **Camadas públicas:** alagamento, drenagem, ocorrência, risco
- **Scanner:** Consulta ArcGIS REST (`FeatureServer/MapServer`)
- **Status v1ii-R1:** NETWORK_UNAVAILABLE — 2 itens pré-auditados

### 2.6 dados.gov.br / S2ID / Atlas

- **Conteúdo:** Evento, COBRADE, município, data, possível geometria
- **Scanner:** `package_search` dados.gov.br CKAN v3
- **Status v1ii-R1:** API_NOT_AVAILABLE — 3 itens pré-auditados

---

## 3. Fluxo de Mineração

```
┌─ Carregar itens pré-auditados (12 known items) ─────────┐
│  (por inspeção de metadados e documentação pública)     │
│                                                         │
├─ Tentar scan via API/catálogo ──────────────────────────┤
│  • Se OK: registrar novos recursos encontrados         │
│  • Se falha: NETWORK_UNAVAILABLE / API_NOT_AVAILABLE   │
│    (pipeline NÃO falha — registra controlado)          │
│                                                         │
├─ Aplicar 10 gates a cada candidato ────────────────────┤
│  gate_01 (fonte oficial) → gate_10 (ground truth)      │
│                                                         │
├─ Download apenas se: pequeno + formato vetorial         │
│  (armazenar em local_runs/ apenas)                     │
│                                                         │
├─ Classificar por status ────────────────────────────────┤
│  OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE                │
│  EVENT_CONFIRMATION_ONLY                               │
│  RISK_SUSCEPTIBILITY_ONLY                              │
│  BLOCKED_NO_DATE / BLOCKED_NO_GEOMETRY                 │
│  CARTOGRAPHIC_LEAD_ONLY                                │
│  SCAN_FAILED_CONTROLLED                                │
│                                                         │
└─ Cruzar com eventos-alvo ──────────────────────────────┘
   PET_2022_02_15 | REC_2022_05_26 | CTB
```

---

## 4. Critérios de Falha Controlada

Um scan pode resultar em:

| Status | Significado |
|--------|-------------|
| `SCAN_OK` | API respondeu e retornou dados |
| `SCAN_EMPTY` | API respondeu mas sem resultados para os termos |
| `SCAN_FAILED_CONTROLLED` | API respondeu mas sem vetores verificáveis |
| `NETWORK_UNAVAILABLE` | Timeout ou erro de conexão |
| `API_NOT_AVAILABLE` | Endpoint sem suporte CKAN/REST ou ausente |
| `PARSE_FAILED_CONTROLLED` | Resposta inválida ou JSON malformado |

**Em todos os casos:** o script termina com código 0 (sem crash). A falha é registrada no scan log.

---

## 5. Invariantes Permanentes

```
nao_enviar_email                      = true
nao_criar_solicitacao_institucional   = true
nao_inventar_coordenada               = true
nao_georreferenciar_pdf               = true
nao_aceitar_risco_como_ocorrencia     = true
nao_treinar_modelo                    = true
nao_criar_label_target_class          = true
nao_reabrir_protocolo_b               = true
nao_versionar_dados_pesados           = true
dados_brutos_apenas_local_runs        = true
publicos_apenas_metadata_registries   = true
markdown_publico_em_portugues         = true
sem_paths_privados_em_arquivos_publicos = true
```

---

## 6. Classificações de Recurso

| Classificação | Significado |
|---------------|-------------|
| `OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE` | Vetor com todos os 10 gates aprovados |
| `OBSERVED_VECTOR_EVENT_REFERENCE` | Vetor com 9/10 gates — referência observacional |
| `GEOCODED_EVENT_TABLE_CANDIDATE` | Tabela com coordenadas, data, fenômeno |
| `EVENT_CONFIRMATION_ONLY` | Confirma evento mas sem geometria patch-level |
| `RISK_SUSCEPTIBILITY_ONLY` | Mapa de risco/susceptibilidade — não ocorrência |
| `MODELLED_LAYER_ONLY` | Modelo prospectivo |
| `CARTOGRAPHIC_LEAD_ONLY` | Mapa/PDF sem vetor direto |
| `DOCUMENTARY_ONLY` | Texto/relatório sem vetor |
| `BLOCKED_NO_GEOMETRY` | Sem geometria vetorial |
| `BLOCKED_NO_DATE` | Sem data de evento explícita |
| `BLOCKED_NO_PHENOMENON` | Sem campo de fenômeno separável |
| `BLOCKED_NOT_OBSERVED_EVENT` | Data incompatível com evento-alvo |
| `BLOCKED_NOT_PATCH_LEVEL` | Geometria municipal, não patch-level |
| `SCAN_FAILED_CONTROLLED` | Falha de scan registrada controladamente |
| `NOT_USABLE` | Recurso inutilizável por qualquer motivo |

---

## 7. Outputs

### Locais (não versionados — `local_runs/protocolo_c/v1ii/`)

```
v1ii_repository_scan_log.csv     — 6 repositórios, status, contadores
v1ii_resource_inventory.csv      — 12 recursos, todos os campos
v1ii_download_audit.csv          — downloads tentados
v1ii_vector_table_audit.csv      — candidatos com geometria ou pendente
v1ii_candidate_decisions.csv     — 12 decisões com gate breakdown
v1ii_qa.csv                      — 7 validações de integridade
v1ii_summary.json                — sumário completo com invariantes
```

### Públicos (`datasets/`)

```
targeted_official_repository_event_vector_registry.csv   — 12 candidatos
schemas/targeted_official_repository_event_vector_registry_schema.csv — 27 campos
```

---

## 8. Resultado da Execução v1ii-R1

**Repositórios consultados:** 6  
**Recursos auditados:** 12  
**Ground truth candidatos:** 0  
**Status operacional:** `BLOCKED`

Distribuição:
```
EVENT_CONFIRMATION_ONLY    : 4
CARTOGRAPHIC_LEAD_ONLY     : 3
BLOCKED_NO_DATE            : 2
SCAN_FAILED_CONTROLLED     : 2
RISK_SUSCEPTIBILITY_ONLY   : 1
```

A lacuna de disponibilidade pública de vetor observado datado em escala patch-level é confirmada. Ver relatório completo em:  
[`protocolo_c_relatorio_mineracao_dirigida_repositorios_oficiais_v1ii.md`](protocolo_c_relatorio_mineracao_dirigida_repositorios_oficiais_v1ii.md)

---

**Status:** Executado (v1ii-R1 com scanners reais)  
**Próxima etapa:** v1ij — Expansão multi-regional dos candidatos observados (condicional a nova fonte com data, fenômeno e geometria)
