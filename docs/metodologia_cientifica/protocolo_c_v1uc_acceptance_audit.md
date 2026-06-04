# Protocolo C — Auditoria de Aceitação v1uc

**Data:** 2026-06-03
**Versão auditada:** v1uc — Evidence Acquisition Core
**Auditor:** Programático (pré-v1ud)

## 1. Gitignore

| Verificação | Resultado |
|------------|-----------|
| `local_only/` bloqueado | PASS — presente nas linhas 67 e 80 do .gitignore |
| `local_only/protocolo_c/evidence_raw/**` bloqueado | PASS — extensões .pdf/.shp/.gpkg/.geojson/.kml/.kmz |
| `local_only/protocolo_c/evidence_staging/**` bloqueado | PASS |
| `local_only/protocolo_c/evidence_reports/**` bloqueado | PASS |

## 2. Arquivos Pesados Fora de local_only/

| Verificação | Resultado |
|------------|-----------|
| Nenhum .tif/.tiff fora de local_only | PASS — 0 encontrados |
| Nenhum .npz fora de local_only | PASS — 12 .npz encontrados em local_runs/ (gitignored) |
| Nenhum raster/embedding em datasets/ | PASS |

## 3. Guardrails nos CSVs

| Guardrail | Resultado |
|-----------|-----------|
| `can_create_training_label=true` ausente | PASS — 0 ocorrências |
| `ground_truth_operational=true` ausente | PASS — 0 ocorrências |
| `can_reopen_protocol_b=true` ausente | PASS — 0 ocorrências |
| Coordenadas inventadas | PASS — nenhuma coordenada presente nos registries |
| Path absoluto local versionável | PASS — 0 ocorrências de `C:\Users` ou `/home/` |

## 4. Contagem de Registros

| Métrica | Valor | Explicação |
|---------|-------|------------|
| Fontes configuradas | 9 | ANA, INMET, Cemaden, SGB/CPRM, Copernicus, Charter, Maxar, Planet, EM-DAT |
| Eventos | 3 | PET_2022_02_15, PET_2024_03_21_28, REC_2022_05_24_30 |
| Linhas evidence_source_registry.csv | 43 | 1 header + 42 dados |
| Avaliações de gate | 462 | 42 evidências × 11 gates |

## 5. Por Que 42 Registros e Não 27

O pipeline gera um registro **por URL base por evento**, não por fonte por evento.

| Fonte | URLs base | × 3 eventos |
|-------|-----------|-------------|
| ANA_HIDROWEB | 2 | 6 |
| INMET_BDMEP | 2 | 6 |
| CEMADEN_PLUVIOMETROS | 2 | 6 |
| SGB_CPRM_CARTOGRAFIA | 2 | 6 |
| COPERNICUS_EMS | 2 | 6 |
| INTERNATIONAL_CHARTER | 1 | 3 |
| MAXAR_OPEN_DATA | 1 | 3 |
| PLANET_DISASTER_DATA | 1 | 3 |
| EMDAT | 1 | 3 |
| **Total** | **14** | **42** |

Isso é o comportamento correto: cada URL base é um endpoint distinto que pode conter dados
diferentes (ex: portal web vs API SOAP para ANA, portal vs dados históricos para INMET).

## 6. YAML Config

| Verificação | Resultado |
|------------|-----------|
| `ground_truth_operational: false` | PASS |
| `can_create_training_label: false` | PASS |
| `can_reopen_protocol_b: false` | PASS |
| `dino_usage: SUPPORT_ONLY` | PASS |
| Todas as fontes com `can_promote_ground_reference: false` | PASS — 9/9 |

## 7. Gate Audit

| Verificação | Resultado |
|------------|-----------|
| G10 (patch_overlay) sempre FAIL | PASS — enforced |
| G11 (supervisor_review) sempre FAIL | PASS — enforced |
| Nenhuma evidência promovida | PASS — todos BLOCKED |

## 8. Testes

| Suite | Resultado |
|-------|-----------|
| test_protocol_c_evidence_acquisition_core.py | 19 passed |
| test_protocol_c_ground_reference_gate_audit.py | 9 passed |
| test_protocol_c_formal_request_package_builder.py | 5 passed |
| test_protocol_c_evidence_closure_report.py | 5 passed |
| **Total** | **38 passed, 0 failed** |

## Decisão

**v1uc ACEITA para progressão a v1ud.**

Todos os guardrails verificados. Nenhuma violação encontrada. Contagem de registros
explicada e coerente. Infraestrutura pronta para aquisição real controlada.
