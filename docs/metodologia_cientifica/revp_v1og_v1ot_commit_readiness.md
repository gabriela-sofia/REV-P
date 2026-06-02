# REV-P v1og-v1ot — Pacote de Commit: Prontidão e Auditoria

## Escopo do commit

Bloco de recuperação temporal Sentinel para Recife, Protocolo C.
Estágios v1og–v1ot, do grafo de proveniência patch-asset à consolidação final auditável.
Resultado científico: **TEMPORAL_RECOVERY_FAIL_CLOSED** (resultado negativo limpo e auditável).

---

## Módulos v1og-v1ot

| Estágio | Script | Função |
|---|---|---|
| v1og | `revp_v1og_rec_patch_provenance_graph_builder.py` | Grafo proveniência patch→asset→produto, normalização aliases |
| v1og-v1ol | `revp_v1og_v1ol_common.py` | Helpers comuns: proveniência, normalização, datas |
| v1oh | `revp_v1oh_local_sentinel_asset_metadata_scanner.py` | Scan local de metadata/sidecars Sentinel (sem pixel) |
| v1oi | `revp_v1oi_sentinel_product_id_mgrs_date_resolver.py` | Resolução de data por ID de produto e MGRS |
| v1oj | `revp_v1oj_rec_patch_scene_date_adjudication_v2.py` | Adjudicação scene_date v2 por patch |
| v1ok | `revp_v1ok_rec_event_patch_temporal_rematch_v2.py` | Rematch temporal evento↔patch v2 |
| v1ol | `revp_v1ol_rec_c3_plus_dino_recheck_after_provenance_recovery.py` | Recheck C3+/C4/DINO pós-proveniência |
| v1om | `revp_v1om_recife_sentinel_sidecar_discovery.py` | Descoberta de sidecars SAFE/MTD/STAC |
| v1om-v1or | `revp_v1om_v1or_common.py` | Helpers v2: parsers, classificadores, filtro fixture |
| v1on | `revp_v1on_sentinel_product_date_parser.py` | Parser de datas de produto Sentinel com bloqueio explícito |
| v1oo | `revp_v1oo_recife_patch_scene_date_resolver_v3.py` | Resolver v3 fail-closed, filtro fixture |
| v1op | `revp_v1op_recife_event_patch_temporal_adjudication_v3.py` | Adjudicação temporal v3 fail-closed |
| v1oq | `revp_v1oq_recife_c3_c4_dino_recheck_after_scene_date_v3.py` | Recheck C3+/C4/DINO pós-resolver v3 |
| v1or | `revp_v1or_scene_date_recovery_v3_bundle.py` | Bundle e sumário v3 |
| v1os | `revp_v1os_fixture_contamination_audit.py` | Auditoria de contaminação por fixture/test |
| v1ot | `revp_v1ot_scene_date_recovery_final_audit_bundle.py` | Manifest final + QC + sumário científico |

---

## Resumo científico

- Patches avaliados: **2.654**
- Product dates confirmadas reais: **0**
- `can_unlock_temporal=true`: **0**
- C3+ candidates: **0**
- C4 formal negatives: **0**
- DINO queue: **0**
- Fixture high-severity: **0**
- QC HIGH FAILs: **0**
- **Status final: TEMPORAL_RECOVERY_FAIL_CLOSED**

O resultado negativo é válido e auditável. A pipeline foi executada com rastreabilidade
completa. Nenhuma cadeia `patch → asset → produto Sentinel oficial → data de aquisição`
foi confirmada com os ativos locais disponíveis.

---

## Outputs principais

### Outputs novos (v1om-v1ot)
- `datasets/recife_sentinel_sidecar_discovery_v1om.csv`
- `datasets/recife_sentinel_product_date_candidates_v1on.csv`
- `datasets/recife_patch_scene_date_resolved_v3_v1oo.csv` (2654 linhas, 0 confirmadas)
- `datasets/recife_event_patch_temporal_adjudication_v3_v1op.csv`
- `datasets/recife_c3_plus_recheck_after_scene_date_v3_v1oq.csv`
- `datasets/recife_c4_status_after_scene_date_v3_v1oq.csv`
- `datasets/recife_dino_review_queue_after_scene_date_v3_v1oq.csv`
- `datasets/recife_scene_date_recovery_v3_master_summary_v1or.csv`
- `datasets/recife_fixture_contamination_audit_v1os.csv` (0 high-severity)
- `datasets/recife_scene_date_recovery_final_manifest_v1ot.csv` (25 artefatos)
- `datasets/recife_scene_date_recovery_final_quality_checks_v1ot.csv` (234 checks, 0 HIGH FAILs)
- `datasets/recife_scene_date_recovery_final_scientific_summary_v1ot.csv` (19 métricas)

### Outputs regenerados/limpos (v1og-v1ol)
- `datasets/recife_patch_provenance_graph_registry.csv` (27.520 linhas, limpo)
- `datasets/recife_sentinel_product_date_resolution_registry.csv` (27.520 linhas)
- `datasets/recife_patch_scene_date_adjudication_v2.csv` (1.410 linhas)
- + 16 arquivos v1og-v1ol regenerados sem fixture

---

## QA executado

- pytest v1og-v1ot: **50/50 PASS**
- v1ot real: `missing=0, high_qc_fails=0, status=TEMPORAL_RECOVERY_FAIL_CLOSED`
- `git diff --cached --name-only`: **vazio**
- Guardrails 37 CSVs: **OK** (1 falso positivo: "local_runs" em texto explicativo de QC)

---

## Guardrails confirmados

- Sem path absoluto Windows em outputs versionáveis ✓
- `local_runs`: falso positivo no QC explicativo — sem referência a path real ✓
- `can_train_model,true`: 0 ✓
- `can_create_operational_label,true`: 0 ✓
- `ground_truth,true`: 0 ✓
- DINO não cria label/target ✓
- `can_unlock_temporal,true` sem PRODUCT_DATE_CONFIRMED: 0 ✓
- Fixture high-severity: 0 ✓

---

## Arquivos recomendados para staging (Grupo A)

### Scripts (16)
```
scripts/protocolo_c/revp_v1og_rec_patch_provenance_graph_builder.py
scripts/protocolo_c/revp_v1og_v1ol_common.py
scripts/protocolo_c/revp_v1oh_local_sentinel_asset_metadata_scanner.py
scripts/protocolo_c/revp_v1oi_sentinel_product_id_mgrs_date_resolver.py
scripts/protocolo_c/revp_v1oj_rec_patch_scene_date_adjudication_v2.py
scripts/protocolo_c/revp_v1ok_rec_event_patch_temporal_rematch_v2.py
scripts/protocolo_c/revp_v1ol_rec_c3_plus_dino_recheck_after_provenance_recovery.py
scripts/protocolo_c/revp_v1om_recife_sentinel_sidecar_discovery.py
scripts/protocolo_c/revp_v1om_v1or_common.py
scripts/protocolo_c/revp_v1on_sentinel_product_date_parser.py
scripts/protocolo_c/revp_v1oo_recife_patch_scene_date_resolver_v3.py
scripts/protocolo_c/revp_v1op_recife_event_patch_temporal_adjudication_v3.py
scripts/protocolo_c/revp_v1oq_recife_c3_c4_dino_recheck_after_scene_date_v3.py
scripts/protocolo_c/revp_v1or_scene_date_recovery_v3_bundle.py
scripts/protocolo_c/revp_v1os_fixture_contamination_audit.py
scripts/protocolo_c/revp_v1ot_scene_date_recovery_final_audit_bundle.py
```

### Testes (8)
```
tests/test_revp_v1og_rec_patch_provenance_graph_builder.py
tests/test_revp_v1oh_local_sentinel_asset_metadata_scanner.py
tests/test_revp_v1oi_sentinel_product_id_mgrs_date_resolver.py
tests/test_revp_v1oj_rec_patch_scene_date_adjudication_v2.py
tests/test_revp_v1ok_rec_event_patch_temporal_rematch_v2.py
tests/test_revp_v1ol_rec_c3_plus_dino_recheck_after_provenance_recovery.py
tests/test_revp_v1om_v1or_scene_date_recovery_v3.py
tests/test_revp_v1ot_scene_date_recovery_final_audit_bundle.py
```

### Documentação (15)
```
docs/metodologia_cientifica/protocolo_c_recife_grafo_linhagem_patch_v1og.md
docs/metodologia_cientifica/protocolo_c_recife_scan_metadata_assets_locais_v1oh.md
docs/metodologia_cientifica/protocolo_c_recife_resolucao_product_id_mgrs_v1oi.md
docs/metodologia_cientifica/protocolo_c_recife_adjudicacao_scene_date_v2_v1oj.md
docs/metodologia_cientifica/protocolo_c_recife_rematch_temporal_v2_v1ok.md
docs/metodologia_cientifica/protocolo_c_recife_linhagem_sentinel_v1og_v1ol.md
docs/metodologia_cientifica/protocolo_c_revp_v1om_sentinel_sidecar_discovery.md
docs/metodologia_cientifica/protocolo_c_revp_v1on_sentinel_product_date_parser.md
docs/metodologia_cientifica/protocolo_c_revp_v1oo_patch_scene_date_resolver_v3.md
docs/metodologia_cientifica/protocolo_c_revp_v1op_temporal_adjudication_v3.md
docs/metodologia_cientifica/protocolo_c_revp_v1oq_c3_c4_dino_recheck_after_v3.md
docs/metodologia_cientifica/revp_v1om_v1or_scene_date_recovery_v3.md
docs/metodologia_cientifica/revp_v1os_fixture_contamination_audit.md
docs/metodologia_cientifica/revp_v1ot_scene_date_recovery_final_audit_bundle.md
docs/metodologia_cientifica/revp_v1og_v1ot_commit_readiness.md
```

### Datasets v1om-v1ot (18 CSV + 18 schemas)
```
datasets/recife_sentinel_sidecar_discovery_v1om.csv
datasets/recife_sentinel_sidecar_discovery_summary_v1om.csv
datasets/recife_sentinel_product_date_candidates_v1on.csv
datasets/recife_sentinel_product_date_summary_v1on.csv
datasets/recife_patch_scene_date_resolved_v3_v1oo.csv
datasets/recife_patch_scene_date_resolution_summary_v3_v1oo.csv
datasets/recife_event_patch_temporal_adjudication_v3_v1op.csv
datasets/recife_temporal_unlock_summary_v3_v1op.csv
datasets/recife_c3_plus_recheck_after_scene_date_v3_v1oq.csv
datasets/recife_c4_status_after_scene_date_v3_v1oq.csv
datasets/recife_dino_review_queue_after_scene_date_v3_v1oq.csv
datasets/recife_scene_date_recovery_v3_master_summary_v1or.csv
datasets/recife_scene_date_recovery_v3_blocking_reasons_v1or.csv
datasets/recife_fixture_contamination_audit_v1os.csv
datasets/recife_fixture_contamination_summary_v1os.csv
datasets/recife_scene_date_recovery_final_manifest_v1ot.csv
datasets/recife_scene_date_recovery_final_quality_checks_v1ot.csv
datasets/recife_scene_date_recovery_final_scientific_summary_v1ot.csv
```
Schemas correspondentes em `datasets/schemas/` (18 arquivos `*_v1om_*` até `*_v1ot_*`).

### Datasets regenerados/limpos v1og-v1ol (19 CSV + 17 schemas)
```
datasets/recife_patch_provenance_graph_registry.csv
datasets/recife_patch_alias_resolution_matrix.csv
datasets/recife_patch_provenance_breakpoint_matrix.csv
datasets/recife_local_sentinel_asset_metadata_inventory.csv
datasets/recife_local_sentinel_asset_date_candidate_registry.csv
datasets/recife_local_asset_scan_summary.csv
datasets/recife_sentinel_product_date_resolution_registry.csv
datasets/recife_sentinel_product_date_confidence_matrix.csv
datasets/recife_patch_scene_date_adjudication_v2.csv
datasets/recife_patch_scene_date_confirmed_registry.csv
datasets/recife_patch_scene_date_unresolved_registry.csv
datasets/recife_event_patch_temporal_rematch_v2_registry.csv
datasets/recife_event_patch_temporal_review_queue_v2.csv
datasets/recife_temporal_unlock_summary_v2.csv
datasets/recife_c3_plus_recheck_after_provenance_recovery.csv
datasets/recife_dino_review_queue_after_provenance_recovery.csv
datasets/recife_c4_status_after_provenance_recovery.csv
datasets/recife_official_positive_candidate_registry.csv
datasets/recife_positive_candidate_date_normalized_registry.csv
```
Schemas correspondentes em `datasets/schemas/` (17 arquivos).

---

## Arquivos explicitamente excluídos (Grupo C)

Modificações antigas não relacionadas a v1og-v1ot:
- `README.md`, `datasets/README.md` — modificações pré-existentes
- `datasets/cicatriz_area_*`, `datasets/consolidated_*`, etc. — bloco anterior
- `docs/metodologia_cientifica/protocolo_c_consolidacao_*.md`, etc. — bloco anterior
- `scripts/protocolo_c/revp_v1if_*.py`, `revp_v1ih_*.py`, etc. — bloco anterior

Arquivos de bloco anterior não commitados:
- `scripts/protocolo_c/revp_v1oe_recife_event_patch_temporal_rematch_after_date_recovery.py`
- `tests/test_revp_v1oe_recife_event_patch_temporal_rematch_after_date_recovery.py`
- `datasets/recife_event_patch_temporal_rematch_registry.csv` (v1nw, não v1og-v1ot)

Centenas de datasets unrelated untracked — não relacionados a v1og-v1ot.

---

## Mensagem de commit sugerida

```
Protocolo C Recife: recuperação temporal Sentinel fail-closed v1og-v1ot

Pipeline completa de recuperação de proveniência temporal Sentinel para patches
Recife. Resultado auditável: TEMPORAL_RECOVERY_FAIL_CLOSED.

- v1og-v1ol: grafo proveniência, scan local, resolução produto, adjudicação v2,
  rematch temporal, recheck C3+/C4/DINO
- v1om-v1or: sidecar discovery, parser de datas, resolver v3, adjudicação v3,
  bundle final — todos fail-closed
- v1os: auditoria de contaminação por fixture/test (0 high-severity)
- v1ot: manifest, quality checks (234 checks, 0 HIGH FAILs), sumário científico

Patches avaliados: 2654 | Datas confirmadas: 0 | C3+: 0 | C4: fechado
Testes: 50/50 PASS | Guardrails: ALL_OK
```

---

## Notas de isolamento de testes

Todos os testes v1og-v1ot foram reescritos para usar `tmp_path` + env vars.
Nenhum teste escreve em `datasets/` real. Verificado por teste guarda-chuva
`test_protocol_c_tests_do_not_write_real_datasets`.
