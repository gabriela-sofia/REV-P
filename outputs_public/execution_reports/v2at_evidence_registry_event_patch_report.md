# v2at - Evidence Registry + Event-Patch Package Engine

## 1. Objetivo
Transformar o REV-P de "patches + embeddings + registries fail-closed" para um sistema
explicito de evidencia observacional: catalogo de fontes externas, observacoes, pacotes
evento-patch com tipagem de fenomeno, janela temporal, forca de evidencia, bloqueios, score
explicavel e decisao de promocao C1/C2/C3/C4 (sempre candidata, nunca label).

Esta etapa NAO treina modelo, NAO cria label binario/operacional, NAO declara ground truth e
NAO transforma o DINOv2 em detector. O DINOv2 permanece apoio de revisao (similaridade,
vizinhanca, PCA, outliers, medoids), nunca validador fisico.

## 2. Entradas usadas
- `protocolo_c/v1us_event_patch_candidate_registry.csv`: 1
- `protocolo_c/v1us_dino_review_support_attachment.csv`: 1
- `protocolo_c/v2aa_sentinel_date_confidence_audit.csv`: 1
- `protocolo_c/v2ab_event_patch_package_validation.csv`: 1
- `protocolo_c/v2bm_cross_region_candidate_registry.csv`: 1
- `protocolo_c/v2bm_cross_region_evidence_scorecard.csv`: 1
- `ground_reference_event_registry.csv`: 1
- `external_evidence_registry.csv`: 1

## 3. Saidas geradas
- `datasets/v2at_external_evidence_source_catalog.csv`: 1
- `datasets/v2at_evidence_observation_registry.csv`: 1
- `datasets/v2at_event_patch_package_registry.csv`: 1
- `datasets/v2at_promotion_gate_decision_audit.csv`: 1
- `datasets/v2at_reviewer_queue_seed.csv`: 1
- `datasets/v2at_operational_label_blocklist.csv`: 1
- `outputs_public/execution_reports/v2at_evidence_registry_event_patch_report.md`: 1
- `outputs_public/execution_reports/v2at_evidence_registry_event_patch_summary.json`: 1
- `outputs_public/logs_summary/v2at_evidence_registry_event_patch.log`: 1
- `outputs_public/logs_summary/v2at_evidence_registry_event_patch.txt`: 1
- `outputs_public/execution_reports/v2at_artifact_index_supplement.md`: 1

## 4. Contagens
- Fontes canonicas no catalogo: **23**
- Observacoes de evidencia: **21**
- Pacotes evento-patch: **172**
- Checagens de gate: **2580**
- Itens na fila de revisao: **172**
- Entradas na blocklist de label: **181**

### 4.1 Por regiao
- `Curitiba`: 1
- `Petropolis`: 116
- `Recife`: 55

### 4.2 Por hazard_type
- `mass_movement`: 116
- `unknown_hazard`: 1
- `urban_flood`: 55

### 4.3 Por source_class (observacoes)
- `context_low`: 2
- `official_geoinfo`: 4
- `official_geological`: 11
- `official_hydromet`: 3
- `operational_mapping`: 1

## 5. Distribuicao de promocao e uso permitido
### promotion_decision
- `C3_CANDIDATE_REFERENCE_HOLD_FOR_OVERLAY`: 55
- `C3_SECONDARY_EVALUATION_HOLD_FOR_OVERLAY`: 116
- `REJECTED_EVENT_REGISTRY_MISSING`: 1

### allowed_use
- `candidate_reference`: 55
- `rejected_context_only`: 1
- `secondary_evaluation_candidate`: 116

- review_only: **0**
- candidate_reference: **55**
- C4_CANDIDATE: **0** (somente candidato; nunca label final)
- bloqueados (com blocking_reason): **172**

## 6. Principais blocking_reason
- `NO_PATCH_EVENT_OVERLAY_GEOMETRY`: 171
- `EVENT_REGISTRY_MISSING_OR_UNTYPED`: 1

## 7. Confirmacoes metodologicas explicitas
- Nenhum label operacional foi criado (`can_create_operational_labels=false`; GATE_15 PASS em todos os pacotes).
- Nenhum modelo foi treinado (`can_train_model=false`).
- Nenhum ground truth foi declarado; ausencia de evidencia nunca virou negativo.
- Benchmark externo nunca virou verdade local (GATE_12); quickview nunca promoveu sozinho (GATE_11).
- Patch boundary nao e geometria de evento; inventario de desastre nao e geometria de desastre.
- C4 aparece apenas como **candidate**, nunca como label final.

## 8. Interpretacao metodologica
O REV-P avanca de review-only para um sistema de evidencia observacional auditavel, mas
continua bloqueado para treino supervisionado. A conclusao correta e:

**EVIDENCE_SYSTEM_READY_FOR_HUMAN_REVIEW_NOT_FOR_TRAINING**

Ou seja: existe agora a infraestrutura explicita (fontes -> observacoes -> pacotes -> gates ->
fila -> blocklist) para que revisao humana, geometria, overlay e evidencia temporal/espacial
forte sejam fechados antes de qualquer referencia operacional.
