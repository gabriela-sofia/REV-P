# Protocolo C — v1ui: Public Official Observed Artifact Discovery and Geometry Acquisition

## Objetivo

Buscar programaticamente artefatos publicos oficiais que possam conter
evidencia observacional de evento (vetores, tabelas georreferenciadas,
anexos tecnicos, servicos ArcGIS/GeoServer, portais de dados abertos).

## Correcao Metodologica

A partir de v1ui, pedidos formais (v1ug/v1uh) sao reclassificados como
caminho secundario/legado. A rota principal e:
PUBLIC_OFFICIAL_DISCOVERY -> PUBLIC_ARTIFACT_DOWNLOAD -> OBSERVED_GEOMETRY_CANDIDATE_AUDIT -> SUPERVISOR_REVIEW_READY

formal_request_path=LEGACY_SECONDARY_ONLY

## Guardrails

- ground_truth_operational=false
- can_create_ground_reference=false
- can_create_training_label=false
- no_overlay_executed=true
- no_coordinates_invented=true
- public_artifact_discovery=true

## Pipeline

1. public_source_discovery — registra fontes publicas oficiais
2. public_portal_crawler — extrai links de artefatos
3. public_artifact_downloader — baixa com allowlist/limites
4. arcgis_geoserver_resolver — registra metadata de servicos
5. public_artifact_inventory — inventaria conteudo
6. observed_geometry_extractor — classifica candidatos
7. event_geometry_candidate_audit — 14 gates (G12-G14 sempre FAIL)
8. public_evidence_gate_delta — compara v1uh vs v1ui
9. supervisor_review_prequeue — fila de revisao humana
10. completion_report — relatorio, manifest, next actions

## 14 Gates

G01-G11: avaliacao de qualidade do candidato
G12: supervisor_review_pending (sempre FAIL em v1ui)
G13: patch_overlay_not_executed (sempre FAIL em v1ui)
G14: label_forbidden (sempre FAIL em v1ui)

Status maximo: OBSERVED_GEOMETRY_CANDIDATE_READY_FOR_SUPERVISOR_REVIEW
