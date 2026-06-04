#!/usr/bin/env python3
"""v1un Recife Human Review Evidence Consolidation Registry."""

import argparse
import csv
import hashlib
import os
from collections import Counter

PROTOCOL_VERSION = "v1un"
EVENT_ID = "REC_2022_05_24_30"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
WRITING_DIR = "docs/writing_support/protocolo_c"
CONFIG_DIR = "configs/protocolo_c"
HUMAN_REVIEW_STATUS = "PREPARED_NOT_OPERATIONAL"
MAX_STATUS = "RECIFE_HUMAN_REVIEW_EVIDENCE_CONSOLIDATED_NON_OPERATIONAL"
FINAL_STATUS = "LOCALITY_ONLY_HUMAN_REVIEW_EVIDENCE_CONSOLIDATED"

CONSOLIDATION_COLUMNS = [
    "consolidation_id", "event_id", "source_scope", "locality_only_candidates",
    "human_review_batches", "aggregation_count", "dominant_hazard_class",
    "contextual_evidence_strength", "locality_evidence_strength",
    "coordinate_evidence_status", "overlay_status", "ground_reference_status",
    "human_review_status", "operational_status", "final_non_operational_status",
    "can_create_ground_reference", "can_create_training_label", "notes",
]

STRENGTH_COLUMNS = [
    "strength_id", "event_id", "evidence_dimension", "classification",
    "basis", "human_review_status", "patch_bound_truth", "operational_validation",
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "notes",
]

CLAIM_COLUMNS = [
    "claim_id", "event_id", "claim_text", "claim_type", "allowed",
    "required_qualifier", "prohibited_reason", "suggested_safe_alternative",
]

LIMITATION_COLUMNS = [
    "limitation_id", "event_id", "limitation", "status", "evidence_count",
    "impact_on_protocol_c", "human_review_status", "patch_bound_truth",
    "operational_validation", "ground_truth_operational",
    "can_create_ground_reference", "can_create_training_label", "notes",
]

TCC_EXPORT_COLUMNS = [
    "export_id", "event_id", "section", "paragraph_id", "claim_scope",
    "safe_text", "uses_only_public_redacted_evidence", "human_review_status",
    "patch_bound_truth", "operational_validation", "can_create_ground_reference",
    "can_create_training_label",
]

STATUS_COLUMNS = [
    "event_id", "previous_status", "new_status", "evidence_level",
    "can_advance_to_overlay", "can_advance_to_ground_reference",
    "can_create_training_label", "protocol_b_status", "required_next_action",
    "notes",
]

BLOCKER_COLUMNS = [
    "blocker_id", "event_id", "blocker", "status", "evidence_count",
    "ground_truth_operational", "patch_bound_truth", "operational_validation",
    "can_create_ground_reference", "can_create_training_label", "notes",
]

NEXT_ACTION_COLUMNS = [
    "action_id", "event_id", "action_type", "priority", "description",
    "target", "status", "notes",
]

MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]

V1UN_ARTIFACTS = [
    "configs/protocolo_c/v1un_recife_evidence_strength_policy.yaml",
    "configs/protocolo_c/v1un_recife_safe_claims_policy.yaml",
    "configs/protocolo_c/v1un_recife_limitations_policy.yaml",
    "configs/protocolo_c/v1un_recife_tcc_export_policy.yaml",
    "configs/protocolo_c/v1un_protocol_c_status_policy.yaml",
    "datasets/protocolo_c/v1un_recife_human_review_evidence_consolidation.csv",
    "datasets/protocolo_c/v1un_recife_evidence_strength_registry.csv",
    "datasets/protocolo_c/v1un_recife_safe_claims_registry.csv",
    "datasets/protocolo_c/v1un_recife_prohibited_claims_registry.csv",
    "datasets/protocolo_c/v1un_recife_limitations_matrix.csv",
    "datasets/protocolo_c/v1un_recife_tcc_evidence_export_registry.csv",
    "datasets/protocolo_c/v1un_recife_protocol_c_status_registry.csv",
    "datasets/protocolo_c/v1un_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v1un_next_actions_registry.csv",
    "datasets/protocolo_c/v1un_versionable_artifacts_manifest.csv",
    "docs/metodologia_cientifica/protocolo_c_v1un_recife_human_review_evidence_consolidation.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v1un_recife_human_review_evidence_consolidation.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v1un.md",
    "docs/metodologia_cientifica/protocolo_c_recife_safe_claims_for_tcc_v1un.md",
    "docs/writing_support/protocolo_c/v1un_recife_tcc_paragraphs.md",
]


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bool_text(value):
    return "true" if bool(value) else "false"


def int_value(value):
    try:
        return int(value or 0)
    except ValueError:
        return 0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_path(path):
    return path.replace("\\", "/")


def v1um_summary():
    evidence = load_csv(os.path.join(DATASET_DIR, "v1um_recife_redacted_evidence_package_registry.csv"))
    batches = load_csv(os.path.join(DATASET_DIR, "v1um_recife_human_review_batch_registry.csv"))
    aggregation = load_csv(os.path.join(DATASET_DIR, "v1um_recife_neighborhood_signal_aggregation.csv"))
    readiness = load_csv(os.path.join(DATASET_DIR, "v1um_recife_non_overlay_readiness_matrix.csv"))
    hazards = Counter(r.get("hazard_class", "") for r in evidence)
    samples = load_csv(os.path.join(DATASET_DIR, "v1um_recife_locality_candidate_sample_registry.csv"))
    return {
        "evidence": evidence,
        "batches": batches,
        "aggregation": aggregation,
        "readiness": readiness,
        "hazards": hazards,
        "samples": samples,
        "locality_only_candidates": len(evidence),
        "human_review_batches": len(batches),
        "aggregation_count": len(aggregation),
        "dominant_hazard_class": hazards.most_common(1)[0][0] if hazards else "NONE",
        "contextual_evidence_strength": "STRONG" if len(evidence) >= 1000 and len(batches) >= 3 else "MODERATE",
        "locality_evidence_strength": "STRONG" if evidence else "ABSENT",
    }


def run_human_review_evidence_consolidator(out_path=None):
    s = v1um_summary()
    rows = [{
        "consolidation_id": "CONSOL_v1un_0000",
        "event_id": EVENT_ID,
        "source_scope": "v1um_human_review_locality_only_outputs",
        "locality_only_candidates": str(s["locality_only_candidates"]),
        "human_review_batches": str(s["human_review_batches"]),
        "aggregation_count": str(s["aggregation_count"]),
        "dominant_hazard_class": s["dominant_hazard_class"],
        "contextual_evidence_strength": s["contextual_evidence_strength"],
        "locality_evidence_strength": s["locality_evidence_strength"],
        "coordinate_evidence_status": "ABSENT",
        "overlay_status": "BLOCKED",
        "ground_reference_status": "NOT_CREATED_BLOCKED",
        "human_review_status": HUMAN_REVIEW_STATUS,
        "operational_status": "NOT_OPERATIONAL",
        "final_non_operational_status": MAX_STATUS,
        "can_create_ground_reference": "false",
        "can_create_training_label": "false",
        "notes": "Human Review Evidence consolidated as Locality-Only Evidence and Non-Overlay Evidence.",
    }]
    out_path = out_path or os.path.join(DATASET_DIR, "v1un_recife_human_review_evidence_consolidation.csv")
    write_csv(out_path, CONSOLIDATION_COLUMNS, rows)
    print(f"[v1un consolidation] rows={len(rows)} -> {out_path}")
    return rows


def run_evidence_strength_classifier(out_path=None):
    s = v1um_summary()
    dims = [
        ("temporal_window_support", "STRONG", "v1uk/v1um event-window candidate rows"),
        ("official_source_support", "STRONG", "public official Recife source lineage from prior registries"),
        ("hazard_textual_support", "MODERATE", f"dominant_hazard_class={s['dominant_hazard_class']}"),
        ("locality_support", "STRONG", f"locality_only_candidates={s['locality_only_candidates']}"),
        ("coordinate_support", "BLOCKED", "no observed coordinate in v1um locality-only evidence"),
        ("overlay_support", "BLOCKED", "Non-Overlay Evidence only"),
        ("supervisor_review_support", "WEAK", "Human Review prepared but not completed"),
        ("ground_reference_support", "BLOCKED", "coordinate and overlay prerequisites absent"),
        ("tcc_contextual_use_support", "STRONG", "safe contextual writing support generated"),
    ]
    rows = []
    for idx, (dim, classification, basis) in enumerate(dims):
        rows.append({
            "strength_id": f"STRENGTH_v1un_{idx:04d}",
            "event_id": EVENT_ID,
            "evidence_dimension": dim,
            "classification": classification,
            "basis": basis,
            "human_review_status": HUMAN_REVIEW_STATUS,
            "patch_bound_truth": "false",
            "operational_validation": "false",
            "ground_truth_operational": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "Evidence strength is non-operational and constrained to Human Review Decision Support.",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1un_recife_evidence_strength_registry.csv")
    write_csv(out_path, STRENGTH_COLUMNS, rows)
    print(f"[v1un strength] rows={len(rows)} -> {out_path}")
    return rows


def run_safe_claims_generator(safe_path=None, prohibited_path=None):
    safe_claims = [
        ("SAFE_v1un_0000", "REC_2022 has public official Locality-Only Evidence compatible with the analyzed temporal window.", "Must state that it is non-operational Human Review Evidence."),
        ("SAFE_v1un_0001", "The records do not provide observed coordinate evidence sufficient for overlay.", "Must retain the Non-Overlay Evidence qualifier."),
        ("SAFE_v1un_0002", "The evidence supports contextual discussion, not operational validation.", "Must keep contextual/non-operational wording."),
        ("SAFE_v1un_0003", "Human Review was structured in auditable Human Review Packages.", "Must state supervisor review remains pending."),
        ("SAFE_v1un_0004", "Protocol C remains blocked for ground reference creation.", "Must mention missing coordinate and overlay prerequisites."),
    ]
    prohibited = [
        ("PROHIB_v1un_0000", "the patch contains observed flooding.", "patch-level observation was not established", "REC_2022 has Locality-Only Evidence for contextual discussion."),
        ("PROHIB_v1un_0001", "the event was validated at patch level.", "patch-bound truth is false", "Human Review Evidence is consolidated as non-operational."),
        ("PROHIB_v1un_0002", "ground truth was created.", "ground_truth_operational=false", "No operational ground reference was created."),
        ("PROHIB_v1un_0003", "a training label was created.", "can_create_training_label=false", "No training target was created."),
        ("PROHIB_v1un_0004", "Protocol B can be reopened.", "can_reopen_protocol_b=false", "Protocol C remains constrained to Human Review Evidence."),
        ("PROHIB_v1un_0005", "Locality-Only Evidence allows overlay.", "coordinate_support=BLOCKED", "Locality-Only Evidence supports non-overlay contextual review."),
    ]
    safe_rows = [{
        "claim_id": cid,
        "event_id": EVENT_ID,
        "claim_text": text,
        "claim_type": "SAFE_CLAIM",
        "allowed": "true",
        "required_qualifier": qualifier,
        "prohibited_reason": "",
        "suggested_safe_alternative": text,
    } for cid, text, qualifier in safe_claims]
    prohibited_rows = [{
        "claim_id": cid,
        "event_id": EVENT_ID,
        "claim_text": text,
        "claim_type": "PROHIBITED_CLAIM",
        "allowed": "false",
        "required_qualifier": "Do not use as a project finding.",
        "prohibited_reason": reason,
        "suggested_safe_alternative": alternative,
    } for cid, text, reason, alternative in prohibited]
    safe_path = safe_path or os.path.join(DATASET_DIR, "v1un_recife_safe_claims_registry.csv")
    prohibited_path = prohibited_path or os.path.join(DATASET_DIR, "v1un_recife_prohibited_claims_registry.csv")
    write_csv(safe_path, CLAIM_COLUMNS, safe_rows)
    write_csv(prohibited_path, CLAIM_COLUMNS, prohibited_rows)
    print(f"[v1un claims] safe={len(safe_rows)} prohibited={len(prohibited_rows)}")
    return safe_rows, prohibited_rows


def run_limitations_matrix_builder(out_path=None):
    s = v1um_summary()
    hazard_ambiguous = sum(1 for r in s["evidence"] if r.get("hazard_class") == "CIVIL_DEFENSE_GENERIC")
    limitations = [
        ("no_coordinates", s["locality_only_candidates"], "blocks overlay and ground reference creation"),
        ("locality_only", s["locality_only_candidates"], "keeps evidence as textual locality support"),
        ("no_overlay", s["locality_only_candidates"], "prevents patch-bound interpretation"),
        ("no_ground_reference", s["locality_only_candidates"], "prevents operational reference construction"),
        ("no_training_label", s["locality_only_candidates"], "prevents supervised target construction"),
        ("no_patch_bound_truth", s["locality_only_candidates"], "keeps findings non-operational"),
        ("no_operational_validation", s["locality_only_candidates"], "limits use to contextual discussion"),
        ("sensitive_values_redacted", s["locality_only_candidates"], "public outputs remain hash/flag based"),
        ("hazard_textual_ambiguity", hazard_ambiguous, "requires Human Review Decision Support"),
        ("civil_defense_generic_dominance", hazard_ambiguous, "reduces hazard specificity"),
    ]
    rows = []
    for idx, (name, count, impact) in enumerate(limitations):
        rows.append({
            "limitation_id": f"LIMIT_v1un_{idx:04d}",
            "event_id": EVENT_ID,
            "limitation": name,
            "status": "ACTIVE" if count else "INACTIVE",
            "evidence_count": str(count),
            "impact_on_protocol_c": impact,
            "human_review_status": HUMAN_REVIEW_STATUS,
            "patch_bound_truth": "false",
            "operational_validation": "false",
            "ground_truth_operational": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "Limitation preserved in v1un Human Review Consolidation.",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1un_recife_limitations_matrix.csv")
    write_csv(out_path, LIMITATION_COLUMNS, rows)
    print(f"[v1un limitations] rows={len(rows)} -> {out_path}")
    return rows


def tcc_paragraphs():
    return [
        ("public_official_evidence", "O REV-P registra evidencia publica oficial para Recife a partir de artefatos CKAN e registries derivados do Protocolo C. Essa evidencia foi consolidada como Human Review Evidence, com uso publico restrito a contagens, hashes, flags e sinteses redigidas."),
        ("locality_only_records", "Os registros consolidados em REC_2022 sao Locality-Only Evidence: apresentam data, sinal textual de hazard e localidade textual redigida, mas nao fornecem coordenada observada suficiente para analise espacial por overlay."),
        ("temporal_window_hazard", "A matriz v1un documenta que a janela temporal de 2022-05-24 a 2022-05-30 possui registros de atendimento compativeis com o evento analisado. A dominancia de sinais genericos de Defesa Civil exige qualificacao metodologica e Human Review Decision Support."),
        ("structured_human_review", "A Revisao Humana e estruturada pelo protocolo por meio de pacotes auditaveis, amostras estratificadas, ranking textual de hazard, agregacoes por localidade/data e matriz de decisao nao-operacional."),
        ("why_not_ground_truth", "A evidencia nao cria patch-bound truth porque permanece sem coordenada observada, sem overlay e sem revisao supervisora concluida. Portanto, nao cria ground truth operacional, ground reference ou label."),
        ("methodological_utility", "Mesmo com esses bloqueios, a consolidacao e util para o REV-P porque sustenta discussao contextual cientifica sobre evidencia documental/administrativa oficial e explicita o limite entre contexto territorial, ocorrencia documentada e ground reference."),
    ]


def run_tcc_text_evidence_exporter(out_path=None, doc_path=None):
    paragraphs = tcc_paragraphs()
    rows = []
    for idx, (section, text) in enumerate(paragraphs):
        rows.append({
            "export_id": f"TCCEXP_v1un_{idx:04d}",
            "event_id": EVENT_ID,
            "section": section,
            "paragraph_id": f"PAR_{idx+1:02d}",
            "claim_scope": "SAFE_CONTEXTUAL_HUMAN_REVIEW_EVIDENCE",
            "safe_text": text,
            "uses_only_public_redacted_evidence": "true",
            "human_review_status": HUMAN_REVIEW_STATUS,
            "patch_bound_truth": "false",
            "operational_validation": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1un_recife_tcc_evidence_export_registry.csv")
    doc_path = doc_path or os.path.join(WRITING_DIR, "v1un_recife_tcc_paragraphs.md")
    write_csv(out_path, TCC_EXPORT_COLUMNS, rows)
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
    lines = ["# v1un Recife TCC Paragraphs", ""]
    for _section, text in paragraphs:
        lines += [text, ""]
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[v1un tcc export] paragraphs={len(rows)} -> {doc_path}")
    return rows


def run_protocol_c_status_updater(out_path=None):
    rows = [{
        "event_id": EVENT_ID,
        "previous_status": "RECIFE_LOCALITY_ONLY_HUMAN_REVIEW_CANDIDATE",
        "new_status": FINAL_STATUS,
        "evidence_level": "STRONG_CONTEXTUAL_LOCALITY_ONLY_NON_OPERATIONAL",
        "can_advance_to_overlay": "false",
        "can_advance_to_ground_reference": "false",
        "can_create_training_label": "false",
        "protocol_b_status": "CLOSED_NOT_REOPENED",
        "required_next_action": "v1uo - Protocolo C Recife Scientific Writing Integration",
        "notes": "REC_2022 consolidated as Human Review Evidence; no overlay, no coordinate inference, no operational validation.",
    }]
    out_path = out_path or os.path.join(DATASET_DIR, "v1un_recife_protocol_c_status_registry.csv")
    write_csv(out_path, STATUS_COLUMNS, rows)
    print(f"[v1un protocol status] rows={len(rows)} -> {out_path}")
    return rows


def write_policy_configs():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    policies = {
        "v1un_recife_evidence_strength_policy.yaml": [
            "protocol_version: v1un",
            "status_max: RECIFE_HUMAN_REVIEW_EVIDENCE_CONSOLIDATED_NON_OPERATIONAL",
            "coordinate_support: BLOCKED",
            "overlay_support: BLOCKED",
            "ground_reference_support: BLOCKED",
        ],
        "v1un_recife_safe_claims_policy.yaml": [
            "protocol_version: v1un",
            "allowed_scope: Human Review Evidence and contextual scientific discussion",
            "operational_claims_allowed: false",
            "patch_bound_truth: false",
        ],
        "v1un_recife_limitations_policy.yaml": [
            "protocol_version: v1un",
            "required_limitations: [no_coordinates, locality_only, no_overlay, no_ground_reference, no_training_label]",
        ],
        "v1un_recife_tcc_export_policy.yaml": [
            "protocol_version: v1un",
            "writing_support_scope: safe contextual paragraphs",
            "raw_sensitive_values_allowed: false",
            "absolute_paths_allowed: false",
        ],
        "v1un_protocol_c_status_policy.yaml": [
            "protocol_version: v1un",
            "final_status: LOCALITY_ONLY_HUMAN_REVIEW_EVIDENCE_CONSOLIDATED",
            "can_reopen_protocol_b: false",
            "operational_validation: false",
        ],
    }
    for name, lines in policies.items():
        with open(os.path.join(CONFIG_DIR, name), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def run_ground_reference_blocker_matrix(out_path=None):
    s = v1um_summary()
    blockers = [
        ("no_coordinates", s["locality_only_candidates"]),
        ("locality_only", s["locality_only_candidates"]),
        ("no_overlay", s["locality_only_candidates"]),
        ("no_ground_reference", s["locality_only_candidates"]),
        ("no_training_label", s["locality_only_candidates"]),
        ("no_patch_bound_truth", s["locality_only_candidates"]),
        ("no_operational_validation", s["locality_only_candidates"]),
    ]
    rows = []
    for idx, (blocker, count) in enumerate(blockers):
        rows.append({
            "blocker_id": f"BLOCK_v1un_{idx:04d}",
            "event_id": EVENT_ID,
            "blocker": blocker,
            "status": "ACTIVE",
            "evidence_count": str(count),
            "ground_truth_operational": "false",
            "patch_bound_truth": "false",
            "operational_validation": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "Blocker preserved after Human Review Evidence Consolidation.",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1un_ground_reference_blocker_matrix.csv")
    write_csv(out_path, BLOCKER_COLUMNS, rows)
    return rows


def run_completion_report():
    write_policy_configs()
    run_ground_reference_blocker_matrix()
    s = v1um_summary()
    next_action = "v1uo - Protocolo C Recife Scientific Writing Integration"
    action_rows = [{
        "action_id": "ACT_v1un_0000",
        "event_id": EVENT_ID,
        "action_type": "SCIENTIFIC_WRITING_INTEGRATION",
        "priority": "1",
        "description": next_action,
        "target": "REC_2022 Human Review Evidence",
        "status": "PENDING",
        "notes": "Do not implement v1uo in v1un.",
    }]
    write_csv(os.path.join(DATASET_DIR, "v1un_next_actions_registry.csv"), NEXT_ACTION_COLUMNS, action_rows)
    manifest = []
    for idx, path in enumerate(V1UN_ARTIFACTS):
        exists = os.path.exists(path)
        manifest.append({
            "artifact_id": f"ART_v1un_{idx:04d}",
            "artifact_path": artifact_path(path),
            "artifact_type": "config" if path.startswith("configs/") else "doc" if path.startswith("docs/") else "dataset",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(path)[:16] if exists else "MISSING",
            "file_size_bytes": str(os.path.getsize(path) if exists else 0),
            "is_versionable": bool_text(exists),
            "reason": "Safe v1un Human Review Evidence artifact" if exists else "File not found",
        })
    write_csv(os.path.join(DATASET_DIR, "v1un_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)
    os.makedirs(DOCS_DIR, exist_ok=True)
    methodology = [
        "# Protocolo C v1un - Recife Human Review Evidence Consolidation",
        "",
        "O protocolo consolida Human Review Evidence para REC_2022 como Locality-Only Evidence e Non-Overlay Evidence.",
        "",
        f"- event_id: {EVENT_ID}",
        f"- status_maximo: {MAX_STATUS}",
        f"- candidatos locality-only: {s['locality_only_candidates']}",
        f"- batches Human Review: {s['human_review_batches']}",
        f"- agregacoes localidade/data: {s['aggregation_count']}",
        "- patch_bound_truth=false",
        "- operational_validation=false",
        "- ground_truth_operational=false",
        "- can_create_ground_reference=false",
        "- can_create_training_label=false",
    ]
    report = [
        "# Relatorio v1un - Recife Human Review Evidence Consolidation",
        "",
        "O REV-P registra a consolidacao final nao-operacional de REC_2022 no Protocolo C.",
        "",
        "## Evidencia Consolidada",
        f"- Locality-Only Evidence: {s['locality_only_candidates']} candidatos.",
        f"- Human Review Package: {s['human_review_batches']} batches auditaveis.",
        f"- Contextual Evidence: {s['aggregation_count']} agregacoes por localidade/data.",
        f"- Hazard dominante: {s['dominant_hazard_class']}.",
        "",
        "## Forca da Evidencia",
        "- temporal_window_support: STRONG",
        "- official_source_support: STRONG",
        "- hazard_textual_support: MODERATE",
        "- locality_support: STRONG",
        "- coordinate_support: BLOCKED",
        "- overlay_support: BLOCKED",
        "- ground_reference_support: BLOCKED",
        "- tcc_contextual_use_support: STRONG",
        "",
        "## Uso Cientifico Seguro",
        "- A Revisao Humana estruturada permite descrever evidencia documental/administrativa locality-only.",
        "- A matriz documenta que a evidencia sustenta discussao contextual, sem validacao operacional.",
        "- O TCC pode usar os paragrafos seguros gerados em writing_support.",
        "",
        "## Proibicoes Preservadas",
        "- Nao ha overlay.",
        "- Nao ha ground reference.",
        "- Nao ha label.",
        "- Nao ha coordenada inventada.",
        "",
        "## Proxima Etapa",
        f"- {next_action}",
    ]
    status = [
        "# Status Atual - Protocolo C v1un",
        "",
        f"event_id={EVENT_ID}",
        f"final_status={FINAL_STATUS}",
        f"locality_only_candidates={s['locality_only_candidates']}",
        f"human_review_batches={s['human_review_batches']}",
        f"aggregation_count={s['aggregation_count']}",
        f"dominant_hazard_class={s['dominant_hazard_class']}",
        "evidence_level=STRONG_CONTEXTUAL_LOCALITY_ONLY_NON_OPERATIONAL",
        "coordinate_support=BLOCKED",
        "overlay_support=BLOCKED",
        "ground_reference_support=BLOCKED",
        "ground_truth_operational=false",
        "patch_bound_truth=false",
        "operational_validation=false",
        "can_create_ground_reference=false",
        "can_create_training_label=false",
        "can_reopen_protocol_b=false",
        "dino_usage=SUPPORT_ONLY",
        "no_overlay_executed=true",
        "no_coordinates_invented=true",
        "human_review_package_created=true",
        f"human_review_status={HUMAN_REVIEW_STATUS}",
        "supervisor_review_completed=false",
        f"max_status={MAX_STATUS}",
        f"next_action={next_action}",
    ]
    safe_doc = [
        "# Recife Safe Claims for TCC - v1un",
        "",
        "## Allowed Claims",
        "- REC_2022 has public official Locality-Only Evidence compatible with the analyzed temporal window.",
        "- The records support contextual scientific discussion and Human Review Decision Support.",
        "- The protocol remains blocked for overlay and ground reference construction.",
        "",
        "## Prohibited Claims",
        "- Do not state patch-level observation.",
        "- Do not state operational validation.",
        "- Do not state supervised training readiness.",
    ]
    files = {
        "protocolo_c_v1un_recife_human_review_evidence_consolidation.md": methodology,
        "protocolo_c_relatorio_v1un_recife_human_review_evidence_consolidation.md": report,
        "protocolo_c_status_atual_v1un.md": status,
        "protocolo_c_recife_safe_claims_for_tcc_v1un.md": safe_doc,
    }
    for name, lines in files.items():
        with open(os.path.join(DOCS_DIR, name), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    print(f"[v1un completion] next_action={next_action}")
    return {
        "locality_only_candidates": s["locality_only_candidates"],
        "human_review_batches": s["human_review_batches"],
        "aggregation_count": s["aggregation_count"],
        "dominant_hazard_class": s["dominant_hazard_class"],
        "next_action": next_action,
    }


def simple_main(fn):
    parser = argparse.ArgumentParser()
    parser.parse_args()
    fn()
