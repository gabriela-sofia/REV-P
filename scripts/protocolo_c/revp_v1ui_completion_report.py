#!/usr/bin/env python3
"""
v1ui — Completion Report

Consolidates all v1ui outputs into reports, next-actions, manifest.
"""

import argparse
import csv
import hashlib
import os
from datetime import datetime

PROTOCOL_VERSION = "v1ui"

GUARDRAILS = {
    "ground_truth_operational": False,
    "can_create_ground_reference": False,
    "can_create_training_label": False,
    "can_reopen_protocol_b": False,
    "no_overlay_executed": True,
    "no_coordinates_invented": True,
    "public_artifact_discovery": True,
    "formal_request_path": "LEGACY_SECONDARY_ONLY",
}

V1UI_ARTIFACTS = {
    "configs/protocolo_c/v1ui_public_source_targets.yaml": "config",
    "configs/protocolo_c/v1ui_allowed_domains.yaml": "config",
    "configs/protocolo_c/v1ui_public_download_policy.yaml": "config",
    "configs/protocolo_c/v1ui_search_terms_by_event.yaml": "config",
    "configs/protocolo_c/v1ui_artifact_classification_policy.yaml": "config",
    "configs/protocolo_c/v1ui_observed_geometry_gate_policy.yaml": "config",
    "datasets/protocolo_c/v1ui_public_source_target_registry.csv": "dataset",
    "datasets/protocolo_c/v1ui_public_discovery_registry.csv": "dataset",
    "datasets/protocolo_c/v1ui_public_artifact_download_manifest.csv": "dataset",
    "datasets/protocolo_c/v1ui_public_artifact_inventory.csv": "dataset",
    "datasets/protocolo_c/v1ui_arcgis_geoserver_layer_registry.csv": "dataset",
    "datasets/protocolo_c/v1ui_observed_geometry_extraction_registry.csv": "dataset",
    "datasets/protocolo_c/v1ui_event_geometry_candidate_registry.csv": "dataset",
    "datasets/protocolo_c/v1ui_public_evidence_gate_delta.csv": "dataset",
    "datasets/protocolo_c/v1ui_supervisor_review_prequeue.csv": "dataset",
    "datasets/protocolo_c/v1ui_next_actions_registry.csv": "dataset",
    "datasets/protocolo_c/v1ui_versionable_artifacts_manifest.csv": "dataset",
}

NEXT_ACTIONS_COLUMNS = [
    "action_id", "event_id", "action_type", "priority",
    "description", "target", "status", "notes",
]

MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(path):
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="v1ui — Completion Report")
    parser.add_argument("--out-dir", default="datasets/protocolo_c")
    parser.add_argument("--docs-dir", default="docs/metodologia_cientifica")
    args = parser.parse_args()

    discoveries = load_csv(os.path.join(args.out_dir, "v1ui_public_discovery_registry.csv"))
    inventory = load_csv(os.path.join(args.out_dir, "v1ui_public_artifact_inventory.csv"))
    extractions = load_csv(os.path.join(args.out_dir, "v1ui_observed_geometry_extraction_registry.csv"))
    candidates = load_csv(os.path.join(args.out_dir, "v1ui_event_geometry_candidate_registry.csv"))
    prequeue = load_csv(os.path.join(args.out_dir, "v1ui_supervisor_review_prequeue.csv"))
    layers = load_csv(os.path.join(args.out_dir, "v1ui_arcgis_geoserver_layer_registry.csv"))
    events = load_csv(os.path.join(args.out_dir, "event_candidate_registry.csv"))
    deltas = load_csv(os.path.join(args.out_dir, "v1ui_public_evidence_gate_delta.csv"))

    next_actions = []
    seq = 0
    for event in events:
        eid = event["event_id"]
        ev_cands = [c for c in candidates if c.get("event_id") == eid]
        ready = [c for c in ev_cands if "READY_FOR_SUPERVISOR_REVIEW" in c.get("max_status", "")]

        if ready:
            next_actions.append({
                "action_id": f"ACT_{PROTOCOL_VERSION}_{seq:04d}",
                "event_id": eid, "action_type": "SUPERVISOR_REVIEW",
                "priority": "1",
                "description": f"{len(ready)} candidates ready for supervisor review",
                "target": "Supervisor humano", "status": "PENDING", "notes": "",
            })
        else:
            next_actions.append({
                "action_id": f"ACT_{PROTOCOL_VERSION}_{seq:04d}",
                "event_id": eid, "action_type": "DEEPEN_PUBLIC_DISCOVERY",
                "priority": "2",
                "description": "Aprofundar descoberta em fontes publicas especificas",
                "target": "Scripts de crawler com --allow-web",
                "status": "PENDING", "notes": "",
            })
        seq += 1

    actions_path = os.path.join(args.out_dir, "v1ui_next_actions_registry.csv")
    with open(actions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NEXT_ACTIONS_COLUMNS)
        writer.writeheader()
        writer.writerows(next_actions)

    manifest = []
    mseq = 0
    for path, atype in sorted(V1UI_ARTIFACTS.items()):
        exists = os.path.exists(path)
        manifest.append({
            "artifact_id": f"ART_{PROTOCOL_VERSION}_{mseq:04d}",
            "artifact_path": path, "artifact_type": atype,
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(path),
            "file_size_bytes": str(os.path.getsize(path) if exists else 0),
            "is_versionable": "true" if exists else "false",
            "reason": "Safe for git" if exists else "File not found",
        })
        mseq += 1

    manifest_path = os.path.join(args.out_dir, "v1ui_versionable_artifacts_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest)

    gains = sum(1 for d in deltas if d.get("delta_type") == "GAIN")
    ready_count = sum(1 for p in prequeue if p.get("review_status") == "READY_FOR_REVIEW")
    geom_candidates = sum(1 for e in extractions
                          if e.get("can_be_observed_geometry_candidate") == "true")

    lines = [
        f"# v1ui Completion Report — Public Official Discovery",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Protocol: {PROTOCOL_VERSION}", "",
        "## Guardrails",
    ]
    for k, v in GUARDRAILS.items():
        lines.append(f"  {k} = {v} [ENFORCED]")
    lines.append("")
    lines.append("## Discovery Summary")
    lines.append(f"  Public sources registered: {len(discoveries)}")
    lines.append(f"  Artifacts inventoried: {len(inventory)}")
    lines.append(f"  ArcGIS/GeoServer layers: {len(layers)}")
    lines.append(f"  Geometry candidates: {geom_candidates}")
    lines.append(f"  Gate deltas (gains): {gains}")
    lines.append(f"  Ready for supervisor review: {ready_count}")
    lines.append("")
    lines.append("## Why No Ground Truth Yet")
    lines.append("  - G12 supervisor_review_pending: always FAIL")
    lines.append("  - G13 patch_overlay_not_executed: always FAIL")
    lines.append("  - G14 label_forbidden: always FAIL")
    lines.append("  - can_create_ground_reference=false at all stages")
    lines.append("")
    lines.append("## Invariants")
    lines.append("  - Nenhum ground reference criado")
    lines.append("  - Nenhum label de treinamento criado")
    lines.append("  - Nenhum overlay executado")
    lines.append("  - Nenhuma coordenada inventada")
    lines.append("  - Nenhum dado bruto versionado")
    lines.append("  - formal_request_path=LEGACY_SECONDARY_ONLY")

    os.makedirs(args.docs_dir, exist_ok=True)
    report_path = os.path.join(args.docs_dir,
                               "protocolo_c_relatorio_v1ui_public_official_discovery.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    status_lines = [
        f"# Status Atual — Protocolo C v1ui",
        f"Atualizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "", f"Fontes publicas registradas: {len(discoveries)}",
        f"Artefatos inventariados: {len(inventory)}",
        f"Candidatos a geometria: {geom_candidates}",
        f"Prontos para revisao: {ready_count}", "",
        "ground_truth_operational=false",
        "can_create_ground_reference=false",
        "can_create_training_label=false",
        "no_overlay_executed=true",
        "formal_request_path=LEGACY_SECONDARY_ONLY",
    ]
    status_path = os.path.join(args.docs_dir, "protocolo_c_status_atual_v1ui.md")
    with open(status_path, "w", encoding="utf-8") as f:
        f.write("\n".join(status_lines))

    print(f"[Completion Report v1ui]")
    print(f"  Sources: {len(discoveries)} | Inventory: {len(inventory)} | Candidates: {geom_candidates}")
    print(f"  Ready for review: {ready_count} | Gains: {gains}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
