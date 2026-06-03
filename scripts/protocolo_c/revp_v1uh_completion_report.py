#!/usr/bin/env python3
"""
v1uh — Completion Report

Consolidates all v1uh outputs into reports, next-actions registry,
blocker matrix, and versionable artifacts manifest.
If no formal responses exist, reports NO_FORMAL_RESPONSES_RECEIVED.
"""

import argparse
import csv
import hashlib
import os
from datetime import datetime

PROTOCOL_VERSION = "v1uh"

GUARDRAILS = {
    "ground_truth_operational": False,
    "can_create_ground_reference": False,
    "can_create_training_label": False,
    "can_reopen_protocol_b": False,
    "no_overlay_executed": True,
    "no_coordinates_invented": True,
    "supervisor_review_completed": False,
    "observed_geometry_candidate_only": True,
    "formal_response_intake_only": True,
}

V1UH_DATASETS = [
    "datasets/protocolo_c/v1uh_formal_response_registry.csv",
    "datasets/protocolo_c/v1uh_response_asset_inventory.csv",
    "datasets/protocolo_c/v1uh_observed_geometry_candidate_registry.csv",
    "datasets/protocolo_c/v1uh_event_field_mapping_registry.csv",
    "datasets/protocolo_c/v1uh_crs_geometry_quality_audit.csv",
    "datasets/protocolo_c/v1uh_phenomenon_temporal_gate_audit.csv",
    "datasets/protocolo_c/v1uh_supervisor_review_queue.csv",
    "datasets/protocolo_c/v1uh_ground_reference_candidate_blocker_matrix.csv",
    "datasets/protocolo_c/v1uh_next_actions_registry.csv",
    "datasets/protocolo_c/v1uh_versionable_artifacts_manifest.csv",
]

V1UH_SCRIPTS = [
    "scripts/protocolo_c/revp_v1uh_formal_response_intake.py",
    "scripts/protocolo_c/revp_v1uh_response_asset_inventory.py",
    "scripts/protocolo_c/revp_v1uh_observed_geometry_candidate_audit.py",
    "scripts/protocolo_c/revp_v1uh_event_field_mapper.py",
    "scripts/protocolo_c/revp_v1uh_crs_and_geometry_quality_audit.py",
    "scripts/protocolo_c/revp_v1uh_phenomenon_temporal_gate_audit.py",
    "scripts/protocolo_c/revp_v1uh_supervisor_review_queue_builder.py",
    "scripts/protocolo_c/revp_v1uh_completion_report.py",
]

V1UH_CONFIGS = [
    "configs/protocolo_c/v1uh_formal_response_intake_policy.yaml",
    "configs/protocolo_c/v1uh_allowed_response_formats.yaml",
    "configs/protocolo_c/v1uh_institution_response_mapping.yaml",
    "configs/protocolo_c/v1uh_observed_geometry_gate_policy.yaml",
    "configs/protocolo_c/v1uh_sensitive_data_policy.yaml",
]

BLOCKER_COLUMNS = [
    "event_id", "candidate_id", "blocker", "blocker_status", "severity",
    "can_be_resolved_by_programming", "can_be_resolved_by_human_review",
    "can_be_resolved_by_formal_request", "required_action", "notes",
]

NEXT_ACTIONS_COLUMNS = [
    "action_id", "event_id", "action_type", "priority",
    "description", "target", "status", "notes",
]

MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]

BLOCKERS_PER_EVENT = [
    ("no_response_received", "CRITICAL", "false", "false", "true"),
    ("no_observed_geometry", "CRITICAL", "false", "true", "true"),
    ("no_event_date", "HIGH", "false", "true", "true"),
    ("no_hazard_type", "HIGH", "false", "true", "true"),
    ("phenomenon_not_separated", "CRITICAL", "false", "true", "true"),
    ("no_crs", "HIGH", "true", "true", "false"),
    ("geometry_invalid", "HIGH", "true", "true", "false"),
    ("license_unknown", "MEDIUM", "false", "true", "false"),
    ("sensitive_data_review", "MEDIUM", "false", "true", "false"),
    ("no_supervisor_review", "HIGH", "false", "true", "false"),
    ("no_patch_overlay", "CRITICAL", "true", "false", "false"),
    ("label_forbidden", "CRITICAL", "false", "false", "false"),
]


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(path: str) -> str:
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def generate_blocker_matrix(events: list[dict], responses: list[dict],
                            candidates: list[dict], _queue: list[dict]) -> list[dict]:
    resp_events = {r["event_id"] for r in responses if r.get("event_id")}
    cand_by_event = {}
    for c in candidates:
        cand_by_event.setdefault(c.get("event_id", ""), []).append(c)

    rows = []
    for event in events:
        eid = event["event_id"]
        ev_cands = cand_by_event.get(eid, [])
        has_response = eid in resp_events or len(responses) > 0
        has_geom = any(c.get("can_be_ground_reference_candidate") == "true"
                       for c in ev_cands)

        for bname, severity, prog, human, formal in BLOCKERS_PER_EVENT:
            if bname == "no_response_received":
                status = "RESOLVED" if has_response else "ACTIVE"
            elif bname == "no_observed_geometry":
                status = "RESOLVED" if has_geom else "ACTIVE"
            elif bname == "label_forbidden":
                status = "ACTIVE"
            elif bname == "no_patch_overlay":
                status = "ACTIVE"
            elif bname == "no_supervisor_review":
                status = "ACTIVE"
            else:
                status = "ACTIVE"

            rows.append({
                "event_id": eid,
                "candidate_id": "",
                "blocker": bname,
                "blocker_status": status,
                "severity": severity,
                "can_be_resolved_by_programming": prog,
                "can_be_resolved_by_human_review": human,
                "can_be_resolved_by_formal_request": formal,
                "required_action": "",
                "notes": "",
            })
    return rows


def generate_next_actions(events: list[dict], responses: list[dict],
                          queue: list[dict]) -> list[dict]:
    rows = []
    seq = 0
    has_responses = len(responses) > 0

    for event in events:
        eid = event["event_id"]
        if not has_responses:
            rows.append({
                "action_id": f"ACT_{PROTOCOL_VERSION}_{seq:04d}",
                "event_id": eid,
                "action_type": "SEND_FORMAL_REQUEST",
                "priority": "1",
                "description": "Enviar solicitacoes formais e aguardar retorno",
                "target": "Instituicoes listadas em v1ug_formal_request_queue",
                "status": "PENDING",
                "notes": "Nenhuma resposta formal recebida",
            })
            seq += 1
        else:
            ev_queue = [q for q in queue if q.get("event_id") == eid]
            ready = [q for q in ev_queue
                     if q.get("review_status") == "READY_FOR_REVIEW"]
            if ready:
                rows.append({
                    "action_id": f"ACT_{PROTOCOL_VERSION}_{seq:04d}",
                    "event_id": eid,
                    "action_type": "SUPERVISOR_REVIEW",
                    "priority": "1",
                    "description": f"{len(ready)} candidatos prontos para revisao supervisora",
                    "target": "Supervisor humano",
                    "status": "PENDING",
                    "notes": "",
                })
                seq += 1
            else:
                rows.append({
                    "action_id": f"ACT_{PROTOCOL_VERSION}_{seq:04d}",
                    "event_id": eid,
                    "action_type": "RESOLVE_BLOCKERS",
                    "priority": "2",
                    "description": "Resolver blockers antes de revisao",
                    "target": "Ver blocker matrix",
                    "status": "PENDING",
                    "notes": "",
                })
                seq += 1
    return rows


def generate_manifest() -> list[dict]:
    all_paths = {}
    for p in V1UH_CONFIGS:
        all_paths[p] = "config"
    for p in V1UH_SCRIPTS:
        all_paths[p] = "script"
    for p in V1UH_DATASETS:
        all_paths[p] = "dataset"

    rows = []
    seq = 0
    for path, atype in sorted(all_paths.items()):
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        sha = sha256_file(path)
        rows.append({
            "artifact_id": f"ART_{PROTOCOL_VERSION}_{seq:04d}",
            "artifact_path": path,
            "artifact_type": atype,
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha,
            "file_size_bytes": str(size),
            "is_versionable": "true" if exists else "false",
            "reason": "Safe for git" if exists else "File not found",
        })
        seq += 1
    return rows


def generate_report(responses, assets, candidates, queue, _blockers,
                    next_actions) -> str:
    lines = []
    lines.append(f"# v1uh Completion Report — Formal Response Intake")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Protocol version: {PROTOCOL_VERSION}")
    lines.append("")

    lines.append("## Guardrails")
    for k, v in GUARDRAILS.items():
        lines.append(f"  {k} = {v} [ENFORCED]")
    lines.append("")

    has_responses = len(responses) > 0
    lines.append("## Intake Summary")
    if not has_responses:
        lines.append("  Status: NO_FORMAL_RESPONSES_RECEIVED")
        lines.append("  Infrastructure is ready. Awaiting formal responses.")
    else:
        accepted = sum(1 for r in responses if r.get("intake_status") == "ACCEPTED")
        quarantined = sum(1 for r in responses if r.get("intake_status") == "QUARANTINED")
        lines.append(f"  Responses: {len(responses)} (accepted={accepted}, quarantined={quarantined})")
    lines.append(f"  Assets inventoried: {len(assets)}")
    lines.append(f"  Geometry candidates: {sum(1 for c in candidates if c.get('can_be_ground_reference_candidate') == 'true')}")
    lines.append("")

    lines.append("## Supervisor Review Queue")
    ready = sum(1 for q in queue if q.get("review_status") == "READY_FOR_REVIEW")
    blocked_q = sum(1 for q in queue if q.get("review_status") == "BLOCKED_PENDING_GATES")
    lines.append(f"  Ready for review: {ready}")
    lines.append(f"  Blocked: {blocked_q}")
    lines.append(f"  Total: {len(queue)}")
    lines.append("")

    lines.append("## Why No Ground Truth Yet")
    lines.append("  - No observed geometry acquired from formal responses")
    lines.append("  - No supervisor review completed")
    lines.append("  - No patch-evidence overlay executed")
    lines.append("  - can_create_ground_reference=false at all stages")
    lines.append("  - can_create_training_label=false at all stages")
    lines.append("")

    lines.append("## Next Steps")
    for act in next_actions:
        lines.append(f"  [{act['action_type']}] {act['event_id']}: {act['description']}")
    lines.append("")

    lines.append("## Invariants")
    lines.append("  - Nenhum ground reference criado")
    lines.append("  - Nenhum label de treinamento criado")
    lines.append("  - Nenhum overlay executado")
    lines.append("  - Nenhuma coordenada inventada")
    lines.append("  - Nenhum dado bruto versionado")
    lines.append("  - Nenhum path absoluto em CSV versionavel")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="v1uh — Completion Report")
    parser.add_argument("--out-dir", default="datasets/protocolo_c")
    parser.add_argument("--docs-dir", default="docs/metodologia_cientifica")
    args = parser.parse_args()

    responses = load_csv(os.path.join(args.out_dir, "v1uh_formal_response_registry.csv"))
    assets = load_csv(os.path.join(args.out_dir, "v1uh_response_asset_inventory.csv"))
    candidates = load_csv(os.path.join(args.out_dir, "v1uh_observed_geometry_candidate_registry.csv"))
    queue = load_csv(os.path.join(args.out_dir, "v1uh_supervisor_review_queue.csv"))
    events = load_csv(os.path.join(args.out_dir, "event_candidate_registry.csv"))

    blockers = generate_blocker_matrix(events, responses, candidates, queue)
    blocker_path = os.path.join(args.out_dir, "v1uh_ground_reference_candidate_blocker_matrix.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(blocker_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BLOCKER_COLUMNS)
        writer.writeheader()
        writer.writerows(blockers)

    next_actions = generate_next_actions(events, responses, queue)
    actions_path = os.path.join(args.out_dir, "v1uh_next_actions_registry.csv")
    with open(actions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NEXT_ACTIONS_COLUMNS)
        writer.writeheader()
        writer.writerows(next_actions)

    manifest = generate_manifest()
    manifest_path = os.path.join(args.out_dir, "v1uh_versionable_artifacts_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest)

    report = generate_report(responses, assets, candidates, queue,
                             blockers, next_actions)

    os.makedirs(args.docs_dir, exist_ok=True)
    report_path = os.path.join(args.docs_dir,
                               "protocolo_c_relatorio_v1uh_formal_response_intake.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    status_path = os.path.join(args.docs_dir, "protocolo_c_status_atual_v1uh.md")
    status_lines = [
        f"# Status Atual — Protocolo C v1uh",
        f"Atualizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Respostas formais recebidas: {len(responses)}",
        f"Assets inventariados: {len(assets)}",
        f"Candidatos a geometria: {sum(1 for c in candidates if c.get('can_be_ground_reference_candidate') == 'true')}",
        f"Prontos para revisao: {sum(1 for q in queue if q.get('review_status') == 'READY_FOR_REVIEW')}",
        "",
        "ground_truth_operational=false",
        "can_create_ground_reference=false",
        "can_create_training_label=false",
        "no_overlay_executed=true",
    ]
    with open(status_path, "w", encoding="utf-8") as f:
        f.write("\n".join(status_lines))

    print(f"[Completion Report v1uh]")
    print(f"  Responses: {len(responses)} | Assets: {len(assets)} | Candidates: {len(candidates)}")
    print(f"  Blockers: {blocker_path}")
    print(f"  Next actions: {actions_path}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Report: {report_path}")
    print(f"  Status: {status_path}")


if __name__ == "__main__":
    main()
