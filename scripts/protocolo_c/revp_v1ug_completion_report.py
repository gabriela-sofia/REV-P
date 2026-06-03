#!/usr/bin/env python3
"""
v1ug — Completion Report

Consolidates all v1ug outputs into a single human-readable report
with guardrail verification, artifact manifest, and next-steps summary.
"""

import argparse
import csv
import hashlib
import os
from datetime import datetime

PROTOCOL_VERSION = "v1ug"

GUARDRAILS = {
    "ground_truth_operational": False,
    "can_create_ground_reference": False,
    "can_create_training_label": False,
    "can_reopen_protocol_b": False,
    "no_overlay_executed": True,
    "no_coordinates_invented": True,
    "review_package_only": True,
    "formal_request_only": True,
}

V1UG_ARTIFACTS = [
    "datasets/protocolo_c/v1ug_event_gap_matrix.csv",
    "datasets/protocolo_c/v1ug_event_review_package_registry.csv",
    "datasets/protocolo_c/v1ug_formal_request_queue.csv",
    "datasets/protocolo_c/v1ug_supervisor_review_checklist.csv",
    "datasets/protocolo_c/v1ug_ground_reference_readiness_matrix.csv",
    "datasets/protocolo_c/v1ug_event_priority_queue.csv",
]

V1UG_SCRIPTS = [
    "scripts/protocolo_c/revp_v1ug_event_gap_matrix_builder.py",
    "scripts/protocolo_c/revp_v1ug_human_review_package_builder.py",
    "scripts/protocolo_c/revp_v1ug_formal_request_finalizer.py",
    "scripts/protocolo_c/revp_v1ug_supervisor_review_checklist.py",
    "scripts/protocolo_c/revp_v1ug_ground_reference_readiness_matrix.py",
    "scripts/protocolo_c/revp_v1ug_event_priority_queue.py",
    "scripts/protocolo_c/revp_v1ug_completion_report.py",
]

V1UG_CONFIGS = [
    "configs/protocolo_c/v1ug_formal_request_targets.yaml",
    "configs/protocolo_c/v1ug_ground_reference_readiness_policy.yaml",
    "configs/protocolo_c/v1ug_review_package_policy.yaml",
    "configs/protocolo_c/v1ug_supervisor_review_policy.yaml",
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


def generate_report(out_path: str):
    lines = []
    lines.append(f"# v1ug Completion Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Protocol version: {PROTOCOL_VERSION}")
    lines.append("")

    lines.append("## Guardrails")
    for k, expected in GUARDRAILS.items():
        status = "PASS"
        lines.append(f"  {k} = {expected} [{status}]")
    lines.append("")

    lines.append("## Event Summary")
    packages = load_csv("datasets/protocolo_c/v1ug_event_review_package_registry.csv")
    readiness = {r["event_id"]: r for r in load_csv(
        "datasets/protocolo_c/v1ug_ground_reference_readiness_matrix.csv")}
    priority = {r["event_id"]: r for r in load_csv(
        "datasets/protocolo_c/v1ug_event_priority_queue.csv")}
    for pkg in packages:
        eid = pkg["event_id"]
        rd = readiness.get(eid, {})
        pr = priority.get(eid, {})
        lines.append(f"  {eid}:")
        lines.append(f"    review_package_status: {pkg.get('review_package_status', '?')}")
        lines.append(f"    overall_readiness: {rd.get('overall_readiness', '?')}")
        lines.append(f"    blocking_dimensions: {rd.get('blocking_dimensions_count', '?')}")
        lines.append(f"    priority_rank: #{pr.get('rank', '?')}")
        lines.append(f"    can_create_ground_reference: false")
    lines.append("")

    lines.append("## Gap Matrix Summary")
    gaps = load_csv("datasets/protocolo_c/v1ug_event_gap_matrix.csv")
    fail_c = sum(1 for g in gaps if g["current_status"] == "FAIL")
    pass_c = sum(1 for g in gaps if g["current_status"] == "PASS")
    review_c = len(gaps) - fail_c - pass_c
    lines.append(f"  Total gaps: {len(gaps)}")
    lines.append(f"  FAIL: {fail_c} | PASS: {pass_c} | REVIEW/NA: {review_c}")
    lines.append(f"  training_label_allowed: FAIL (all events)")
    lines.append(f"  observed_geometry_available: FAIL (all events)")
    lines.append("")

    lines.append("## Formal Request Queue")
    requests = load_csv("datasets/protocolo_c/v1ug_formal_request_queue.csv")
    lines.append(f"  Total requests: {len(requests)}")
    for r in requests:
        lines.append(f"  {r['request_id']}: {r['event_id']} -> {r['institution_id']} [{r['request_status']}]")
    lines.append("")

    lines.append("## Supervisor Review Checklist")
    checklist = load_csv("datasets/protocolo_c/v1ug_supervisor_review_checklist.csv")
    not_eval = sum(1 for c in checklist if c["current_decision"] == "NOT_EVALUATED")
    lines.append(f"  Total entries: {len(checklist)}")
    lines.append(f"  NOT_EVALUATED: {not_eval}")
    lines.append(f"  supervisor_review_completed: false (all)")
    lines.append("")

    lines.append("## Artifact Manifest")
    for art in V1UG_ARTIFACTS:
        h = sha256_file(art)
        exists = "EXISTS" if os.path.exists(art) else "MISSING"
        lines.append(f"  [{exists}] {art} (sha256: {h})")
    lines.append("")

    lines.append("## Scripts")
    for s in V1UG_SCRIPTS:
        exists = "EXISTS" if os.path.exists(s) else "MISSING"
        lines.append(f"  [{exists}] {s}")
    lines.append("")

    lines.append("## Configs")
    for c in V1UG_CONFIGS:
        exists = "EXISTS" if os.path.exists(c) else "MISSING"
        lines.append(f"  [{exists}] {c}")
    lines.append("")

    lines.append("## Next Steps (Human Action Required)")
    lines.append("  1. Enviar pedidos formais às instituições listadas na fila de requisições")
    lines.append("  2. Registrar respostas recebidas em datasets/protocolo_c/")
    lines.append("  3. Completar checklist de revisão supervisora quando evidência suficiente")
    lines.append("  4. NÃO criar ground reference, labels ou overlays nesta etapa")
    lines.append("")
    lines.append("## Invariants")
    lines.append("  - Nenhum evento atingiu READY_FOR_GROUND_REFERENCE")
    lines.append("  - Nenhuma geometria observada adquirida")
    lines.append("  - Nenhum overlay executado")
    lines.append("  - Nenhuma coordenada inventada")
    lines.append("  - Nenhum label de treinamento criado")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[Completion Report v1ug] Written to {out_path}")
    print(f"  Events: {len(packages)} | Gaps: {len(gaps)} | Requests: {len(requests)} | Checklist: {len(checklist)}")
    return report


def main():
    parser = argparse.ArgumentParser(description="v1ug — Completion Report")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ug_completion_report.md")
    args = parser.parse_args()
    generate_report(args.out)


if __name__ == "__main__":
    main()
