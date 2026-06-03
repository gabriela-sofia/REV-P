#!/usr/bin/env python3
"""
v1ug — Versionable Artifacts Manifest

Lists all v1ug artifacts that are safe to commit (no raw data,
no embeddings, no local_runs content). Includes SHA256 hashes.
"""

import argparse
import csv
import hashlib
import os

PROTOCOL_VERSION = "v1ug"

COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]

VERSIONABLE_PATTERNS = {
    "configs/protocolo_c/v1ug_formal_request_targets.yaml": "config",
    "configs/protocolo_c/v1ug_ground_reference_readiness_policy.yaml": "config",
    "configs/protocolo_c/v1ug_review_package_policy.yaml": "config",
    "configs/protocolo_c/v1ug_supervisor_review_policy.yaml": "config",
    "scripts/protocolo_c/revp_v1ug_event_gap_matrix_builder.py": "script",
    "scripts/protocolo_c/revp_v1ug_human_review_package_builder.py": "script",
    "scripts/protocolo_c/revp_v1ug_formal_request_finalizer.py": "script",
    "scripts/protocolo_c/revp_v1ug_supervisor_review_checklist.py": "script",
    "scripts/protocolo_c/revp_v1ug_ground_reference_readiness_matrix.py": "script",
    "scripts/protocolo_c/revp_v1ug_event_priority_queue.py": "script",
    "scripts/protocolo_c/revp_v1ug_completion_report.py": "script",
    "scripts/protocolo_c/revp_v1ug_versionable_artifacts_manifest.py": "script",
    "datasets/protocolo_c/v1ug_event_gap_matrix.csv": "dataset",
    "datasets/protocolo_c/v1ug_event_review_package_registry.csv": "dataset",
    "datasets/protocolo_c/v1ug_formal_request_queue.csv": "dataset",
    "datasets/protocolo_c/v1ug_supervisor_review_checklist.csv": "dataset",
    "datasets/protocolo_c/v1ug_ground_reference_readiness_matrix.csv": "dataset",
    "datasets/protocolo_c/v1ug_event_priority_queue.csv": "dataset",
    "datasets/protocolo_c/v1ug_completion_report.md": "report",
    "datasets/protocolo_c/v1ug_versionable_artifacts_manifest.csv": "manifest",
}


def sha256_file(path: str) -> str:
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="v1ug — Versionable Artifacts Manifest")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ug_versionable_artifacts_manifest.csv")
    args = parser.parse_args()

    rows = []
    seq = 0
    for path, art_type in sorted(VERSIONABLE_PATTERNS.items()):
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        sha = sha256_file(path)
        rows.append({
            "artifact_id": f"ART_{PROTOCOL_VERSION}_{seq:04d}",
            "artifact_path": path,
            "artifact_type": art_type,
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha,
            "file_size_bytes": str(size),
            "is_versionable": "true" if exists else "false",
            "reason": "Safe for git" if exists else "File not found",
        })
        seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    found = sum(1 for r in rows if r["is_versionable"] == "true")
    print(f"[Versionable Artifacts Manifest v1ug] {found}/{len(rows)} artifacts found")
    for r in rows:
        tag = "OK" if r["is_versionable"] == "true" else "MISSING"
        print(f"  [{tag}] {r['artifact_path']}")
    print(f"\nManifest: {args.out}")


if __name__ == "__main__":
    main()
