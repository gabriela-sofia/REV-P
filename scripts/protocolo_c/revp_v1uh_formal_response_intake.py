#!/usr/bin/env python3
"""
v1uh — Formal Response Intake

Scans inbox for formal responses, registers files, computes SHA256,
classifies by extension/mime, moves allowed files to staging,
quarantines disallowed. Empty inbox produces valid empty registries.
Never deletes originals. Never versions raw files.
"""

import argparse
import csv
import hashlib
import mimetypes
import os
import shutil
from datetime import datetime

PROTOCOL_VERSION = "v1uh"

RESPONSE_COLUMNS = [
    "response_id", "institution", "event_id", "received_date",
    "source_channel", "original_filename", "local_raw_path_hash",
    "sha256", "file_size_bytes", "mime_type", "extension",
    "intake_status", "quarantine_reason", "sensitive_review_required",
    "license_status", "redistribution_allowed", "notes",
]

BLOCKED_EXTENSIONS = {
    ".exe", ".bat", ".cmd", ".ps1", ".sh", ".py", ".js",
    ".dll", ".so", ".msi", ".scr", ".com", ".vbs", ".wsf",
}

ALLOWED_EXTENSIONS = {
    ".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx",
    ".gpkg", ".geojson", ".kml", ".kmz",
    ".csv", ".xlsx", ".xls",
    ".pdf", ".zip",
    ".png", ".jpg", ".jpeg",
    ".json",
}

MAX_FILE_SIZE = 524288000  # 500 MB

SENSITIVE_PATTERNS = ["cpf", "vitima", "cadastro_pessoal", "rg_"]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_path(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]


def detect_sensitive(filename: str) -> bool:
    lower = filename.lower()
    return any(p in lower for p in SENSITIVE_PATTERNS)


def classify_file(filepath: str, filename: str) -> tuple[str, str]:
    ext = os.path.splitext(filename)[1].lower()
    size = os.path.getsize(filepath)

    if ext in BLOCKED_EXTENSIONS:
        return "QUARANTINED", "suspicious_extension"
    if ext not in ALLOWED_EXTENSIONS:
        return "QUARANTINED", "format_not_allowed"
    if size > MAX_FILE_SIZE:
        return "QUARANTINED", "file_too_large"
    if detect_sensitive(filename):
        return "QUARANTINED", "sensitive_data_suspected"

    try:
        if ext == ".zip":
            import zipfile
            with zipfile.ZipFile(filepath, "r") as zf:
                for name in zf.namelist():
                    if ".." in name or name.startswith("/"):
                        return "QUARANTINED", "zip_path_traversal"
                    zext = os.path.splitext(name)[1].lower()
                    if zext in BLOCKED_EXTENSIONS:
                        return "QUARANTINED", "zip_dangerous_content"
    except Exception:
        return "QUARANTINED", "file_corrupted"

    return "ACCEPTED", ""


def scan_inbox(inbox: str, staging: str, quarantine: str) -> list[dict]:
    rows = []
    if not os.path.isdir(inbox):
        return rows

    seq = 0
    for filename in sorted(os.listdir(inbox)):
        filepath = os.path.join(inbox, filename)
        if not os.path.isfile(filepath):
            continue

        ext = os.path.splitext(filename)[1].lower()
        size = os.path.getsize(filepath)
        sha = sha256_file(filepath)
        mime, _ = mimetypes.guess_type(filename)
        status, q_reason = classify_file(filepath, filename)
        sensitive = detect_sensitive(filename)

        if status == "ACCEPTED":
            dest = os.path.join(staging, filename)
            os.makedirs(staging, exist_ok=True)
            if not os.path.exists(dest):
                shutil.copy2(filepath, dest)
        elif status == "QUARANTINED":
            dest = os.path.join(quarantine, filename)
            os.makedirs(quarantine, exist_ok=True)
            if not os.path.exists(dest):
                shutil.copy2(filepath, dest)

        rows.append({
            "response_id": f"RESP_{PROTOCOL_VERSION}_{seq:04d}",
            "institution": "",
            "event_id": "",
            "received_date": datetime.now().strftime("%Y-%m-%d"),
            "source_channel": "",
            "original_filename": filename,
            "local_raw_path_hash": hash_path(filepath),
            "sha256": sha,
            "file_size_bytes": str(size),
            "mime_type": mime or "application/octet-stream",
            "extension": ext,
            "intake_status": status,
            "quarantine_reason": q_reason,
            "sensitive_review_required": str(sensitive).lower(),
            "license_status": "UNKNOWN",
            "redistribution_allowed": "false",
            "notes": "",
        })
        seq += 1

    return rows


def main():
    parser = argparse.ArgumentParser(description="v1uh — Formal Response Intake")
    parser.add_argument("--inbox",
                        default="local_only/protocolo_c/formal_responses/inbox")
    parser.add_argument("--staging",
                        default="local_only/protocolo_c/formal_responses/staging")
    parser.add_argument("--quarantine",
                        default="local_only/protocolo_c/formal_responses/quarantine")
    parser.add_argument("--out-dir", default="datasets/protocolo_c")
    args = parser.parse_args()

    rows = scan_inbox(args.inbox, args.staging, args.quarantine)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "v1uh_formal_response_registry.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESPONSE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    accepted = sum(1 for r in rows if r["intake_status"] == "ACCEPTED")
    quarantined = sum(1 for r in rows if r["intake_status"] == "QUARANTINED")

    if not rows:
        print(f"[Formal Response Intake v1uh] NO_RESPONSES_RECEIVED")
        print(f"  Inbox: {args.inbox} (empty)")
    else:
        print(f"[Formal Response Intake v1uh] {len(rows)} files scanned")
        print(f"  ACCEPTED: {accepted} | QUARANTINED: {quarantined}")

    print(f"  can_create_ground_reference=false")
    print(f"\nRegistry: {out_path}")


if __name__ == "__main__":
    main()
