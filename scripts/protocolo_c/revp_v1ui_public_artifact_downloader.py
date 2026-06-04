#!/usr/bin/env python3
"""
v1ui — Public Artifact Downloader

Downloads public artifacts from the crawl manifest.
Respects allowlist, size limits, extension filters.
Stores in local_only. Never versions raw. Computes SHA256.
Default: dry_run=true unless --download is passed.
"""

import argparse
import csv
import hashlib
import os
from urllib.parse import urlparse

try:
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False

try:
    import yaml
except ImportError:
    yaml = None

PROTOCOL_VERSION = "v1ui"

DOWNLOAD_COLUMNS = [
    "artifact_id", "event_id", "source_id", "url", "local_path_hash",
    "sha256", "file_size_bytes", "mime_type", "extension",
    "download_status", "license_status", "public_access_status",
    "sensitive_review_required", "notes",
]

ALLOWED_EXTENSIONS = {
    ".zip", ".shp", ".shx", ".dbf", ".prj", ".gpkg", ".geojson",
    ".kml", ".kmz", ".csv", ".xlsx", ".xls", ".pdf", ".json", ".xml",
}
BLOCKED_EXTENSIONS = {".exe", ".bat", ".cmd", ".ps1", ".sh", ".py", ".dll", ".msi"}
MAX_SIZE = 104857600  # 100 MB


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_path(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_yaml(path):
    if yaml is None or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_allowed_domains(path):
    cfg = load_yaml(path)
    domains = set()
    for group in cfg.get("allowed_domains", {}).values():
        if isinstance(group, list):
            domains.update(group)
    return domains


def download_file(url, dest, timeout=60):
    if not HAS_URLLIB:
        return False, "NO_URLLIB"
    try:
        req = urllib.request.Request(url,
            headers={"User-Agent": "REV-P-Academic-Research/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
        return True, "OK"
    except Exception as e:
        return False, str(e)[:200]


def main():
    parser = argparse.ArgumentParser(description="v1ui — Public Artifact Downloader")
    parser.add_argument("--manifest", default="datasets/protocolo_c/v1ui_public_artifact_download_manifest.csv")
    parser.add_argument("--allowed-domains", default="configs/protocolo_c/v1ui_allowed_domains.yaml")
    parser.add_argument("--local-only-dir", default="local_only/protocolo_c/public_official_artifacts/raw/v1ui")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ui_public_artifact_inventory_download.csv")
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-download-mb", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    manifest = load_csv(args.manifest)
    allowed = load_allowed_domains(args.allowed_domains)
    do_download = args.allow_web and args.download

    rows = []
    seq = 0
    total_bytes = 0

    for entry in manifest:
        url = entry.get("discovered_url", "")
        event_id = entry.get("event_id", "")
        source_id = entry.get("source_id", "")
        ext = entry.get("extension", "")
        status_in = entry.get("artifact_candidate_status", "")

        if status_in not in ("CANDIDATE",) and ext not in ALLOWED_EXTENSIONS:
            continue

        domain = urlparse(url).hostname or ""
        domain_ok = any(domain.endswith(d) for d in allowed) if allowed else False
        ext_ok = ext in ALLOWED_EXTENSIONS and ext not in BLOCKED_EXTENSIONS

        dl_status = "DRY_RUN"
        sha = ""
        size = 0
        mime = ""
        path_hash = hash_path(url)

        if do_download and domain_ok and ext_ok:
            dest = os.path.join(args.local_only_dir, source_id, event_id,
                                os.path.basename(urlparse(url).path) or "artifact")
            if not os.path.exists(dest):
                ok, msg = download_file(url, dest, args.timeout)
                if ok:
                    size = os.path.getsize(dest)
                    if size > args.max_download_mb * 1024 * 1024:
                        os.remove(dest)
                        dl_status = "REJECTED_TOO_LARGE"
                    else:
                        sha = sha256_file(dest)
                        dl_status = "DOWNLOADED"
                        total_bytes += size
                else:
                    dl_status = f"FAILED:{msg[:50]}"
            else:
                size = os.path.getsize(dest)
                sha = sha256_file(dest)
                dl_status = "ALREADY_EXISTS"
        elif not domain_ok:
            dl_status = "BLOCKED_DOMAIN"
        elif not ext_ok:
            dl_status = "BLOCKED_EXTENSION"

        rows.append({
            "artifact_id": f"PART_{PROTOCOL_VERSION}_{seq:04d}",
            "event_id": event_id, "source_id": source_id,
            "url": url, "local_path_hash": path_hash,
            "sha256": sha, "file_size_bytes": str(size),
            "mime_type": mime, "extension": ext,
            "download_status": dl_status,
            "license_status": "PUBLIC_ASSUMED",
            "public_access_status": "PUBLIC" if domain_ok else "BLOCKED",
            "sensitive_review_required": "false",
            "notes": "",
        })
        seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DOWNLOAD_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    downloaded = sum(1 for r in rows if r["download_status"] == "DOWNLOADED")
    print(f"[Public Artifact Downloader v1ui] {len(rows)} entries | downloaded={downloaded} | {total_bytes/1024/1024:.1f}MB")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
