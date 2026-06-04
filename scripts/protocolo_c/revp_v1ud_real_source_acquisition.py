#!/usr/bin/env python3
"""
v1ud — Real Source Acquisition

Downloads files from the manifest where download_allowed=True.
Saves raw files to local_only/, computes SHA256, records metadata.
Outputs evidence_extraction_registry.csv and updates integrity.
"""

import argparse
import csv
import hashlib
import os
import sys
import time
from datetime import datetime
from urllib.parse import urlparse

try:
    import yaml
except ImportError:
    yaml = None

try:
    import requests
except ImportError:
    requests = None

PROTOCOL_VERSION = "v1ud"
USER_AGENT = (
    "REV-P-AcademicResearch/1.0 "
    "(TCC ground reference evidence acquisition; contact: academic use only)"
)

EXTRACTION_COLUMNS = [
    "extraction_id", "manifest_id", "source_id", "event_id", "url",
    "acquisition_status", "local_raw_path", "sha256", "file_size_bytes",
    "mime_type", "content_type_header", "acquisition_timestamp",
    "download_duration_ms", "http_status", "error_detail",
    "can_create_training_label", "ground_truth_operational", "notes",
]


def load_yaml(path: str) -> dict:
    if yaml is None:
        print("[ERROR] pyyaml not installed")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_mime(filepath: str) -> str:
    import mimetypes
    mt, _ = mimetypes.guess_type(filepath)
    return mt or "application/octet-stream"


def derive_filename(url: str, content_type: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    basename = os.path.basename(path)
    if basename and "." in basename:
        return basename[:200]
    ext_map = {
        "text/html": ".html", "application/pdf": ".pdf",
        "application/zip": ".zip", "text/csv": ".csv",
        "application/json": ".json", "text/xml": ".xml",
        "application/xml": ".xml",
    }
    ct_lower = (content_type or "").split(";")[0].strip().lower()
    ext = ext_map.get(ct_lower, ".bin")
    safe_name = parsed.hostname or "download"
    safe_name = safe_name.replace(".", "_")[:50]
    return f"{safe_name}{ext}"


def download_file(
    url: str, dest_dir: str, max_mb: float, timeout: int, dry_run: bool
) -> dict:
    if dry_run:
        return {
            "status": "DRY_RUN",
            "local_path": "",
            "sha256": "",
            "file_size": 0,
            "mime_type": "",
            "content_type": "",
            "duration_ms": 0,
            "http_status": "",
            "error": "",
        }

    if requests is None:
        return {
            "status": "DEPENDENCY_MISSING",
            "local_path": "", "sha256": "", "file_size": 0,
            "mime_type": "", "content_type": "", "duration_ms": 0,
            "http_status": "", "error": "requests not installed",
        }

    headers = {"User-Agent": USER_AGENT}
    start = time.monotonic()
    try:
        resp = requests.get(
            url, headers=headers, timeout=timeout,
            stream=True, allow_redirects=True,
        )
        resp.raise_for_status()

        ct = resp.headers.get("Content-Type", "")
        cl = resp.headers.get("Content-Length")
        if cl and int(cl) > max_mb * 1024 * 1024:
            return {
                "status": "REJECTED_TOO_LARGE",
                "local_path": "", "sha256": "", "file_size": int(cl),
                "mime_type": "", "content_type": ct,
                "duration_ms": int((time.monotonic() - start) * 1000),
                "http_status": str(resp.status_code),
                "error": f"Content-Length {cl} > {max_mb}MB",
            }

        os.makedirs(dest_dir, exist_ok=True)
        filename = derive_filename(url, ct)
        dest_path = os.path.join(dest_dir, filename)

        limit_bytes = int(max_mb * 1024 * 1024)
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                downloaded += len(chunk)
                if downloaded > limit_bytes:
                    f.close()
                    os.remove(dest_path)
                    return {
                        "status": "REJECTED_TOO_LARGE",
                        "local_path": "", "sha256": "", "file_size": downloaded,
                        "mime_type": "", "content_type": ct,
                        "duration_ms": int((time.monotonic() - start) * 1000),
                        "http_status": str(resp.status_code),
                        "error": f"Mid-stream exceeded {max_mb}MB",
                    }
                f.write(chunk)

        duration = int((time.monotonic() - start) * 1000)
        file_hash = sha256_file(dest_path)
        mime = detect_mime(dest_path)

        return {
            "status": "DOWNLOADED",
            "local_path": dest_path,
            "sha256": file_hash,
            "file_size": downloaded,
            "mime_type": mime,
            "content_type": ct,
            "duration_ms": duration,
            "http_status": str(resp.status_code),
            "error": "",
        }

    except requests.exceptions.RequestException as e:
        return {
            "status": "NETWORK_ERROR",
            "local_path": "", "sha256": "", "file_size": 0,
            "mime_type": "", "content_type": "",
            "duration_ms": int((time.monotonic() - start) * 1000),
            "http_status": "", "error": str(e)[:300],
        }


def main():
    parser = argparse.ArgumentParser(description="v1ud — Real Source Acquisition")
    parser.add_argument("--config", default="configs/protocolo_c/ground_reference_evidence_sources.yaml")
    parser.add_argument("--targets", default="configs/protocolo_c/v1ud_real_acquisition_targets.yaml")
    parser.add_argument("--events", default="datasets/protocolo_c/event_candidate_registry.csv")
    parser.add_argument("--manifest", default="datasets/protocolo_c/v1ud_download_manifest.csv")
    parser.add_argument("--out-dir", default="datasets/protocolo_c")
    parser.add_argument("--local-only-dir", default="local_only/protocolo_c")
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-download-mb", type=float, default=25.0)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    manifest = load_csv(args.manifest)
    if not manifest:
        print("[v1ud] No manifest found. Run download_manifest_builder first.")
        print("  Generating from targets in dry-run mode...")
        manifest = []

    print(f"[Real Acquisition v1ud] Manifest entries: {len(manifest)}")
    print(f"  dry_run={args.dry_run} allow_web={args.allow_web} download={args.download}")
    print(f"  max_download_mb={args.max_download_mb} timeout={args.timeout}")
    print(f"  ground_truth_operational=false")
    print(f"  can_create_training_label=false")

    results = []
    rate_delay = 1.5

    for idx, entry in enumerate(manifest):
        source_id = entry["source_id"]
        event_id = entry["event_id"]
        url = entry["url"]
        allowed = entry["download_allowed"] == "True"
        max_mb = min(float(entry.get("max_download_mb", args.max_download_mb)), args.max_download_mb)

        if args.dry_run:
            dl_result = download_file(url, "", max_mb, args.timeout, dry_run=True)
        elif allowed and args.allow_web and args.download:
            raw_dir = os.path.join(args.local_only_dir, entry.get("local_raw_target", f"evidence_raw/v1ud/{source_id}/{event_id}"))
            dl_result = download_file(url, raw_dir, max_mb, args.timeout, dry_run=False)
            if idx < len(manifest) - 1:
                time.sleep(rate_delay)
        else:
            dl_result = {
                "status": "SKIPPED",
                "local_path": "", "sha256": "", "file_size": 0,
                "mime_type": "", "content_type": "", "duration_ms": 0,
                "http_status": "", "error": entry.get("reason", "Not allowed"),
            }

        local_path_safe = ""
        if dl_result["local_path"]:
            local_path_safe = os.path.relpath(dl_result["local_path"], start=".")

        results.append({
            "extraction_id": f"EXT_{PROTOCOL_VERSION}_{idx:04d}",
            "manifest_id": entry.get("manifest_id", ""),
            "source_id": source_id,
            "event_id": event_id,
            "url": url,
            "acquisition_status": dl_result["status"],
            "local_raw_path": local_path_safe,
            "sha256": dl_result["sha256"],
            "file_size_bytes": str(dl_result["file_size"]),
            "mime_type": dl_result["mime_type"],
            "content_type_header": dl_result["content_type"],
            "acquisition_timestamp": datetime.now().isoformat() if dl_result["status"] == "DOWNLOADED" else "",
            "download_duration_ms": str(dl_result["duration_ms"]),
            "http_status": dl_result["http_status"],
            "error_detail": dl_result["error"],
            "can_create_training_label": "false",
            "ground_truth_operational": "false",
            "notes": "",
        })

    out_path = os.path.join(args.out_dir, "v1ud_evidence_extraction_registry.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXTRACTION_COLUMNS)
        writer.writeheader()
        writer.writerows(results)

    status_counts = {}
    for r in results:
        s = r["acquisition_status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    print(f"\n[Results] {len(results)} entries processed")
    for s, c in sorted(status_counts.items()):
        print(f"  {s}: {c}")
    print(f"\nRegistry: {out_path}")


if __name__ == "__main__":
    main()
