#!/usr/bin/env python3
"""
v1uf — Station-Resolved Official Data Acquisition

Builds a large-download manifest from v1ue INMET year-specific datasets and,
when --allow-large-official-download is set, downloads the official ZIPs
(streaming, size-limited, allowlisted source only) into local_only/.

Never versions raw ZIPs. Never invents coordinates. Fail-closed on errors.
"""

import argparse
import csv
import hashlib
import os
import sys
import time
from urllib.parse import urlparse

try:
    import yaml
except ImportError:
    yaml = None

try:
    import requests
except ImportError:
    requests = None

PROTOCOL_VERSION = "v1uf"
USER_AGENT = (
    "REV-P-AcademicResearch/1.0 "
    "(TCC ground reference evidence acquisition; contact: academic use only)"
)

MANIFEST_COLUMNS = [
    "manifest_id", "event_id", "source_id", "url", "expected_format",
    "content_length_bytes", "max_download_mb", "source_allowlisted",
    "license_status", "download_decision", "download_status",
    "zip_local_path", "zip_sha256", "zip_size_bytes", "blocking_reason",
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


def rel_path(path: str) -> str:
    try:
        return os.path.relpath(path, start=".").replace("\\", "/")
    except ValueError:
        return path.replace("\\", "/")


def stream_download(url: str, dest: str, max_mb: float, timeout: int) -> dict:
    if requests is None:
        return {"status": "DEPENDENCY_MISSING", "error": "requests not installed"}
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True)
        resp.raise_for_status()
        cl = resp.headers.get("Content-Length")
        if cl and int(cl) > max_mb * 1024 * 1024:
            return {"status": "REJECTED_TOO_LARGE", "error": f"Content-Length {cl} > {max_mb}MB"}
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        limit = int(max_mb * 1024 * 1024)
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                downloaded += len(chunk)
                if downloaded > limit:
                    f.close()
                    os.remove(dest)
                    return {"status": "REJECTED_TOO_LARGE", "error": f"Mid-stream exceeded {max_mb}MB"}
                f.write(chunk)
        return {"status": "DOWNLOADED", "size": downloaded, "sha256": sha256_file(dest)}
    except requests.exceptions.RequestException as e:
        return {"status": "NETWORK_ERROR", "error": str(e)[:300]}


def main():
    parser = argparse.ArgumentParser(description="v1uf — Station-Resolved Official Data Acquisition")
    parser.add_argument("--v1ue-resolution", default="datasets/protocolo_c/v1ue_official_dataset_resolution_registry.csv")
    parser.add_argument("--stations", default="datasets/protocolo_c/v1ue_station_candidate_registry.csv")
    parser.add_argument("--policy", default="configs/protocolo_c/v1uf_large_official_download_policy.yaml")
    parser.add_argument("--out-dir", default="datasets/protocolo_c")
    parser.add_argument("--local-only-dir", default="local_only/protocolo_c")
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--allow-large-official-download", action="store_true")
    parser.add_argument("--max-download-mb", type=float, default=150.0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    policy = load_yaml(args.policy)
    resolutions = load_csv(args.v1ue_resolution)

    allow_map = {}
    for s in policy.get("large_download_allowed_sources", []):
        allow_map[s["source_id"]] = s

    # INMET year-specific downloadable datasets
    targets = [
        r for r in resolutions
        if r.get("source_id") == "INMET_BDMEP"
        and r.get("http_status") == "200"
        and r.get("is_year_specific") == "true"
        and r.get("is_downloadable") == "true"
    ]

    print(f"[v1uf Acquisition] INMET year-specific targets: {len(targets)}")
    print(f"  dry_run={args.dry_run} allow_web={args.allow_web} "
          f"allow_large={args.allow_large_official_download} max_mb={args.max_download_mb}")
    print(f"  ground_truth_operational=false can_create_ground_reference=false")

    url_cache: dict[str, dict] = {}  # url -> {path, sha, size}
    rows = []
    seq = 0
    for tgt in targets:
        source_id = tgt["source_id"]
        event_id = tgt["event_id"]
        url = tgt["candidate_url"]
        content_length = tgt.get("content_length", "")
        license_status = tgt.get("license_status", "UNKNOWN_NEEDS_REVIEW")

        allow_src = allow_map.get(source_id)
        allowlisted = allow_src is not None
        src_max_mb = min(allow_src.get("max_download_mb", args.max_download_mb), args.max_download_mb) if allow_src else 0

        # Decision
        path_ok = True
        if allow_src and allow_src.get("allowed_path_prefix"):
            path_ok = allow_src["allowed_path_prefix"] in urlparse(url).path

        if not allowlisted:
            decision = "BLOCKED_SOURCE_NOT_ALLOWED"
            blocking = f"{source_id} not in large_download_allowed_sources"
        elif not path_ok:
            decision = "BLOCKED_PATH"
            blocking = "URL path not under allowed prefix"
        elif license_status == "UNKNOWN_NEEDS_REVIEW":
            decision = "LICENSE_REVIEW_REQUIRED"
            blocking = "License not reviewed"
        elif content_length and int(content_length) > src_max_mb * 1024 * 1024:
            decision = "BLOCKED_TOO_LARGE"
            blocking = f"content_length {content_length} > {src_max_mb}MB"
        else:
            decision = "DOWNLOAD_ALLOWED"
            blocking = ""

        download_status = "NOT_ATTEMPTED"
        zip_local = ""
        zip_sha = ""
        zip_size = ""

        do_download = (
            decision == "DOWNLOAD_ALLOWED"
            and args.allow_web
            and args.allow_large_official_download
            and not args.dry_run
        )

        if args.dry_run:
            download_status = "DRY_RUN"
        elif do_download:
            if url in url_cache:
                cached = url_cache[url]
                download_status = "DOWNLOADED_CACHED"
                zip_local = cached["path"]
                zip_sha = cached["sha"]
                zip_size = cached["size"]
            else:
                dest = os.path.join(
                    args.local_only_dir, "evidence_raw", "v1uf", source_id, event_id,
                    os.path.basename(urlparse(url).path) or "dataset.zip",
                )
                result = stream_download(url, dest, src_max_mb, args.timeout)
                download_status = result["status"]
                if result["status"] == "DOWNLOADED":
                    zip_local = rel_path(dest)
                    zip_sha = result["sha256"]
                    zip_size = str(result["size"])
                    url_cache[url] = {"path": zip_local, "sha": zip_sha, "size": zip_size}
                else:
                    blocking = result.get("error", download_status)
                time.sleep(2.0)
        else:
            download_status = "SKIPPED"

        rows.append({
            "manifest_id": f"LDM_{PROTOCOL_VERSION}_{seq:04d}",
            "event_id": event_id,
            "source_id": source_id,
            "url": url,
            "expected_format": tgt.get("dataset_type", "ZIP"),
            "content_length_bytes": content_length,
            "max_download_mb": str(src_max_mb),
            "source_allowlisted": str(allowlisted).lower(),
            "license_status": license_status,
            "download_decision": decision,
            "download_status": download_status,
            "zip_local_path": zip_local,
            "zip_sha256": zip_sha,
            "zip_size_bytes": zip_size,
            "blocking_reason": blocking,
        })
        seq += 1

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "v1uf_large_download_manifest.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    statuses = {}
    for r in rows:
        statuses[r["download_status"]] = statuses.get(r["download_status"], 0) + 1
    print(f"\n[Results] {len(rows)} manifest entries")
    for s, c in sorted(statuses.items()):
        print(f"  {s}: {c}")
    print(f"\nManifest: {out_path}")


if __name__ == "__main__":
    main()
