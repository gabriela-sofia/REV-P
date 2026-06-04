#!/usr/bin/env python3
"""
v1ud — Download Manifest Builder

Creates a download manifest from resolved URLs. Nothing downloads
without being in the manifest first.
"""

import argparse
import csv
import os
import sys
from urllib.parse import urlparse

try:
    import yaml
except ImportError:
    yaml = None

PROTOCOL_VERSION = "v1ud"

MANIFEST_COLUMNS = [
    "manifest_id", "source_id", "event_id", "url", "expected_type",
    "max_download_mb", "local_raw_target", "local_staging_target",
    "download_allowed", "reason", "priority",
]


def load_yaml(path: str) -> dict:
    if yaml is None:
        print("[ERROR] pyyaml not installed")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def infer_expected_type(url: str, content_type: str) -> str:
    lower_url = url.lower()
    if content_type:
        ct = content_type.lower()
        if "pdf" in ct:
            return "PDF"
        if "zip" in ct:
            return "ZIP"
        if "json" in ct:
            return "JSON"
        if "html" in ct:
            return "HTML"
        if "csv" in ct or "text/plain" in ct:
            return "CSV"
        if "xml" in ct:
            return "XML"
    for ext, typ in [
        (".pdf", "PDF"), (".zip", "ZIP"), (".csv", "CSV"),
        (".geojson", "GEOJSON"), (".gpkg", "GPKG"), (".shp", "SHP"),
        (".kml", "KML"), (".kmz", "KMZ"), (".json", "JSON"),
        (".html", "HTML"), (".xml", "XML"),
    ]:
        if ext in lower_url:
            return typ
    return "HTML"


def main():
    parser = argparse.ArgumentParser(description="v1ud — Download Manifest Builder")
    parser.add_argument("--resolution-registry", default="datasets/protocolo_c/v1ud_source_resolution_registry.csv")
    parser.add_argument("--policy-config", default="configs/protocolo_c/v1ud_download_policy.yaml")
    parser.add_argument("--domains-config", default="configs/protocolo_c/v1ud_allowed_domains.yaml")
    parser.add_argument("--local-only-dir", default="local_only/protocolo_c")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ud_download_manifest.csv")
    args = parser.parse_args()

    resolutions = load_csv(args.resolution_registry)
    policy = load_yaml(args.policy_config)
    domains_config = load_yaml(args.domains_config)

    domain_map = {d["domain"]: d for d in domains_config.get("allowed_domains", [])}
    global_max = policy.get("global_limits", {}).get("max_download_mb", 25)

    rows = []
    for idx, res in enumerate(resolutions):
        source_id = res["source_id"]
        event_id = res["event_id"]
        url = res["candidate_url"]
        decision = res["acquisition_decision"]

        parsed = urlparse(url)
        host = parsed.hostname or ""
        domain_info = domain_map.get(host, {})
        max_mb = min(domain_info.get("max_download_mb", global_max), global_max)

        raw_dir = f"evidence_raw/v1ud/{source_id}/{event_id}"
        staging_dir = f"evidence_staging/v1ud/{source_id}/{event_id}"

        if decision == "DOWNLOAD_CANDIDATE":
            allowed = True
            reason = "Resolved and approved for download"
        elif decision == "METADATA_ONLY":
            allowed = False
            reason = "Domain approved for metadata only"
        elif decision == "DRY_RUN":
            allowed = False
            reason = "Dry run — no download"
        else:
            allowed = False
            reason = res.get("blocking_reason", decision)

        rows.append({
            "manifest_id": f"MAN_{PROTOCOL_VERSION}_{idx:04d}",
            "source_id": source_id,
            "event_id": event_id,
            "url": url,
            "expected_type": infer_expected_type(url, res.get("content_type", "")),
            "max_download_mb": str(max_mb),
            "local_raw_target": raw_dir,
            "local_staging_target": staging_dir,
            "download_allowed": str(allowed),
            "reason": reason,
            "priority": res.get("priority", ""),
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    allowed_count = sum(1 for r in rows if r["download_allowed"] == "True")
    blocked_count = len(rows) - allowed_count

    print(f"[Manifest Builder v1ud] {len(rows)} entries")
    print(f"  download_allowed=True: {allowed_count}")
    print(f"  download_allowed=False: {blocked_count}")
    print(f"\nManifest: {args.out}")


if __name__ == "__main__":
    main()
