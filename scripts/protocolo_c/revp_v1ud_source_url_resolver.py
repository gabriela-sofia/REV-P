#!/usr/bin/env python3
"""
v1ud — Source URL Resolver

Resolves base URLs to candidate URLs, checks HTTP status, content type,
and domain allowlists. Outputs source_resolution_registry.csv.
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

try:
    import requests
except ImportError:
    requests = None

PROTOCOL_VERSION = "v1ud"
USER_AGENT = (
    "REV-P-AcademicResearch/1.0 "
    "(TCC ground reference evidence acquisition; contact: academic use only)"
)

RESOLUTION_COLUMNS = [
    "resolution_id", "source_id", "event_id", "priority", "base_url",
    "candidate_url", "http_status", "content_type", "content_length",
    "resolved_method", "allowed_domain", "domain_category",
    "license_status", "acquisition_decision", "blocking_reason", "notes",
]


def load_yaml(path: str) -> dict:
    if yaml is None:
        print("[ERROR] pyyaml not installed")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_domain_index(domains_config: dict) -> dict:
    index = {}
    for entry in domains_config.get("allowed_domains", []):
        index[entry["domain"]] = entry
    return index


def check_domain(url: str, domain_index: dict) -> tuple[bool, str, dict]:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if host in domain_index:
        return True, host, domain_index[host]
    for domain, info in domain_index.items():
        if host.endswith("." + domain) or host == domain:
            return True, host, info
    return False, host, {}


def resolve_url(url: str, timeout: int, dry_run: bool) -> dict:
    if dry_run:
        return {
            "http_status": "DRY_RUN",
            "content_type": "",
            "content_length": "",
            "resolved_method": "dry_run",
        }
    if requests is None:
        return {
            "http_status": "DEPENDENCY_MISSING",
            "content_type": "",
            "content_length": "",
            "resolved_method": "none",
        }
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        return {
            "http_status": str(resp.status_code),
            "content_type": resp.headers.get("Content-Type", ""),
            "content_length": resp.headers.get("Content-Length", ""),
            "resolved_method": "HEAD",
        }
    except requests.exceptions.RequestException as e:
        return {
            "http_status": f"ERROR:{type(e).__name__}",
            "content_type": "",
            "content_length": "",
            "resolved_method": "HEAD_FAILED",
        }


def main():
    parser = argparse.ArgumentParser(description="v1ud — Source URL Resolver")
    parser.add_argument("--sources-config", default="configs/protocolo_c/ground_reference_evidence_sources.yaml")
    parser.add_argument("--targets-config", default="configs/protocolo_c/v1ud_real_acquisition_targets.yaml")
    parser.add_argument("--domains-config", default="configs/protocolo_c/v1ud_allowed_domains.yaml")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ud_source_resolution_registry.csv")
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sources_config = load_yaml(args.sources_config)
    targets_config = load_yaml(args.targets_config)
    domains_config = load_yaml(args.domains_config)
    domain_index = build_domain_index(domains_config)

    source_map = {s["source_id"]: s for s in sources_config.get("sources", [])}

    rows = []
    seq = 0
    for priority_key in ["priority_1", "priority_2", "priority_3"]:
        priority_num = priority_key.split("_")[1]
        for source_block in targets_config.get(priority_key, []):
            source_id = source_block["source_id"]
            source_info = source_map.get(source_id, {})
            license_status = source_info.get("license_status", "UNKNOWN_NEEDS_REVIEW")

            for target in source_block.get("targets", []):
                event_id = target["event_id"]
                for url in target.get("candidate_urls", []):
                    allowed, host, domain_info = check_domain(url, domain_index)
                    resolution = resolve_url(url, args.timeout, args.dry_run)

                    if not allowed:
                        decision = "BLOCKED_DOMAIN"
                        blocking = f"Domain {host} not in allowed_domains"
                    elif resolution["http_status"].startswith("ERROR"):
                        decision = "RESOLUTION_FAILED"
                        blocking = resolution["http_status"]
                    elif resolution["http_status"] == "DRY_RUN":
                        decision = "DRY_RUN"
                        blocking = ""
                    elif resolution["http_status"].startswith(("4", "5")):
                        decision = "HTTP_ERROR"
                        blocking = f"HTTP {resolution['http_status']}"
                    elif license_status in ("UNKNOWN_NEEDS_REVIEW",):
                        decision = "LICENSE_NEEDS_REVIEW"
                        blocking = "License not reviewed"
                    elif not domain_info.get("download_allowed", False):
                        decision = "METADATA_ONLY"
                        blocking = "Domain not approved for download"
                    else:
                        decision = "DOWNLOAD_CANDIDATE"
                        blocking = ""

                    rows.append({
                        "resolution_id": f"RES_{PROTOCOL_VERSION}_{seq:04d}",
                        "source_id": source_id,
                        "event_id": event_id,
                        "priority": priority_num,
                        "base_url": url,
                        "candidate_url": url,
                        "http_status": resolution["http_status"],
                        "content_type": resolution["content_type"],
                        "content_length": resolution["content_length"],
                        "resolved_method": resolution["resolved_method"],
                        "allowed_domain": str(allowed),
                        "domain_category": domain_info.get("category", "UNKNOWN"),
                        "license_status": license_status,
                        "acquisition_decision": decision,
                        "blocking_reason": blocking,
                        "notes": target.get("notes", ""),
                    })
                    seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESOLUTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    decisions = {}
    for r in rows:
        d = r["acquisition_decision"]
        decisions[d] = decisions.get(d, 0) + 1

    print(f"[URL Resolver v1ud] Resolved {len(rows)} URLs")
    for d, c in sorted(decisions.items()):
        print(f"  {d}: {c}")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
