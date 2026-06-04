#!/usr/bin/env python3
"""
v1ui — Public Source Discovery

Reads events and public source targets to build a discovery registry.
Resolves URLs when --allow-web is set; otherwise registers as DRY_RUN.
formal_request_path=LEGACY_SECONDARY_ONLY.
"""

import argparse
import csv
import os
from urllib.parse import urlparse

try:
    import yaml
except ImportError:
    yaml = None

try:
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False

PROTOCOL_VERSION = "v1ui"

DISCOVERY_COLUMNS = [
    "discovery_id", "event_id", "region", "source_id", "source_name",
    "source_class", "query_terms", "candidate_url", "domain",
    "http_status", "content_type", "content_length",
    "discovery_method", "public_access_status", "candidate_class",
    "event_specificity", "download_priority", "blocking_reason", "notes",
]

SOURCE_TARGET_COLUMNS = [
    "source_id", "source_name", "base_url", "applicable_events",
    "service_type", "priority", "status",
]


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
    allowed = cfg.get("allowed_domains", {})
    for group in allowed.values():
        if isinstance(group, list):
            domains.update(group)
    return domains


def probe_url(url, timeout=30):
    if not HAS_URLLIB:
        return 0, "", 0
    try:
        req = urllib.request.Request(url, method="HEAD",
            headers={"User-Agent": "REV-P-Academic-Research/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            ctype = resp.headers.get("Content-Type", "")
            clength = resp.headers.get("Content-Length", "0")
            return status, ctype, int(clength) if clength.isdigit() else 0
    except Exception:
        return 0, "", 0


def classify_source(source, _event):
    stype = source.get("service_type", "")
    if stype == "arcgis_portal":
        return "ARCGIS_REST_CANDIDATE"
    if stype == "ckan_portal":
        return "OPEN_DATA_PORTAL_CANDIDATE"
    if stype == "document_repository":
        return "DOCUMENT_REPOSITORY_CANDIDATE"
    return "CITY_PORTAL_PUBLIC_CANDIDATE"


def main():
    parser = argparse.ArgumentParser(description="v1ui — Public Source Discovery")
    parser.add_argument("--config", default="configs/protocolo_c/v1ui_public_source_targets.yaml")
    parser.add_argument("--events", default="datasets/protocolo_c/event_candidate_registry.csv")
    parser.add_argument("--allowed-domains", default="configs/protocolo_c/v1ui_allowed_domains.yaml")
    parser.add_argument("--out-dir", default="datasets/protocolo_c")
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    events = load_csv(args.events)
    allowed = load_allowed_domains(args.allowed_domains)
    sources = cfg.get("sources", [])
    event_ids = {e["event_id"]: e for e in events}

    discovery_rows = []
    target_rows = []
    seq = 0

    for source in sources:
        sid = source["source_id"]
        sname = source.get("name", sid)
        applicable = source.get("applicable_events", [])
        base_urls = source.get("base_urls", [])
        stype = source.get("service_type", "portal_publico")
        priority = source.get("priority", 5)

        for url in base_urls:
            target_rows.append({
                "source_id": sid, "source_name": sname,
                "base_url": url,
                "applicable_events": "|".join(applicable),
                "service_type": stype,
                "priority": str(priority), "status": "REGISTERED",
            })

        for ev_id in applicable:
            event = event_ids.get(ev_id, {})
            region = event.get("region", "")
            for url in base_urls:
                domain = urlparse(url).hostname or ""
                domain_ok = any(domain.endswith(d) for d in allowed) if allowed else True

                http_status, ctype, clength = 0, "", 0
                method = "DRY_RUN"
                if args.allow_web and domain_ok:
                    http_status, ctype, clength = probe_url(url, args.timeout)
                    method = "HEAD_PROBE"

                cclass = classify_source(source, event)
                blocking = "" if domain_ok else "domain_not_allowed"

                discovery_rows.append({
                    "discovery_id": f"DISC_{PROTOCOL_VERSION}_{seq:04d}",
                    "event_id": ev_id, "region": region,
                    "source_id": sid, "source_name": sname,
                    "source_class": stype,
                    "query_terms": event.get("city", ""),
                    "candidate_url": url, "domain": domain,
                    "http_status": str(http_status),
                    "content_type": ctype,
                    "content_length": str(clength),
                    "discovery_method": method,
                    "public_access_status": "PUBLIC" if domain_ok else "BLOCKED",
                    "candidate_class": cclass,
                    "event_specificity": "SOURCE_LEVEL",
                    "download_priority": str(priority),
                    "blocking_reason": blocking, "notes": "",
                })
                seq += 1

    os.makedirs(args.out_dir, exist_ok=True)

    target_path = os.path.join(args.out_dir, "v1ui_public_source_target_registry.csv")
    with open(target_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SOURCE_TARGET_COLUMNS)
        writer.writeheader()
        writer.writerows(target_rows)

    disc_path = os.path.join(args.out_dir, "v1ui_public_discovery_registry.csv")
    with open(disc_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DISCOVERY_COLUMNS)
        writer.writeheader()
        writer.writerows(discovery_rows)

    print(f"[Public Source Discovery v1ui] {len(target_rows)} targets | {len(discovery_rows)} discoveries")
    print(f"  formal_request_path=LEGACY_SECONDARY_ONLY")
    print(f"  can_create_ground_reference=false")
    print(f"\nTargets: {target_path}")
    print(f"Discovery: {disc_path}")


if __name__ == "__main__":
    main()
