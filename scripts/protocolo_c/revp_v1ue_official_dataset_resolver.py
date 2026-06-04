#!/usr/bin/env python3
"""
v1ue — Official Dataset Resolver

Resolves event/year/city-specific datasets (not just homepages).
Checks HTTP status, content type, classifies event/year/city specificity.
Never downloads here — only resolves and classifies.
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

PROTOCOL_VERSION = "v1ue"
USER_AGENT = (
    "REV-P-AcademicResearch/1.0 "
    "(TCC ground reference evidence acquisition; contact: academic use only)"
)

RESOLUTION_COLUMNS = [
    "dataset_resolution_id", "event_id", "source_id", "target_year",
    "target_city", "target_hazard", "query_terms", "candidate_url",
    "http_status", "content_type", "content_length", "dataset_type",
    "is_event_specific", "is_year_specific", "is_city_specific",
    "is_downloadable", "license_status", "resolution_status", "blocking_reason",
]


def load_yaml(path: str) -> dict:
    if yaml is None:
        print("[ERROR] pyyaml not installed")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sources_license(sources_config: dict) -> dict:
    out = {}
    for s in sources_config.get("sources", []):
        out[s["source_id"]] = s.get("license_status", "UNKNOWN_NEEDS_REVIEW")
    return out


def resolve(url: str, timeout: int, allow_web: bool) -> dict:
    if not allow_web:
        return {"http_status": "NOT_RESOLVED", "content_type": "", "content_length": ""}
    if requests is None:
        return {"http_status": "DEPENDENCY_MISSING", "content_type": "", "content_length": ""}
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        return {
            "http_status": str(resp.status_code),
            "content_type": resp.headers.get("Content-Type", ""),
            "content_length": resp.headers.get("Content-Length", ""),
        }
    except requests.exceptions.RequestException as e:
        return {"http_status": f"ERROR:{type(e).__name__}", "content_type": "", "content_length": ""}


def classify_specificity(target: dict, url: str) -> dict:
    year = target.get("target_year", "")
    city = target.get("target_city", "")
    hazard = target.get("target_hazard", "")
    lower_url = url.lower()

    is_year = year in url if year else False
    is_city = city.lower() in lower_url if city else False
    is_event = is_year and (is_city or hazard.lower() in lower_url)

    return {
        "is_year_specific": str(is_year).lower(),
        "is_city_specific": str(is_city).lower(),
        "is_event_specific": str(is_event).lower(),
    }


def main():
    parser = argparse.ArgumentParser(description="v1ue — Official Dataset Resolver")
    parser.add_argument("--targets-config", default="configs/protocolo_c/v1ue_official_dataset_targets.yaml")
    parser.add_argument("--sources-config", default="configs/protocolo_c/ground_reference_evidence_sources.yaml")
    parser.add_argument("--domains-config", default="configs/protocolo_c/v1ud_allowed_domains.yaml")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ue_official_dataset_resolution_registry.csv")
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--max-download-mb", type=float, default=50.0)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets_config = load_yaml(args.targets_config)
    sources_config = load_yaml(args.sources_config)
    domains_config = load_yaml(args.domains_config)

    license_map = load_sources_license(sources_config)
    domain_map = {d["domain"]: d for d in domains_config.get("allowed_domains", [])}

    rows = []
    seq = 0
    for target in targets_config.get("targets", []):
        source_id = target["source_id"]
        license_status = license_map.get(source_id, "UNKNOWN_NEEDS_REVIEW")

        for url in target.get("candidate_urls", []):
            parsed = urlparse(url)
            host = parsed.hostname or ""
            domain_info = domain_map.get(host, {})

            if args.dry_run:
                res = {"http_status": "DRY_RUN", "content_type": "", "content_length": ""}
            else:
                res = resolve(url, args.timeout, args.allow_web)

            spec = classify_specificity(target, url)

            expected_format = target.get("expected_format", "")
            is_downloadable = "false"
            if expected_format in ("ZIP", "CSV") and domain_info.get("download_allowed", False):
                is_downloadable = "true"

            if res["http_status"] == "DRY_RUN":
                resolution_status = "DRY_RUN"
                blocking = ""
            elif res["http_status"] == "NOT_RESOLVED":
                resolution_status = "NOT_RESOLVED"
                blocking = "Web access not enabled"
            elif res["http_status"].startswith("ERROR"):
                resolution_status = "RESOLUTION_FAILED"
                blocking = res["http_status"]
            elif res["http_status"].startswith(("4", "5")):
                resolution_status = "HTTP_ERROR"
                blocking = f"HTTP {res['http_status']}"
            elif spec["is_event_specific"] == "true":
                resolution_status = "EVENT_SPECIFIC_RESOLVED"
                blocking = ""
            elif spec["is_year_specific"] == "true":
                resolution_status = "YEAR_SPECIFIC_RESOLVED"
                blocking = ""
            else:
                resolution_status = "GENERIC_PORTAL"
                blocking = "Resolved but not event/year specific"

            if license_status == "UNKNOWN_NEEDS_REVIEW":
                blocking = (blocking + "; LICENSE_NEEDS_REVIEW").strip("; ")

            rows.append({
                "dataset_resolution_id": f"DSR_{PROTOCOL_VERSION}_{seq:04d}",
                "event_id": target.get("event_id", ""),
                "source_id": source_id,
                "target_year": target.get("target_year", ""),
                "target_city": target.get("target_city", ""),
                "target_hazard": target.get("target_hazard", ""),
                "query_terms": "|".join(target.get("query_terms", [])),
                "candidate_url": url,
                "http_status": res["http_status"],
                "content_type": res["content_type"],
                "content_length": res["content_length"],
                "dataset_type": target.get("dataset_type", ""),
                "is_event_specific": spec["is_event_specific"],
                "is_year_specific": spec["is_year_specific"],
                "is_city_specific": spec["is_city_specific"],
                "is_downloadable": is_downloadable,
                "license_status": license_status,
                "resolution_status": resolution_status,
                "blocking_reason": blocking,
            })
            seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESOLUTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    statuses = {}
    for r in rows:
        statuses[r["resolution_status"]] = statuses.get(r["resolution_status"], 0) + 1

    print(f"[Official Dataset Resolver v1ue] {len(rows)} datasets resolved")
    for s, c in sorted(statuses.items()):
        print(f"  {s}: {c}")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
