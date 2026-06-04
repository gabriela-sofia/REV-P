#!/usr/bin/env python3
"""
v1uj — CKAN / Dados Abertos Resolver

Detecta APIs CKAN e roda package_search por evento/regiao, lista resources e
classifica por formato. SHP/GeoJSON/GPKG/KML/KMZ/CSV/XLSX viram candidatos.
Infraestrutura/drenagem generica NAO e ocorrencia observada (is_contextual_only).
Download so depois, pelo downloader. DRY_RUN sem --allow-web.
"""

import argparse
import csv
import json
import os
from urllib.parse import urlencode

try:
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False

try:
    import yaml
except ImportError:
    yaml = None

PROTOCOL_VERSION = "v1uj"

CKAN_COLUMNS = [
    "ckan_record_id", "event_id", "portal_url", "package_id", "package_title",
    "resource_id", "resource_name", "resource_url", "resource_format",
    "is_geospatial_candidate", "is_event_specific", "is_contextual_only",
    "download_priority", "blocking_reason",
]


def load_yaml(path):
    if yaml is None or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def fetch_json(url, timeout=30):
    if not HAS_URLLIB:
        return None
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "REV-P-Academic-Research/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read(2_000_000).decode("utf-8"))
    except Exception:
        return None


def classify_resource(fmt, package_text, cfg):
    """Classifica resource CKAN por formato e contexto. Funcao pura."""
    fmt_u = (fmt or "").strip().upper().lstrip(".")
    geo = {f.upper() for f in cfg.get("geospatial_formats", [])}
    tab = {f.upper() for f in cfg.get("tabular_formats", [])}
    context_terms = [t.lower() for t in cfg.get("contextual_only_terms", [])]

    is_geo = fmt_u in geo
    is_tab = fmt_u in tab
    is_candidate = is_geo or is_tab
    is_contextual = any(t in (package_text or "").lower() for t in context_terms)

    if is_geo and not is_contextual:
        priority = "1"
    elif is_geo and is_contextual:
        priority = "3"
    elif is_tab:
        priority = "2"
    else:
        priority = "9"

    return is_candidate, is_contextual, priority


def parse_package_search(doc):
    """Extrai (package_id, title, notes, resources[]) de um package_search. Pura."""
    out = []
    result = doc.get("result", {}) if isinstance(doc, dict) else {}
    for pkg in result.get("results", []):
        out.append((
            pkg.get("id", ""),
            pkg.get("title", ""),
            (pkg.get("title", "") + " " + pkg.get("notes", "")),
            pkg.get("resources", []),
        ))
    return out


def main():
    parser = argparse.ArgumentParser(description="v1uj — CKAN Open Data Resolver")
    parser.add_argument("--config", default="configs/protocolo_c/v1uj_ckan_targets.yaml")
    parser.add_argument("--out", default="datasets/protocolo_c/v1uj_ckan_resource_registry.csv")
    parser.add_argument("--search-fixture", default="",
                        help="JSON local de package_search (testes offline)")
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    portals = cfg.get("portals", [])
    endpoints = cfg.get("ckan_endpoints", {})
    search_path = endpoints.get("package_search", "/api/3/action/package_search")
    queries_by_event = cfg.get("search_queries", {})
    max_rows = cfg.get("max_resources_per_package", 30)

    fixture_doc = None
    if args.search_fixture and os.path.exists(args.search_fixture):
        with open(args.search_fixture, "r", encoding="utf-8") as f:
            fixture_doc = json.load(f)

    rows = []
    seq = 0

    for portal in portals:
        portal_url = portal.get("base_url", "")
        applicable = portal.get("applicable_events", [])
        for ev_id in applicable:
            for query in queries_by_event.get(ev_id, []):
                doc = fixture_doc
                if doc is None and args.allow_web:
                    qs = urlencode({"q": query, "rows": cfg.get("max_packages_per_query", 25)})
                    url = f"{portal_url.rstrip('/')}{search_path}?{qs}"
                    doc = fetch_json(url, args.timeout)

                if not doc:
                    rows.append({
                        "ckan_record_id": f"CKAN_{PROTOCOL_VERSION}_{seq:04d}",
                        "event_id": ev_id, "portal_url": portal_url,
                        "package_id": "", "package_title": query,
                        "resource_id": "", "resource_name": "",
                        "resource_url": "", "resource_format": "",
                        "is_geospatial_candidate": "false",
                        "is_event_specific": "false",
                        "is_contextual_only": "false",
                        "download_priority": "9",
                        "blocking_reason": "DRY_RUN" if not args.allow_web else "NO_RESPONSE",
                    })
                    seq += 1
                    if fixture_doc is None:
                        continue
                    break  # fixture: nao iterar todas as queries

                for pkg_id, title, text, resources in parse_package_search(doc):
                    for res in resources[:max_rows]:
                        fmt = res.get("format", "")
                        is_cand, is_ctx, priority = classify_resource(fmt, text, cfg)
                        blocking = ""
                        if is_ctx:
                            blocking = "generic_infrastructure_not_occurrence"
                        elif not is_cand:
                            blocking = "non_candidate_format"
                        rows.append({
                            "ckan_record_id": f"CKAN_{PROTOCOL_VERSION}_{seq:04d}",
                            "event_id": ev_id, "portal_url": portal_url,
                            "package_id": pkg_id, "package_title": title[:200],
                            "resource_id": res.get("id", ""),
                            "resource_name": res.get("name", "")[:200],
                            "resource_url": res.get("url", ""),
                            "resource_format": fmt,
                            "is_geospatial_candidate": str(is_cand).lower(),
                            "is_event_specific": "true",
                            "is_contextual_only": str(is_ctx).lower(),
                            "download_priority": priority,
                            "blocking_reason": blocking,
                        })
                        seq += 1
                if fixture_doc is not None:
                    break  # fixture: uma query por portal/evento basta
            if fixture_doc is not None:
                break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CKAN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    candidates = sum(1 for r in rows if r["is_geospatial_candidate"] == "true")
    print(f"[CKAN Open Data Resolver v1uj] {len(rows)} resource records | candidates={candidates}")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
