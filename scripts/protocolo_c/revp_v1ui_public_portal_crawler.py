#!/usr/bin/env python3
"""
v1ui — Public Portal Crawler

Crawls discovered public portal URLs to extract artifact links.
Depth limited. Respects allowlist. No aggressive scraping.
When --allow-web is not set, runs in DRY_RUN mode.
"""

import argparse
import csv
import os
import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

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

CRAWL_COLUMNS = [
    "crawl_id", "event_id", "source_id", "parent_url", "discovered_url",
    "link_text", "extension", "detected_service_type",
    "relevance_score", "event_term_match", "locality_term_match",
    "hazard_term_match", "artifact_candidate_status", "notes",
]

ARTIFACT_EXTENSIONS = {
    ".zip", ".shp", ".gpkg", ".geojson", ".kml", ".kmz",
    ".csv", ".xlsx", ".xls", ".pdf", ".json", ".xml",
}

SERVICE_PATTERNS = [
    (r"/FeatureServer", "arcgis_featureserver"),
    (r"/MapServer", "arcgis_mapserver"),
    (r"/ows\b", "geoserver_ows"),
    (r"/wfs\b", "geoserver_wfs"),
    (r"/wms\b", "geoserver_wms"),
    (r"/package/", "ckan_package"),
    (r"/resource/", "ckan_resource"),
]

HAZARD_TERMS = {"inundacao", "alagamento", "enchente", "deslizamento",
                "flood", "landslide", "movimento", "transbordamento"}


class LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self._current_href = None
        self._current_text = ""

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self._current_href = v
                    self._current_text = ""

    def handle_data(self, data):
        if self._current_href is not None:
            self._current_text += data.strip()

    def handle_endtag(self, tag):
        if tag == "a" and self._current_href is not None:
            self.links.append((self._current_href, self._current_text))
            self._current_href = None
            self._current_text = ""


def fetch_html(url, timeout=30):
    if not HAS_URLLIB:
        return ""
    try:
        req = urllib.request.Request(url,
            headers={"User-Agent": "REV-P-Academic-Research/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if "text/html" not in resp.headers.get("Content-Type", ""):
                return ""
            return resp.read(500_000).decode("utf-8", errors="replace")
    except Exception:
        return ""


def score_link(url, text, search_terms):
    score = 0
    lower_url = url.lower()
    lower_text = text.lower()
    combined = lower_url + " " + lower_text

    ext = os.path.splitext(urlparse(url).path)[1].lower()
    if ext in ARTIFACT_EXTENSIONS:
        score += 30

    for pat, _ in SERVICE_PATTERNS:
        if re.search(pat, url, re.IGNORECASE):
            score += 25
            break

    event_match = hazard_match = locality_match = False
    terms = search_terms if search_terms else {}
    for t in terms.get("terms_pt", []) + terms.get("terms_en", []):
        if t.lower() in combined:
            score += 10
            event_match = True
            break

    for h in HAZARD_TERMS:
        if h in combined:
            score += 5
            hazard_match = True
            break

    city = terms.get("city", "").lower()
    if city and city in combined:
        score += 5
        locality_match = True

    return score, event_match, hazard_match, locality_match


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


def main():
    parser = argparse.ArgumentParser(description="v1ui — Public Portal Crawler")
    parser.add_argument("--discovery", default="datasets/protocolo_c/v1ui_public_discovery_registry.csv")
    parser.add_argument("--search-terms", default="configs/protocolo_c/v1ui_search_terms_by_event.yaml")
    parser.add_argument("--allowed-domains", default="configs/protocolo_c/v1ui_allowed_domains.yaml")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ui_public_artifact_download_manifest.csv")
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    discoveries = load_csv(args.discovery)
    terms_cfg = load_yaml(args.search_terms)
    allowed = load_allowed_domains(args.allowed_domains)
    events_terms = terms_cfg.get("events", {})

    rows = []
    seq = 0
    seen_urls = set()

    for disc in discoveries:
        parent_url = disc.get("candidate_url", "")
        event_id = disc.get("event_id", "")
        source_id = disc.get("source_id", "")
        ev_terms = events_terms.get(event_id, {})

        if not args.allow_web:
            rows.append({
                "crawl_id": f"CRAWL_{PROTOCOL_VERSION}_{seq:04d}",
                "event_id": event_id, "source_id": source_id,
                "parent_url": parent_url, "discovered_url": parent_url,
                "link_text": "", "extension": "",
                "detected_service_type": "DRY_RUN",
                "relevance_score": "0",
                "event_term_match": "false", "locality_term_match": "false",
                "hazard_term_match": "false",
                "artifact_candidate_status": "DRY_RUN", "notes": "",
            })
            seq += 1
            continue

        html = fetch_html(parent_url, args.timeout)
        if not html:
            rows.append({
                "crawl_id": f"CRAWL_{PROTOCOL_VERSION}_{seq:04d}",
                "event_id": event_id, "source_id": source_id,
                "parent_url": parent_url, "discovered_url": parent_url,
                "link_text": "", "extension": "",
                "detected_service_type": "FETCH_FAILED",
                "relevance_score": "0",
                "event_term_match": "false", "locality_term_match": "false",
                "hazard_term_match": "false",
                "artifact_candidate_status": "NO_HTML", "notes": "",
            })
            seq += 1
            continue

        extractor = LinkExtractor()
        extractor.feed(html)

        for href, text in extractor.links:
            full_url = urljoin(parent_url, href)
            domain = urlparse(full_url).hostname or ""
            if not any(domain.endswith(d) for d in allowed):
                continue
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            ext = os.path.splitext(urlparse(full_url).path)[1].lower()
            svc_type = ""
            for pat, stype in SERVICE_PATTERNS:
                if re.search(pat, full_url, re.IGNORECASE):
                    svc_type = stype
                    break

            score, ev_match, haz_match, loc_match = score_link(
                full_url, text, ev_terms)

            status = "CANDIDATE" if score >= 10 else "LOW_RELEVANCE"

            rows.append({
                "crawl_id": f"CRAWL_{PROTOCOL_VERSION}_{seq:04d}",
                "event_id": event_id, "source_id": source_id,
                "parent_url": parent_url, "discovered_url": full_url,
                "link_text": text[:200], "extension": ext,
                "detected_service_type": svc_type or ext or "html",
                "relevance_score": str(score),
                "event_term_match": str(ev_match).lower(),
                "locality_term_match": str(loc_match).lower(),
                "hazard_term_match": str(haz_match).lower(),
                "artifact_candidate_status": status, "notes": "",
            })
            seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CRAWL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    candidates = sum(1 for r in rows if r["artifact_candidate_status"] == "CANDIDATE")
    print(f"[Public Portal Crawler v1ui] {len(rows)} links | {candidates} candidates")
    print(f"\nManifest: {args.out}")


if __name__ == "__main__":
    main()
