#!/usr/bin/env python3
"""
v1uj — Copernicus EMS Resolver

Aprofunda ativacoes de Copernicus EMS Rapid Mapping a partir do baseline
v1ui-live. Descobre paginas de ativacao, produtos e downloads e classifica
cada produto (delineation / grading / reference / map_pdf / vector_package /
raster_package / quicklook / metadata_only).

Quicklook NUNCA vira ground truth. Produto operacional publico pode virar no
maximo OBSERVED_GEOMETRY_CANDIDATE em etapa posterior, nunca ground reference.
Sem login, sem bypass, sem scraping agressivo. DRY_RUN sem --allow-web.
"""

import argparse
import csv
import os
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

PROTOCOL_VERSION = "v1uj"

EMS_COLUMNS = [
    "ems_record_id", "event_id", "activation_id", "activation_url",
    "product_url", "product_name", "product_type", "format_hint",
    "is_vector_candidate", "is_event_specific", "download_allowed",
    "blocking_reason", "notes",
]

VECTOR_FORMATS = {".zip", ".gpkg", ".geojson", ".shp", ".json", ".gml"}
RASTER_FORMATS = {".tif", ".tiff", ".jp2"}
QUICKLOOK_FORMATS = {".png", ".jpg", ".jpeg"}


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


class LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self._href = None
        self._text = ""

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self._href = v
                    self._text = ""

    def handle_data(self, data):
        if self._href is not None:
            self._text += data.strip()

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append((self._href, self._text))
            self._href = None
            self._text = ""


def classify_product(name, url):
    """Classifica produto Copernicus EMS por nome/URL. Funcao pura."""
    combined = (name + " " + url).upper()
    ext = os.path.splitext(urlparse(url).path)[1].lower()

    if ext in QUICKLOOK_FORMATS or "QUICKLOOK" in combined or "QUICKVIEW" in combined:
        return "quicklook", ext
    if ext in RASTER_FORMATS:
        return "raster_package", ext
    if ext in (".json", ".xml") or "METADATA" in combined:
        return "metadata_only", ext

    if "_DEL_" in combined or "DELINEATION" in combined:
        ptype = "delineation"
    elif "_GRA_" in combined or "GRADING" in combined:
        ptype = "grading"
    elif "_REF_" in combined or "REFERENCE" in combined:
        ptype = "reference"
    elif ext == ".pdf" or "MAP" in combined:
        ptype = "map_pdf"
    elif ext in VECTOR_FORMATS:
        ptype = "vector_package"
    else:
        ptype = "metadata_only"

    if ext == ".pdf" and ptype in ("delineation", "grading", "reference"):
        ptype = "map_pdf"
    return ptype, ext


def is_vector(product_type, ext):
    return (product_type in ("delineation", "grading", "reference", "vector_package")
            and ext in VECTOR_FORMATS)


def parse_products(html, base_url):
    """Extrai (url, name, type, ext) de uma pagina de ativacao. Funcao pura."""
    extractor = LinkExtractor()
    extractor.feed(html)
    out = []
    for href, text in extractor.links:
        full = urljoin(base_url, href)
        if "/download/" not in full and "/mapping/" not in full:
            ext = os.path.splitext(urlparse(full).path)[1].lower()
            if ext not in (VECTOR_FORMATS | RASTER_FORMATS | QUICKLOOK_FORMATS
                           | {".pdf", ".json", ".xml"}):
                continue
        ptype, ext = classify_product(text or full, full)
        out.append((full, text, ptype, ext))
    return out


def fetch_html(url, timeout=30):
    if not HAS_URLLIB:
        return ""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "REV-P-Academic-Research/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(1_000_000).decode("utf-8", errors="replace")
    except Exception:
        return ""


def download_allowed_for(ptype, cfg):
    dl = cfg.get("download", {})
    if ptype in ("delineation", "grading", "reference", "vector_package"):
        return dl.get("allow_vector_package", True)
    if ptype == "map_pdf":
        return dl.get("allow_map_pdf", True)
    if ptype == "quicklook":
        return dl.get("allow_quicklook", False)
    if ptype == "raster_package":
        return dl.get("allow_raster_package", False)
    return False


def main():
    parser = argparse.ArgumentParser(description="v1uj — Copernicus EMS Resolver")
    parser.add_argument("--config", default="configs/protocolo_c/v1uj_copernicus_ems_targets.yaml")
    parser.add_argument("--out", default="datasets/protocolo_c/v1uj_copernicus_ems_registry.csv")
    parser.add_argument("--html-fixture", default="",
                        help="HTML local para parse offline (testes)")
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    activations = cfg.get("candidate_activations", [])
    detail_tpls = cfg.get("base_urls", {}).get("activation_detail_template",
                                               ["https://emergency.copernicus.eu/mapping/list-of-components/{activation_id}"])
    detail_tpl = detail_tpls[0] if detail_tpls else ""

    fixture_html = ""
    if args.html_fixture and os.path.exists(args.html_fixture):
        with open(args.html_fixture, "r", encoding="utf-8") as f:
            fixture_html = f.read()

    rows = []
    seq = 0
    for act in activations:
        aid = act.get("activation_id", "")
        event_id = act.get("event_id", "")
        is_event_specific = act.get("is_event_specific", False)
        activation_url = detail_tpl.format(activation_id=aid)

        html = fixture_html
        if not html and args.allow_web:
            html = fetch_html(activation_url, args.timeout)

        if not html:
            rows.append({
                "ems_record_id": f"EMS_{PROTOCOL_VERSION}_{seq:04d}",
                "event_id": event_id, "activation_id": aid,
                "activation_url": activation_url, "product_url": "",
                "product_name": "", "product_type": "",
                "format_hint": "", "is_vector_candidate": "false",
                "is_event_specific": str(bool(is_event_specific)).lower(),
                "download_allowed": "false",
                "blocking_reason": "DRY_RUN" if not args.allow_web else "NO_HTML",
                "notes": "",
            })
            seq += 1
            continue

        for url, name, ptype, ext in parse_products(html, activation_url):
            vec = is_vector(ptype, ext)
            dl_ok = download_allowed_for(ptype, cfg)
            blocking = ""
            if ptype == "quicklook":
                blocking = "quicklook_not_ground_truth"
            elif ptype == "raster_package":
                blocking = "raster_not_allowed_this_stage"
            rows.append({
                "ems_record_id": f"EMS_{PROTOCOL_VERSION}_{seq:04d}",
                "event_id": event_id, "activation_id": aid,
                "activation_url": activation_url, "product_url": url,
                "product_name": name[:200], "product_type": ptype,
                "format_hint": ext, "is_vector_candidate": str(vec).lower(),
                "is_event_specific": str(bool(is_event_specific)).lower(),
                "download_allowed": str(bool(dl_ok)).lower(),
                "blocking_reason": blocking, "notes": "",
            })
            seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EMS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    vectors = sum(1 for r in rows if r["is_vector_candidate"] == "true")
    print(f"[Copernicus EMS Resolver v1uj] {len(rows)} products | vector_candidates={vectors}")
    print(f"  quickview_is_not_ground_truth=true")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
