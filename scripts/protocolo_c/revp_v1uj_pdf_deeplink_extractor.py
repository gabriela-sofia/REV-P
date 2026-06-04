#!/usr/bin/env python3
"""
v1uj — PDF Deep Link Extractor

Le PDFs baixados na v1ui-live (e v1uj) em local_only, extrai texto com pypdf e
procura URLs internas e mencoes a anexos / shapefile / geodados / geojson /
kmz / kml / gpkg / mapa interativo / ArcGIS / GeoServer / dados abertos.

NAO faz OCR massivo. Aceita .txt como texto ja extraido (fixtures/testes).
Registra PDF_LINK_CANDIDATE. Sem rede.
"""

import argparse
import csv
import os
import re
from urllib.parse import urlparse

try:
    import yaml
except ImportError:
    yaml = None

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

PROTOCOL_VERSION = "v1uj"

DEEPLINK_COLUMNS = [
    "deeplink_id", "source_file", "event_id", "link_url", "link_domain",
    "domain_allowed", "term_matches", "has_geodata_term",
    "is_pdf_link_candidate", "text_extract_status", "blocking_reason", "notes",
]

GEODATA_TERMS = [
    "anexo", "shapefile", "geodados", "geojson", "kmz", "kml", "gpkg",
    "mapa interativo", "arcgis", "geoserver", "dados abertos", "shp", "geopackage",
]

URL_RE = re.compile(r"https?://[^\s\)\]\}<>\"']+", re.IGNORECASE)


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


def extract_text_from_pdf(path):
    """Extrai texto de um PDF via pypdf. Retorna (texto, status)."""
    if not HAS_PYPDF:
        return "", "NO_PYPDF"
    try:
        reader = pypdf.PdfReader(path)
        parts = []
        for page in reader.pages[:50]:
            parts.append(page.extract_text() or "")
        return "\n".join(parts), "OK"
    except Exception as e:
        return "", f"ERROR:{str(e)[:50]}"


def find_links(text):
    """Encontra URLs http(s) no texto. Funcao pura."""
    seen = []
    for m in URL_RE.findall(text or ""):
        url = m.rstrip(".,;")
        if url not in seen:
            seen.append(url)
    return seen


def find_terms(text):
    """Encontra termos de geodados no texto. Funcao pura."""
    lower = (text or "").lower()
    return [t for t in GEODATA_TERMS if t in lower]


def event_from_path(path, raw_root):
    """Deriva event_id a partir do layout .../<source>/<event>/file."""
    rel = os.path.relpath(path, raw_root)
    parts = rel.replace("\\", "/").split("/")
    for p in parts:
        if p.startswith(("PET_", "REC_")):
            return p
    return ""


def iter_source_files(raw_dirs):
    for raw_dir in raw_dirs:
        if not os.path.isdir(raw_dir):
            continue
        for root, _dirs, files in os.walk(raw_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in (".pdf", ".txt"):
                    yield os.path.join(root, fname), raw_dir


def main():
    parser = argparse.ArgumentParser(description="v1uj — PDF Deep Link Extractor")
    parser.add_argument("--raw-dirs", nargs="*", default=[
        "local_only/protocolo_c/public_official_artifacts",
        "local_only/protocolo_c/focused_public_artifacts/raw/v1uj",
    ])
    parser.add_argument("--allowed-domains", default="configs/protocolo_c/v1ui_allowed_domains.yaml")
    parser.add_argument("--out", default="datasets/protocolo_c/v1uj_pdf_deeplink_registry.csv")
    args = parser.parse_args()

    allowed = load_allowed_domains(args.allowed_domains)

    rows = []
    seq = 0
    for fpath, raw_root in iter_source_files(args.raw_dirs):
        ext = os.path.splitext(fpath)[1].lower()
        if ext == ".pdf":
            text, status = extract_text_from_pdf(fpath)
        else:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            status = "OK"

        event_id = event_from_path(fpath, raw_root)
        links = find_links(text)
        terms = find_terms(text)
        has_geo = bool(terms)

        if not links and not terms:
            rows.append({
                "deeplink_id": f"PDFL_{PROTOCOL_VERSION}_{seq:04d}",
                "source_file": os.path.basename(fpath), "event_id": event_id,
                "link_url": "", "link_domain": "", "domain_allowed": "",
                "term_matches": "", "has_geodata_term": "false",
                "is_pdf_link_candidate": "false",
                "text_extract_status": status,
                "blocking_reason": "no_links_or_terms" if status == "OK" else status,
                "notes": "",
            })
            seq += 1
            continue

        if not links and terms:
            rows.append({
                "deeplink_id": f"PDFL_{PROTOCOL_VERSION}_{seq:04d}",
                "source_file": os.path.basename(fpath), "event_id": event_id,
                "link_url": "", "link_domain": "", "domain_allowed": "",
                "term_matches": "|".join(terms), "has_geodata_term": "true",
                "is_pdf_link_candidate": "true",
                "text_extract_status": status,
                "blocking_reason": "term_mention_no_link", "notes": "",
            })
            seq += 1
            continue

        for url in links:
            domain = urlparse(url).hostname or ""
            domain_ok = any(domain.endswith(d) for d in allowed) if allowed else False
            rows.append({
                "deeplink_id": f"PDFL_{PROTOCOL_VERSION}_{seq:04d}",
                "source_file": os.path.basename(fpath), "event_id": event_id,
                "link_url": url, "link_domain": domain,
                "domain_allowed": str(domain_ok).lower(),
                "term_matches": "|".join(terms),
                "has_geodata_term": str(has_geo).lower(),
                "is_pdf_link_candidate": "true",
                "text_extract_status": status,
                "blocking_reason": "" if domain_ok else "domain_not_allowed",
                "notes": "",
            })
            seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DEEPLINK_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    candidates = sum(1 for r in rows if r["is_pdf_link_candidate"] == "true")
    print(f"[PDF Deep Link Extractor v1uj] {len(rows)} entries | link_candidates={candidates}")
    print(f"  pypdf={'available' if HAS_PYPDF else 'MISSING'} | no_massive_ocr=true")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
