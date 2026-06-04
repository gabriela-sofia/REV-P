#!/usr/bin/env python3
"""
v1uj — RIGeo / SGB Bitstream Resolver

Aprofunda o repositorio tecnico RIGeo (DSpace): resolve item pages e
bitstreams, extrai links de download, registra tamanho quando disponivel e
classifica por extensao. Diferencia relatorio PDF de pacote de anexos.

ZIP so com PDF = DOCUMENT_ONLY (auditado no inventory). ZIP com vetor = auditar.
Sem login, sem bypass. DRY_RUN sem --allow-web. Download via downloader.
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

RIGEO_COLUMNS = [
    "rigeo_record_id", "event_id", "item_url", "item_title",
    "bitstream_url", "bitstream_name", "bitstream_extension",
    "bitstream_class", "size_hint", "is_attachment_package",
    "is_geodata_candidate", "download_allowed", "blocking_reason",
]


def load_yaml(path):
    if yaml is None or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class BitstreamExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.title = ""
        self._href = None
        self._text = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self._href = v
                    self._text = ""
        elif tag in ("h1", "title"):
            self._in_title = True

    def handle_data(self, data):
        if self._href is not None:
            self._text += data.strip()
        if self._in_title and not self.title:
            self.title += data.strip()

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append((self._href, self._text))
            self._href = None
            self._text = ""
        elif tag in ("h1", "title"):
            self._in_title = False


def classify_bitstream(ext, cfg):
    """Classifica bitstream por extensao usando config. Funcao pura."""
    classes = cfg.get("bitstream_classes", {})
    for cls, exts in classes.items():
        if ext in [e.lower() for e in exts]:
            return cls
    return "other"


def download_allowed_for(cls, cfg):
    dl = cfg.get("download", {})
    if cls == "technical_report":
        return dl.get("allow_pdf", True)
    if cls == "attachment_package":
        return dl.get("allow_zip", True)
    if cls in ("geodata", "tabular"):
        return dl.get("allow_geodata", True)
    return False


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


def main():
    parser = argparse.ArgumentParser(description="v1uj — RIGeo Bitstream Resolver")
    parser.add_argument("--config", default="configs/protocolo_c/v1uj_rigeo_targets.yaml")
    parser.add_argument("--out", default="datasets/protocolo_c/v1uj_rigeo_bitstream_registry.csv")
    parser.add_argument("--item-fixture", default="",
                        help="HTML local de item page (testes offline)")
    parser.add_argument("--item-event", default="PET_2022_02_15",
                        help="event_id associado ao item fixture")
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    base_url = cfg.get("base_url", "https://rigeo.sgb.gov.br")
    geodata_exts = [e.lower() for e in cfg.get("bitstream_classes", {}).get("geodata", [])]

    rows = []
    seq = 0

    fixture_html = ""
    if args.item_fixture and os.path.exists(args.item_fixture):
        with open(args.item_fixture, "r", encoding="utf-8") as f:
            fixture_html = f.read()

    if not fixture_html and not args.allow_web:
        # DRY_RUN: registra termos de busca como itens nao resolvidos
        for ev_id, terms in cfg.get("search_terms", {}).items():
            rows.append({
                "rigeo_record_id": f"RIGEO_{PROTOCOL_VERSION}_{seq:04d}",
                "event_id": ev_id if ev_id != "general" else "",
                "item_url": cfg.get("discovery_paths", {}).get("simple_search", ""),
                "item_title": "; ".join(terms)[:200],
                "bitstream_url": "", "bitstream_name": "",
                "bitstream_extension": "", "bitstream_class": "",
                "size_hint": "", "is_attachment_package": "false",
                "is_geodata_candidate": "false", "download_allowed": "false",
                "blocking_reason": "DRY_RUN",
            })
            seq += 1
    else:
        item_pages = []
        if fixture_html:
            item_pages.append(("local_item_fixture", args.item_event, fixture_html))
        # web discovery de item pages reais ficaria aqui (mantido conservador)

        for item_url, event_id, html in item_pages:
            extractor = BitstreamExtractor()
            extractor.feed(html)
            title = extractor.title
            for href, text in extractor.links:
                full = urljoin(base_url, href)
                if "bitstream" not in full.lower():
                    continue
                ext = os.path.splitext(urlparse(full).path)[1].lower()
                cls = classify_bitstream(ext, cfg)
                is_pkg = cls == "attachment_package"
                # .shp.zip e geodado em pacote
                is_geo = ext in geodata_exts or (
                    is_pkg and any(g in full.lower() for g in ("shp", "geo", "setores", "risco")))
                dl_ok = download_allowed_for(cls, cfg)
                rows.append({
                    "rigeo_record_id": f"RIGEO_{PROTOCOL_VERSION}_{seq:04d}",
                    "event_id": event_id, "item_url": item_url,
                    "item_title": title[:200], "bitstream_url": full,
                    "bitstream_name": (text or os.path.basename(urlparse(full).path))[:200],
                    "bitstream_extension": ext, "bitstream_class": cls,
                    "size_hint": "", "is_attachment_package": str(is_pkg).lower(),
                    "is_geodata_candidate": str(bool(is_geo)).lower(),
                    "download_allowed": str(bool(dl_ok)).lower(),
                    "blocking_reason": "",
                })
                seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RIGEO_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    geo = sum(1 for r in rows if r["is_geodata_candidate"] == "true")
    print(f"[RIGeo Bitstream Resolver v1uj] {len(rows)} bitstream records | geodata_candidates={geo}")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
