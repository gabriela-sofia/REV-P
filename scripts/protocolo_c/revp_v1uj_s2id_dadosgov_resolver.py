#!/usr/bin/env python3
"""
v1uj — S2iD / dados.gov.br Resolver

Busca recursos publicos de desastres reconhecidos, ocorrencias, COBRADE,
decretos e registros municipais. Resolve via CKAN (dados.gov.br) e/ou analisa
CSV/XLSX tabular.

Classificacao: disaster_event_registry / recognition_record /
municipal_level_context / no_geometry / table_with_coordinates_candidate.

Municipio NAO e geometria de inundacao. CSV/XLSX com geocodigo municipal/data
e evidencia documental/temporal, nunca geometria de ocorrencia patch-level.
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

S2ID_COLUMNS = [
    "s2id_record_id", "event_id", "portal_url", "package_id",
    "resource_name", "resource_url", "resource_format", "record_class",
    "has_coordinate_columns", "municipio_field_present", "cobrade_field_present",
    "date_field_present", "is_event_specific", "is_geometry_of_occurrence",
    "download_priority", "blocking_reason",
]

COORD_TERMS = {"lat", "latitude", "lon", "long", "longitude", "x", "y"}
MUNI_TERMS = {"municipio", "município", "ibge", "ibge_code", "cod_ibge", "geocodigo"}
DATE_TERMS = {"data", "data_registro", "data_ocorr", "data_evento", "ano"}
COBRADE_TERMS = {"cobrade", "cod_cobrade", "codigo_cobrade"}


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


def classify_table(header, sample_rows):
    """Classifica uma tabela S2iD por colunas. Funcao pura.

    Municipio nunca e geometria; coordenadas com valores -> candidate.
    """
    hl = [h.strip().lower() for h in (header or [])]
    has_coord_cols = any(h in COORD_TERMS for h in hl)
    has_muni = any(h in MUNI_TERMS for h in hl)
    has_cobrade = any(h in COBRADE_TERMS for h in hl)
    has_date = any(h in DATE_TERMS for h in hl)

    # coordenadas precisam ter valor preenchido em pelo menos uma linha
    coord_filled = False
    if has_coord_cols and sample_rows:
        coord_idx = [i for i, h in enumerate(hl) if h in COORD_TERMS]
        for r in sample_rows:
            if any(i < len(r) and str(r[i]).strip() not in ("", "nan", "none") for i in coord_idx):
                coord_filled = True
                break

    if coord_filled:
        record_class = "table_with_coordinates_candidate"
    elif has_cobrade and has_muni:
        record_class = "disaster_event_registry"
    elif has_muni and has_date:
        record_class = "recognition_record"
    elif has_muni:
        record_class = "municipal_level_context"
    else:
        record_class = "no_geometry"

    return {
        "record_class": record_class,
        "has_coord_cols": coord_filled,
        "has_muni": has_muni,
        "has_cobrade": has_cobrade,
        "has_date": has_date,
    }


def read_csv_header_and_rows(path, max_rows=50):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = []
        for i, r in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(r)
    return header, rows


def main():
    parser = argparse.ArgumentParser(description="v1uj — S2iD / dados.gov.br Resolver")
    parser.add_argument("--config", default="configs/protocolo_c/v1uj_s2id_targets.yaml")
    parser.add_argument("--out", default="datasets/protocolo_c/v1uj_s2id_resource_registry.csv")
    parser.add_argument("--csv-fixture", default="",
                        help="CSV tabular local (testes offline)")
    parser.add_argument("--search-fixture", default="",
                        help="JSON local de package_search CKAN (testes offline)")
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    portals = cfg.get("portals", [])
    queries = cfg.get("search_queries", [])
    municipalities = cfg.get("event_municipalities", {})

    rows = []
    seq = 0

    # (1) Analise tabular direta (CSV fixture ou recurso tabular conhecido)
    if args.csv_fixture and os.path.exists(args.csv_fixture):
        header, sample = read_csv_header_and_rows(args.csv_fixture)
        cl = classify_table(header, sample)
        # mapear evento por municipio presente na amostra
        muni_text = " ".join(" ".join(r) for r in sample).lower()
        matched_event = ""
        for ev_id, info in municipalities.items():
            if str(info.get("municipio", "")).lower() in muni_text:
                matched_event = ev_id
                break
        blocking = "municipality_is_not_occurrence_geometry"
        if cl["record_class"] == "table_with_coordinates_candidate":
            blocking = ""
        rows.append({
            "s2id_record_id": f"S2ID_{PROTOCOL_VERSION}_{seq:04d}",
            "event_id": matched_event,
            "portal_url": "local_csv_fixture",
            "package_id": "", "resource_name": os.path.basename(args.csv_fixture),
            "resource_url": "", "resource_format": "CSV",
            "record_class": cl["record_class"],
            "has_coordinate_columns": str(cl["has_coord_cols"]).lower(),
            "municipio_field_present": str(cl["has_muni"]).lower(),
            "cobrade_field_present": str(cl["has_cobrade"]).lower(),
            "date_field_present": str(cl["has_date"]).lower(),
            "is_event_specific": str(bool(matched_event)).lower(),
            "is_geometry_of_occurrence": "false",
            "download_priority": "2" if cl["record_class"] == "table_with_coordinates_candidate" else "5",
            "blocking_reason": blocking,
        })
        seq += 1

    # (2) Descoberta CKAN em dados.gov.br
    search_doc = None
    if args.search_fixture and os.path.exists(args.search_fixture):
        with open(args.search_fixture, "r", encoding="utf-8") as f:
            search_doc = json.load(f)

    for portal in portals:
        api_path = portal.get("api_action_path")
        if not api_path:
            continue
        portal_url = portal.get("base_url", "")
        for query in queries:
            doc = search_doc
            if doc is None and args.allow_web:
                qs = urlencode({"q": query, "rows": cfg.get("max_packages_per_query", 25)})
                url = f"{portal_url.rstrip('/')}{api_path}/package_search?{qs}"
                doc = fetch_json(url, args.timeout)
            if not doc:
                if not args.allow_web and search_doc is None:
                    rows.append({
                        "s2id_record_id": f"S2ID_{PROTOCOL_VERSION}_{seq:04d}",
                        "event_id": "", "portal_url": portal_url,
                        "package_id": "", "resource_name": query,
                        "resource_url": "", "resource_format": "",
                        "record_class": "no_geometry",
                        "has_coordinate_columns": "false",
                        "municipio_field_present": "false",
                        "cobrade_field_present": "false",
                        "date_field_present": "false",
                        "is_event_specific": "false",
                        "is_geometry_of_occurrence": "false",
                        "download_priority": "9", "blocking_reason": "DRY_RUN",
                    })
                    seq += 1
                continue
            result = doc.get("result", {})
            for pkg in result.get("results", []):
                for res in pkg.get("resources", []):
                    fmt = (res.get("format", "") or "").upper()
                    rclass = "no_geometry"
                    if fmt in ("CSV", "XLSX", "XLS", "JSON"):
                        rclass = "municipal_level_context"
                    rows.append({
                        "s2id_record_id": f"S2ID_{PROTOCOL_VERSION}_{seq:04d}",
                        "event_id": "", "portal_url": portal_url,
                        "package_id": pkg.get("id", ""),
                        "resource_name": res.get("name", "")[:200],
                        "resource_url": res.get("url", ""),
                        "resource_format": fmt, "record_class": rclass,
                        "has_coordinate_columns": "false",
                        "municipio_field_present": "false",
                        "cobrade_field_present": "false",
                        "date_field_present": "false",
                        "is_event_specific": "false",
                        "is_geometry_of_occurrence": "false",
                        "download_priority": "4",
                        "blocking_reason": "municipality_is_not_occurrence_geometry",
                    })
                    seq += 1
            if search_doc is not None:
                break
        if search_doc is not None:
            break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=S2ID_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    coord = sum(1 for r in rows if r["record_class"] == "table_with_coordinates_candidate")
    print(f"[S2iD/dados.gov Resolver v1uj] {len(rows)} records | coord_table_candidates={coord}")
    print(f"  municipality_is_not_occurrence_geometry=true")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
