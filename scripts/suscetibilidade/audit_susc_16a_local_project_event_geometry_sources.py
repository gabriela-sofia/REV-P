"""SUSC-16A local/project footprint rescue utilities and phase 1 runner.

This module intentionally keeps SUSC-16A fail-closed. It scans local and public
metadata sources, records only manifests/reports, and never copies heavy raw
data, creates ground truth, trains a model, or creates score v7.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DAT = ROOT / "datasets" / "suscetibilidade"
MAN = ROOT / "manifests" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"

LOCAL_AUDIT = OUT / "SUSC_16A_local_project_event_geometry_source_audit.csv"
LOCAL_REPORT = OUT / "SUSC_16A_local_project_event_geometry_source_report.md"
LOCAL_CANDIDATES = DAT / "susc_16a_local_event_geometry_candidates_v1.csv"
LOCAL_PARSE_AUDIT = OUT / "SUSC_16A_local_event_geometry_parse_audit.csv"
EXTERNAL_DISCOVERY = OUT / "SUSC_16A_external_flood_footprint_discovery_report.md"
EXTERNAL_MANIFEST = MAN / "susc_16a_external_flood_footprint_download_manifest_v1.csv"
EXTERNAL_DIR = DAT / "external_flood_footprints_susc16a"
WINDOW_PLAN = DAT / "susc_16a_sentinel_event_window_plan_v1.csv"
WINDOW_REPORT = OUT / "SUSC_16A_sentinel_event_window_plan_report.md"
S1_TARGETS = MAN / "susc_16a_sentinel1_flood_mapping_targets_v1.csv"
S1_TARGETS_REPORT = OUT / "SUSC_16A_sentinel1_flood_mapping_targets_report.md"
GEE_STUB = HERE / "susc_16a_gee_sentinel1_flood_mapping_stub.js"
STAC_STUB = HERE / "susc_16a_stac_sentinel1_query_stub.py"
SAR_CANDIDATES = DAT / "susc_16a_local_sar_flood_footprint_candidates_v1.csv"
SAR_REPORT = OUT / "SUSC_16A_local_sar_flood_canary_report.md"
FOOTPRINT_CATALOG = DAT / "susc_16a_observed_footprint_catalog_v1.csv"
FOOTPRINT_MANIFEST = MAN / "susc_16a_observed_footprint_catalog_manifest_v1.json"
FOOTPRINT_AUDIT = OUT / "SUSC_16A_observed_footprint_catalog_audit.csv"
LINKAGE = DAT / "susc_16a_footprint_patch_linkage_v1.csv"
LINKAGE_SUMMARY = OUT / "SUSC_16A_footprint_patch_linkage_summary.csv"
LINKAGE_LIMITS = OUT / "SUSC_16A_footprint_patch_linkage_limitations.json"
LINKAGE_GEOJSON = OUT / "SUSC_16A_footprint_patch_linkage_geojson.geojson"
SCORE_DIAG = OUT / "SUSC_16A_footprint_score_diagnostics.csv"
SCORE_DIAG_REGION = OUT / "SUSC_16A_footprint_score_diagnostics_by_region.csv"
SCORE_SUMMARY = OUT / "SUSC_16A_footprint_score_diagnostics_summary.json"
READINESS_MD = OUT / "SUSC_16A_observational_readiness.md"
FINAL_REPORT = OUT / "SUSC_16A_observed_footprint_rescue_report.md"

MATRIX = DAT / "susc_features_by_patch_v1.csv"
SCORE = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
S13C = DAT / "susc_13c_live_observed_events_parsed_v1.csv"
S15A = DAT / "susc_15a_precision_geocoded_occurrences_v1.csv"
S15B = DAT / "susc_15b_precision_event_catalog_v1.csv"

MANDATORY = (
    "O SUSC-16A substitui a tentativa de geocodificacao textual por uma "
    "estrategia de footprints observacionais, combinando geometrias locais, "
    "fontes oficiais/tecnicas e planejamento Sentinel/SAR. A etapa mantem "
    "todos os vinculos review-only, nao cria ground truth, nao libera treino "
    "supervisionado e nao cria score v7 automatico."
)
GOV = {
    "allowed_for_training": "false",
    "can_be_ground_truth": "false",
    "can_be_used_as_ground_truth": "false",
    "review_only": "true",
}

SCAN_EXTS = {
    ".geojson", ".json", ".gpkg", ".shp", ".dbf", ".prj", ".kml", ".kmz",
    ".csv", ".xlsx", ".gml", ".xml", ".wkt", ".pdf",
}
SAR_EXTS = {".tif", ".tiff", ".npy", ".npz"}
EVENT_TERMS = [
    "alagamento", "alagamentos", "inundacao", "inundação", "enchente",
    "enxurrada", "cheia", "area alagada", "área alagada", "area atingida",
    "área atingida", "ocorrencia", "ocorrência", "defesa civil",
    "ponto critico", "ponto crítico", "flood", "flood_extent",
    "flood_footprint",
]
PRIORITY_SUBPATHS = [
    "data/raw/recife/seced_defesa_civil",
    "data/raw/recife/official_address_base",
    "data/raw/recife/sgb_cprm",
    "data/raw/recife/esig_emlurb_drenagem",
    "data/raw/recife/esig_emlurb_macrodrenagem",
    "data/raw/curitiba/geodc_defesa_civil",
    "data/raw/curitiba/geocuritiba",
    "data/raw/curitiba/sgb_cprm",
    "data/raw/petropolis/sgb_cprm",
    "data/raw/petropolis/geinea_rj25",
]
CITY_BBOX = {
    "recife": (-35.05, -8.20, -34.80, -7.90),
    "curitiba": (-49.45, -25.65, -49.15, -25.30),
    "petropolis": (-43.35, -22.60, -43.00, -22.20),
}


def ensure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    ensure(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, obj: dict) -> None:
    ensure(path)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    ensure(path)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def sha256_file(path: Path, limit_mb: int = 250) -> str:
    try:
        if path.stat().st_size > limit_mb * 1024 * 1024:
            return ""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def path_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def safe_path(path: Path, root: Path | None = None) -> str:
    try:
        if root:
            return "PROJECT_ROOT/" + str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except (ValueError, OSError):
        return "EXTERNAL_PATH/" + path.name


def norm(text: str) -> str:
    return (text or "").lower().replace("ã", "a").replace("á", "a").replace("â", "a").replace("ç", "c").replace("ê", "e").replace("é", "e")


def infer_region(text: str) -> str:
    t = norm(text)
    for region in ("recife", "curitiba", "petropolis"):
        if region in t or (region == "petropolis" and "petrópolis" in text.lower()):
            return region
    return "unknown"


def source_class(path: Path, text_hint: str) -> str:
    t = norm(str(path) + " " + text_hint)
    if any(x in t for x in ("alag", "inund", "enchente", "cheia", "flood_extent", "flood_footprint", "area atingida")):
        if path.suffix.lower() in {".geojson", ".json", ".gpkg", ".shp", ".kml", ".kmz", ".wkt", ".gml"}:
            return "official_flood_footprint_candidate"
        return "local_event_polygon_candidate"
    if "ponto" in t and any(x in t for x in ("critico", "ocorr", "defesa civil")):
        return "local_event_point_candidate"
    if "risco" in t:
        return "official_risk_area_only"
    if "suscet" in t or "suscept" in t:
        return "official_susceptibility_map_only"
    if "dren" in t:
        return "official_drainage_context"
    if "address" in t or "endereco" in t or "cadastro" in t:
        return "official_address_reference"
    if path.suffix.lower() in {".tif", ".tiff"} or path.stat().st_size > 250 * 1024 * 1024:
        return "unsupported_or_heavy"
    return "non_event_context"


def text_sample(path: Path, max_bytes: int = 65536) -> str:
    if path.suffix.lower() in {".gpkg", ".shp", ".dbf", ".kmz", ".xlsx", ".pdf"}:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_bytes]
    except OSError:
        return ""


def find_project_roots() -> list[Path]:
    candidates = [
        ROOT.parent / "PROJETO",
        Path("C:/Users/gabriela/Documents/PROJETO"),
        ROOT / "PROJETO",
    ]
    roots = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved not in roots:
            roots.append(resolved)
    return roots


def audit_local_sources() -> int:
    fields = [
        "source_id", "root_label", "source_path", "path_sha256", "extension",
        "size_bytes", "sha256", "region", "priority_path_match",
        "term_match_count", "matched_terms", "source_class", "can_parse_geometry",
        "blocked_reason", "review_only",
    ]
    rows = []
    roots = find_project_roots()
    scan_roots = [(root, "PROJETO") for root in roots]
    scan_roots.append((ROOT, "REV-P"))
    idx = 0
    for root, label in scan_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext not in SCAN_EXTS:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            sample = text_sample(path)
            haystack = norm(str(path) + " " + sample[:4096])
            terms = [term for term in EVENT_TERMS if norm(term) in haystack]
            priority = any(str(path).replace("\\", "/").lower().endswith(p.lower()) or p.lower() in str(path).replace("\\", "/").lower() for p in PRIORITY_SUBPATHS)
            cls = source_class(path, sample if terms else "")
            if not terms and not priority and cls in {"non_event_context", "official_address_reference"}:
                continue
            can_parse = ext in {".geojson", ".json", ".csv", ".kml", ".kmz", ".wkt", ".gml", ".xml"}
            blocked = ""
            if ext in {".gpkg", ".shp", ".dbf", ".xlsx", ".pdf"}:
                blocked = "optional_dependency_or_manual_parse_required"
            if size > 250 * 1024 * 1024:
                blocked = "blocked_heavy_gt_250mb"
                cls = "unsupported_or_heavy"
                can_parse = False
            idx += 1
            rows.append({
                "source_id": f"S16ALOC_{idx:05d}",
                "root_label": label,
                "source_path": safe_path(path, root if label == "PROJETO" else ROOT),
                "path_sha256": path_hash(str(path.resolve())),
                "extension": ext,
                "size_bytes": size,
                "sha256": sha256_file(path) if size <= 250 * 1024 * 1024 else "",
                "region": infer_region(str(path)),
                "priority_path_match": str(priority).lower(),
                "term_match_count": len(terms),
                "matched_terms": ";".join(sorted(set(terms))),
                "source_class": cls,
                "can_parse_geometry": str(can_parse).lower(),
                "blocked_reason": blocked,
                "review_only": "true",
            })
    write_csv(LOCAL_AUDIT, rows, fields)
    counts = Counter(row["source_class"] for row in rows)
    write_md(LOCAL_REPORT, f"""# SUSC-16A - auditoria local de fontes de geometria de evento

Status: review-only. `allowed_for_training=false`; `can_be_ground_truth=false`.

{MANDATORY}

## Escopo minerado
- Raizes PROJETO encontradas: {len(roots)}
- REV-P tambem foi escaneado para copias locais.
- Arquivos candidatos registrados: {len(rows)}

## Classificacao
{json.dumps(dict(counts), ensure_ascii=False, indent=2)}

## Decisao
O manifesto registra caminhos sanitizados, tamanho e SHA256 quando viavel. Nenhum dado bruto pesado foi copiado para o REV-P, e nenhuma fonte local foi promovida automaticamente a ground truth ou treino.
""")
    print(f"local sources audited: {len(rows)}")
    return 0


def parse_bbox_values(values: list[float]) -> str:
    if len(values) < 2:
        return ""
    xs = values[0::2]
    ys = values[1::2]
    if not xs or not ys:
        return ""
    return f"{min(xs):.8f};{min(ys):.8f};{max(xs):.8f};{max(ys):.8f}"


def coords_from_geojson_obj(obj) -> list[float]:
    values: list[float] = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "FeatureCollection":
                for feat in node.get("features", []):
                    walk(feat)
            elif node.get("type") == "Feature":
                walk(node.get("geometry"))
            elif "coordinates" in node:
                walk(node.get("coordinates"))
            else:
                for value in node.values():
                    walk(value)
        elif isinstance(node, list):
            if len(node) >= 2 and all(isinstance(x, (int, float)) for x in node[:2]):
                lon, lat = float(node[0]), float(node[1])
                if -75 <= lon <= -30 and -35 <= lat <= 6:
                    values.extend([lon, lat])
            else:
                for item in node:
                    walk(item)

    walk(obj)
    return values


def parse_geojson(path: Path) -> tuple[str, str, str, str]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, json.JSONDecodeError):
        return "", "", "", "json_parse_failed"
    coords = coords_from_geojson_obj(obj)
    bbox = parse_bbox_values(coords)
    gtype = obj.get("type", "") if isinstance(obj, dict) else ""
    if gtype == "FeatureCollection":
        features = obj.get("features", [])
        if features and isinstance(features[0], dict):
            geom = features[0].get("geometry") or {}
            gtype = geom.get("type", gtype)
    lat = lon = ""
    if len(coords) >= 2 and (max(coords[0::2]) - min(coords[0::2]) < 0.0001) and (max(coords[1::2]) - min(coords[1::2]) < 0.0001):
        lon, lat = f"{coords[0]:.8f}", f"{coords[1]:.8f}"
    return gtype, lat, lon, bbox


def parse_csv_geometry(path: Path) -> tuple[str, str, str, str]:
    rows = read_csv(path)
    if not rows:
        return "", "", "", "empty_csv"
    fields = {str(name).lower(): name for name in rows[0].keys() if name}
    lat_key = next((fields[k] for k in fields if k in {"lat", "latitude", "y"} or "latitude" in k), None)
    lon_key = next((fields[k] for k in fields if k in {"lon", "lng", "longitude", "x"} or "longitude" in k), None)
    wkt_key = next((fields[k] for k in fields if "wkt" in k), None)
    coords: list[float] = []
    for row in rows[:5000]:
        if lat_key and lon_key:
            try:
                lat = float(row.get(lat_key, ""))
                lon = float(row.get(lon_key, ""))
                if -75 <= lon <= -30 and -35 <= lat <= 6:
                    coords.extend([lon, lat])
            except ValueError:
                pass
        if wkt_key:
            coords.extend(coords_from_wkt(row.get(wkt_key, "")))
    bbox = parse_bbox_values(coords)
    if len(coords) >= 2 and len(coords) <= 2:
        return "Point", f"{coords[1]:.8f}", f"{coords[0]:.8f}", bbox
    return ("point_set" if coords else "", "", "", bbox)


def coords_from_wkt(text: str) -> list[float]:
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text or "")]
    coords: list[float] = []
    for lon, lat in zip(nums[0::2], nums[1::2]):
        if -75 <= lon <= -30 and -35 <= lat <= 6:
            coords.extend([lon, lat])
    return coords


def parse_kml(path: Path) -> tuple[str, str, str, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return "", "", "", "read_failed"
    coords: list[float] = []
    for block in re.findall(r"<coordinates[^>]*>(.*?)</coordinates>", text, flags=re.S | re.I):
        for token in re.split(r"\s+", block.strip()):
            parts = token.split(",")
            if len(parts) >= 2:
                try:
                    lon, lat = float(parts[0]), float(parts[1])
                    if -75 <= lon <= -30 and -35 <= lat <= 6:
                        coords.extend([lon, lat])
                except ValueError:
                    pass
    return ("kml_geometry" if coords else "", "", "", parse_bbox_values(coords))


def parse_local_sources() -> int:
    fields = [
        "local_geometry_id", "source_path", "region", "municipality", "source_name",
        "source_type", "event_type", "event_date", "event_period_start",
        "event_period_end", "geometry_type", "lat", "lon", "bbox", "wkt",
        "geojson_ref", "crs", "evidence_tier", "source_confidence",
        "parse_confidence", "is_direct_event_geometry", "is_flood_footprint",
        "is_risk_area", "is_susceptibility_reference",
        "calibration_eligible_candidate", "review_only", "notes",
    ]
    audit_fields = [
        "source_id", "source_path", "parse_status", "geometry_type", "bbox",
        "evidence_tier", "calibration_eligible_candidate", "notes", "review_only",
    ]
    rows, audits = [], []
    for item in read_csv(LOCAL_AUDIT):
        source_path = item.get("source_path", "")
        path = None
        if source_path.startswith("PROJECT_ROOT/"):
            roots = find_project_roots()
            if roots:
                path = roots[0] / source_path.replace("PROJECT_ROOT/", "", 1)
        elif source_path and not source_path.startswith("EXTERNAL_PATH/"):
            path = ROOT / source_path
        if not path or not path.exists() or item.get("can_parse_geometry") != "true":
            audits.append({**{k: item.get(k, "") for k in ("source_id", "source_path")}, "parse_status": "blocked_or_missing", "geometry_type": "", "bbox": "", "evidence_tier": "rejected_non_event", "calibration_eligible_candidate": "false", "notes": item.get("blocked_reason", ""), "review_only": "true"})
            continue
        ext = path.suffix.lower()
        gtype = lat = lon = bbox = ""
        status = "parsed_no_geometry"
        if ext in {".geojson", ".json", ".gml", ".xml"}:
            gtype, lat, lon, bbox = parse_geojson(path)
        elif ext == ".csv":
            gtype, lat, lon, bbox = parse_csv_geometry(path)
        elif ext == ".wkt":
            coords = coords_from_wkt(text_sample(path))
            bbox = parse_bbox_values(coords)
            gtype = "wkt_geometry" if coords else ""
        elif ext == ".kml":
            gtype, lat, lon, bbox = parse_kml(path)
        elif ext == ".kmz":
            try:
                with zipfile.ZipFile(path) as zf:
                    name = next((n for n in zf.namelist() if n.lower().endswith(".kml")), "")
                    text = zf.read(name).decode("utf-8", errors="ignore") if name else ""
                coords = []
                for block in re.findall(r"<coordinates[^>]*>(.*?)</coordinates>", text, flags=re.S | re.I):
                    for token in re.split(r"\s+", block.strip()):
                        parts = token.split(",")
                        if len(parts) >= 2:
                            coords.extend([float(parts[0]), float(parts[1])])
                bbox = parse_bbox_values(coords)
                gtype = "kmz_geometry" if coords else ""
            except (OSError, zipfile.BadZipFile, ValueError):
                pass
        if bbox or (lat and lon):
            status = "parsed_geometry"
        cls = item.get("source_class", "")
        is_footprint = cls in {"official_flood_footprint_candidate", "local_event_polygon_candidate"}
        is_direct = cls in {"local_event_point_candidate", "local_event_polygon_candidate", "official_flood_footprint_candidate"}
        is_risk = cls == "official_risk_area_only"
        is_susc = cls == "official_susceptibility_map_only"
        tier = "rejected_non_event"
        if is_direct and is_footprint:
            tier = "T0_local_official_flood_footprint"
        elif is_direct and "point" in cls:
            tier = "T1_local_official_event_point"
        elif is_risk:
            tier = "T2_local_official_risk_area_context"
        elif is_susc:
            tier = "T3_local_susceptibility_reference"
        elif cls == "official_drainage_context":
            tier = "T4_local_drainage_context"
        eligible = bool((bbox or (lat and lon)) and is_direct and not is_risk and not is_susc)
        row = {
            "local_geometry_id": f"S16ALOCG_{len(rows):05d}",
            "source_path": source_path,
            "region": item.get("region", "unknown"),
            "municipality": item.get("region", "unknown"),
            "source_name": Path(source_path).name,
            "source_type": cls,
            "event_type": "flood" if is_direct else "context",
            "event_date": "",
            "event_period_start": "",
            "event_period_end": "",
            "geometry_type": gtype,
            "lat": lat,
            "lon": lon,
            "bbox": bbox,
            "wkt": "",
            "geojson_ref": source_path if ext in {".geojson", ".json"} and bbox else "",
            "crs": "EPSG:4326_or_source_unknown",
            "evidence_tier": tier,
            "source_confidence": "medium" if item.get("root_label") == "PROJETO" else "low",
            "parse_confidence": "medium" if eligible else "low",
            "is_direct_event_geometry": str(is_direct).lower(),
            "is_flood_footprint": str(is_footprint).lower(),
            "is_risk_area": str(is_risk).lower(),
            "is_susceptibility_reference": str(is_susc).lower(),
            "calibration_eligible_candidate": str(eligible).lower(),
            "review_only": "true",
            "notes": "sem data explicita extraida; elegibilidade permanece candidata review-only" if eligible else "contexto ou geometria insuficiente",
        }
        rows.append(row)
        audits.append({
            "source_id": item.get("source_id", ""),
            "source_path": source_path,
            "parse_status": status,
            "geometry_type": gtype,
            "bbox": bbox,
            "evidence_tier": tier,
            "calibration_eligible_candidate": str(eligible).lower(),
            "notes": row["notes"],
            "review_only": "true",
        })
    write_csv(LOCAL_CANDIDATES, rows, fields)
    write_csv(LOCAL_PARSE_AUDIT, audits, audit_fields)
    print(f"local geometry candidates: {len(rows)}")
    return 0


EXTERNAL_SOURCES = [
    ("S16AEXT_001", "Copernicus EMS On Demand Mapping", "https://mapping.emergency.copernicus.eu/", "technical_flood_footprint", "portal_search_required"),
    ("S16AEXT_002", "Dartmouth Flood Observatory flood records", "https://floodobservatory.colorado.edu/wiki/FloodRecords", "technical_flood_footprint", "global_records_filter_required"),
    ("S16AEXT_003", "Global Flood Database MODIS events", "https://developers.google.com/earth-engine/datasets/catalog/GLOBAL_FLOOD_DB_MODIS_EVENTS_V1", "remote_sensing_candidate_footprint", "gee_execution_required"),
    ("S16AEXT_004", "Recife Defesa Civil dados abertos", "https://dados.recife.pe.gov.br/dataset/?organization=secretaria-executiva-de-defesa-civil", "official_observed_event_point", "ckan_resource_filter_required"),
    ("S16AEXT_005", "Copernicus GloFAS Global Flood Monitoring", "https://global-flood.emergency.copernicus.eu/react/technical-information/glofas-gfm/", "remote_sensing_candidate_footprint", "service_query_required"),
    ("S16AEXT_006", "CEMADEN mapa interativo", "https://mapainterativo.cemaden.gov.br/", "official_rain_hydro_context", "manual_form_required"),
    ("S16AEXT_007", "SGB/CPRM geoportal", "https://geoportal.sgb.gov.br/", "official_risk_context", "manual_layer_discovery_required"),
    ("S16AEXT_008", "INEA dados geoespaciais", "https://www.inea.rj.gov.br/", "official_risk_context", "manual_layer_discovery_required"),
]


def discover_external_sources() -> int:
    lines = ["# SUSC-16A - descoberta de footprints externos", "", "Status: review-only.", "", MANDATORY, "", "## Fontes registradas"]
    for sid, name, url, cls, blocker in EXTERNAL_SOURCES:
        lines.append(f"- `{sid}` {name}: {url} ({cls}; {blocker}).")
    lines.extend([
        "",
        "## Decisao",
        "As fontes foram registradas como candidatas externas. Nenhum raster pesado, imagem bruta ou HTML sem vetor foi promovido. Aquisicao vetorial direta fica bloqueada ate haver URL pequena e auditavel.",
    ])
    write_md(EXTERNAL_DISCOVERY, "\n".join(lines))
    print(f"external sources registered: {len(EXTERNAL_SOURCES)}")
    return 0


def acquire_external_sources() -> int:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    (EXTERNAL_DIR / ".gitignore").write_text("*\n!.gitignore\n!README.md\n", encoding="utf-8")
    (EXTERNAL_DIR / "README.md").write_text(
        "Diretorio reservado para footprints externos SUSC-16A. Dados brutos pesados permanecem fora do Git; apenas manifestos publicos sao versionados.\n",
        encoding="utf-8",
    )
    fields = [
        "download_id", "source_id", "source_name", "source_url", "accepted_format",
        "download_attempted", "download_status", "http_status", "local_artifact_path",
        "size_bytes", "sha256", "blocker", "review_only",
    ]
    rows = []
    for idx, (sid, name, url, cls, blocker) in enumerate(EXTERNAL_SOURCES, start=1):
        rows.append({
            "download_id": f"S16ADL_{idx:04d}",
            "source_id": sid,
            "source_name": name,
            "source_url": url,
            "accepted_format": "GeoJSON/SHP ZIP/GPKG/KML/KMZ/WKT/CSV latlon/WFS/FeatureServer",
            "download_attempted": "false",
            "download_status": "BLOCKED_NO_DIRECT_SMALL_VECTOR_URL",
            "http_status": "",
            "local_artifact_path": "",
            "size_bytes": "",
            "sha256": "",
            "blocker": blocker,
            "review_only": "true",
        })
    write_csv(EXTERNAL_MANIFEST, rows, fields)
    print(f"external acquisition manifest rows: {len(rows)}")
    return 0


def _date(text: str) -> datetime | None:
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def _patch_region_bbox() -> dict[str, tuple[float, float, float, float]]:
    out: dict[str, list[float]] = {}
    for row in read_csv(MATRIX):
        region = (row.get("regiao") or "").lower()
        try:
            vals = [float(row[c]) for c in ("xmin", "ymin", "xmax", "ymax")]
        except (KeyError, ValueError):
            continue
        cur = out.setdefault(region, [vals[0], vals[1], vals[2], vals[3]])
        cur[0], cur[1], cur[2], cur[3] = min(cur[0], vals[0]), min(cur[1], vals[1]), max(cur[2], vals[2]), max(cur[3], vals[3])
    return {k: tuple(v) for k, v in out.items()}


def build_window_plan() -> int:
    fields = [
        "event_window_id", "region", "municipality", "event_date",
        "event_period_start", "event_period_end", "pre_window_start",
        "pre_window_end", "post_window_start", "post_window_end",
        "aoi_strategy", "aoi_bbox", "source_event_count",
        "eligible_for_sentinel1", "eligible_for_sentinel2", "notes",
    ]
    groups: dict[tuple[str, str], int] = defaultdict(int)
    for path in (S15A, S15B, S13C):
        for row in read_csv(path):
            region = (row.get("region") or "unknown").lower()
            event_date = row.get("event_date") or row.get("event_period_start") or ""
            if region == "unknown" or not _date(event_date):
                continue
            groups[(region, event_date)] += 1
    patch_bbox = _patch_region_bbox()
    rows = []
    for idx, ((region, day), count) in enumerate(sorted(groups.items()), start=1):
        d = _date(day)
        if not d:
            continue
        bbox = patch_bbox.get(region) or CITY_BBOX.get(region)
        bbox_text = ";".join(f"{x:.8f}" for x in bbox) if bbox else ""
        rows.append({
            "event_window_id": f"S16AWIN_{idx:05d}",
            "region": region,
            "municipality": region,
            "event_date": day,
            "event_period_start": day,
            "event_period_end": day,
            "pre_window_start": (d - timedelta(days=14)).strftime("%Y-%m-%d"),
            "pre_window_end": (d - timedelta(days=1)).strftime("%Y-%m-%d"),
            "post_window_start": d.strftime("%Y-%m-%d"),
            "post_window_end": (d + timedelta(days=7)).strftime("%Y-%m-%d"),
            "aoi_strategy": "patch_region_extent" if region in patch_bbox else "municipality_bbox_fallback",
            "aoi_bbox": bbox_text,
            "source_event_count": count,
            "eligible_for_sentinel1": str(bool(bbox_text)).lower(),
            "eligible_for_sentinel2": str(bool(bbox_text)).lower(),
            "notes": "AOI regional/municipal nao vira coordenada de ocorrencia; apenas guia footprint orbital candidato.",
        })
    write_csv(WINDOW_PLAN, rows, fields)
    write_md(WINDOW_REPORT, f"""# SUSC-16A - plano de janelas Sentinel/SAR por evento

Status: review-only. `score_v7_created=false`.

{MANDATORY}

- Janelas geradas: {len(rows)}
- Eventos fonte agregados: {sum(int(row['source_event_count']) for row in rows)}
- Estrategia principal: usar datas oficiais como gatilho temporal e AOI por extensao dos patches/regiao.

AOI municipal/regional nao e coordenada de ocorrencia. O unico objeto que podera ser cruzado com patches e o footprint orbital candidato produzido por metodo documentado.
""")
    print(f"sentinel event windows: {len(rows)}")
    return 0


def build_sentinel1_targets() -> int:
    fields = [
        "target_id", "event_window_id", "region", "aoi_bbox", "pre_window",
        "post_window", "sensor", "collection", "expected_method",
        "requires_external_execution", "gee_script_stub_path",
        "stac_query_stub_path", "output_expected", "review_only",
    ]
    rows = []
    for idx, win in enumerate(read_csv(WINDOW_PLAN), start=1):
        if win.get("eligible_for_sentinel1") != "true":
            continue
        rows.append({
            "target_id": f"S16AS1_{idx:05d}",
            "event_window_id": win.get("event_window_id", ""),
            "region": win.get("region", ""),
            "aoi_bbox": win.get("aoi_bbox", ""),
            "pre_window": f"{win.get('pre_window_start')}..{win.get('pre_window_end')}",
            "post_window": f"{win.get('post_window_start')}..{win.get('post_window_end')}",
            "sensor": "Sentinel-1",
            "collection": "COPERNICUS/S1_GRD",
            "expected_method": "VV/VH backscatter change; speckle filtering; permanent water, HAND/slope and optional urban masks; MMU; polygonization",
            "requires_external_execution": "true",
            "gee_script_stub_path": str(GEE_STUB.relative_to(ROOT)).replace("\\", "/"),
            "stac_query_stub_path": str(STAC_STUB.relative_to(ROOT)).replace("\\", "/"),
            "output_expected": "candidate_flood_water_polygon_review_only",
            "review_only": "true",
        })
    write_csv(S1_TARGETS, rows, fields)
    write_md(S1_TARGETS_REPORT, f"""# SUSC-16A - targets Sentinel-1/SAR

Status: review-only. Nenhum raster pesado foi baixado.

{MANDATORY}

- Targets Sentinel-1 gerados: {len(rows)}
- Execucao externa requerida: true
- Metodo esperado: Sentinel-1 GRD before/after, mudanca VV/VH, filtro speckle, remocao de agua permanente, mascaras HAND/slope/urbana, unidade minima de mapeamento e poligonizacao.
""")
    write_stubs()
    print(f"sentinel1 targets: {len(rows)}")
    return 0


def write_stubs() -> None:
    GEE_STUB.write_text("""// SUSC-16A Sentinel-1 flood mapping stub (review-only).
// Requires authenticated Earth Engine environment. This stub does not run in
// Codex by itself and must not persist raster outputs into the public repo.
var target = {
  event_window_id: 'S16AWIN_EXAMPLE',
  aoi_bbox: [-35.05, -8.20, -34.80, -7.90],
  pre_start: 'YYYY-MM-DD',
  pre_end: 'YYYY-MM-DD',
  post_start: 'YYYY-MM-DD',
  post_end: 'YYYY-MM-DD'
};
var aoi = ee.Geometry.Rectangle(target.aoi_bbox);
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));
var pre = s1.filterDate(target.pre_start, target.pre_end).median();
var post = s1.filterDate(target.post_start, target.post_end).median();
var change = post.select('VV').subtract(pre.select('VV'))
  .add(post.select('VH').subtract(pre.select('VH'))).rename('vv_vh_change');
var candidateFlood = change.lt(-2.0).selfMask();
// Export candidateFlood polygonization to a local-only destination after manual review.
""", encoding="utf-8")
    STAC_STUB.write_text('''"""SUSC-16A Sentinel-1 STAC query stub (review-only).

This stub prints the query shape only. It does not authenticate, download, or
write raster data unless a future operator explicitly implements those steps.
"""

from __future__ import annotations

import json


def build_query(aoi_bbox, pre_window, post_window):
    return {
        "collections": ["sentinel-1-grd"],
        "bbox": aoi_bbox,
        "datetime": f"{pre_window[0]}/{post_window[1]}",
        "query": {"sar:instrument_mode": {"eq": "IW"}},
        "review_only": True,
    }


if __name__ == "__main__":
    print(json.dumps(build_query([-35.05, -8.20, -34.80, -7.90], ("YYYY-MM-DD", "YYYY-MM-DD"), ("YYYY-MM-DD", "YYYY-MM-DD")), indent=2))
''', encoding="utf-8")


def run_sar_canary() -> int:
    fields = [
        "sar_candidate_id", "source_path", "region", "event_window_id",
        "sensor", "input_type", "canary_status", "geometry_type", "bbox",
        "method", "eligible_for_observational_evaluation", "can_be_ground_truth",
        "allowed_for_training", "review_only", "notes",
    ]
    candidates = []
    search_roots = find_project_roots() + [ROOT]
    found = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and (path.suffix.lower() in SAR_EXTS or any(term in path.name.lower() for term in ("sentinel-1", "sar", "s1_vv", "s1_vh"))):
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if size <= 250 * 1024 * 1024:
                    found.append((root, path, size))
                if len(found) >= 20:
                    break
        if len(found) >= 20:
            break
    if found:
        for idx, (root, path, _size) in enumerate(found[:3], start=1):
            candidates.append({
                "sar_candidate_id": f"S16ASAR_{idx:04d}",
                "source_path": safe_path(path, root if root != ROOT else ROOT),
                "region": infer_region(str(path)),
                "event_window_id": "",
                "sensor": "Sentinel-1_or_local_sar",
                "input_type": path.suffix.lower(),
                "canary_status": "BLOCKED_NO_OFFLINE_RASTER_PARSER_IN_STAGE",
                "geometry_type": "",
                "bbox": "",
                "method": "local raster canary not executed; parser intentionally not added without raster dependency contract",
                "eligible_for_observational_evaluation": "false",
                "can_be_ground_truth": "false",
                "allowed_for_training": "false",
                "review_only": "true",
                "notes": "arquivo local encontrado, mas sem execucao raster automatica para evitar artefato pesado/ambiente incerto",
            })
    write_csv(SAR_CANDIDATES, candidates, fields)
    status = "local_sar_candidates_found_but_blocked" if candidates else "no_local_sar_raster_available"
    write_md(SAR_REPORT, f"""# SUSC-16A - canary SAR local

Status: {status}

{MANDATORY}

- Arquivos SAR/raster locais compativeis encontrados: {len(found)}
- Candidatos de footprint SAR gerados: {len(candidates)}

Nenhum raster pesado foi copiado ou processado automaticamente. Quando houver contrato de ambiente raster, a execucao devera ocorrer localmente e manter apenas geometrias/manifestos review-only no repositorio publico.
""")
    print(f"sar canary status: {status}; candidates={len(candidates)}")
    return 0


def build_footprint_catalog() -> int:
    fields = [
        "footprint_id", "event_id", "source_family", "source_path", "region",
        "municipality", "event_date", "event_period_start", "event_period_end",
        "footprint_type", "geometry_type", "lat", "lon", "bbox", "wkt",
        "geojson_ref", "evidence_strength", "method", "eligible_for_patch_observational_evaluation",
        "can_be_ground_truth", "can_be_used_as_ground_truth", "allowed_for_training",
        "review_only", "limitations",
    ]
    rows = []

    def add(base: dict) -> None:
        base = {**base, **GOV}
        base["footprint_id"] = f"S16AFP_{len(rows):05d}"
        rows.append({field: base.get(field, "") for field in fields})

    for row in read_csv(LOCAL_CANDIDATES):
        eligible = row.get("calibration_eligible_candidate") == "true" and bool(row.get("bbox") or (row.get("lat") and row.get("lon")))
        ftype = "official_flood_footprint" if row.get("is_flood_footprint") == "true" else "official_observed_event_point" if row.get("is_direct_event_geometry") == "true" else "official_risk_context" if row.get("is_risk_area") == "true" else "official_susceptibility_reference" if row.get("is_susceptibility_reference") == "true" else "insufficient"
        add({
            "event_id": row.get("local_geometry_id", ""),
            "source_family": "local_project",
            "source_path": row.get("source_path", ""),
            "region": row.get("region", ""),
            "municipality": row.get("municipality", ""),
            "event_date": row.get("event_date", ""),
            "event_period_start": row.get("event_period_start", ""),
            "event_period_end": row.get("event_period_end", ""),
            "footprint_type": ftype,
            "geometry_type": row.get("geometry_type", ""),
            "lat": row.get("lat", ""),
            "lon": row.get("lon", ""),
            "bbox": row.get("bbox", ""),
            "wkt": row.get("wkt", ""),
            "geojson_ref": row.get("geojson_ref", ""),
            "evidence_strength": "moderate_candidate" if eligible else "context_or_insufficient",
            "method": "local_geometry_parse",
            "eligible_for_patch_observational_evaluation": str(eligible).lower(),
            "limitations": row.get("notes", ""),
        })
    for row in read_csv(SAR_CANDIDATES):
        eligible = row.get("eligible_for_observational_evaluation") == "true"
        add({
            "event_id": row.get("sar_candidate_id", ""),
            "source_family": "local_sar_canary",
            "source_path": row.get("source_path", ""),
            "region": row.get("region", ""),
            "footprint_type": "remote_sensing_candidate_footprint" if eligible else "insufficient",
            "geometry_type": row.get("geometry_type", ""),
            "bbox": row.get("bbox", ""),
            "evidence_strength": "remote_sensing_candidate" if eligible else "blocked",
            "method": row.get("method", ""),
            "eligible_for_patch_observational_evaluation": str(eligible).lower(),
            "limitations": row.get("notes", ""),
        })
    for row in read_csv(S13C):
        is_observed = row.get("is_observed_event") == "true"
        has_geom = bool(row.get("bbox") or (row.get("lat") and row.get("lon")) or row.get("wkt") or row.get("geojson_ref"))
        eligible = is_observed and has_geom and bool(row.get("event_date") or row.get("event_period_start"))
        ftype = "official_observed_event_polygon" if "polygon" in row.get("geometry_type", "").lower() else "official_observed_event_point" if is_observed else "official_risk_context" if row.get("is_risk_area") == "true" else "insufficient"
        add({
            "event_id": row.get("event_id", ""),
            "source_family": "susc_13c_existing",
            "source_path": row.get("local_file_path", "") or row.get("source_url", ""),
            "region": row.get("region", ""),
            "municipality": row.get("municipality", ""),
            "event_date": row.get("event_date", ""),
            "event_period_start": row.get("event_period_start", ""),
            "event_period_end": row.get("event_period_end", ""),
            "footprint_type": ftype,
            "geometry_type": row.get("geometry_type", ""),
            "lat": row.get("lat", ""),
            "lon": row.get("lon", ""),
            "bbox": row.get("bbox", ""),
            "wkt": row.get("wkt", ""),
            "geojson_ref": row.get("geojson_ref", ""),
            "evidence_strength": "existing_observed_candidate" if eligible else "context_or_insufficient",
            "method": "existing_susc_13c_catalog",
            "eligible_for_patch_observational_evaluation": str(eligible).lower(),
            "limitations": "SUSC-13C reaproveitado como evidencia observacional review-only.",
        })
    for row in read_csv(S15A)[:0] + read_csv(S15B)[:0]:
        pass
    write_csv(FOOTPRINT_CATALOG, rows, fields)
    write_csv(FOOTPRINT_AUDIT, [
        {
            "footprint_id": row["footprint_id"],
            "footprint_type": row["footprint_type"],
            "eligible_for_patch_observational_evaluation": row["eligible_for_patch_observational_evaluation"],
            "review_only": "true",
            "decision": "eligible_review_only" if row["eligible_for_patch_observational_evaluation"] == "true" else "blocked_or_context",
        }
        for row in rows
    ], ["footprint_id", "footprint_type", "eligible_for_patch_observational_evaluation", "review_only", "decision"])
    write_json(FOOTPRINT_MANIFEST, {
        "artifact": "SUSC-16A observed footprint catalog",
        "rows": len(rows),
        "eligible_footprints_count": sum(1 for row in rows if row.get("eligible_for_patch_observational_evaluation") == "true"),
        "footprint_type_counts": dict(Counter(row.get("footprint_type", "") for row in rows)),
        "score_v7_created": False,
        "can_be_ground_truth": False,
        "can_be_used_as_ground_truth": False,
        "allowed_for_training": False,
        "review_only": True,
    })
    print(f"footprint catalog rows: {len(rows)}")
    return 0


def _bbox(row: dict) -> tuple[float, float, float, float] | None:
    text = row.get("bbox", "")
    try:
        parts = [float(x) for x in re.split(r"[;,]", text) if x != ""]
    except ValueError:
        return None
    if len(parts) == 4:
        return parts[0], parts[1], parts[2], parts[3]
    return None


def _intersects(a, b) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _overlap_ratio(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1e-12)
    return round(inter / area_b, 6)


def _haversine(lon1, lat1, lon2, lat2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_linkage() -> int:
    fields = [
        "linkage_id", "footprint_id", "event_id", "patch_id", "region",
        "event_date", "footprint_type", "spatial_relation", "overlap_ratio",
        "distance_m", "score_v6", "score_v6_class", "evidence_strength",
        "eligible_for_observational_evaluation", "can_be_ground_truth",
        "allowed_for_training", "review_only", "interpretation", "limitations",
    ]
    patches = []
    for row in read_csv(MATRIX):
        bb = _bbox({"bbox": ";".join(row.get(c, "") for c in ("xmin", "ymin", "xmax", "ymax"))})
        if bb:
            patches.append((row["patch_id"], (row.get("regiao") or "").lower(), bb, ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)))
    scores = {row["patch_id"]: row for row in read_csv(SCORE)}
    rows = []
    for fp in read_csv(FOOTPRINT_CATALOG):
        fp_bb = _bbox(fp)
        region = (fp.get("region") or "").lower()
        eligible_fp = fp.get("eligible_for_patch_observational_evaluation") == "true"
        made = False
        for pid, pregion, pbb, cen in patches:
            if region not in {"", "unknown"} and pregion != region:
                continue
            relation = "insufficient_for_patch_link"
            overlap = ""
            distance = ""
            if fp_bb and _intersects(fp_bb, pbb):
                overlap_v = _overlap_ratio(fp_bb, pbb)
                overlap = str(overlap_v)
                if fp.get("footprint_type") == "remote_sensing_candidate_footprint":
                    relation = "remote_sensing_candidate_intersects_patch"
                elif fp.get("footprint_type") in {"official_flood_footprint", "technical_flood_footprint", "official_observed_event_polygon"}:
                    relation = "footprint_polygon_intersects_patch" if overlap_v < 0.8 else "footprint_polygon_covers_patch"
                elif fp.get("footprint_type") == "official_risk_context":
                    relation = "risk_context_intersects_patch"
                elif fp.get("footprint_type") == "official_susceptibility_reference":
                    relation = "susceptibility_reference_intersects_patch"
            elif fp.get("lat") and fp.get("lon"):
                try:
                    lon, lat = float(fp["lon"]), float(fp["lat"])
                    inside = pbb[0] <= lon <= pbb[2] and pbb[1] <= lat <= pbb[3]
                    distance_v = _haversine(lon, lat, cen[0], cen[1])
                    if inside:
                        relation = "event_point_inside_patch"
                    elif distance_v <= 500:
                        relation = "event_point_near_patch"
                    distance = str(round(distance_v, 1))
                except ValueError:
                    pass
            if relation == "insufficient_for_patch_link":
                continue
            can_eval = eligible_fp and relation in {
                "footprint_polygon_intersects_patch", "footprint_polygon_covers_patch",
                "event_point_inside_patch", "event_point_near_patch",
                "remote_sensing_candidate_intersects_patch",
            }
            score = scores.get(pid, {})
            rows.append({
                "linkage_id": f"S16ALINK_{len(rows):06d}",
                "footprint_id": fp.get("footprint_id", ""),
                "event_id": fp.get("event_id", ""),
                "patch_id": pid,
                "region": pregion,
                "event_date": fp.get("event_date", "") or fp.get("event_period_start", ""),
                "footprint_type": fp.get("footprint_type", ""),
                "spatial_relation": relation,
                "overlap_ratio": overlap,
                "distance_m": distance,
                "score_v6": score.get("susc_score_v6_candidate", ""),
                "score_v6_class": score.get("susc_class_v6_candidate", ""),
                "evidence_strength": fp.get("evidence_strength", ""),
                "eligible_for_observational_evaluation": str(can_eval).lower(),
                "can_be_ground_truth": "false",
                "allowed_for_training": "false",
                "review_only": "true",
                "interpretation": "Vinculo espacial footprint-patch para avaliacao observacional review-only." if can_eval else "Contexto espacial nao elegivel para avaliacao observacional.",
                "limitations": fp.get("limitations", ""),
            })
            made = True
        if not made:
            rows.append({
                "linkage_id": f"S16ALINK_{len(rows):06d}",
                "footprint_id": fp.get("footprint_id", ""),
                "event_id": fp.get("event_id", ""),
                "patch_id": "REGION_LEVEL_NO_PATCH_RESOLUTION",
                "region": region,
                "event_date": fp.get("event_date", "") or fp.get("event_period_start", ""),
                "footprint_type": fp.get("footprint_type", ""),
                "spatial_relation": "insufficient_for_patch_link",
                "overlap_ratio": "",
                "distance_m": "",
                "score_v6": "",
                "score_v6_class": "",
                "evidence_strength": fp.get("evidence_strength", ""),
                "eligible_for_observational_evaluation": "false",
                "can_be_ground_truth": "false",
                "allowed_for_training": "false",
                "review_only": "true",
                "interpretation": "Sem geometria ou sem intersecao com patch.",
                "limitations": fp.get("limitations", ""),
            })
    write_csv(LINKAGE, rows, fields)
    eval_rows = [row for row in rows if row["eligible_for_observational_evaluation"] == "true"]
    write_csv(LINKAGE_SUMMARY, [
        {"metric": "footprint_patch_links_total", "value": len(rows), "review_only": "true"},
        {"metric": "eligible_patch_links_count", "value": len(eval_rows), "review_only": "true"},
        {"metric": "unique_observational_patches", "value": len({row["patch_id"] for row in eval_rows}), "review_only": "true"},
        {"metric": "relations", "value": json.dumps(dict(Counter(row["spatial_relation"] for row in rows)), ensure_ascii=False, sort_keys=True), "review_only": "true"},
    ], ["metric", "value", "review_only"])
    write_json(LINKAGE_LIMITS, {
        "artifact": "SUSC-16A footprint patch linkage",
        "links": len(rows),
        "eligible_patch_links_count": len(eval_rows),
        "unique_observational_patches": len({row["patch_id"] for row in eval_rows}),
        "score_v7_created": False,
        "can_be_ground_truth": False,
        "allowed_for_training": False,
        "review_only": True,
        "limitations": ["Bairro-only nao e footprint elegivel.", "Risk/susceptibility context nao e evento observado.", "Footprints candidatos continuam review-only."],
    })
    features = []
    for row in eval_rows[:5000]:
        score = scores.get(row["patch_id"], {})
        pbb = next((p[2] for p in patches if p[0] == row["patch_id"]), None)
        if pbb:
            features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [(pbb[0] + pbb[2]) / 2, (pbb[1] + pbb[3]) / 2]}, "properties": row})
    write_json(LINKAGE_GEOJSON, {"type": "FeatureCollection", "features": features, "review_only": True, "can_be_ground_truth": False, "allowed_for_training": False})
    print(f"footprint patch links: {len(rows)}; eligible={len(eval_rows)}")
    return 0


def run_score_diagnostics() -> int:
    links = read_csv(LINKAGE)
    eval_links = [row for row in links if row.get("eligible_for_observational_evaluation") == "true"]
    scores = []
    classes = Counter()
    regions = defaultdict(list)
    for row in eval_links:
        try:
            scores.append(float(row.get("score_v6", "")))
        except ValueError:
            pass
        if row.get("score_v6_class"):
            classes[row["score_v6_class"]] += 1
        regions[row.get("region", "")].append(row)
    all_scores = sorted([(float(row["susc_score_v6_candidate"]), row["patch_id"]) for row in read_csv(SCORE)], reverse=True)
    obs_patches = {row["patch_id"] for row in eval_links if row.get("patch_id") and row["patch_id"] != "REGION_LEVEL_NO_PATCH_RESOLUTION"}

    def hit(n: int) -> float:
        top = {pid for _score, pid in all_scores[:n]}
        return round(len(top & obs_patches) / n, 4) if n else 0.0

    mean = round(sum(scores) / len(scores), 6) if scores else None
    median = None
    if scores:
        s = sorted(scores)
        median = s[len(s) // 2] if len(s) % 2 else round((s[len(s) // 2 - 1] + s[len(s) // 2]) / 2, 6)
    eligible_footprints = sum(1 for row in read_csv(FOOTPRINT_CATALOG) if row.get("eligible_for_patch_observational_evaluation") == "true")
    ready_16b = len(eval_links) >= 10
    ready_v7_discussion = len(obs_patches) >= 20 and len([r for r in regions if r]) >= 2 and bool(scores)
    metrics = {
        "eligible_footprints_count": eligible_footprints,
        "eligible_patch_links_count": len(eval_links),
        "unique_observational_patches": len(obs_patches),
        "mean_score_footprint_patches": mean,
        "median_score_footprint_patches": median,
        "score_class_distribution": dict(classes),
        "hit_rate_top_10": hit(10),
        "hit_rate_top_20": hit(20),
        "hit_rate_top_30": hit(30),
        "low_score_footprint_cases": sum(1 for value in scores if value < 0.33),
        "high_score_footprint_cases": sum(1 for value in scores if value >= 0.67),
        "ready_for_16B_score_evaluation": ready_16b,
        "ready_for_score_v7_discussion": ready_v7_discussion,
        "score_v7_created": False,
        "can_be_ground_truth": False,
        "allowed_for_training": False,
        "review_only": True,
    }
    write_csv(SCORE_DIAG, [{"metric": k, "value": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v, "review_only": "true"} for k, v in metrics.items()], ["metric", "value", "review_only"])
    write_csv(SCORE_DIAG_REGION, [{"region": region, "eligible_patch_links_count": len(rows), "unique_observational_patches": len({row["patch_id"] for row in rows}), "review_only": "true"} for region, rows in sorted(regions.items())], ["region", "eligible_patch_links_count", "unique_observational_patches", "review_only"])
    write_json(SCORE_SUMMARY, metrics)
    write_md(READINESS_MD, f"""# SUSC-16A - prontidao observacional por footprints

Status: review-only. `score_v7_created=false`.

{MANDATORY}

- Footprints elegiveis: {eligible_footprints}
- Links footprint-patch elegiveis: {len(eval_links)}
- Patches observacionais unicos: {len(obs_patches)}
- Media score v6 dos patches: {mean}
- Mediana score v6 dos patches: {median}
- hit@10: {hit(10)}
- hit@20: {hit(20)}
- hit@30: {hit(30)}
- ready_for_16B_score_evaluation: {ready_16b}
- ready_for_score_v7_discussion: {ready_v7_discussion}

Score v7 nao foi criado.
""")
    write_final_report(metrics)
    print(f"score diagnostics: links={len(eval_links)} patches={len(obs_patches)} ready16b={ready_16b}")
    return 0


def write_final_report(metrics: dict) -> None:
    local_count = len(read_csv(LOCAL_AUDIT))
    local_geom = len(read_csv(LOCAL_CANDIDATES))
    external_count = len(read_csv(EXTERNAL_MANIFEST))
    windows = len(read_csv(WINDOW_PLAN))
    targets = len(read_csv(S1_TARGETS))
    sar = len(read_csv(SAR_CANDIDATES))
    catalog = read_csv(FOOTPRINT_CATALOG)
    links = read_csv(LINKAGE)
    write_md(FINAL_REPORT, f"""# SUSC-16A - Footprint observacional preciso por dados locais, fontes oficiais e Sentinel/SAR

Status: review-only. `allowed_for_training=false`; `can_be_ground_truth=false`; `score_v7_created=false`.

{MANDATORY}

## 1. Por que 15A/15B bloquearam
SUSC-15A/15B mantiveram 4412 ocorrencias oficiais reais, mas nao encontraram lat/lon, poligono, lote, intersecao ou faixa numerica controlada suficiente para calibracao. O resultado permaneceu 0 eventos elegiveis e 0 links patch-evento precisos.

## 2. Por que mudar para footprint observacional
A geocodificacao textual chegou ao limite. O SUSC-16A troca a busca por coordenada textual por footprints observacionais: geometrias locais, fontes oficiais/tecnicas e planejamento Sentinel/SAR.

## 3. Dados locais do PROJETO minerados
Fontes locais registradas: {local_count}. Caminhos publicos foram sanitizados e nenhum bruto pesado foi copiado.

## 4. Geometrias locais encontradas
Candidatos parseados: {local_geom}. Candidatos elegiveis permanecem review-only e dependem de geometria direta e data/metodo documentados.

## 5. Fontes externas consultadas
Fontes externas registradas: {external_count}. Todas ficaram bloqueadas para aquisicao automatica por falta de URL vetorial direta pequena e auditavel nesta execucao.

## 6. Footprints oficiais/tecnicos encontrados
Footprints oficiais/tecnicos materializados automaticamente: {metrics['eligible_footprints_count']}. Fontes externas sem vetor direto foram mantidas como manifestos bloqueados.

## 7. Plano Sentinel/SAR criado
Janelas Sentinel/SAR: {windows}. Targets Sentinel-1: {targets}. Stubs GEE/STAC foram criados sem autenticar ou baixar raster.

## 8. Canary SAR local executado ou bloqueado
Candidatos SAR locais: {sar}. O canary permanece bloqueado quando nao ha raster/array local com contrato de parsing seguro.

## 9. Catalogo de footprints
Linhas no catalogo unificado: {len(catalog)}. O catalogo unifica fontes locais, canary SAR e evidencias existentes SUSC-13C, sem ground truth.

## 10. Linkage footprint-patch
Links totais: {len(links)}. Links elegiveis: {metrics['eligible_patch_links_count']}. Patches observacionais unicos: {metrics['unique_observational_patches']}.

## 11. Score v6 x footprints
Media score v6: {metrics['mean_score_footprint_patches']}. Mediana: {metrics['median_score_footprint_patches']}. hit@10={metrics['hit_rate_top_10']}; hit@20={metrics['hit_rate_top_20']}; hit@30={metrics['hit_rate_top_30']}.

## 12. Readiness
ready_for_16B_score_evaluation={metrics['ready_for_16B_score_evaluation']}; ready_for_score_v7_discussion={metrics['ready_for_score_v7_discussion']}.

## 13. O que pode ser afirmado
Podemos afirmar que existe um pipeline auditavel para buscar footprints observacionais e cruzar candidatos com patches em modo review-only.

## 14. O que nao pode ser afirmado
Nao ha ground truth, treino supervisionado, ocorrencia operacional confirmada por patch, score v7, ou validacao automatica de bairro-only como calibracao.

## 15. Proximo marco
Se houver pelo menos 10 links elegiveis, executar SUSC-16B para avaliacao de score v6 com footprints. Caso contrario, priorizar aquisicao manual/controlada dos vetores externos bloqueados e execucao Sentinel/SAR fora do repo publico.
""")


if __name__ == "__main__":
    raise SystemExit(audit_local_sources())
