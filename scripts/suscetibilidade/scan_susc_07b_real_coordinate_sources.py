"""
SUSC-07B Real Coordinate Source Scanner + External Acquisition Queue (offline)

(1) Scans the repository for files that may carry REAL, traceable coordinates
    (geojson/csv lat-lon/kml/wkt/shp/gpkg/pdf/xlsx) and records, per file, the
    coordinate signals found.
(2) Builds an external acquisition queue per evidence/region with official-source
    targets and suggested search queries. Nothing heavy is downloaded here.

Outputs:
  - outputs_public/suscetibilidade/SUSC_07B_real_coordinate_source_scan.csv
  - outputs_public/suscetibilidade/SUSC_07B_real_coordinate_source_scan_summary.json
  - outputs_public/suscetibilidade/SUSC_07B_external_coordinate_acquisition_queue.csv

Governance: review_only, no ground truth, no training. No coordinate is invented.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CATALOG = ROOT / "datasets" / "suscetibilidade" / "susc_07_event_evidence_catalog_v1.csv"
OUT_SCAN = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_real_coordinate_source_scan.csv"
OUT_SUMMARY = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_real_coordinate_source_scan_summary.json"
)
OUT_QUEUE = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_external_coordinate_acquisition_queue.csv"
)

SCAN_ROOTS = [ROOT / "datasets", ROOT / "outputs_public", ROOT / "docs",
              ROOT / "manifests", ROOT / "configs", ROOT / "scripts"]
PRUNE = {".git", ".claude", "worktrees", "__pycache__", "node_modules",
         "local_runs", "local_only", ".venv", "venv", ".pytest_cache"}
PRUNE_PREFIX = ("figures", "figuras", "tmp_")

GEO_EXTS = {".geojson", ".json", ".csv", ".tsv", ".wkt", ".kml", ".kmz",
            ".shp", ".dbf", ".shx", ".prj", ".gpkg", ".geopackage", ".pdf", ".xlsx"}
PARSEABLE_TEXT = {".geojson", ".json", ".csv", ".tsv", ".wkt", ".kml"}
MAX_FILE_BYTES = 5_000_000

COORD_PATTERNS = [
    "latitude", "longitude", "\"lat\"", "\"lon\"", "coordenada", "bbox",
    "bounding box", '"geometry"', "geometria", "POINT(", "POLYGON(",
    "MULTIPOLYGON(", '"coordinates"', "sirgas", "wgs84", "epsg",
]

REGION_BOUNDS = {
    "recife": (-35.20, -34.80, -8.50, -7.85),
    "petropolis": (-43.45, -43.00, -22.75, -22.20),
    "curitiba": (-49.55, -49.00, -25.75, -25.20),
}

# Priority official sources per region (institution, dataset/doc hint).
SOURCE_TARGETS = {
    "petropolis": [
        ("SGB/CPRM", "setores de risco / relatorio tecnico 2022", "geojson|shapefile|pdf"),
        ("Defesa Civil Petropolis/RJ", "ocorrencias inundacao 2022", "csv|pdf"),
        ("INEA/RJ", "areas atingidas / mancha de inundacao", "shapefile|geojson"),
        ("CEMADEN", "ocorrencia inundacao Petropolis lat lon", "csv"),
        ("ANA/Hidroweb", "estacoes/cota fluviometrica", "csv"),
    ],
    "recife": [
        ("APAC", "alagamento/inundacao shapefile", "shapefile|geojson"),
        ("Prefeitura do Recife", "base alagamento municipal", "geojson|csv"),
        ("Defesa Civil Recife/PE", "ocorrencias coordenadas", "csv|pdf"),
        ("CPRM/SGB", "setores de risco RMR", "shapefile|pdf"),
        ("ANA/Hidroweb", "cota/estacoes", "csv"),
    ],
    "curitiba": [
        ("GeoCuritiba", "alagamento/area de risco geojson", "geojson"),
        ("IPPUC/Prefeitura de Curitiba", "bases vetoriais alagamento", "shapefile|geojson"),
        ("Aguas Parana/IAT", "bacias/rios/areas de risco", "shapefile"),
        ("Defesa Civil Curitiba/PR", "ocorrencias inundacao", "csv|pdf"),
        ("CEMADEN", "ocorrencia inundacao Curitiba lat lon", "csv"),
    ],
}

NUM_RE = re.compile(r"-?\d{1,3}\.\d{3,}")


def _prune(n):
    return n in PRUNE or any(n.startswith(p) for p in PRUNE_PREFIX)


def iter_files():
    seen = set()
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if not _prune(d)]
            for fn in fns:
                p = Path(dp) / fn
                if p.suffix.lower() not in GEO_EXTS:
                    continue
                if p.name.lower().startswith(("susc_07b", "susc_07a")):
                    continue
                if p in seen:
                    continue
                seen.add(p)
                yield p


def rel(p):
    return str(p.relative_to(ROOT)).replace("\\", "/")


def region_of_coord(lon, lat):
    for reg, (lo, hi, la, ha) in REGION_BOUNDS.items():
        if lo <= lon <= hi and la <= lat <= ha:
            return reg
    return "unknown"


def main() -> int:
    print("=" * 60)
    print("SUSC-07B Real Coordinate Source Scanner + Queue (offline)")
    print("=" * 60)

    scan_rows = []
    for p in iter_files():
        ext = p.suffix.lower()
        try:
            size = p.stat().st_size
        except OSError:
            continue
        parseable = ext in PARSEABLE_TEXT and size <= MAX_FILE_BYTES
        matched = []
        has_real_coord = False
        region_guess = "unknown"
        if parseable:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                text = ""
            tl = text.lower()
            matched = [pat for pat in COORD_PATTERNS if pat.lower() in tl]
            # detect a plausible WGS84 coordinate pair and infer region
            nums = NUM_RE.findall(text)
            floats = [float(n) for n in nums[:2000]]
            for i in range(len(floats) - 1):
                lon, lat = floats[i], floats[i + 1]
                if -75 <= lon <= -30 and -35 <= lat <= 6:
                    r = region_of_coord(lon, lat)
                    if r != "unknown":
                        has_real_coord = True
                        region_guess = r
                        break
        else:
            matched = ["binary_or_large_no_offline_parse"]

        if not matched and not has_real_coord:
            continue
        if ext in {".shp", ".gpkg", ".geopackage", ".kmz", ".pdf", ".xlsx", ".dbf", ".shx", ".prj"}:
            status = "vector_or_doc_requires_manual_parser"
        elif has_real_coord:
            status = "real_coordinate_candidate"
        elif matched:
            status = "coordinate_signal_no_region_match"
        else:
            status = "no_coordinate"

        scan_rows.append({
            "file_path": rel(p),
            "ext": ext,
            "size_bytes": size,
            "matched_patterns": ";".join(matched[:8]),
            "has_real_coordinate": str(has_real_coord).lower(),
            "region_guess": region_guess,
            "parser": "pure_python" if parseable else "manual_required",
            "candidate_status": status,
            "notes": "offline scan; coordinate only accepted from traceable file",
        })

    OUT_SCAN.parent.mkdir(parents=True, exist_ok=True)
    sfields = ["file_path", "ext", "size_bytes", "matched_patterns",
               "has_real_coordinate", "region_guess", "parser",
               "candidate_status", "notes"]
    with OUT_SCAN.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields, lineterminator="\n")
        w.writeheader()
        w.writerows(scan_rows)

    # external acquisition queue (per evidence with a known region)
    queue_rows = []
    if CATALOG.exists():
        with CATALOG.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            catalog = list(csv.DictReader(f))
        for ev in catalog:
            region = ev["region"]
            if region not in SOURCE_TARGETS:
                continue
            date = ev.get("date_or_period", "unknown")
            for prio, (inst, dataset, gtype) in enumerate(SOURCE_TARGETS[region], start=1):
                query = f"{inst} {region} {date if date!='unknown' else ''} {ev.get('event_type','')} coordenadas {gtype}".strip()
                query = re.sub(r"\s+", " ", query)
                queue_rows.append({
                    "evidence_id": ev["evidence_id"],
                    "region": region,
                    "event_type": ev.get("event_type", ""),
                    "date_or_period": date,
                    "source_priority": prio,
                    "target_institution": inst,
                    "target_dataset_or_document": dataset,
                    "suggested_search_query": query,
                    "expected_geometry_type": gtype,
                    "expected_confidence_if_found": "medium" if prio <= 2 else "low",
                    "download_allowed": "false",
                    "manual_review_required": "true",
                    "notes": "external acquisition not performed automatically; offline plan only",
                })
    qfields = ["evidence_id", "region", "event_type", "date_or_period", "source_priority",
               "target_institution", "target_dataset_or_document", "suggested_search_query",
               "expected_geometry_type", "expected_confidence_if_found", "download_allowed",
               "manual_review_required", "notes"]
    with OUT_QUEUE.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=qfields, lineterminator="\n")
        w.writeheader()
        w.writerows(queue_rows)

    real = sum(1 for r in scan_rows if r["has_real_coordinate"] == "true")
    summary = {
        "schema_id": "SUSC-07B-SCAN",
        "files_with_signals": len(scan_rows),
        "real_coordinate_candidates": real,
        "status_counts": dict(Counter(r["candidate_status"] for r in scan_rows)),
        "region_counts": dict(Counter(r["region_guess"] for r in scan_rows if r["region_guess"] != "unknown")),
        "external_queue_rows": len(queue_rows),
        "governance": {"allowed_for_training": False, "review_only": True,
                       "can_create_ground_truth": False},
        "note": "Coordinate accepted only from traceable file; no invention; no centroid/geocoding.",
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nfiles with coordinate signals: {len(scan_rows)} | real-coordinate candidates: {real}")
    print("status:", summary["status_counts"])
    print("region (real coords):", summary["region_counts"])
    print(f"external acquisition queue rows: {len(queue_rows)}")
    print("\nNo coordinate invented. review-only. No ground truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
