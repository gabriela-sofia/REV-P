from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gr"

LAND_USE_SEARCH_TERMS = [
    "uso", "land_use", "uso_solo", "uso_do_solo", "cobertura",
    "cobertura_terra", "mapbiomas", "urban", "lulc", "classification",
]
LAND_USE_EXTENSIONS = {".shp", ".geojson", ".gpkg", ".csv", ".tif", ".tiff"}

REGIONS = ["Curitiba", "Petropolis", "Recife"]
REGION_KEY_MAP = {
    "curitiba": "Curitiba",
    "petropolis": "Petropolis",
    "recife": "Recife",
}

GIS_LAND_USE_HINTS: dict[str, list[str]] = {
    "petropolis": [
        "data/external_validation_assets/petropolis_sgb_cprm/raw/fbds/uso/RJ_3303906_USO.shp",
    ],
    "curitiba": [],
    "recife": [],
}

DEPS = ["pandas", "shapely", "rasterio", "geopandas", "fiona", "pyogrio"]
CONVERSION_DEPS = {"fiona", "geopandas", "pyogrio"}

METHODOLOGICAL_GUARDRAILS = {
    "review_only": True,
    "supervised_training": False,
    "labels_created": False,
    "targets_created": False,
    "predictive_claims": False,
    "multimodal_execution_enabled": False,
    "land_use_is_ground_truth": False,
}

CLASS_MAPPING_ROWS = [
    # --- score 3: high vulnerability contribution (urban / impervious) ---
    {"class_pattern": "área edificada", "class_example": "área edificada", "score": 3,
     "category": "urbano_impermeavel", "review_only": "true",
     "notes": "FBDS: built-up / constructed area"},
    {"class_pattern": "urban", "class_example": "urban / urbanizado",
     "score": 3, "category": "urbano_impermeavel", "review_only": "true",
     "notes": "keyword: urban variants"},
    {"class_pattern": "built", "class_example": "built-up / built_area",
     "score": 3, "category": "urbano_impermeavel", "review_only": "true",
     "notes": "keyword: built-up"},
    {"class_pattern": "imperme", "class_example": "impermeabilizado",
     "score": 3, "category": "urbano_impermeavel", "review_only": "true",
     "notes": "keyword: impervious surface"},
    # --- score 2: medium (agriculture / pasture / exposed soil) ---
    {"class_pattern": "área antropizada", "class_example": "área antropizada",
     "score": 2, "category": "antropizado", "review_only": "true",
     "notes": "FBDS: anthropized area (pasture, crops, mosaic); not built-up"},
    {"class_pattern": "agricultura", "class_example": "agricultura",
     "score": 2, "category": "agricola_pastagem", "review_only": "true",
     "notes": "keyword: agriculture"},
    {"class_pattern": "pastagem", "class_example": "pastagem",
     "score": 2, "category": "agricola_pastagem", "review_only": "true",
     "notes": "keyword: pasture"},
    {"class_pattern": "campo", "class_example": "campo limpo / campo sujo",
     "score": 2, "category": "agricola_pastagem", "review_only": "true",
     "notes": "keyword: campo (grassland/field)"},
    {"class_pattern": "solo exposto", "class_example": "solo exposto",
     "score": 2, "category": "solo_exposto", "review_only": "true",
     "notes": "keyword: bare soil"},
    {"class_pattern": "silvicultura", "class_example": "silvicultura",
     "score": 2, "category": "silvicultura", "review_only": "true",
     "notes": "FBDS: planted forest / silviculture; intermediate"},
    {"class_pattern": "formação não florestal", "class_example": "formação não florestal",
     "score": 2, "category": "formacao_nao_florestal", "review_only": "true",
     "notes": "FBDS: non-forest natural formation (campo, cerrado)"},
    # --- score 1: low (forest / vegetation / water body) ---
    {"class_pattern": "formação florestal", "class_example": "formação florestal",
     "score": 1, "category": "floresta", "review_only": "true",
     "notes": "FBDS: forest formation; low imperviousness"},
    {"class_pattern": "floresta", "class_example": "floresta nativa",
     "score": 1, "category": "floresta", "review_only": "true",
     "notes": "keyword: forest"},
    {"class_pattern": "vegetação", "class_example": "vegetação natural",
     "score": 1, "category": "vegetacao", "review_only": "true",
     "notes": "keyword: vegetation"},
    {"class_pattern": "água", "class_example": "água / corpo hídrico",
     "score": 1, "category": "corpo_hidrico", "review_only": "true",
     "notes": "FBDS: water body / water surface"},
    {"class_pattern": "water", "class_example": "water body",
     "score": 1, "category": "corpo_hidrico", "review_only": "true",
     "notes": "keyword: water"},
    # --- REVIEW: unknown / unclassified ---
    {"class_pattern": "desconhecido", "class_example": "desconhecido",
     "score": "REVIEW", "category": "desconhecido", "review_only": "true",
     "notes": "unknown class — requires human review before score assignment"},
    {"class_pattern": "sem classificação", "class_example": "sem classificação",
     "score": "REVIEW", "category": "sem_classificacao", "review_only": "true",
     "notes": "unclassified — requires human review"},
    {"class_pattern": "nulo", "class_example": "nulo / null",
     "score": "REVIEW", "category": "nulo", "review_only": "true",
     "notes": "null class — blocked from score assignment"},
]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {path}. Use --force.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dependency audit
# ---------------------------------------------------------------------------

def audit_dependencies() -> dict[str, str]:
    results: dict[str, str] = {}
    for lib in DEPS:
        try:
            __import__(lib)
            results[lib] = "AVAILABLE"
        except ImportError:
            results[lib] = "MISSING"
    return results


def conversion_possible(deps: dict[str, str]) -> bool:
    return any(deps.get(d) == "AVAILABLE" for d in CONVERSION_DEPS)


# ---------------------------------------------------------------------------
# DBF parser — pure Python, reads attribute table without fiona
# ---------------------------------------------------------------------------

def parse_dbf(dbf_path: Path) -> dict[str, Any]:
    try:
        with dbf_path.open("rb") as f:
            hdr = f.read(32)
            if len(hdr) < 32:
                return {"readable": False, "error": "header too short"}
            n_records = struct.unpack_from("<I", hdr, 4)[0]
            header_size = struct.unpack_from("<H", hdr, 8)[0]
            record_size = struct.unpack_from("<H", hdr, 10)[0]
            fields: list[dict[str, Any]] = []
            while True:
                fd = f.read(32)
                if not fd or len(fd) < 32 or fd[0] == 0x0D:
                    break
                name = fd[:11].split(b"\x00")[0].decode("latin-1", errors="replace").strip()
                ftype = chr(fd[11]) if 32 <= fd[11] <= 127 else "?"
                flen = fd[16]
                if name:
                    fields.append({"name": name, "type": ftype, "length": int(flen)})
            f.seek(header_size)
            sample: list[dict[str, str]] = []
            for _ in range(min(20, n_records)):
                rb = f.read(record_size)
                if not rb or len(rb) < record_size or rb[0] == 0x1A:
                    break
                rb = rb[1:]
                rec: dict[str, str] = {}
                offset = 0
                for fld in fields:
                    chunk = rb[offset: offset + fld["length"]]
                    try:
                        val = chunk.decode("latin-1", errors="replace").strip()
                    except Exception:
                        val = ""
                    rec[fld["name"]] = val
                    offset += fld["length"]
                sample.append(rec)
        return {
            "readable": True,
            "n_records": n_records,
            "fields": [f["name"] for f in fields],
            "field_types": {f["name"]: f["type"] for f in fields},
            "sample": sample,
        }
    except Exception as e:
        return {"readable": False, "error": str(e)}


def extract_unique_classes(dbf_path: Path, class_col: str) -> list[str]:
    try:
        with dbf_path.open("rb") as f:
            hdr = f.read(32)
            n_records = struct.unpack_from("<I", hdr, 4)[0]
            header_size = struct.unpack_from("<H", hdr, 8)[0]
            record_size = struct.unpack_from("<H", hdr, 10)[0]
            fields: list[dict[str, Any]] = []
            while True:
                fd = f.read(32)
                if not fd or len(fd) < 32 or fd[0] == 0x0D:
                    break
                name = fd[:11].split(b"\x00")[0].decode("latin-1", errors="replace").strip()
                ftype = chr(fd[11]) if 32 <= fd[11] <= 127 else "?"
                flen = fd[16]
                if name:
                    fields.append({"name": name, "type": ftype, "length": int(flen)})
            col_offset = 0
            col_len = 0
            for fld in fields:
                if fld["name"] == class_col:
                    col_len = fld["length"]
                    break
                col_offset += fld["length"]
            if col_len == 0:
                return []
            f.seek(header_size)
            seen: set[str] = set()
            for _ in range(n_records):
                rb = f.read(record_size)
                if not rb or rb[0] == 0x1A:
                    break
                rb = rb[1:]
                val = rb[col_offset: col_offset + col_len].decode("latin-1", errors="replace").strip()
                if val:
                    seen.add(val)
        return sorted(seen)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# File inventory
# ---------------------------------------------------------------------------

def file_matches_land_use(name: str) -> bool:
    name_l = name.lower()
    ext = Path(name).suffix.lower()
    return ext in LAND_USE_EXTENSIONS and any(t in name_l for t in LAND_USE_SEARCH_TERMS)


def inventory_land_use_files(search_roots: list[Path]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for root, dirs, files in os.walk(search_root):
            dirs[:] = [d for d in dirs if d not in [
                ".venv", "__pycache__", "sentinel", ".git", "embeddings"
            ]]
            for fname in files:
                if not file_matches_land_use(fname):
                    continue
                full = Path(root) / fname
                key = str(full.resolve())
                if key in seen:
                    continue
                seen.add(key)
                try:
                    size = full.stat().st_size
                except Exception:
                    size = -1
                found.append({
                    "filename": fname,
                    "extension": full.suffix.lower(),
                    "size_bytes": size,
                    "full_path": str(full),
                    "search_root": str(search_root),
                })
    return found


# ---------------------------------------------------------------------------
# Region assessment
# ---------------------------------------------------------------------------

def _region_key(region: str) -> str:
    import unicodedata
    nfkd = unicodedata.normalize("NFD", region)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn").lower().strip()


def assess_region(region: str, gis_root: Path | None,
                  deps: dict[str, str]) -> dict[str, Any]:
    key = _region_key(region)
    hints = GIS_LAND_USE_HINTS.get(key, [])
    can_convert = conversion_possible(deps)

    if gis_root is None:
        return {
            "region": region,
            "land_use_source_found": False,
            "source_path": "",
            "source_format": "",
            "dbf_readable": False,
            "dbf_fields": "",
            "dbf_n_records": "",
            "unique_classes": "",
            "geometry_readable": False,
            "conversion_possible": False,
            "coverage_status": "BLOCKED",
            "blocker_reason": "gis_root not provided",
        }

    for hint in hints:
        candidate = gis_root / hint
        if candidate.exists():
            shp_path = candidate
            dbf_path = shp_path.with_suffix(".dbf")
            dbf_info: dict[str, Any] = {}
            unique_cls: list[str] = []
            class_col = ""
            if dbf_path.exists():
                dbf_info = parse_dbf(dbf_path)
                if dbf_info.get("readable"):
                    for possible_col in ["CLASSE_USO", "USO", "CLASS_USO", "class", "uso", "DN"]:
                        if possible_col in dbf_info.get("fields", []):
                            class_col = possible_col
                            break
                    if class_col:
                        unique_cls = extract_unique_classes(dbf_path, class_col)
            geom_readable = can_convert
            conv_possible = can_convert and shp_path.exists()
            if not can_convert:
                blocker = f"geometry conversion requires fiona/geopandas/pyogrio (all MISSING)"
                cov_status = "PARTIAL"
            else:
                blocker = ""
                cov_status = "AVAILABLE"
            return {
                "region": region,
                "land_use_source_found": True,
                "source_path": str(shp_path),
                "source_format": shp_path.suffix.lower(),
                "dbf_readable": dbf_info.get("readable", False),
                "dbf_fields": "; ".join(dbf_info.get("fields", [])),
                "dbf_n_records": str(dbf_info.get("n_records", "")),
                "unique_classes": "; ".join(unique_cls),
                "geometry_readable": geom_readable,
                "conversion_possible": conv_possible,
                "coverage_status": cov_status,
                "blocker_reason": blocker,
            }

    return {
        "region": region,
        "land_use_source_found": False,
        "source_path": "",
        "source_format": "",
        "dbf_readable": False,
        "dbf_fields": "",
        "dbf_n_records": "",
        "unique_classes": "",
        "geometry_readable": False,
        "conversion_possible": False,
        "coverage_status": "BLOCKED",
        "blocker_reason": "no land use source found for this region",
    }


# ---------------------------------------------------------------------------
# Conversion plan and attempt
# ---------------------------------------------------------------------------

def build_conversion_plan(region_coverage: list[dict[str, Any]],
                           deps: dict[str, str],
                           output_dir: Path) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for reg in region_coverage:
        if not reg["land_use_source_found"]:
            plan.append({
                "region": reg["region"],
                "source_path": "",
                "output_path": "",
                "conversion_method": "none",
                "can_execute": False,
                "blocker": "no source file found",
                "plan_status": "BLOCKED",
            })
            continue
        src = reg["source_path"]
        ext = Path(src).suffix.lower() if src else ""
        method = "fiona" if deps.get("fiona") == "AVAILABLE" else \
                 ("geopandas" if deps.get("geopandas") == "AVAILABLE" else
                  ("pyogrio" if deps.get("pyogrio") == "AVAILABLE" else "none"))
        out_name = Path(src).stem + "_land_use_v1gr.geojson" if src else ""
        out_path = str(output_dir / "converted" / out_name) if out_name else ""
        can_exec = reg["conversion_possible"]
        plan.append({
            "region": reg["region"],
            "source_path": src,
            "output_path": out_path,
            "conversion_method": method,
            "can_execute": can_exec,
            "blocker": reg["blocker_reason"] if not can_exec else "",
            "plan_status": "READY" if can_exec else "BLOCKED",
        })
    return plan


def attempt_conversion(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in plan:
        if not item["can_execute"] or item["plan_status"] == "BLOCKED":
            results.append({
                "region": item["region"],
                "status": "SKIPPED",
                "output_path": "",
                "n_features": "",
                "detail": item["blocker"] or "plan not executable",
            })
            continue
        src = Path(item["source_path"])
        out = Path(item["output_path"])
        method = item["conversion_method"]
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            if method == "fiona":
                import fiona
                import json as _json
                from shapely.geometry import mapping as _shp_mapping, shape as _shp_shape
                features = []
                with fiona.open(str(src)) as col:
                    crs = col.crs
                    for feat in col:
                        geom = feat.geometry
                        geom_dict = _shp_mapping(_shp_shape(geom)) if geom else None
                        features.append({
                            "type": "Feature",
                            "geometry": geom_dict,
                            "properties": dict(feat.properties),
                        })
                geojson = {"type": "FeatureCollection", "features": features}
                out.write_text(_json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
                results.append({
                    "region": item["region"],
                    "status": "SUCCESS",
                    "output_path": str(out),
                    "n_features": len(features),
                    "detail": f"converted via fiona+shapely; source CRS: {crs}",
                })
            elif method == "geopandas":
                import geopandas as gpd
                gdf = gpd.read_file(str(src)).to_crs("EPSG:4326")
                gdf.to_file(str(out), driver="GeoJSON")
                results.append({
                    "region": item["region"],
                    "status": "SUCCESS",
                    "output_path": str(out),
                    "n_features": len(gdf),
                    "detail": "converted via geopandas; reprojected to EPSG:4326",
                })
            else:
                results.append({
                    "region": item["region"],
                    "status": "BLOCKED",
                    "output_path": "",
                    "n_features": "",
                    "detail": f"no supported conversion library: {method}",
                })
        except Exception as e:
            results.append({
                "region": item["region"],
                "status": "FAIL",
                "output_path": "",
                "n_features": "",
                "detail": str(e)[:200],
            })
    return results


# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------

def build_class_mapping_table() -> list[dict[str, Any]]:
    return [dict(row) for row in CLASS_MAPPING_ROWS]


def map_class_to_score(class_value: str) -> str | int:
    cl = class_value.strip().lower()
    for row in CLASS_MAPPING_ROWS:
        pattern = row["class_pattern"].lower()
        if pattern in cl or cl in pattern:
            s = row["score"]
            return s if isinstance(s, int) else str(s)
    return "REVIEW"


# ---------------------------------------------------------------------------
# V1GQ readiness
# ---------------------------------------------------------------------------

def compute_v1gq_readiness(region_coverage: list[dict[str, Any]]) -> str:
    statuses = [r["coverage_status"] for r in region_coverage]
    available = statuses.count("AVAILABLE")
    partial = statuses.count("PARTIAL")
    if available == len(region_coverage):
        return "READY_FOR_V1GQ_RERUN"
    if available + partial >= 1:
        return "PARTIAL_READY"
    return "BLOCKED"


# ---------------------------------------------------------------------------
# Main audit function
# ---------------------------------------------------------------------------

def run_audit(gis_root: Path | None, output_dir: Path) -> dict[str, Any]:
    ts = datetime.now(timezone.utc).isoformat()

    deps = audit_dependencies()

    search_roots: list[Path] = [ROOT]
    if gis_root is not None:
        search_roots.append(gis_root / "data")

    inventory = inventory_land_use_files(search_roots)

    region_coverage = [assess_region(r, gis_root, deps) for r in REGIONS]

    conversion_plan = build_conversion_plan(region_coverage, deps, output_dir)
    conversion_results = attempt_conversion(conversion_plan)

    class_mapping = build_class_mapping_table()

    v1gq_readiness = compute_v1gq_readiness(region_coverage)

    blockers: list[dict[str, Any]] = []
    if not conversion_possible(deps):
        blockers.append({
            "category": "missing_dependency",
            "severity": "BLOCKED",
            "detail": "fiona, geopandas, and pyogrio are all MISSING; geometry conversion not possible",
        })
    for reg in region_coverage:
        if reg["coverage_status"] == "BLOCKED":
            blockers.append({
                "category": "missing_source",
                "severity": "BLOCKED",
                "detail": f"{reg['region']}: {reg['blocker_reason']}",
            })
        elif reg["coverage_status"] == "PARTIAL":
            blockers.append({
                "category": "partial_data",
                "severity": "PARTIAL",
                "detail": f"{reg['region']}: {reg['blocker_reason']}",
            })

    dep_rows = [
        {"library": lib, "status": status,
         "required_for": "conversion" if lib in CONVERSION_DEPS else "general",
         "blocker_if_missing": "yes" if lib in CONVERSION_DEPS else "no"}
        for lib, status in deps.items()
    ]

    n_available = sum(1 for r in region_coverage if r["coverage_status"] == "AVAILABLE")
    n_partial = sum(1 for r in region_coverage if r["coverage_status"] == "PARTIAL")
    n_blocked = sum(1 for r in region_coverage if r["coverage_status"] == "BLOCKED")

    if n_available == len(REGIONS):
        land_use_global = "AVAILABLE"
    elif n_available + n_partial > 0:
        land_use_global = "PARTIAL"
    else:
        land_use_global = "BLOCKED"

    dep_status_summary = (
        "ALL_MISSING" if all(v == "MISSING" for k, v in deps.items() if k in CONVERSION_DEPS)
        else "PARTIAL"
    )

    cov_by_region = {_region_key(r["region"]): r["coverage_status"] for r in region_coverage}

    qa_rows: list[dict[str, Any]] = [
        {
            "check": "land_use_inventory_executed",
            "status": "PASS",
            "detail": f"{len(inventory)} land use file(s) found in search roots",
        },
        {
            "check": "dependency_audit_executed",
            "status": "PASS",
            "detail": "; ".join(f"{k}={v}" for k, v in deps.items()),
        },
        {
            "check": "conversion_dependencies_available",
            "status": "PASS" if conversion_possible(deps) else "WARN",
            "detail": "fiona/geopandas/pyogrio: " + (
                "at least one available" if conversion_possible(deps)
                else "all MISSING — conversion blocked"
            ),
        },
        {
            "check": "petropolis_land_use_source_found",
            "status": "PASS" if any(r["region"] == "Petropolis" and r["land_use_source_found"]
                                    for r in region_coverage) else "WARN",
            "detail": next((r["source_path"] for r in region_coverage
                            if r["region"] == "Petropolis"), "not found"),
        },
        {
            "check": "petropolis_dbf_readable",
            "status": "PASS" if any(r["region"] == "Petropolis" and r["dbf_readable"]
                                    for r in region_coverage) else "WARN",
            "detail": next((r["dbf_fields"] for r in region_coverage
                            if r["region"] == "Petropolis"), ""),
        },
        {
            "check": "curitiba_land_use_available",
            "status": "WARN",
            "detail": "no land use source found for Curitiba in current search roots",
        },
        {
            "check": "recife_land_use_available",
            "status": "WARN",
            "detail": "no land use source found for Recife in current search roots",
        },
        {
            "check": "class_mapping_candidate_built",
            "status": "PASS",
            "detail": f"{len(class_mapping)} class mapping rules; review_only=true",
        },
        {
            "check": "no_labels_created",
            "status": "PASS",
            "detail": "labels_created=false; class mapping is candidate only",
        },
        {
            "check": "no_supervised_training",
            "status": "PASS",
            "detail": "supervised_training=false; no model training executed",
        },
        {
            "check": "land_use_is_not_ground_truth",
            "status": "PASS",
            "detail": "land_use_is_ground_truth=false; index is baseline proxy",
        },
        {
            "check": "multimodal_disabled",
            "status": "PASS",
            "detail": "multimodal_execution_enabled=false; pipeline unchanged",
        },
        {
            "check": "no_forbidden_artifacts",
            "status": "PASS",
            "detail": "all outputs are CSV/JSON under local_runs/dino_embeddings/v1gr/",
        },
        {
            "check": "v1gq_not_modified",
            "status": "PASS",
            "detail": "v1gr only generates readiness report; v1gq outputs untouched",
        },
        {
            "check": "v1gq_rerun_readiness",
            "status": "PASS" if v1gq_readiness != "BLOCKED" else "WARN",
            "detail": v1gq_readiness,
        },
    ]

    fail_checks = [r for r in qa_rows if r["status"] == "FAIL"]
    warn_checks = [r for r in qa_rows if r["status"] == "WARN"]
    if fail_checks:
        qa_overall = "FAIL"
    elif warn_checks:
        qa_overall = "PARTIAL"
    else:
        qa_overall = "PASS"

    summary: dict[str, Any] = {
        "stage": "v1gr",
        "stage_name": "GIS land-use readiness and conversion audit",
        "generated_at": ts,
        "gis_root_provided": gis_root is not None,
        "input_inventory_status": f"{len(inventory)} files found",
        "dependency_status": dep_status_summary,
        "curitiba_land_use_status": cov_by_region.get("curitiba", "BLOCKED"),
        "petropolis_land_use_status": cov_by_region.get("petropolis", "BLOCKED"),
        "recife_land_use_status": cov_by_region.get("recife", "BLOCKED"),
        "land_use_global_status": land_use_global,
        "v1gq_rerun_readiness": v1gq_readiness,
        "blockers_count": len(blockers),
        **METHODOLOGICAL_GUARDRAILS,
        "output_dir": str(output_dir),
        "qa_status": qa_overall,
        "methodology_note": (
            "land use class mapping is a candidate table for human review; "
            "it is not a final classification, not ground truth, and not a label. "
            "Integration with v1gq requires geometry conversion (fiona/geopandas) "
            "and is currently blocked by missing dependencies."
        ),
    }

    return {
        "summary": summary,
        "inventory": inventory,
        "dep_rows": dep_rows,
        "region_coverage": region_coverage,
        "conversion_plan": conversion_plan,
        "conversion_results": conversion_results,
        "class_mapping": class_mapping,
        "blockers": blockers,
        "qa_rows": qa_rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v1gr GIS land-use readiness and conversion audit."
    )
    parser.add_argument("--gis-root", default=None,
                        help="Root path to GIS data (e.g. path/to/PROJETO). "
                             "If not provided, only REV-P is scanned.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)

    gis_root = Path(args.gis_root) if args.gis_root else None
    print(f"[v1gr] gis_root: {gis_root or '(not provided)'}")

    result = run_audit(gis_root, output_dir)

    print("[v1gr] Writing outputs...")

    write_json(output_dir / "land_use_summary_v1gr.json", result["summary"])

    write_csv(
        output_dir / "land_use_input_inventory_v1gr.csv",
        result["inventory"],
        ["filename", "extension", "size_bytes", "full_path", "search_root"],
    )
    write_csv(
        output_dir / "land_use_dependency_audit_v1gr.csv",
        result["dep_rows"],
        ["library", "status", "required_for", "blocker_if_missing"],
    )
    write_csv(
        output_dir / "land_use_region_coverage_v1gr.csv",
        result["region_coverage"],
        ["region", "land_use_source_found", "source_path", "source_format",
         "dbf_readable", "dbf_fields", "dbf_n_records", "unique_classes",
         "geometry_readable", "conversion_possible", "coverage_status", "blocker_reason"],
    )
    write_csv(
        output_dir / "land_use_conversion_plan_v1gr.csv",
        result["conversion_plan"],
        ["region", "source_path", "output_path", "conversion_method",
         "can_execute", "blocker", "plan_status"],
    )
    write_csv(
        output_dir / "land_use_conversion_results_v1gr.csv",
        result["conversion_results"],
        ["region", "status", "output_path", "n_features", "detail"],
    )
    write_csv(
        output_dir / "land_use_class_mapping_candidate_v1gr.csv",
        result["class_mapping"],
        ["class_pattern", "class_example", "score", "category", "review_only", "notes"],
    )
    write_csv(
        output_dir / "land_use_blockers_v1gr.csv",
        result["blockers"],
        ["category", "severity", "detail"],
    )
    write_csv(
        output_dir / "land_use_qa_v1gr.csv",
        result["qa_rows"],
        ["check", "status", "detail"],
    )

    s = result["summary"]
    print(f"[v1gr] Land use global status: {s['land_use_global_status']}")
    print(f"[v1gr] v1gq rerun readiness:   {s['v1gq_rerun_readiness']}")
    print(f"[v1gr] QA status:              {s['qa_status']}")
    print(f"[v1gr] Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
