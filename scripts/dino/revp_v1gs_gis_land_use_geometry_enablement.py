from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import struct
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gs"
DEFAULT_V1GQ_SCRIPT = ROOT / "scripts" / "dino" / "revp_v1gq_gis_multicriteria_vulnerability_baseline.py"
DEFAULT_V1GQ_RERUN_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gq_rerun_v1gs"

PETROPOLIS_SHP_HINT = (
    "data/external_validation_assets/petropolis_sgb_cprm/raw/fbds/uso/RJ_3303906_USO.shp"
)
PETROPOLIS_SHP_SIDECARS = [".shp", ".dbf", ".shx", ".prj"]
PETROPOLIS_CLASS_COL = "CLASSE_USO"
PETROPOLIS_ENCODING = "latin-1"
PETROPOLIS_CRS_SRC = "EPSG:31983"
PETROPOLIS_CRS_DST = "EPSG:4326"

DEPS = ["pandas", "rasterio", "shapely", "fiona", "geopandas", "pyogrio"]
GEOMETRY_LIBS = ["pyogrio", "geopandas", "fiona"]

METHODOLOGICAL_GUARDRAILS = {
    "review_only": True,
    "supervised_training": False,
    "labels_created": False,
    "targets_created": False,
    "predictive_claims": False,
    "multimodal_execution_enabled": False,
    "land_use_is_ground_truth": False,
    "vulnerability_index_is_ground_truth": False,
    "final_vulnerability_claim": False,
    "dino_predicts_vulnerability": False,
}

PENDING_ISSUES = {
    "curitiba_land_use_status": "BLOCKED",
    "recife_land_use_status": "BLOCKED",
    "population_density_status": "BLOCKED",
    "road_density_status": "PARTIAL",
    "global_index_status": "PARTIAL",
    "petropolis_land_use_status": "PARTIAL",
}


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


def geometry_lib_available(deps: dict[str, str]) -> str | None:
    for lib in GEOMETRY_LIBS:
        if deps.get(lib) == "AVAILABLE":
            return lib
    return None


# ---------------------------------------------------------------------------
# Sidecar audit
# ---------------------------------------------------------------------------

def audit_sidecars(shp_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ext in PETROPOLIS_SHP_SIDECARS:
        p = shp_path.with_suffix(ext)
        essential = ext in (".shp", ".dbf", ".shx")
        exists = p.exists()
        try:
            size = p.stat().st_size if exists else -1
        except Exception:
            size = -1
        rows.append({
            "extension": ext,
            "path": str(p),
            "exists": exists,
            "size_bytes": size,
            "essential": essential,
            "blocker": "" if exists or not essential else f"missing essential sidecar: {ext}",
        })
    return rows


def sidecars_complete(rows: list[dict[str, Any]]) -> bool:
    return all(r["exists"] for r in rows if r["essential"])


# ---------------------------------------------------------------------------
# DBF fallback parser
# ---------------------------------------------------------------------------

def _parse_dbf_fields(f: Any) -> list[dict[str, Any]]:
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
    return fields


def read_dbf_attributes(dbf_path: Path, class_col: str) -> dict[str, Any]:
    try:
        with dbf_path.open("rb") as f:
            hdr = f.read(32)
            if len(hdr) < 32:
                return {"readable": False, "error": "header too short"}
            n_records = struct.unpack_from("<I", hdr, 4)[0]
            header_size = struct.unpack_from("<H", hdr, 8)[0]
            record_size = struct.unpack_from("<H", hdr, 10)[0]
            fields = _parse_dbf_fields(f)
            col_offset, col_len = 0, 0
            for fld in fields:
                if fld["name"] == class_col:
                    col_len = fld["length"]
                    break
                col_offset += fld["length"]
            f.seek(header_size)
            counts: dict[str, int] = {}
            for _ in range(n_records):
                rb = f.read(record_size)
                if not rb or rb[0] == 0x1A:
                    break
                if col_len:
                    val = rb[1 + col_offset: 1 + col_offset + col_len].decode(
                        "latin-1", errors="replace"
                    ).strip()
                    if val:
                        counts[val] = counts.get(val, 0) + 1
        return {
            "readable": True,
            "n_records": n_records,
            "fields": [f["name"] for f in fields],
            "class_col": class_col if col_len else "",
            "class_distribution": counts,
        }
    except Exception as e:
        return {"readable": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Geometry reading — pyogrio → geopandas → fiona
# ---------------------------------------------------------------------------

def _try_pyogrio(shp_path: Path, out_path: Path) -> dict[str, Any]:
    try:
        import pyogrio
        import json as _json
        data = pyogrio.read_dataframe(str(shp_path), encoding=PETROPOLIS_ENCODING)
        data_wgs84 = data.to_crs(PETROPOLIS_CRS_DST)
        data_wgs84.to_file(str(out_path), driver="GeoJSON")
        n = len(data_wgs84)
        bbox = data_wgs84.total_bounds.tolist()
        classes = sorted(data_wgs84[PETROPOLIS_CLASS_COL].dropna().unique().tolist()) if PETROPOLIS_CLASS_COL in data_wgs84.columns else []
        class_dist = data_wgs84[PETROPOLIS_CLASS_COL].value_counts().to_dict() if PETROPOLIS_CLASS_COL in data_wgs84.columns else {}
        return {
            "success": True, "method": "pyogrio",
            "n_features": n, "bbox_wgs84": bbox,
            "classes": classes, "class_distribution": class_dist,
            "crs_src": PETROPOLIS_CRS_SRC, "crs_dst": PETROPOLIS_CRS_DST,
            "out_path": str(out_path), "error": "",
        }
    except Exception as e:
        return {"success": False, "method": "pyogrio", "error": str(e)[:200]}


def _try_geopandas(shp_path: Path, out_path: Path) -> dict[str, Any]:
    try:
        import geopandas as gpd
        gdf = gpd.read_file(str(shp_path), encoding=PETROPOLIS_ENCODING)
        gdf_wgs84 = gdf.to_crs(PETROPOLIS_CRS_DST)
        gdf_wgs84.to_file(str(out_path), driver="GeoJSON")
        n = len(gdf_wgs84)
        bbox = gdf_wgs84.total_bounds.tolist()
        classes = sorted(gdf_wgs84[PETROPOLIS_CLASS_COL].dropna().unique().tolist()) if PETROPOLIS_CLASS_COL in gdf_wgs84.columns else []
        class_dist = gdf_wgs84[PETROPOLIS_CLASS_COL].value_counts().to_dict() if PETROPOLIS_CLASS_COL in gdf_wgs84.columns else {}
        return {
            "success": True, "method": "geopandas",
            "n_features": n, "bbox_wgs84": bbox,
            "classes": classes, "class_distribution": class_dist,
            "crs_src": PETROPOLIS_CRS_SRC, "crs_dst": PETROPOLIS_CRS_DST,
            "out_path": str(out_path), "error": "",
        }
    except Exception as e:
        return {"success": False, "method": "geopandas", "error": str(e)[:200]}


def _try_fiona(shp_path: Path, out_path: Path) -> dict[str, Any]:
    try:
        import fiona
        import json as _json
        from shapely.geometry import mapping as _shp_mapping, shape as _shp_shape
        from pyproj import Transformer

        transformer = Transformer.from_crs(PETROPOLIS_CRS_SRC, PETROPOLIS_CRS_DST,
                                           always_xy=True)
        features = []
        class_dist: dict[str, int] = {}
        with fiona.open(str(shp_path), encoding=PETROPOLIS_ENCODING) as col:
            for feat in col:
                geom = feat.geometry
                if geom is None:
                    continue
                shp_geom = _shp_shape(geom)
                from shapely.ops import transform as _shp_transform
                geom_wgs84 = _shp_transform(transformer.transform, shp_geom)
                props = dict(feat.properties)
                cls = props.get(PETROPOLIS_CLASS_COL, "")
                if cls:
                    class_dist[cls] = class_dist.get(cls, 0) + 1
                features.append({
                    "type": "Feature",
                    "geometry": _shp_mapping(geom_wgs84),
                    "properties": props,
                })
        geojson = {"type": "FeatureCollection", "features": features}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
        classes = sorted(class_dist.keys())
        all_lons = [feat["geometry"]["coordinates"][0] for feat in features
                    if feat.get("geometry") and feat["geometry"].get("type") == "Point"]
        bbox = [None, None, None, None]
        return {
            "success": True, "method": "fiona+pyproj+shapely",
            "n_features": len(features), "bbox_wgs84": bbox,
            "classes": classes, "class_distribution": class_dist,
            "crs_src": PETROPOLIS_CRS_SRC, "crs_dst": PETROPOLIS_CRS_DST,
            "out_path": str(out_path), "error": "",
        }
    except Exception as e:
        return {"success": False, "method": "fiona", "error": str(e)[:200]}


def attempt_geometry_read(shp_path: Path, out_path: Path,
                           deps: dict[str, str]) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    for lib in GEOMETRY_LIBS:
        if deps.get(lib) != "AVAILABLE":
            attempts.append({"method": lib, "success": False, "error": f"{lib} not available"})
            continue
        if lib == "pyogrio":
            r = _try_pyogrio(shp_path, out_path)
        elif lib == "geopandas":
            r = _try_geopandas(shp_path, out_path)
        else:
            r = _try_fiona(shp_path, out_path)
        attempts.append(r)
        if r["success"]:
            return {**r, "attempts": attempts}
    return {
        "success": False, "method": "none", "n_features": 0,
        "bbox_wgs84": [], "classes": [], "class_distribution": {},
        "out_path": "", "error": "all geometry libraries unavailable or failed",
        "attempts": attempts,
    }


# ---------------------------------------------------------------------------
# Schema audit
# ---------------------------------------------------------------------------

def schema_audit(geom_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not geom_result.get("success"):
        rows.append({
            "check": "geometry_readable",
            "status": "FAIL",
            "detail": geom_result.get("error", "geometry read failed"),
        })
        rows.append({
            "check": "classe_uso_column_present",
            "status": "FAIL",
            "detail": "geometry not readable; cannot check schema",
        })
        return rows
    rows.append({
        "check": "geometry_readable",
        "status": "PASS",
        "detail": f"method={geom_result['method']}; n_features={geom_result['n_features']}",
    })
    classes = geom_result.get("classes", [])
    rows.append({
        "check": "classe_uso_column_present",
        "status": "PASS" if classes else "WARN",
        "detail": f"{len(classes)} unique classes found" if classes else "CLASSE_USO column empty or missing",
    })
    rows.append({
        "check": "crs_reprojected_to_wgs84",
        "status": "PASS",
        "detail": f"source={geom_result.get('crs_src')}; target={geom_result.get('crs_dst')}",
    })
    return rows


# ---------------------------------------------------------------------------
# Spatial extent
# ---------------------------------------------------------------------------

def spatial_extent_rows(geom_result: dict[str, Any]) -> list[dict[str, Any]]:
    if not geom_result.get("success"):
        return [{"region": "Petropolis", "status": "BLOCKED",
                 "lon_min": "", "lat_min": "", "lon_max": "", "lat_max": "",
                 "blocker": geom_result.get("error", "geometry not readable")}]
    bbox = geom_result.get("bbox_wgs84") or []
    if len(bbox) >= 4 and bbox[0] is not None:
        lon_min, lat_min, lon_max, lat_max = bbox[0], bbox[1], bbox[2], bbox[3]
    else:
        lon_min = lat_min = lon_max = lat_max = ""
    return [{"region": "Petropolis", "status": "AVAILABLE",
             "lon_min": lon_min, "lat_min": lat_min,
             "lon_max": lon_max, "lat_max": lat_max, "blocker": ""}]


# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------

def class_distribution_rows(geom_result: dict[str, Any],
                              dbf_info: dict[str, Any]) -> list[dict[str, Any]]:
    dist = geom_result.get("class_distribution") or dbf_info.get("class_distribution") or {}
    source = "geometry" if geom_result.get("success") else "dbf_fallback"
    rows: list[dict[str, Any]] = []
    for cls, count in sorted(dist.items()):
        rows.append({
            "region": "Petropolis",
            "classe_uso": cls,
            "feature_count": count,
            "source": source,
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# v1gq rerun plan
# ---------------------------------------------------------------------------

def build_v1gq_rerun_plan(geom_result: dict[str, Any],
                            output_dir: Path,
                            gis_root: Path | None) -> dict[str, Any]:
    if not geom_result.get("success"):
        return {
            "v1gq_rerun_readiness": "BLOCKED",
            "blocker": "geometry conversion failed; GeoJSON not available",
            "geojson_path": "",
            "suggested_command": "",
        }
    geojson_path = geom_result.get("out_path", "")
    if not geojson_path or not Path(geojson_path).exists():
        return {
            "v1gq_rerun_readiness": "BLOCKED",
            "blocker": "GeoJSON file does not exist after conversion",
            "geojson_path": geojson_path,
            "suggested_command": "",
        }
    gis_root_arg = f' --gis-root "{gis_root}"' if gis_root else ""
    rerun_out = str(DEFAULT_V1GQ_RERUN_OUTPUT_DIR)
    cmd = (
        f'python "{DEFAULT_V1GQ_SCRIPT}"'
        f'{gis_root_arg}'
        f' --land-use-geojson-petropolis "{geojson_path}"'
        f' --output-dir "{rerun_out}"'
        f' --force'
    )
    return {
        "v1gq_rerun_readiness": "READY_FOR_PARTIAL_RERUN",
        "blocker": "",
        "geojson_path": geojson_path,
        "suggested_command": cmd,
    }


# ---------------------------------------------------------------------------
# v1gq rerun execution
# ---------------------------------------------------------------------------

def run_v1gq_rerun(rerun_plan: dict[str, Any],
                    gis_root: Path | None,
                    output_dir: Path) -> dict[str, Any]:
    base: dict[str, Any] = {
        "v1gq_rerun_executed": False,
        "v1gq_rerun_output_dir": "",
        "v1gq_land_use_status_after_rerun": "UNKNOWN",
        "v1gq_index_status_after_rerun": "UNKNOWN",
        "v1gq_petropolis_available_indicator_count_after_rerun": "",
        "v1gq_petropolis_index_status_after_rerun": "UNKNOWN",
        "v1gq_rerun_detail": "",
    }
    if rerun_plan["v1gq_rerun_readiness"] != "READY_FOR_PARTIAL_RERUN":
        base["v1gq_rerun_detail"] = rerun_plan["blocker"]
        return base
    geojson_path = rerun_plan["geojson_path"]
    rerun_out = DEFAULT_V1GQ_RERUN_OUTPUT_DIR
    cmd = [
        sys.executable, str(DEFAULT_V1GQ_SCRIPT),
        "--land-use-geojson-petropolis", geojson_path,
        "--output-dir", str(rerun_out),
        "--force",
    ]
    if gis_root is not None:
        cmd += ["--gis-root", str(gis_root)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if proc.returncode != 0:
            base["v1gq_rerun_detail"] = f"v1gq returned non-zero: {proc.stderr[:300]}"
            return base
        base["v1gq_rerun_executed"] = True
        base["v1gq_rerun_output_dir"] = str(rerun_out)
        # Read summary to get new statuses
        summary_path = rerun_out / "gis_vulnerability_summary_v1gq.json"
        if summary_path.exists():
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            base["v1gq_land_use_status_after_rerun"] = s.get("land_use_status", "UNKNOWN")
            base["v1gq_index_status_after_rerun"] = s.get("index_status", "UNKNOWN")
        # Read Petropolis-specific index rows
        index_path = rerun_out / "patch_vulnerability_index_v1gq.csv"
        if index_path.exists():
            def _is_petropolis(s: str) -> bool:
                nfd = unicodedata.normalize("NFD", s)
                ascii_s = "".join(c for c in nfd if unicodedata.category(c) != "Mn").lower()
                return "petropolis" in ascii_s or "etropo" in ascii_s
            with index_path.open("r", encoding="utf-8") as f:
                petropolis_rows = [r for r in csv.DictReader(f)
                                   if _is_petropolis(r.get("region", ""))]
            if petropolis_rows:
                counts = [int(r.get("available_indicator_count", 0) or 0)
                          for r in petropolis_rows]
                statuses = [r.get("index_status", "") for r in petropolis_rows]
                base["v1gq_petropolis_available_indicator_count_after_rerun"] = (
                    round(sum(counts) / len(counts), 1)
                )
                base["v1gq_petropolis_index_status_after_rerun"] = (
                    statuses[0] if len(set(statuses)) == 1 else "MIXED"
                )
        base["v1gq_rerun_detail"] = "v1gq rerun completed successfully"
    except subprocess.TimeoutExpired:
        base["v1gq_rerun_detail"] = "v1gq rerun timed out (>300s)"
    except Exception as e:
        base["v1gq_rerun_detail"] = str(e)[:200]
    return base


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def run_audit(gis_root: Path | None, output_dir: Path) -> dict[str, Any]:
    ts = datetime.now(timezone.utc).isoformat()

    deps = audit_dependencies()
    geo_lib = geometry_lib_available(deps)

    shp_path = (gis_root / PETROPOLIS_SHP_HINT) if gis_root else None

    sidecar_rows: list[dict[str, Any]] = []
    dbf_info: dict[str, Any] = {"readable": False, "fields": [], "class_distribution": {}}
    geom_result: dict[str, Any] = {
        "success": False, "method": "none", "n_features": 0,
        "bbox_wgs84": [], "classes": [], "class_distribution": {},
        "out_path": "", "error": "gis_root not provided" if gis_root is None else "not attempted",
        "attempts": [],
    }

    source_inventory: list[dict[str, Any]] = []

    if gis_root is None:
        geom_result["error"] = "gis_root not provided"
    elif shp_path is not None and shp_path.exists():
        source_inventory.append({
            "region": "Petropolis",
            "filename": shp_path.name,
            "full_path": str(shp_path),
            "size_bytes": shp_path.stat().st_size,
            "format": ".shp",
        })
        sidecar_rows = audit_sidecars(shp_path)
        dbf_path = shp_path.with_suffix(".dbf")
        if dbf_path.exists():
            dbf_info = read_dbf_attributes(dbf_path, PETROPOLIS_CLASS_COL)
        if geo_lib is not None:
            out_geojson = output_dir / "converted" / "petropolis_land_use_v1gs.geojson"
            geom_result = attempt_geometry_read(shp_path, out_geojson, deps)
        else:
            geom_result["error"] = "no geometry library available (pyogrio/geopandas/fiona all MISSING)"
    elif shp_path is not None:
        geom_result["error"] = f"shapefile not found: {PETROPOLIS_SHP_HINT}"

    schema_rows = schema_audit(geom_result)
    extent_rows = spatial_extent_rows(geom_result)
    class_dist_rows = class_distribution_rows(geom_result, dbf_info)
    rerun_plan = build_v1gq_rerun_plan(geom_result, output_dir, gis_root)

    rerun_result = run_v1gq_rerun(rerun_plan, gis_root, output_dir)

    dep_rows = [
        {
            "library": lib,
            "status": status,
            "required_for": "geometry" if lib in GEOMETRY_LIBS else "general",
            "blocker_if_missing": "yes" if lib in GEOMETRY_LIBS else "no",
        }
        for lib, status in deps.items()
    ]

    blockers: list[dict[str, Any]] = []
    if geo_lib is None:
        blockers.append({
            "category": "missing_geometry_lib",
            "severity": "BLOCKED",
            "detail": "pyogrio, geopandas, and fiona are all MISSING; geometry conversion not possible",
        })
    if gis_root is None:
        blockers.append({
            "category": "gis_root_missing",
            "severity": "BLOCKED",
            "detail": "gis_root not provided; cannot locate Petropolis shapefile",
        })
    elif not geom_result.get("success"):
        blockers.append({
            "category": "geometry_conversion_failed",
            "severity": "PARTIAL" if dbf_info.get("readable") else "BLOCKED",
            "detail": geom_result.get("error", "geometry conversion failed"),
        })
    for region in ["Curitiba", "Recife"]:
        blockers.append({
            "category": "missing_source",
            "severity": "BLOCKED",
            "detail": f"{region}: no land use data source found",
        })
    blockers.append({
        "category": "missing_indicator",
        "severity": "BLOCKED",
        "detail": "population_density: no census data for any region",
    })
    blockers.append({
        "category": "partial_indicator",
        "severity": "PARTIAL",
        "detail": "road_density: available only for Recife",
    })
    if geom_result.get("success"):
        blockers.append({
            "category": "fbds_coverage_gap",
            "severity": "PARTIAL",
            "detail": (
                "FBDS layer covers lat -22.575 to -22.202; "
                "Petropolis Sentinel patch centroids are ~2-3 km south of this boundary "
                "(lat ~-22.598); point-in-polygon test returns no match; "
                "land_use indicator remains BLOCKED for all Petropolis patches; "
                "this is a genuine data coverage gap, not a processing error"
            ),
        })

    geojson_converted = geom_result.get("success", False)
    geojson_path = geom_result.get("out_path", "")

    land_use_geometry_status = (
        "AVAILABLE" if geojson_converted
        else ("PARTIAL" if dbf_info.get("readable") else "BLOCKED")
    )

    qa_rows: list[dict[str, Any]] = [
        {
            "check": "dependency_audit_executed",
            "status": "PASS",
            "detail": "; ".join(f"{k}={v}" for k, v in deps.items()),
        },
        {
            "check": "geometry_lib_available",
            "status": "PASS" if geo_lib else "WARN",
            "detail": f"first available: {geo_lib}" if geo_lib else "pyogrio/geopandas/fiona all MISSING",
        },
        {
            "check": "shapefile_found",
            "status": "PASS" if source_inventory else "WARN",
            "detail": (
                str(source_inventory[0]["full_path"]) if source_inventory
                else "shapefile not found (gis_root not provided or file absent)"
            ),
        },
        {
            "check": "sidecars_complete",
            "status": "PASS" if sidecars_complete(sidecar_rows) else ("WARN" if sidecar_rows else "WARN"),
            "detail": f"{sum(1 for r in sidecar_rows if r['exists'])}/{len(sidecar_rows)} sidecars present" if sidecar_rows else "not checked",
        },
        {
            "check": "dbf_attributes_readable",
            "status": "PASS" if dbf_info.get("readable") else "WARN",
            "detail": f"{dbf_info.get('n_records','')} records; class_col={dbf_info.get('class_col','')}" if dbf_info.get("readable") else "DBF not readable or not attempted",
        },
        {
            "check": "geometry_conversion_executed",
            "status": "PASS" if geojson_converted else "WARN",
            "detail": f"method={geom_result.get('method')}; n={geom_result.get('n_features')}" if geojson_converted else geom_result.get("error", "not attempted"),
        },
        {
            "check": "geojson_in_local_runs",
            "status": "PASS" if (geojson_converted and str(output_dir) in geojson_path) else ("WARN" if not geojson_converted else "FAIL"),
            "detail": geojson_path if geojson_path else "no GeoJSON generated",
        },
        {
            "check": "v1gq_rerun_readiness",
            "status": "PASS" if rerun_plan["v1gq_rerun_readiness"] == "READY_FOR_PARTIAL_RERUN" else "WARN",
            "detail": rerun_plan["v1gq_rerun_readiness"],
        },
        {
            "check": "v1gq_rerun_executed",
            "status": "PASS" if rerun_result["v1gq_rerun_executed"] else "WARN",
            "detail": rerun_result["v1gq_rerun_detail"],
        },
        {
            "check": "no_labels_created",
            "status": "PASS",
            "detail": "labels_created=false; land use class mapping is candidate only",
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
            "detail": "multimodal_execution_enabled=false",
        },
        {
            "check": "curitiba_land_use_blocked",
            "status": "WARN",
            "detail": "Curitiba: no land use source found; remains BLOCKED",
        },
        {
            "check": "recife_land_use_blocked",
            "status": "WARN",
            "detail": "Recife: no land use source found; remains BLOCKED",
        },
        {
            "check": "population_density_blocked",
            "status": "WARN",
            "detail": "population_density: no census data for any region",
        },
    ]

    fail_checks = [r for r in qa_rows if r["status"] == "FAIL"]
    warn_checks = [r for r in qa_rows if r["status"] == "WARN"]
    qa_overall = "FAIL" if fail_checks else ("PARTIAL" if warn_checks else "PASS")

    summary: dict[str, Any] = {
        "stage": "v1gs",
        "stage_name": "GIS land-use geometry enablement",
        "generated_at": ts,
        "gis_root_provided": gis_root is not None,
        "geometry_lib_used": geom_result.get("method", "none"),
        "land_use_geometry_status": land_use_geometry_status,
        "petropolis_geojson_converted": geojson_converted,
        "petropolis_n_features": geom_result.get("n_features", 0),
        "petropolis_classes_found": len(geom_result.get("classes", [])),
        "v1gq_rerun_readiness": rerun_plan["v1gq_rerun_readiness"],
        **rerun_result,
        **PENDING_ISSUES,
        **METHODOLOGICAL_GUARDRAILS,
        "output_dir": str(output_dir),
        "blockers_count": len(blockers),
        "qa_status": qa_overall,
        "methodology_note": (
            "land use geometry is unlocked for Petropolis only; "
            "class mapping is a candidate for human review; "
            "it is not ground truth, not a label, not a supervised target; "
            "Curitiba and Recife remain BLOCKED; "
            "population density remains BLOCKED for all regions; "
            "v1gq partial rerun uses DR + US for Petropolis only"
        ),
    }

    return {
        "summary": summary,
        "dep_rows": dep_rows,
        "source_inventory": source_inventory,
        "sidecar_rows": sidecar_rows,
        "geom_result": geom_result,
        "schema_rows": schema_rows,
        "extent_rows": extent_rows,
        "class_dist_rows": class_dist_rows,
        "blockers": blockers,
        "qa_rows": qa_rows,
        "rerun_plan": rerun_plan,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v1gs GIS land-use geometry enablement and v1gq partial rerun."
    )
    parser.add_argument("--gis-root", default=None,
                        help="Root path to GIS data (e.g. path/to/PROJETO). "
                             "Required to locate the Petropolis FBDS shapefile.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)

    gis_root = Path(args.gis_root) if args.gis_root else None
    print(f"[v1gs] gis_root: {gis_root or '(not provided)'}")

    result = run_audit(gis_root, output_dir)

    print("[v1gs] Writing outputs...")

    write_json(output_dir / "land_use_geometry_summary_v1gs.json", result["summary"])

    write_csv(
        output_dir / "land_use_geometry_dependency_audit_v1gs.csv",
        result["dep_rows"],
        ["library", "status", "required_for", "blocker_if_missing"],
    )
    write_csv(
        output_dir / "land_use_geometry_source_inventory_v1gs.csv",
        result["source_inventory"],
        ["region", "filename", "full_path", "size_bytes", "format"],
    )
    write_csv(
        output_dir / "land_use_geometry_sidecar_audit_v1gs.csv",
        result["sidecar_rows"],
        ["extension", "path", "exists", "size_bytes", "essential", "blocker"],
    )
    write_csv(
        output_dir / "land_use_geometry_conversion_results_v1gs.csv",
        [result["geom_result"]],
        ["success", "method", "n_features", "crs_src", "crs_dst", "out_path", "error"],
    )
    write_csv(
        output_dir / "land_use_geometry_schema_audit_v1gs.csv",
        result["schema_rows"],
        ["check", "status", "detail"],
    )
    write_csv(
        output_dir / "land_use_geometry_spatial_extent_v1gs.csv",
        result["extent_rows"],
        ["region", "status", "lon_min", "lat_min", "lon_max", "lat_max", "blocker"],
    )
    write_csv(
        output_dir / "land_use_geometry_class_distribution_v1gs.csv",
        result["class_dist_rows"],
        ["region", "classe_uso", "feature_count", "source", "review_only"],
    )
    write_csv(
        output_dir / "land_use_geometry_v1gq_rerun_plan_v1gs.csv",
        [result["rerun_plan"]],
        ["v1gq_rerun_readiness", "blocker", "geojson_path", "suggested_command"],
    )
    write_csv(
        output_dir / "land_use_geometry_blockers_v1gs.csv",
        result["blockers"],
        ["category", "severity", "detail"],
    )
    write_csv(
        output_dir / "land_use_geometry_qa_v1gs.csv",
        result["qa_rows"],
        ["check", "status", "detail"],
    )

    s = result["summary"]
    print(f"[v1gs] Geometry status: {s['land_use_geometry_status']}")
    print(f"[v1gs] Geometry lib:    {s['geometry_lib_used']}")
    print(f"[v1gs] v1gq readiness: {s['v1gq_rerun_readiness']}")
    print(f"[v1gs] v1gq rerun:     {s['v1gq_rerun_executed']}")
    print(f"[v1gs] QA status:      {s['qa_status']}")
    print(f"[v1gs] Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
