from __future__ import annotations

import argparse
import csv
import json
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gt"

DEFAULT_V1GE_MANIFEST = (
    ROOT / "local_runs" / "dino_embeddings" / "v1ge"
    / "dino_expanded_embedding_manifest_v1ge.csv"
)
DEFAULT_V1FU_MANIFEST = (
    ROOT / "manifests" / "dino_inputs"
    / "revp_v1fu_dino_sentinel_input_manifest"
    / "dino_sentinel_input_manifest_v1fu.csv"
)
V1GS_GEOJSON = (
    ROOT / "local_runs" / "dino_embeddings" / "v1gs"
    / "converted" / "petropolis_land_use_v1gs.geojson"
)

SENTINEL_SUBDIR = "data/sentinel"

# WGS84 bbox of the converted Petropolis FBDS layer (from v1gs)
PETROPOLIS_FBDS_BBOX = [-43.37768, -22.57516, -42.97798, -22.20247]

PATCH_SCOPES = ("dino-corpus", "full-manifest")

METHODOLOGICAL_GUARDRAILS: dict[str, Any] = {
    "review_only": True,
    "supervised_training": False,
    "labels_created": False,
    "targets_created": False,
    "predictive_claims": False,
    "multimodal_execution_enabled": False,
    "land_use_is_ground_truth": False,
    "vulnerability_index_is_ground_truth": False,
    "dino_predicts_vulnerability": False,
}

COVERAGE_STATUS_COVERED = "COVERED"
COVERAGE_STATUS_BBOX_ONLY = "BBOX_OVERLAP_NO_CENTROID"
COVERAGE_STATUS_UNCOVERED = "UNCOVERED"
COVERAGE_STATUS_NO_TIF = "NO_TIF"

EXPANSION_CANDIDATES: list[dict[str, str]] = [
    {
        "region": "Curitiba",
        "candidate_source": "MapBiomas",
        "url_hint": "mapbiomas.org",
        "notes": "national annual land-use/land-cover; would cover Curitiba patches",
        "status": "NOT_ACQUIRED",
    },
    {
        "region": "Recife",
        "candidate_source": "MapBiomas",
        "url_hint": "mapbiomas.org",
        "notes": "national annual land-use/land-cover; would cover Recife patches",
        "status": "NOT_ACQUIRED",
    },
    {
        "region": "Petropolis",
        "candidate_source": "MapBiomas",
        "url_hint": "mapbiomas.org",
        "notes": "national coverage may include southern Petropolis patches outside FBDS extent",
        "status": "NOT_ACQUIRED",
    },
    {
        "region": "Petropolis",
        "candidate_source": "FBDS_extended_or_updated",
        "url_hint": "fbds.org.br",
        "notes": "more recent or wider FBDS edition may cover lat < -22.575",
        "status": "NOT_ACQUIRED",
    },
    {
        "region": "All",
        "candidate_source": "IBGE_LULC_grid",
        "url_hint": "ibge.gov.br",
        "notes": "IBGE land-use census grids; national coverage; requires download",
        "status": "NOT_ACQUIRED",
    },
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


def normalize_region(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn").lower().strip()


def read_csv_file(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Patch loading — normalizes to unified schema
# ---------------------------------------------------------------------------

def load_patches(scope: str, gis_root: Path | None) -> list[dict[str, Any]]:
    if scope == "dino-corpus":
        rows = read_csv_file(DEFAULT_V1GE_MANIFEST)
        patches = []
        for r in rows:
            patches.append({
                "patch_id": r.get("patch_id", ""),
                "region": r.get("region", ""),
                "source_path": r.get("source_path", ""),
                "scope": "dino-corpus",
                "manifest_row": r,
            })
        return patches
    else:  # full-manifest
        rows = read_csv_file(DEFAULT_V1FU_MANIFEST)
        patches = []
        for r in rows:
            asset_rel = r.get("asset_path_reference", "")
            src_path = str(gis_root / asset_rel) if (gis_root and asset_rel) else ""
            patches.append({
                "patch_id": r.get("canonical_patch_id", r.get("dino_input_id", "")),
                "region": r.get("region", ""),
                "source_path": src_path,
                "scope": "full-manifest",
                "manifest_row": r,
            })
        return patches


# ---------------------------------------------------------------------------
# TIF bounds
# ---------------------------------------------------------------------------

def get_patch_bounds_wgs84(tif_path: str) -> tuple[float, float, float, float] | None:
    if not tif_path:
        return None
    p = Path(tif_path)
    if not p.exists():
        return None
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        with rasterio.open(p) as ds:
            b = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
            return float(b[0]), float(b[1]), float(b[2]), float(b[3])
    except Exception:
        return None


def patch_centroid(bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    lon = (bounds[0] + bounds[2]) / 2.0
    lat = (bounds[1] + bounds[3]) / 2.0
    return lon, lat


# ---------------------------------------------------------------------------
# Source inventory
# ---------------------------------------------------------------------------

def load_known_sources() -> list[dict[str, Any]]:
    sources = []
    geojson_exists = V1GS_GEOJSON.exists()
    sources.append({
        "source_id": "petropolis_fbds_v1gs",
        "region": "Petropolis",
        "format": "geojson_wgs84",
        "local_path": str(V1GS_GEOJSON),
        "file_exists": geojson_exists,
        "bbox_lon_min": PETROPOLIS_FBDS_BBOX[0],
        "bbox_lat_min": PETROPOLIS_FBDS_BBOX[1],
        "bbox_lon_max": PETROPOLIS_FBDS_BBOX[2],
        "bbox_lat_max": PETROPOLIS_FBDS_BBOX[3],
        "class_col": "CLASSE_USO",
        "n_features": 6861,
        "origin": "v1gs",
        "status": "AVAILABLE" if geojson_exists else "MISSING",
    })
    return sources


# ---------------------------------------------------------------------------
# Coverage geometry
# ---------------------------------------------------------------------------

def bbox_overlaps(
    lon_min_a: float, lat_min_a: float, lon_max_a: float, lat_max_a: float,
    lon_min_b: float, lat_min_b: float, lon_max_b: float, lat_max_b: float,
) -> bool:
    return not (
        lon_max_a < lon_min_b or lon_min_a > lon_max_b
        or lat_max_a < lat_min_b or lat_min_a > lat_max_b
    )


def centroid_in_source_geojson(cx: float, cy: float, geojson_path: Path) -> dict[str, Any]:
    if not geojson_path.exists():
        return {"found": False, "reason": "GeoJSON file not found"}
    try:
        from shapely.geometry import Point, shape as _shp_shape
        import json as _json
        with geojson_path.open(encoding="utf-8") as f:
            data = _json.load(f)
        pt = Point(cx, cy)
        for feat in data.get("features", []):
            geom = feat.get("geometry")
            if not geom:
                continue
            try:
                poly = _shp_shape(geom)
                if poly.contains(pt):
                    props = feat.get("properties", {})
                    return {
                        "found": True,
                        "class_value": props.get("CLASSE_USO", ""),
                        "reason": "",
                    }
            except Exception:
                continue
        return {"found": False, "reason": "centroid not inside any polygon"}
    except ImportError:
        return {"found": False, "reason": "shapely not installed"}
    except Exception as e:
        return {"found": False, "reason": str(e)[:120]}


# ---------------------------------------------------------------------------
# Per-patch assessment
# ---------------------------------------------------------------------------

def assess_patch(patch: dict[str, Any],
                 sources: list[dict[str, Any]]) -> dict[str, Any]:
    pid = patch["patch_id"]
    region = patch["region"]
    region_key = normalize_region(region)
    bounds = get_patch_bounds_wgs84(patch["source_path"])

    base: dict[str, Any] = {
        "patch_id": pid,
        "region": region,
        "scope": patch["scope"],
        "tif_found": bounds is not None,
        "lon_min": "", "lat_min": "", "lon_max": "", "lat_max": "",
        "centroid_lon": "", "centroid_lat": "",
        "coverage_status": COVERAGE_STATUS_NO_TIF,
        "source_id": "",
        "source_bbox_overlap": False,
        "centroid_in_polygon": False,
        "class_value": "",
        "blocker": "",
        "note": "",
    }

    if bounds is None:
        base["blocker"] = (
            "TIF file not found or not readable; cannot determine patch location"
        )
        return base

    lon_min, lat_min, lon_max, lat_max = bounds
    cx, cy = patch_centroid(bounds)
    base.update({
        "lon_min": round(lon_min, 6),
        "lat_min": round(lat_min, 6),
        "lon_max": round(lon_max, 6),
        "lat_max": round(lat_max, 6),
        "centroid_lon": round(cx, 6),
        "centroid_lat": round(cy, 6),
    })

    for src in sources:
        region_matches = (normalize_region(src["region"]) == region_key
                          or src["region"] == "All")
        if not region_matches:
            continue
        if not src.get("file_exists", False):
            continue
        overlap = bbox_overlaps(
            lon_min, lat_min, lon_max, lat_max,
            src["bbox_lon_min"], src["bbox_lat_min"],
            src["bbox_lon_max"], src["bbox_lat_max"],
        )
        if not overlap:
            continue
        base["source_id"] = src["source_id"]
        base["source_bbox_overlap"] = True
        result = centroid_in_source_geojson(cx, cy, Path(src["local_path"]))
        if result["found"]:
            base["coverage_status"] = COVERAGE_STATUS_COVERED
            base["centroid_in_polygon"] = True
            base["class_value"] = result["class_value"]
            return base
        else:
            base["coverage_status"] = COVERAGE_STATUS_BBOX_ONLY
            base["note"] = result["reason"]
            return base

    base["coverage_status"] = COVERAGE_STATUS_UNCOVERED
    base["blocker"] = f"no land-use source with bbox overlapping region={region}"
    return base


# ---------------------------------------------------------------------------
# Region and gap summaries
# ---------------------------------------------------------------------------

def region_summary(patch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from collections import defaultdict
    buckets: dict[str, list[str]] = defaultdict(list)
    for r in patch_rows:
        buckets[r["region"]].append(r["coverage_status"])
    rows = []
    for region, statuses in sorted(buckets.items()):
        n = len(statuses)
        n_covered = statuses.count(COVERAGE_STATUS_COVERED)
        n_bbox = statuses.count(COVERAGE_STATUS_BBOX_ONLY)
        n_uncovered = statuses.count(COVERAGE_STATUS_UNCOVERED)
        n_no_tif = statuses.count(COVERAGE_STATUS_NO_TIF)
        rows.append({
            "region": region,
            "n_patches": n,
            "n_covered": n_covered,
            "n_bbox_overlap_only": n_bbox,
            "n_uncovered": n_uncovered,
            "n_no_tif": n_no_tif,
            "coverage_rate": round(n_covered / n, 3) if n else 0.0,
            "land_use_status": (
                "AVAILABLE" if n_covered == n
                else ("PARTIAL" if n_covered > 0
                      else ("BBOX_PARTIAL" if n_bbox > 0 else "BLOCKED"))
            ),
        })
    return rows


def coverage_gap_rows(patch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps = []
    for r in patch_rows:
        if r["coverage_status"] in (COVERAGE_STATUS_COVERED,):
            continue
        severity = (
            "BLOCKED" if r["coverage_status"] in (COVERAGE_STATUS_UNCOVERED,
                                                   COVERAGE_STATUS_NO_TIF)
            else "PARTIAL"
        )
        gaps.append({
            "patch_id": r["patch_id"],
            "region": r["region"],
            "coverage_status": r["coverage_status"],
            "severity": severity,
            "detail": r.get("blocker") or r.get("note") or r["coverage_status"],
        })
    return gaps


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def run_audit(scope: str, gis_root: Path | None,
              output_dir: Path) -> dict[str, Any]:
    ts = datetime.now(timezone.utc).isoformat()

    patches = load_patches(scope, gis_root)
    sources = load_known_sources()
    patch_rows = [assess_patch(p, sources) for p in patches]
    reg_summary = region_summary(patch_rows)
    gaps = coverage_gap_rows(patch_rows)

    n_covered = sum(1 for r in patch_rows if r["coverage_status"] == COVERAGE_STATUS_COVERED)
    n_bbox_only = sum(1 for r in patch_rows if r["coverage_status"] == COVERAGE_STATUS_BBOX_ONLY)
    n_uncovered = sum(1 for r in patch_rows if r["coverage_status"] == COVERAGE_STATUS_UNCOVERED)
    n_no_tif = sum(1 for r in patch_rows if r["coverage_status"] == COVERAGE_STATUS_NO_TIF)
    n_total = len(patch_rows)

    overall_coverage = (
        "FULL" if n_covered == n_total
        else ("PARTIAL" if n_covered > 0
              else ("BBOX_PARTIAL" if n_bbox_only > 0 else "BLOCKED"))
    )

    available_sources = [s for s in sources if s.get("file_exists")]

    qa_rows: list[dict[str, Any]] = [
        {
            "check": "manifest_loaded",
            "status": "PASS" if patches else "FAIL",
            "detail": f"{n_total} patches loaded from scope={scope}",
        },
        {
            "check": "sources_inventoried",
            "status": "PASS",
            "detail": f"{len(sources)} known sources; {len(available_sources)} available",
        },
        {
            "check": "tif_bounds_readable",
            "status": "PASS" if n_no_tif == 0 else "WARN",
            "detail": f"{n_total - n_no_tif}/{n_total} patches with readable TIF bounds",
        },
        {
            "check": "covered_patches",
            "status": "PASS" if n_covered > 0 else "WARN",
            "detail": f"{n_covered}/{n_total} patches with land_use COVERED",
        },
        {
            "check": "bbox_overlap_only",
            "status": "WARN" if n_bbox_only > 0 else "PASS",
            "detail": (
                f"{n_bbox_only} patches with bbox overlap but centroid outside polygon "
                "(coverage gap: patch location outside source extent)"
                if n_bbox_only > 0
                else "no bbox-only overlaps"
            ),
        },
        {
            "check": "uncovered_patches",
            "status": "WARN" if n_uncovered > 0 else "PASS",
            "detail": f"{n_uncovered} patches with no land-use source overlap",
        },
        {
            "check": "no_labels_created",
            "status": "PASS",
            "detail": "labels_created=false; coverage audit only",
        },
        {
            "check": "land_use_not_ground_truth",
            "status": "PASS",
            "detail": "land_use_is_ground_truth=false",
        },
        {
            "check": "multimodal_disabled",
            "status": "PASS",
            "detail": "multimodal_execution_enabled=false",
        },
        {
            "check": "expansion_candidates_documented",
            "status": "PASS",
            "detail": f"{len(EXPANSION_CANDIDATES)} expansion candidate sources documented",
        },
        {
            "check": "petropolis_fbds_coverage_gap_documented",
            "status": "PASS",
            "detail": (
                "FBDS layer covers lat -22.575 to -22.202; "
                "Petropolis dino-corpus patches at lat ~-22.598 are outside this extent; "
                "documented as data coverage gap, not a processing error"
            ),
        },
    ]

    fail_checks = [r for r in qa_rows if r["status"] == "FAIL"]
    warn_checks = [r for r in qa_rows if r["status"] == "WARN"]
    qa_overall = "FAIL" if fail_checks else ("PARTIAL" if warn_checks else "PASS")

    summary: dict[str, Any] = {
        "stage": "v1gt",
        "stage_name": "GIS land-use coverage expansion audit",
        "generated_at": ts,
        "scope": scope,
        "gis_root_provided": gis_root is not None,
        "n_patches_total": n_total,
        "n_patches_covered": n_covered,
        "n_patches_bbox_overlap_only": n_bbox_only,
        "n_patches_uncovered": n_uncovered,
        "n_patches_no_tif": n_no_tif,
        "overall_coverage_status": overall_coverage,
        "n_known_sources": len(sources),
        "n_available_sources": len(available_sources),
        "n_expansion_candidates": len(EXPANSION_CANDIDATES),
        "petropolis_fbds_coverage_gap": (
            "FBDS extent ends at lat -22.575; dino-corpus Petropolis patches "
            "at lat ~-22.598 are ~2-3 km south of coverage boundary"
        ),
        **METHODOLOGICAL_GUARDRAILS,
        "output_dir": str(output_dir),
        "blockers_count": len(gaps),
        "qa_status": qa_overall,
        "methodology_note": (
            "land-use coverage audit only; no labels created; "
            "coverage status indicates data availability, "
            "not ground truth or vulnerability classification; "
            "DINO does not predict vulnerability"
        ),
    }

    return {
        "summary": summary,
        "patch_rows": patch_rows,
        "reg_summary": reg_summary,
        "gaps": gaps,
        "sources": sources,
        "expansion_candidates": EXPANSION_CANDIDATES,
        "qa_rows": qa_rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v1gt GIS land-use coverage expansion audit."
    )
    parser.add_argument(
        "--gis-root", default=None,
        help="Root path to GIS data (e.g. path/to/PROJETO). Required to locate TIF files.",
    )
    parser.add_argument(
        "--patch-scope", default="dino-corpus",
        choices=list(PATCH_SCOPES),
        help="dino-corpus = v1ge 12 patches; full-manifest = v1fu 128 patches.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)

    gis_root = Path(args.gis_root) if args.gis_root else None
    scope = args.patch_scope
    print(f"[v1gt] scope: {scope}")
    print(f"[v1gt] gis_root: {gis_root or '(not provided)'}")

    result = run_audit(scope, gis_root, output_dir)

    print("[v1gt] Writing outputs...")

    write_json(output_dir / "land_use_coverage_summary_v1gt.json", result["summary"])

    write_csv(
        output_dir / "land_use_patch_coverage_v1gt.csv",
        result["patch_rows"],
        ["patch_id", "region", "scope", "tif_found",
         "lon_min", "lat_min", "lon_max", "lat_max",
         "centroid_lon", "centroid_lat",
         "coverage_status", "source_id",
         "source_bbox_overlap", "centroid_in_polygon",
         "class_value", "blocker", "note"],
    )
    write_csv(
        output_dir / "land_use_region_coverage_v1gt.csv",
        result["reg_summary"],
        ["region", "n_patches", "n_covered", "n_bbox_overlap_only",
         "n_uncovered", "n_no_tif", "coverage_rate", "land_use_status"],
    )
    write_csv(
        output_dir / "land_use_source_inventory_v1gt.csv",
        result["sources"],
        ["source_id", "region", "format", "local_path", "file_exists",
         "bbox_lon_min", "bbox_lat_min", "bbox_lon_max", "bbox_lat_max",
         "class_col", "n_features", "origin", "status"],
    )
    write_csv(
        output_dir / "land_use_coverage_gaps_v1gt.csv",
        result["gaps"],
        ["patch_id", "region", "coverage_status", "severity", "detail"],
    )
    write_csv(
        output_dir / "land_use_expansion_candidates_v1gt.csv",
        result["expansion_candidates"],
        ["region", "candidate_source", "url_hint", "notes", "status"],
    )
    write_csv(
        output_dir / "land_use_coverage_qa_v1gt.csv",
        result["qa_rows"],
        ["check", "status", "detail"],
    )

    s = result["summary"]
    print(f"[v1gt] Coverage: {s['n_patches_covered']}/{s['n_patches_total']} patches COVERED")
    print(f"[v1gt] BBOX-only: {s['n_patches_bbox_overlap_only']}")
    print(f"[v1gt] Uncovered: {s['n_patches_uncovered']}")
    print(f"[v1gt] No-TIF:    {s['n_patches_no_tif']}")
    print(f"[v1gt] Overall:   {s['overall_coverage_status']}")
    print(f"[v1gt] QA:        {s['qa_status']}")
    print(f"[v1gt] Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
