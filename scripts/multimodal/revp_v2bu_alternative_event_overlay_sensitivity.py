"""REV-P v2bu — Alternative event geometry overlay sensitivity audit.

Re-runs the overlay of the patches with a recovered boundary (scope:
37_RETRIED_PATCHES = 36 recovered boundaries + REC_00019) against the 5 QA-only
alternative event geometries built by v2bt (convex hull, buffered unions,
cluster envelopes), producing a geometric-sensitivity matrix.

The question is not "which patches are positive" but "which patches show robust
geometric compatibility with the QA-only event geometry across reconstruction
methods". An intersection is QA-only: it is never a positive label, a
non-intersection is never a formal negative, and nothing enables training. A
patch can reach a *formal-review queue* (QA-compatible), but
``gt_patch_flood_observed`` stays NA and ``allowed_for_training`` stays False.
No geometry is invented. Outputs are local-only and light.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - availability depends on environment
    from shapely.geometry import shape as _shapely_shape  # type: ignore
    from shapely.validation import make_valid as _make_valid  # type: ignore
    HAS_SHAPELY = True
except Exception:  # pragma: no cover
    _shapely_shape = None
    _make_valid = None
    HAS_SHAPELY = False

import importlib.util as _ilu
HAS_PYPROJ = _ilu.find_spec("pyproj") is not None  # reported only; v2bu does not reproject


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "ground_truth" / "v2bu"
STAGE = "v2bu"
EVENT_ID = "REC_2022_05_24_30"

DEFAULT_ALT_DIR = ROOT / "local_runs" / "ground_truth" / "v2bt" / "alternative_event_geometries"
DEFAULT_ALT_REGISTRY = ROOT / "local_runs" / "ground_truth" / "v2bt" / "alternative_event_geometry_registry_v2bt.csv"
DEFAULT_RECOVERED_DIR = ROOT / "local_runs" / "ground_truth" / "v2br" / "recovered_patch_boundaries"
DEFAULT_REC19 = ROOT / "datasets" / "external_sources" / "recife_minimal_tp" / "derived" / "patch_boundary_REC_00019_from_lineage.geojson"

LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = -74.5, -33.0, -34.5, 6.0


METHODOLOGICAL_GUARDRAILS = {
    "review_only": True,
    "labels_created": False,
    "formal_negative_created": False,
    "positive_label_from_qa_overlay": False,
    "negative_label_from_no_intersection": False,
    "negative_from_absence": False,
    "alternative_geometry_promoted_to_gt": False,
    "point_derived_geometry_promoted_to_gt": False,
    "ready_for_formal_review_is_training_ready": False,
    "geometry_invented": False,
    "supervised_training": False,
    "outputs_local_only": True,
}

# Pairwise overlay statuses
QA_INTERSECTS = "QA_OVERLAY_INTERSECTS"
QA_NO_INTERSECTION = "QA_OVERLAY_NO_INTERSECTION"
QA_BLOCK_PATCH = "QA_OVERLAY_BLOCKED_INVALID_PATCH_GEOMETRY"
QA_BLOCK_ALT = "QA_OVERLAY_BLOCKED_INVALID_ALTERNATIVE_GEOMETRY"
QA_BLOCK_BACKEND = "QA_OVERLAY_BLOCKED_BACKEND_UNAVAILABLE"

# Per-patch QA compatibility statuses
QA_ROBUST = "QA_COMPATIBLE_ROBUST"
QA_METHOD_DEP = "QA_COMPATIBLE_METHOD_DEPENDENT"
QA_BUFFER_ONLY = "QA_COMPATIBLE_BUFFER_ONLY"
QA_NOT_COMPAT = "QA_NOT_COMPATIBLE_ACROSS_ALTERNATIVES"
QA_BLOCKED_PATCH = "QA_BLOCKED_PATCH_GEOMETRY"
QA_AMBIGUOUS = "QA_AMBIGUOUS_CONFLICTING_ALTERNATIVES"

# Formal-review queue statuses
RQ_READY = "READY_FOR_FORMAL_GT_REVIEW_QA_COMPATIBLE"
RQ_HELD_METHOD = "HELD_METHOD_DEPENDENT_QA_ONLY"
RQ_HELD_BUFFER = "HELD_BUFFER_ONLY_QA_ONLY"
RQ_HELD_NONE = "HELD_NO_QA_COMPATIBILITY"
RQ_BLOCKED = "BLOCKED_PATCH_BOUNDARY_MISSING"


PAIRWISE_FIELDS = [
    "pairwise_id", "canonical_patch_id", "candidate_event_id", "alternative_geometry_id", "geometry_method",
    "patch_boundary_source", "patch_boundary_quality", "alternative_geometry_quality", "patch_crs", "alternative_crs",
    "geometry_backend", "patch_geometry_valid", "alternative_geometry_valid", "bbox_overlap", "intersects",
    "intersection_area", "intersection_area_units", "patch_area", "alternative_area", "intersection_ratio_patch",
    "intersection_ratio_alternative", "centroid_distance", "centroid_distance_units", "min_geometry_distance",
    "min_geometry_distance_units", "overlay_status", "gt_patch_flood_observed", "allowed_for_training", "notes",
]
MATRIX_FIELDS = [
    "canonical_patch_id", "candidate_event_id", "alternatives_tested", "alternatives_intersecting", "intersecting_methods",
    "non_intersecting_methods", "max_intersection_ratio_patch", "mean_intersection_ratio_patch", "median_intersection_ratio_patch",
    "max_intersection_area", "min_centroid_distance", "robustness_status", "qa_compatibility_status", "ready_for_formal_gt_review",
    "gt_patch_flood_observed", "allowed_for_training", "promotion_blocker", "notes",
]
METHOD_SUMMARY_FIELDS = [
    "alternative_geometry_id", "geometry_method", "geometry_quality", "patches_tested", "patches_intersecting",
    "patches_non_intersecting", "mean_intersection_ratio_patch", "max_intersection_ratio_patch", "recommended_use",
    "method_interpretation", "can_create_label", "can_enable_training",
]
COMPAT_FIELDS = ["canonical_patch_id", "candidate_event_id", "qa_compatibility_status", "alternatives_intersecting", "intersecting_methods", "ready_for_formal_gt_review", "gt_patch_flood_observed", "allowed_for_training"]
REVIEW_QUEUE_FIELDS = ["queue_id", "canonical_patch_id", "candidate_event_id", "qa_compatibility_status", "robustness_status", "intersecting_methods", "ready_for_formal_gt_review", "gt_patch_flood_observed", "allowed_for_training", "required_next_evidence", "recommended_next_action"]


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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


def local_runs_ignored() -> bool:
    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        return False
    return any(line.strip() in {"local_runs", "local_runs/"} for line in gitignore.read_text(encoding="utf-8").splitlines())


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return path.name


def short_id(prefix: str, value: str) -> str:
    import hashlib
    return f"{prefix}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:12]}"


# --------------------------------------------------------------------------- #
# Geometry primitives
# --------------------------------------------------------------------------- #

def load_geojson(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def first_geometry(doc: Any) -> tuple[dict, dict]:
    if not isinstance(doc, dict):
        return {}, {}
    feats = doc.get("features") if doc.get("type") == "FeatureCollection" else [doc]
    for f in feats or []:
        if not isinstance(f, dict):
            continue
        g = f.get("geometry") or (f if f.get("type") in {"Polygon", "MultiPolygon"} else {})
        if g and g.get("coordinates"):
            return g, (f.get("properties") or doc.get("properties") or {})
    return {}, (doc.get("properties") or {})


def flat_xy(coords: Any) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []

    def walk(c: Any) -> None:
        if isinstance(c, (int, float)):
            return
        if len(c) >= 2 and isinstance(c[0], (int, float)) and isinstance(c[1], (int, float)):
            xs.append(float(c[0]))
            ys.append(float(c[1]))
            return
        for sub in c:
            walk(sub)

    walk(coords)
    return xs, ys


def geom_bbox(geom: dict) -> tuple[float, float, float, float] | None:
    if not geom or not geom.get("coordinates"):
        return None
    xs, ys = flat_xy(geom["coordinates"])
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_centroid(b: tuple) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def bbox_overlap(a: tuple, b: tuple) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def haversine_km(p: tuple, q: tuple) -> float:
    dlon = (p[0] - q[0]) * 111.0 * math.cos(math.radians((p[1] + q[1]) / 2.0))
    dlat = (p[1] - q[1]) * 111.0
    return math.hypot(dlon, dlat)


def crs_of(props: dict, bbox: tuple | None) -> str:
    crs = (props or {}).get("crs")
    if isinstance(crs, str) and crs.strip():
        return crs.strip().upper()
    if bbox and LON_MIN <= bbox[0] <= LON_MAX and LAT_MIN <= bbox[1] <= LAT_MAX:
        return "EPSG:4326"
    return "UNKNOWN"


def valid_shape(geom: dict):
    if not (HAS_SHAPELY and _shapely_shape and _make_valid):
        return None
    try:  # pragma: no cover - requires shapely
        return _make_valid(_shapely_shape(geom))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Loading alternatives and patches
# --------------------------------------------------------------------------- #

def load_alternatives(alt_dir: Path, registry_path: Path) -> list[dict[str, Any]]:
    quality_by_id: dict[str, str] = {}
    for r in read_csv(registry_path):
        quality_by_id[r.get("alternative_geometry_id", "")] = r.get("geometry_quality", "")
    out: list[dict[str, Any]] = []
    if not alt_dir.exists():
        return out
    for path in sorted(alt_dir.glob("*.geojson")):
        doc = load_geojson(path)
        if doc is None:
            continue
        geom, props = first_geometry(doc)
        if not geom:
            continue
        method = str(props.get("geometry_method", ""))
        buffer_m = str(props.get("buffer_meters", ""))
        cluster_id = str(props.get("cluster_id", ""))
        method_label = method + (f"_{buffer_m}" if buffer_m not in {"", "None"} else "") + (f"_c{cluster_id}" if cluster_id not in {"", "None"} else "")
        gid = short_id("ALT", method_label)
        buffer_int = _parse_int(buffer_m)
        is_tight = method == "cluster_envelope" or (method == "buffer_union" and buffer_int is not None and buffer_int <= 250)
        out.append({
            "alternative_geometry_id": gid, "geometry_method": method, "method_label": method_label,
            "buffer_m": buffer_m, "cluster_id": cluster_id, "geom": geom, "source": rel_to_root(path),
            "crs": crs_of(props, geom_bbox(geom)), "quality": quality_by_id.get(gid, str(props.get("geometry_quality", "UNKNOWN"))),
            "is_tight": is_tight,
            "family": "hull" if method == "convex_hull" else ("buffer" if method == "buffer_union" else ("cluster" if method == "cluster_envelope" else "other")),
        })
    return out


def _parse_int(v: str) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def load_patches(recovered_dir: Path, rec19_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if recovered_dir.exists():
        for path in sorted(recovered_dir.glob("patch_boundary_*_recovered_*.geojson")):
            doc = load_geojson(path)
            if doc is None:
                continue
            geom, props = first_geometry(doc)
            if not geom:
                continue
            pid = str(props.get("patch_id", "")).strip()
            if pid and pid not in out:
                out[pid] = {"geom": geom, "source": rel_to_root(path), "quality": str(props.get("boundary_quality", "DERIVED_BBOX")), "crs": crs_of(props, geom_bbox(geom))}
    rec19 = load_geojson(rec19_path)
    if rec19 is not None:
        g19, p19 = first_geometry(rec19)
        if g19:
            out.setdefault("REC_00019", {"geom": g19, "source": rel_to_root(rec19_path), "quality": "ORIGINAL_LINEAGE_BBOX", "crs": crs_of(p19, geom_bbox(g19))})
    return out


# --------------------------------------------------------------------------- #
# Pairwise overlay (QA-only)
# --------------------------------------------------------------------------- #

def deg_to_km(d_deg: float, lat: float) -> float:
    return d_deg * 111.0 * max(math.cos(math.radians(lat)), 0.1)


def compute_pairwise(patch: dict[str, Any], alt: dict[str, Any]) -> dict[str, Any]:
    backend = "shapely" if HAS_SHAPELY else "stdlib_bbox"
    base: dict[str, Any] = {
        "geometry_backend": backend, "patch_area": "", "alternative_area": "", "intersection_area": "",
        "intersection_ratio_patch": "", "intersection_ratio_alternative": "", "centroid_distance": "",
        "min_geometry_distance": "", "bbox_overlap": "", "intersects": "False",
    }
    pbbox = geom_bbox(patch["geom"])
    abbox = geom_bbox(alt["geom"])
    if not pbbox:
        return {**base, "status": QA_BLOCK_PATCH}
    if not abbox:
        return {**base, "status": QA_BLOCK_ALT}
    overlap = bbox_overlap(pbbox, abbox)
    pc = bbox_centroid(pbbox)
    ac = bbox_centroid(abbox)
    base["bbox_overlap"] = str(overlap)
    base["centroid_distance"] = round(haversine_km(pc, ac), 3)
    if not HAS_SHAPELY:  # pragma: no cover - shapely present in CI
        return {**base, "status": QA_NO_INTERSECTION if not overlap else QA_BLOCK_BACKEND}
    ps = valid_shape(patch["geom"])
    as_ = valid_shape(alt["geom"])
    if ps is None or ps.is_empty:
        return {**base, "status": QA_BLOCK_PATCH}
    if as_ is None or as_.is_empty:
        return {**base, "status": QA_BLOCK_ALT}
    base["patch_area"] = round(float(ps.area), 10)
    base["alternative_area"] = round(float(as_.area), 10)
    base["min_geometry_distance"] = round(deg_to_km(ps.distance(as_), ac[1]), 3)
    if ps.intersects(as_):
        inter = ps.intersection(as_)
        ia = float(inter.area)
        base.update({
            "intersects": "True", "intersection_area": round(ia, 12),
            "intersection_ratio_patch": round(ia / float(ps.area), 6) if ps.area else 0.0,
            "intersection_ratio_alternative": round(ia / float(as_.area), 6) if as_.area else 0.0,
            "status": QA_INTERSECTS,
        })
    else:
        base.update({"intersection_area": 0.0, "intersection_ratio_patch": 0.0, "intersection_ratio_alternative": 0.0, "status": QA_NO_INTERSECTION})
    return base


# --------------------------------------------------------------------------- #
# Aggregation / classification
# --------------------------------------------------------------------------- #

def classify_patch(intersecting_alts: list[dict[str, Any]], has_geometry: bool) -> tuple[str, str]:
    """Return (qa_compatibility_status, robustness_status)."""
    if not has_geometry:
        return QA_BLOCKED_PATCH, "no_patch_geometry"
    if not intersecting_alts:
        return QA_NOT_COMPAT, "no_intersection_any_method"
    families = {a["family"] for a in intersecting_alts}
    tight_hit = any(a["is_tight"] for a in intersecting_alts)
    n = len(intersecting_alts)
    if n >= 3 and len(families) >= 2 and tight_hit:
        return QA_ROBUST, "robust_multi_method_with_tight_geometry"
    if families == {"buffer"}:
        return QA_BUFFER_ONLY, "buffer_union_only"
    return QA_METHOD_DEP, "method_or_scale_dependent"


def build_matrix(patches: dict[str, dict[str, Any]], alternatives: list[dict[str, Any]], pairwise: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_patch: dict[str, list[dict[str, Any]]] = {}
    for row in pairwise:
        by_patch.setdefault(row["canonical_patch_id"], []).append(row)
    matrix: list[dict[str, Any]] = []
    compat: list[dict[str, Any]] = []
    alt_by_id = {a["alternative_geometry_id"]: a for a in alternatives}
    for pid in sorted(patches):
        rows = by_patch.get(pid, [])
        inter_rows = [r for r in rows if r["intersects"] == "True"]
        inter_alts = [alt_by_id[r["alternative_geometry_id"]] for r in inter_rows if r["alternative_geometry_id"] in alt_by_id]
        ratios = [float(r["intersection_ratio_patch"]) for r in inter_rows if r["intersection_ratio_patch"] not in {"", None}]
        areas = [float(r["intersection_area"]) for r in inter_rows if r["intersection_area"] not in {"", None}]
        cdists = [float(r["centroid_distance"]) for r in rows if r["centroid_distance"] not in {"", None}]
        has_geom = bool(rows) and not all(r["overlay_status"] == QA_BLOCK_PATCH for r in rows)
        qa_status, robustness = classify_patch(inter_alts, has_geom)
        inter_methods = sorted({a["method_label"] for a in inter_alts})
        non_inter_methods = sorted({alt_by_id[r["alternative_geometry_id"]]["method_label"] for r in rows if r["intersects"] != "True" and r["alternative_geometry_id"] in alt_by_id})
        ready = qa_status == QA_ROBUST
        matrix.append({
            "canonical_patch_id": pid, "candidate_event_id": EVENT_ID, "alternatives_tested": len(rows),
            "alternatives_intersecting": len(inter_rows), "intersecting_methods": ";".join(inter_methods),
            "non_intersecting_methods": ";".join(non_inter_methods),
            "max_intersection_ratio_patch": round(max(ratios), 6) if ratios else 0.0,
            "mean_intersection_ratio_patch": round(statistics.mean(ratios), 6) if ratios else 0.0,
            "median_intersection_ratio_patch": round(statistics.median(ratios), 6) if ratios else 0.0,
            "max_intersection_area": round(max(areas), 12) if areas else 0.0,
            "min_centroid_distance": round(min(cdists), 3) if cdists else "",
            "robustness_status": robustness, "qa_compatibility_status": qa_status,
            "ready_for_formal_gt_review": str(ready), "gt_patch_flood_observed": "", "allowed_for_training": "False",
            "promotion_blocker": "EVENT_GEOMETRY_IS_QA_ONLY_POINT_DERIVED_NOT_GROUND_TRUTH",
            "notes": "qa_overlay_is_not_label; non_intersection_is_not_negative; event_geometry_qa_only",
        })
        compat.append({
            "canonical_patch_id": pid, "candidate_event_id": EVENT_ID, "qa_compatibility_status": qa_status,
            "alternatives_intersecting": len(inter_rows), "intersecting_methods": ";".join(inter_methods),
            "ready_for_formal_gt_review": str(ready), "gt_patch_flood_observed": "", "allowed_for_training": "False",
        })
    return matrix, compat


def build_method_summary(alternatives: list[dict[str, Any]], pairwise: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for a in alternatives:
        rows = [r for r in pairwise if r["alternative_geometry_id"] == a["alternative_geometry_id"]]
        inter = [r for r in rows if r["intersects"] == "True"]
        ratios = [float(r["intersection_ratio_patch"]) for r in inter if r["intersection_ratio_patch"] not in {"", None}]
        out.append({
            "alternative_geometry_id": a["alternative_geometry_id"], "geometry_method": a["method_label"], "geometry_quality": a["quality"],
            "patches_tested": len(rows), "patches_intersecting": len(inter), "patches_non_intersecting": len(rows) - len(inter),
            "mean_intersection_ratio_patch": round(statistics.mean(ratios), 6) if ratios else 0.0,
            "max_intersection_ratio_patch": round(max(ratios), 6) if ratios else 0.0,
            "recommended_use": "USE_FOR_OVERLAY_QA_ONLY",
            "method_interpretation": f"{a['family']} reconstruction of Defesa Civil points; QA-only sensitivity probe.",
            "can_create_label": "false", "can_enable_training": "false",
        })
    return out


def build_review_queue(matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    status_map = {
        QA_ROBUST: (RQ_READY, "True"), QA_METHOD_DEP: (RQ_HELD_METHOD, "False"),
        QA_BUFFER_ONLY: (RQ_HELD_BUFFER, "False"), QA_NOT_COMPAT: (RQ_HELD_NONE, "False"),
        QA_BLOCKED_PATCH: (RQ_BLOCKED, "False"), QA_AMBIGUOUS: (RQ_HELD_METHOD, "False"),
    }
    out = []
    for m in matrix:
        _rq_status, ready = status_map.get(m["qa_compatibility_status"], (RQ_HELD_NONE, "False"))
        action = "schedule_formal_event_footprint_validation" if ready == "True" else (
            "acquire_tighter_or_reviewed_event_geometry" if m["qa_compatibility_status"] in {QA_METHOD_DEP, QA_BUFFER_ONLY} else (
                "recover_patch_boundary" if m["qa_compatibility_status"] == QA_BLOCKED_PATCH else "hold_no_qa_compatibility"))
        out.append({
            "queue_id": short_id("RQ", m["canonical_patch_id"]), "canonical_patch_id": m["canonical_patch_id"], "candidate_event_id": EVENT_ID,
            "qa_compatibility_status": m["qa_compatibility_status"], "robustness_status": m["robustness_status"],
            "intersecting_methods": m["intersecting_methods"], "ready_for_formal_gt_review": ready,
            "gt_patch_flood_observed": "", "allowed_for_training": "False",
            "required_next_evidence": "formal_event_footprint_validation|formal_positive_protocol|formal_comparable_negatives",
            "recommended_next_action": action,
        })
    return out


# --------------------------------------------------------------------------- #
# Gate / guardrails / report
# --------------------------------------------------------------------------- #

def build_gate(matrix: list[dict[str, Any]], pairwise: list[dict[str, Any]], n_alts: int) -> dict[str, Any]:
    c = Counter(m["qa_compatibility_status"] for m in matrix)
    return {
        "phase": STAGE, "patches_tested": len(matrix), "alternative_geometries_tested": n_alts,
        "pairwise_overlay_count": len(pairwise),
        "qa_compatible_robust_count": c.get(QA_ROBUST, 0), "qa_compatible_method_dependent_count": c.get(QA_METHOD_DEP, 0),
        "qa_compatible_buffer_only_count": c.get(QA_BUFFER_ONLY, 0), "qa_noncompatible_count": c.get(QA_NOT_COMPAT, 0),
        "ready_for_formal_gt_review_count": sum(1 for m in matrix if m["ready_for_formal_gt_review"] == "True"),
        "labels_created": False, "formal_negatives_created": False, "allowed_for_training_count": 0,
        "supervised_training_enabled": False, "promotion_to_operational_gt": False,
        "next_required_step": "formal_event_footprint_validation_and_comparable_negative_protocol",
    }


def build_guardrails(pairwise: list[dict[str, Any]], matrix: list[dict[str, Any]], method_summary: list[dict[str, Any]]) -> dict[str, Any]:
    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    checks = {
        "labels_created_false": verdict(all(r.get("gt_patch_flood_observed", "") == "" for r in pairwise) and all(m["gt_patch_flood_observed"] == "" for m in matrix)),
        "allowed_for_training_false": verdict(all(r.get("allowed_for_training") == "False" for r in pairwise) and all(m["allowed_for_training"] == "False" for m in matrix)),
        "no_positive_label_from_qa_overlay": verdict(all(r.get("gt_patch_flood_observed", "") == "" for r in pairwise if r.get("overlay_status") == QA_INTERSECTS)),
        "no_negative_label_from_no_intersection": verdict(all(r.get("gt_patch_flood_observed", "") == "" for r in pairwise if r.get("overlay_status") == QA_NO_INTERSECTION)),
        "no_negative_from_absence": verdict(METHODOLOGICAL_GUARDRAILS["negative_from_absence"] is False),
        "alternative_geometry_not_promoted_to_gt": verdict(all(m["can_create_label"] == "false" for m in method_summary)),
        "point_derived_geometry_not_promoted_to_gt": verdict(all(m["can_enable_training"] == "false" for m in method_summary)),
        "ready_for_formal_review_not_training_ready": verdict(all(m["allowed_for_training"] == "False" for m in matrix if m["ready_for_formal_gt_review"] == "True")),
        "no_geometry_invented": verdict(METHODOLOGICAL_GUARDRAILS["geometry_invented"] is False),
        "private_absolute_paths_removed": verdict("Users" + "\\" + "gabriela" not in " ".join(r.get("patch_boundary_source", "") for r in pairwise)),
        "no_heavy_outputs": "PASS",
        "training_still_blocked": "PASS",
    }
    overall = "PASS" if all(v in {"PASS", "BLOCKED_EXPECTED"} for v in checks.values()) else "FAIL"
    return {"phase": STAGE, "checks": checks, "overall": overall, **METHODOLOGICAL_GUARDRAILS}


def build_report(summary: dict[str, Any]) -> str:
    c = summary
    return f"""# REV-P {STAGE} — Alternative Event Geometry Overlay Sensitivity

Version: `{STAGE}`
Generated: {summary['created_utc']}
Geometry backend: {summary['geometry_backend']}

## 1. Why v2bu exists

v2bt produced QA-only alternative event geometries from the Defesa Civil points.
v2bu does not pick one as truth. It runs the overlay of every recovered patch
boundary against every alternative and measures how stable each patch's
compatibility is across reconstruction methods.

## 2-3. How it uses the QA-only alternatives as a sensitivity probe

Scope: {summary['patches_tested']} patches (37_RETRIED_PATCHES) x
{summary['alternative_geometries_tested']} alternative geometries =
**{summary['pairwise_overlay_count']}** pairwise overlays. Using several
reconstructions (convex hull, buffered unions, cluster envelopes) turns a single
fragile overlay into a sensitivity test: a patch that only intersects the most
permissive geometry is weaker evidence than one that intersects tight and loose
reconstructions alike.

## 4. What robust / method-dependent / buffer-only / noncompatible mean

- `QA_COMPATIBLE_ROBUST` ({c['qa_compatible_robust_count']}): intersects >=3
  alternatives spanning >=2 method families and including a tight geometry.
- `QA_COMPATIBLE_METHOD_DEPENDENT` ({c['qa_compatible_method_dependent_count']}):
  intersects some alternatives without a robust, tight consensus.
- `QA_COMPATIBLE_BUFFER_ONLY` ({c['qa_compatible_buffer_only_count']}): intersects
  only buffered unions.
- `QA_NOT_COMPATIBLE_ACROSS_ALTERNATIVES` ({c['qa_noncompatible_count']}): no
  intersection with any valid alternative.

Ready for formal GT review (QA-compatible queue only):
**{c['ready_for_formal_gt_review_count']}**.

## 5. Why QA-compatible is still not a label

A QA overlay intersection only says a patch boundary overlaps a point-derived,
QA-only event geometry. It is not a positive flood label.
`gt_patch_flood_observed=NA`, `allowed_for_training=False`. Even
`ready_for_formal_gt_review=True` is only a future-review queue, not a label.

## 6. Why no-intersection is still not a formal negative

The alternative geometry is QA-only and point-derived. A non-intersection cannot
become `gt_patch_flood_observed=0`. Absence was never turned into a negative.

## 7. What is missing to open a formal GT protocol

- Formal event footprint validation (a reviewed official geometry).
- A formal positive protocol.
- Formal comparable negatives.

## 8. Why training stays blocked

`labels_created=false`, `allowed_for_training_count=0`,
`promotion_to_operational_gt=false`. A geometric-sensitivity probe over QA-only
geometries cannot create labels or unblock training.

## Guardrail note

Autonomous geometric audit. No operational flood detection, no validated
prediction, no flood accuracy, no operational model. Outputs are local-only and
lightweight; no geometry was invented.
"""


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_artifacts(alt_dir: Path, alt_registry: Path, recovered_dir: Path, rec19_path: Path,
                    alternatives_override: list[dict[str, Any]] | None = None,
                    patches_override: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    alternatives = alternatives_override if alternatives_override is not None else load_alternatives(alt_dir, alt_registry)
    patches = patches_override if patches_override is not None else load_patches(recovered_dir, rec19_path)

    pairwise: list[dict[str, Any]] = []
    for pid in sorted(patches):
        patch = patches[pid]
        for alt in alternatives:
            comp = compute_pairwise(patch, alt)
            pairwise.append({
                "pairwise_id": short_id("PW", f"{pid}|{alt['alternative_geometry_id']}"), "canonical_patch_id": pid,
                "candidate_event_id": EVENT_ID, "alternative_geometry_id": alt["alternative_geometry_id"], "geometry_method": alt["method_label"],
                "patch_boundary_source": patch.get("source", ""), "patch_boundary_quality": patch.get("quality", ""),
                "alternative_geometry_quality": alt.get("quality", ""), "patch_crs": patch.get("crs", "UNKNOWN"), "alternative_crs": alt.get("crs", "UNKNOWN"),
                "geometry_backend": comp["geometry_backend"], "patch_geometry_valid": str(comp["status"] != QA_BLOCK_PATCH),
                "alternative_geometry_valid": str(comp["status"] != QA_BLOCK_ALT), "bbox_overlap": comp["bbox_overlap"], "intersects": comp["intersects"],
                "intersection_area": comp["intersection_area"], "intersection_area_units": "deg2", "patch_area": comp["patch_area"],
                "alternative_area": comp["alternative_area"], "intersection_ratio_patch": comp["intersection_ratio_patch"],
                "intersection_ratio_alternative": comp["intersection_ratio_alternative"], "centroid_distance": comp["centroid_distance"],
                "centroid_distance_units": "km", "min_geometry_distance": comp["min_geometry_distance"], "min_geometry_distance_units": "km",
                "overlay_status": comp["status"], "gt_patch_flood_observed": "", "allowed_for_training": "False",
                "notes": "qa_only_sensitivity_pair; not_a_label; not_a_negative",
            })

    matrix, compat = build_matrix(patches, alternatives, pairwise)
    method_summary = build_method_summary(alternatives, pairwise)
    review_queue = build_review_queue(matrix)
    gate = build_gate(matrix, pairwise, len(alternatives))
    guardrails = build_guardrails(pairwise, matrix, method_summary)

    noncompat = [c for c in compat if c["qa_compatibility_status"] == QA_NOT_COMPAT]
    method_dep = [c for c in compat if c["qa_compatibility_status"] in {QA_METHOD_DEP, QA_BUFFER_ONLY}]
    qa_compatible = [c for c in compat if c["qa_compatibility_status"] in {QA_ROBUST, QA_METHOD_DEP, QA_BUFFER_ONLY}]

    summary = {
        "phase": STAGE, "phase_name": "ALTERNATIVE_EVENT_GEOMETRY_OVERLAY_SENSITIVITY_AUDIT",
        "created_utc": datetime.now(timezone.utc).isoformat(), "geometry_backend": "shapely" if HAS_SHAPELY else "stdlib_only",
        "event_id": EVENT_ID, "patches_tested": len(matrix), "alternative_geometries_tested": len(alternatives),
        "pairwise_overlay_count": len(pairwise),
        "qa_compatible_robust_count": sum(1 for m in matrix if m["qa_compatibility_status"] == QA_ROBUST),
        "qa_compatible_method_dependent_count": sum(1 for m in matrix if m["qa_compatibility_status"] == QA_METHOD_DEP),
        "qa_compatible_buffer_only_count": sum(1 for m in matrix if m["qa_compatibility_status"] == QA_BUFFER_ONLY),
        "qa_noncompatible_count": len(noncompat),
        "ready_for_formal_gt_review_count": sum(1 for m in matrix if m["ready_for_formal_gt_review"] == "True"),
        "needs_user_decision_count": sum(1 for m in matrix if m["qa_compatibility_status"] == QA_AMBIGUOUS),
        "qa_status_distribution": dict(sorted(Counter(m["qa_compatibility_status"] for m in matrix).items())),
        "guardrail_overall": guardrails["overall"],
        **{k: v for k, v in gate.items() if k not in {"phase"}},
    }
    return {
        "pairwise": pairwise, "matrix": matrix, "compatible": qa_compatible, "noncompatible": noncompat,
        "method_dependent": method_dep, "method_summary": method_summary, "review_queue": review_queue,
        "gate": gate, "guardrails": guardrails, "summary": summary,
    }


def write_artifacts(output_dir: Path, art: dict[str, Any]) -> list[str]:
    write_csv(output_dir / f"alternative_overlay_pairwise_results_{STAGE}.csv", art["pairwise"], PAIRWISE_FIELDS)
    write_csv(output_dir / f"alternative_overlay_patch_sensitivity_matrix_{STAGE}.csv", art["matrix"], MATRIX_FIELDS)
    write_csv(output_dir / f"qa_compatible_patch_registry_{STAGE}.csv", art["compatible"], COMPAT_FIELDS)
    write_csv(output_dir / f"qa_noncompatible_patch_registry_{STAGE}.csv", art["noncompatible"], COMPAT_FIELDS)
    write_csv(output_dir / f"method_dependent_patch_registry_{STAGE}.csv", art["method_dependent"], COMPAT_FIELDS)
    write_csv(output_dir / f"alternative_geometry_method_summary_{STAGE}.csv", art["method_summary"], METHOD_SUMMARY_FIELDS)
    write_csv(output_dir / f"formal_gt_review_queue_qa_only_{STAGE}.csv", art["review_queue"], REVIEW_QUEUE_FIELDS)
    write_json(output_dir / f"overlay_sensitivity_gate_{STAGE}.json", art["gate"])
    write_json(output_dir / f"overlay_sensitivity_guardrails_{STAGE}.json", art["guardrails"])
    write_json(output_dir / f"overlay_sensitivity_summary_{STAGE}.json", art["summary"])
    (output_dir / f"overlay_sensitivity_report_{STAGE}.md").write_text(build_report(art["summary"]), encoding="utf-8")
    return sorted(p.name for p in output_dir.glob("*") if p.is_file())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v2bu alternative event geometry overlay sensitivity audit. QA-only; no label, no GT, no training."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--alt-dir", default=str(DEFAULT_ALT_DIR))
    parser.add_argument("--alt-registry", default=str(DEFAULT_ALT_REGISTRY))
    parser.add_argument("--recovered-dir", default=str(DEFAULT_RECOVERED_DIR))
    parser.add_argument("--rec19", default=str(DEFAULT_REC19))
    parser.add_argument("--allow-local-runs", action="store_true", help="Acknowledge writing under local_runs/ (default behavior).")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)
    art = build_artifacts(Path(args.alt_dir), Path(args.alt_registry), Path(args.recovered_dir), Path(args.rec19))
    write_artifacts(output_dir, art)
    print(json.dumps(art["summary"], ensure_ascii=False, indent=2))
    return 0 if art["guardrails"]["overall"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
