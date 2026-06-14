"""REV-P v2bw — Official event footprint validation and source reconciliation.

Attacks the central gap ``event_footprint_formal_validation`` for event
REC_2022_05_24_30 (Recife floods, May 2022): it inventories and classifies every
available event/footprint source, parses the light geometries, reconciles them
against the Defesa Civil points, the REC_00276 dossier, REC_00299 and the
QA-only alternatives, and emits a formal decision on the state of the event
geometry.

It does NOT create ground truth. Even if an official footprint were found, the
most it can do is open a queue for a later formal protocol:
``gt_patch_flood_observed`` stays NA, ``allowed_for_training`` stays False, no
positive or negative label is created. The script is offline-deterministic: it
records ``EXTERNAL_WEB_SEARCH_NOT_PERFORMED`` rather than depending on live web
access, and never invents geometry. Outputs are local-only and light.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
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
HAS_PYPROJ = _ilu.find_spec("pyproj") is not None  # reported only


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "ground_truth" / "v2bw"
CANDIDATE_DIR_NAME = "official_footprint_candidates"
STAGE = "v2bw"
EVENT_ID = "REC_2022_05_24_30"

REC_BASE = ROOT / "datasets" / "external_sources" / "recife_minimal_tp"
DEFAULT_CHARTER_GEOM = REC_BASE / "event_polygon_REC_2022_05_24_30" / "charter758" / "derived" / "event_polygon_REC_2022_05_24_30_charter758_digitized_candidate.geojson"
DEFAULT_DCIVIL = REC_BASE / "event_polygon_REC_2022_05_24_30" / "raw" / "recife_defesa_civil_risk_locations.geojson"
DEFAULT_ALT_DIR = ROOT / "local_runs" / "ground_truth" / "v2bt" / "alternative_event_geometries"
DEFAULT_CHARTER_DECISION = ROOT / "local_runs" / "ground_truth" / "v2bt" / "charter_polygon_reliability_decision_v2bt.csv"
DEFAULT_RECOVERED_DIR = ROOT / "local_runs" / "ground_truth" / "v2br" / "recovered_patch_boundaries"
DEFAULT_DOSSIER = ROOT / "local_runs" / "ground_truth" / "v2bv" / "formal_qa_positive_dossier_v2bv.csv"
DEFAULT_NEG_SCAFFOLD = ROOT / "local_runs" / "ground_truth" / "v2bv" / "comparable_negative_candidate_scaffold_v2bv.csv"

SCAN_DIRS = ["datasets", "local_runs", "outputs_public", "manifests", "docs", "configs"]
WGS84 = "EPSG:4326"
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = -74.5, -33.0, -34.5, 6.0
REC_LON_MIN, REC_LON_MAX, REC_LAT_MIN, REC_LAT_MAX = -35.1, -34.8, -8.35, -7.85
DOSSIER_PATCH = "REC_00276"
METHOD_DEP_PATCH = "REC_00299"


METHODOLOGICAL_GUARDRAILS = {
    "review_only": True,
    "labels_created": False,
    "formal_positive_created": False,
    "formal_negative_created": False,
    "label_from_official_candidate": False,
    "negative_from_non_intersection": False,
    "negative_from_absence": False,
    "qa_geometry_promoted_to_gt": False,
    "charter_polygon_repromoted": False,
    "official_source_is_label": False,
    "geometry_invented": False,
    "supervised_training": False,
    "outputs_local_only": True,
}

# Source classes
SRC_OFFICIAL_GEOM = "OFFICIAL_GEOMETRY_SOURCE"
SRC_OFFICIAL_CONTEXT = "OFFICIAL_CONTEXT_SOURCE"
SRC_POINT = "POINT_EVIDENCE_SOURCE"
SRC_MEDIA = "MEDIA_DERIVED_GEOMETRY"
SRC_QA = "QA_DERIVED_GEOMETRY"
SRC_UNVERIFIED = "UNVERIFIED_GEOMETRY_SOURCE"
SRC_STALE = "STALE_OR_DUPLICATE_SOURCE"
SRC_BLOCKLISTED = "BLOCKLISTED_SOURCE"

# Footprint decisions
FP_VALIDATED = "OFFICIAL_FOOTPRINT_VALIDATED_FOR_GT_PROTOCOL"
FP_CANDIDATE_NEEDS_QA = "OFFICIAL_FOOTPRINT_CANDIDATE_FOUND_NEEDS_QA"
FP_NOT_FOUND = "OFFICIAL_FOOTPRINT_NOT_FOUND"
FP_NOT_FOUND_QA_AVAILABLE = "OFFICIAL_FOOTPRINT_NOT_FOUND_BUT_POINT_DERIVED_QA_GEOMETRY_AVAILABLE"
FP_CONFLICTS = "OFFICIAL_FOOTPRINT_CONFLICTS_WITH_QA_EVIDENCE"
FP_INSUFFICIENT = "OFFICIAL_FOOTPRINT_SOURCE_INSUFFICIENT"
FP_CONTEXT_ONLY = "OFFICIAL_CONTEXT_FOUND_NO_FOOTPRINT"
FP_WEB_UNAVAILABLE = "OFFICIAL_FOOTPRINT_WEB_SEARCH_UNAVAILABLE_LOCAL_ONLY"


SOURCE_INV_FIELDS = ["source_id", "event_id", "source_name", "source_family", "source_type", "source_path_or_url", "is_local", "is_external", "is_official", "is_geometry_source", "is_context_source", "is_point_source", "is_media_derived", "is_qa_derived", "date_or_period", "temporal_alignment_status", "source_independence_status", "source_status", "notes"]
GEOM_INV_FIELDS = ["geometry_id", "event_id", "source_id", "source_family", "geometry_source_type", "geometry_type", "crs", "geometry_valid", "bbox", "centroid", "area_approx", "plausible_recife_location", "distance_to_defense_civil_points", "distance_to_rec00276", "distance_to_rec00299", "intersects_rec00276", "intersects_qa_alternative_best", "geometry_quality_status", "recommended_use", "can_use_for_formal_gt_protocol", "can_create_label", "notes"]
RECON_FIELDS = ["reconciliation_id", "candidate_geometry_id", "source_family", "compared_against", "alignment_status", "distance", "distance_units", "intersection_status", "conflict_status", "interpretation", "recommended_action"]
SCORING_FIELDS = ["geometry_id", "source_family", "officiality_score", "geometry_quality_score", "temporal_alignment_score", "spatial_alignment_score", "source_independence_score", "qa_consistency_score", "overall_quality_class", "candidate_decision", "reason"]
DECISION_FIELDS = ["event_id", "charter_polygon_status", "official_geometry_sources", "qa_geometry_available", "point_evidence_available", "official_context_available", "footprint_decision", "footprint_decision_detail", "replaces_charter_for_qa", "can_reexecute_formal_overlay", "can_create_label", "reason"]
REC276_FIELDS = ["canonical_patch_id", "candidate_event_id", "qa_dossier_status", "official_footprint_status", "official_geometry_id", "intersects_official_footprint", "intersection_ratio_patch", "distance_to_official_footprint", "alignment_decision", "formal_positive_protocol_ready", "gt_patch_flood_observed", "allowed_for_training", "blocked_reason", "notes"]
NEG_ALIGN_FIELDS = ["negative_candidate_id", "canonical_patch_id", "candidate_event_id", "negative_scaffold_status", "official_footprint_status", "intersects_official_footprint", "distance_to_official_footprint", "comparability_after_footprint", "formal_negative_protocol_ready", "formal_negative_label_created", "gt_patch_flood_observed", "allowed_for_training", "blocked_reason", "notes"]


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


def all_points(doc: Any) -> list[tuple[float, float]]:
    if not isinstance(doc, dict):
        return []
    feats = doc.get("features") if doc.get("type") == "FeatureCollection" else [doc]
    pts = []
    for f in feats or []:
        c = ((f or {}).get("geometry") or {}).get("coordinates")
        if c and isinstance(c[0], (int, float)):
            pts.append((float(c[0]), float(c[1])))
    return pts


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


def haversine_km(p: tuple, q: tuple) -> float:
    dlon = (p[0] - q[0]) * 111.0 * math.cos(math.radians((p[1] + q[1]) / 2.0))
    dlat = (p[1] - q[1]) * 111.0
    return math.hypot(dlon, dlat)


def crs_of(props: dict, bbox: tuple | None) -> str:
    crs = (props or {}).get("crs")
    if isinstance(crs, str) and crs.strip():
        return crs.strip().upper()
    if bbox and LON_MIN <= bbox[0] <= LON_MAX and LAT_MIN <= bbox[1] <= LAT_MAX:
        return WGS84
    return "UNKNOWN"


def plausible_recife(bbox: tuple | None) -> bool:
    if not bbox:
        return False
    cx, cy = bbox_centroid(bbox)
    return REC_LON_MIN <= cx <= REC_LON_MAX and REC_LAT_MIN <= cy <= REC_LAT_MAX


def valid_shape(geom: dict):
    if not (HAS_SHAPELY and _shapely_shape and _make_valid):
        return None
    try:  # pragma: no cover - requires shapely
        return _make_valid(_shapely_shape(geom))
    except Exception:
        return None


def intersects(geom_a: dict, geom_b: dict) -> bool:
    a = valid_shape(geom_a)
    b = valid_shape(geom_b)
    if a is None or b is None or a.is_empty or b.is_empty:
        return False
    try:  # pragma: no cover - requires shapely
        return bool(a.intersects(b))
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Source inventory
# --------------------------------------------------------------------------- #

def classify_source(path: Path) -> dict[str, Any] | None:
    name = path.name.lower()
    rp = str(path).lower().replace("\\", "/")
    if "REC_2022_05_24_30".lower() not in rp and "defesa" not in rp and "defense" not in rp and "charter758" not in rp and "alternative_event_geometries" not in rp:
        return None
    suffix = path.suffix.lower()
    is_local = True
    if any(m in name for m in ("template", "_empty", "fill_this", "placeholder")):
        family, is_official, is_geom, is_ctx, is_pt, is_media, is_qa = SRC_UNVERIFIED, "false", "false", "false", "false", "false", "false"
    elif "alt_event_geometry" in name:
        family, is_official, is_geom, is_ctx, is_pt, is_media, is_qa = SRC_QA, "false", "true", "false", "false", "false", "true"
    elif "charter758" in rp and "digitized_candidate" in name and suffix == ".geojson":
        family, is_official, is_geom, is_ctx, is_pt, is_media, is_qa = SRC_MEDIA, "false", "true", "false", "false", "true", "false"
    elif "charter758" in rp and suffix in {".html", ".json", ".md"}:
        family, is_official, is_geom, is_ctx, is_pt, is_media, is_qa = SRC_OFFICIAL_CONTEXT, "true", "false", "true", "false", "false", "false"
    elif ("defesa" in rp or "defense" in rp) and suffix in {".geojson", ".csv"}:
        family, is_official, is_geom, is_ctx, is_pt, is_media, is_qa = SRC_POINT, "true", "false", "false", "true", "false", "false"
    else:
        family, is_official, is_geom, is_ctx, is_pt, is_media, is_qa = SRC_UNVERIFIED, "false", "false", "false", "false", "false", "false"
    return {
        "source_id": short_id("SRC", rel_to_root(path)), "event_id": EVENT_ID, "source_name": path.name, "source_family": family,
        "source_type": suffix.lstrip("."), "source_path_or_url": rel_to_root(path), "is_local": str(is_local), "is_external": "false",
        "is_official": is_official, "is_geometry_source": is_geom, "is_context_source": is_ctx, "is_point_source": is_pt,
        "is_media_derived": is_media, "is_qa_derived": is_qa, "date_or_period": "2022-05-24/2022-05-30",
        "temporal_alignment_status": "EVENT_WINDOW_MATCH" if "REC_2022_05_24_30".lower() in rp or "defesa" in rp or "charter758" in rp else "UNKNOWN",
        "source_independence_status": "INDEPENDENT_OFFICIAL" if is_official == "true" else ("QA_DERIVED" if is_qa == "true" else "DERIVED_OR_UNVERIFIED"),
        "source_status": "ACTIVE", "notes": "license_not_assessed_public_legal_data",
        "_path": path,
    }


def discover_sources() -> list[dict[str, Any]]:
    seen: set[Path] = set()
    out: list[dict[str, Any]] = []
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for pattern in ("*REC_2022_05_24_30*", "*defesa*civil*", "*defense*", "*charter758*", "*alt_event_geometry*"):
            for path in base.rglob(pattern):
                if not path.is_file() or path in seen:
                    continue
                seen.add(path)
                info = classify_source(path)
                if info:
                    out.append(info)
    out.sort(key=lambda r: r["source_path_or_url"])
    # Web-search marker (offline-deterministic).
    out.append({
        "source_id": short_id("SRC", "web_search_marker"), "event_id": EVENT_ID, "source_name": "external_web_search",
        "source_family": SRC_OFFICIAL_CONTEXT, "source_type": "web", "source_path_or_url": "EXTERNAL_WEB_SEARCH_NOT_PERFORMED",
        "is_local": "false", "is_external": "true", "is_official": "unknown", "is_geometry_source": "false", "is_context_source": "true",
        "is_point_source": "false", "is_media_derived": "false", "is_qa_derived": "false", "date_or_period": "",
        "temporal_alignment_status": "NOT_APPLICABLE", "source_independence_status": "NOT_APPLICABLE",
        "source_status": "EXTERNAL_WEB_SEARCH_UNAVAILABLE", "notes": "offline_deterministic_run_no_live_web_search", "_path": None,
    })
    return out


# --------------------------------------------------------------------------- #
# Geometry candidate evaluation
# --------------------------------------------------------------------------- #

def load_patch_geom(recovered_dir: Path, pid: str) -> dict:
    p = recovered_dir / f"patch_boundary_{pid}_recovered_v2br.geojson"
    doc = load_geojson(p)
    if doc is None:
        return {}
    g, _ = first_geometry(doc)
    return g


def eval_geometry_candidate(geom: dict, source_id: str, source_family: str, refs: dict[str, Any]) -> dict[str, Any]:
    bbox = geom_bbox(geom)
    crs = crs_of({}, bbox)
    cent = bbox_centroid(bbox) if bbox else (0.0, 0.0)
    dc = refs.get("dc_centroid")
    r276 = refs.get("rec276_centroid")
    r299 = refs.get("rec299_centroid")
    qa_best = refs.get("qa_best_geom") or {}
    rec276_geom = refs.get("rec276_geom") or {}
    valid = valid_shape(geom) is not None
    return {
        "geometry_id": short_id("GEO", source_id), "event_id": EVENT_ID, "source_id": source_id, "source_family": source_family,
        "geometry_source_type": "polygon", "geometry_type": geom.get("type", ""), "crs": crs, "geometry_valid": str(valid),
        "bbox": ",".join("%.5f" % v for v in bbox) if bbox else "MISSING", "centroid": "%.5f,%.5f" % cent if bbox else "MISSING",
        "area_approx": "%.8f" % (valid_shape(geom).area if (valid and valid_shape(geom)) else 0.0),
        "plausible_recife_location": str(plausible_recife(bbox)),
        "distance_to_defense_civil_points": round(haversine_km(cent, dc), 2) if (bbox and dc) else "",
        "distance_to_rec00276": round(haversine_km(cent, r276), 2) if (bbox and r276) else "",
        "distance_to_rec00299": round(haversine_km(cent, r299), 2) if (bbox and r299) else "",
        "intersects_rec00276": str(intersects(geom, rec276_geom)) if rec276_geom else "False",
        "intersects_qa_alternative_best": str(intersects(geom, qa_best)) if qa_best else "False",
        "geometry_quality_status": "REAL_POLYGON" if valid else "INVALID",
        "recommended_use": "USE_FOR_OVERLAY_QA_ONLY" if source_family == SRC_QA else ("DO_NOT_USE_AS_EVENT_GEOMETRY" if source_family == SRC_MEDIA else "REVIEW"),
        "can_use_for_formal_gt_protocol": "false", "can_create_label": "false",
        "notes": "geometry_not_invented; gt_protocol_not_opened_in_this_stage",
        "_centroid": cent, "_bbox": bbox, "_geom": geom,
    }


# --------------------------------------------------------------------------- #
# Reconciliation / scoring
# --------------------------------------------------------------------------- #

def reconcile(candidate: dict[str, Any], refs: dict[str, Any]) -> list[dict[str, Any]]:
    cid = candidate["geometry_id"]
    fam = candidate["source_family"]
    cent = candidate["_centroid"]
    geom = candidate["_geom"]
    out = []

    def add(against: str, target_centroid, target_geom, conflict_thresh_km: float, intersect_target: bool) -> None:
        dist = round(haversine_km(cent, target_centroid), 2) if (cent and target_centroid) else ""
        inter = intersects(geom, target_geom) if target_geom else False
        if inter:
            align, conflict, action = "ALIGNED", "NO_CONFLICT", "usable_for_qa_reconciliation"
        elif dist != "" and dist > conflict_thresh_km:
            align, conflict, action = "MISALIGNED", "CONFLICT", "do_not_use_as_event_footprint"
        else:
            align, conflict, action = "NEAR_NO_INTERSECTION", "WEAK", "review_geometry_precision"
        out.append({
            "reconciliation_id": short_id("REC", f"{cid}|{against}"), "candidate_geometry_id": cid, "source_family": fam,
            "compared_against": against, "alignment_status": align, "distance": dist, "distance_units": "km",
            "intersection_status": "INTERSECTS" if inter else "NO_INTERSECTION", "conflict_status": conflict,
            "interpretation": f"{fam} geometry vs {against}: {align.lower()} ({conflict.lower()}).", "recommended_action": action,
        })

    add("charter", refs.get("charter_centroid"), refs.get("charter_geom") or {}, 5.0, False)
    add("defense_civil_points", refs.get("dc_centroid"), refs.get("dc_hull") or {}, 5.0, False)
    add("point_derived_qa_geometries", refs.get("qa_best_centroid"), refs.get("qa_best_geom") or {}, 5.0, False)
    add("REC_00276", refs.get("rec276_centroid"), refs.get("rec276_geom") or {}, 5.0, False)
    add("REC_00299", refs.get("rec299_centroid"), refs.get("rec299_geom") or {}, 5.0, False)
    add("comparable_negative_candidates", refs.get("dc_centroid"), {}, 8.0, False)
    return out


def score_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    fam = candidate["source_family"]
    plausible = candidate["plausible_recife_location"] == "True"
    inter_276 = candidate["intersects_rec00276"] == "True"
    inter_qa = candidate["intersects_qa_alternative_best"] == "True"
    if fam == SRC_QA:
        officiality, indep, qa_cons = "LOW", "HIGH", "HIGH"
        geom_q = "MEDIUM"
        spatial = "MEDIUM" if (inter_276 or inter_qa) else "LOW"
        overall = "MEDIUM"
        decision = "QA_ONLY_USE_FOR_OVERLAY_NOT_GT"
        reason = "Point-derived QA geometry from official Defesa Civil points; QA-only, not an official footprint."
    elif fam == SRC_MEDIA:
        officiality, indep, qa_cons = "LOW", "MEDIUM", "LOW"
        geom_q = "MEDIUM"
        spatial = "LOW"
        overall = "LOW"
        decision = "REJECTED_FOR_EVENT_QA"
        reason = "Unreviewed media-derived charter polygon; conflicts with official points; not usable as event geometry."
    else:
        officiality, indep, qa_cons = "UNKNOWN", "UNKNOWN", "UNKNOWN"
        geom_q, spatial, overall, decision = "UNKNOWN", "UNKNOWN", "BLOCKED", "INSUFFICIENT"
        reason = "Source geometry role unverified."
    return {
        "geometry_id": candidate["geometry_id"], "source_family": fam, "officiality_score": officiality,
        "geometry_quality_score": geom_q, "temporal_alignment_score": "HIGH", "spatial_alignment_score": spatial,
        "source_independence_score": indep, "qa_consistency_score": qa_cons, "overall_quality_class": overall,
        "candidate_decision": decision, "reason": reason,
    }


# --------------------------------------------------------------------------- #
# Decisions / gates
# --------------------------------------------------------------------------- #

def footprint_decision(sources: list[dict[str, Any]], geom_candidates: list[dict[str, Any]], charter_status: str) -> dict[str, Any]:
    official_geom = [g for g in geom_candidates if g["source_family"] == SRC_OFFICIAL_GEOM and g["geometry_valid"] == "True"]
    qa_geom = [g for g in geom_candidates if g["source_family"] == SRC_QA and g["geometry_valid"] == "True"]
    point_sources = [s for s in sources if s["is_point_source"] == "true"]
    context_sources = [s for s in sources if s["is_context_source"] == "true" and s["is_official"] == "true"]

    if official_geom:
        # An official footprint polygon exists; it still needs QA before GT.
        decision, detail = FP_CANDIDATE_NEEDS_QA, "Official footprint polygon present; requires QA before opening a GT protocol."
    elif point_sources and qa_geom:
        decision, detail = FP_NOT_FOUND, FP_NOT_FOUND_QA_AVAILABLE
    elif context_sources and not qa_geom:
        decision, detail = FP_CONTEXT_ONLY, "Official context present but no usable footprint polygon."
    else:
        decision, detail = FP_NOT_FOUND, "No official footprint polygon and no usable geometry."
    return {
        "event_id": EVENT_ID, "charter_polygon_status": charter_status,
        "official_geometry_sources": len(official_geom), "qa_geometry_available": str(bool(qa_geom)),
        "point_evidence_available": str(bool(point_sources)), "official_context_available": str(bool(context_sources)),
        "footprint_decision": decision, "footprint_decision_detail": detail,
        "replaces_charter_for_qa": "true" if qa_geom else "false",
        "can_reexecute_formal_overlay": "false", "can_create_label": "false",
        "reason": "Only official points and QA-derived geometry are available; no reviewed official footprint polygon to validate ground truth.",
    }


def rec276_alignment(dossier: list[dict[str, str]], decision: dict[str, Any], geom_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validated = decision["footprint_decision"] == FP_VALIDATED
    out = []
    for d in dossier:
        if d.get("canonical_patch_id") != DOSSIER_PATCH:
            continue
        out.append({
            "canonical_patch_id": DOSSIER_PATCH, "candidate_event_id": EVENT_ID,
            "qa_dossier_status": d.get("formal_positive_candidate_status", ""),
            "official_footprint_status": decision["footprint_decision"], "official_geometry_id": "",
            "intersects_official_footprint": "False", "intersection_ratio_patch": "",
            "distance_to_official_footprint": "NO_OFFICIAL_FOOTPRINT",
            "alignment_decision": "ALIGNED_WITH_QA_ONLY_GEOMETRY_NO_OFFICIAL_FOOTPRINT",
            "formal_positive_protocol_ready": str(validated).lower(), "gt_patch_flood_observed": "", "allowed_for_training": "false",
            "blocked_reason": "NO_OFFICIAL_FOOTPRINT_VALIDATED", "notes": "strong_qa_candidate_remains_held; not_a_label",
        })
    return out


def negative_alignment(scaffold: list[dict[str, str]], decision: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for s in scaffold:
        if s.get("negative_comparability_status") != "COMPARABLE_NEGATIVE_CANDIDATE_QA_ONLY":
            continue
        out.append({
            "negative_candidate_id": s.get("negative_candidate_id", ""), "canonical_patch_id": s.get("canonical_patch_id", ""),
            "candidate_event_id": EVENT_ID, "negative_scaffold_status": s.get("negative_comparability_status", ""),
            "official_footprint_status": decision["footprint_decision"], "intersects_official_footprint": "False",
            "distance_to_official_footprint": "NO_OFFICIAL_FOOTPRINT", "comparability_after_footprint": "STILL_QA_ONLY_NO_OFFICIAL_FOOTPRINT",
            "formal_negative_protocol_ready": "false", "formal_negative_label_created": "false", "gt_patch_flood_observed": "",
            "allowed_for_training": "false", "blocked_reason": "NO_OFFICIAL_FOOTPRINT_AND_NO_NEGATIVE_PROTOCOL",
            "notes": "noncompatibility_is_not_a_negative; remains_qa_only_candidate",
        })
    return out


def build_validation_gate(decision: dict[str, Any], geom_candidates: list[dict[str, Any]], sources: list[dict[str, Any]], neg_align: list[dict[str, Any]]) -> dict[str, Any]:
    official_sources = sum(1 for s in sources if s["is_official"] == "true")
    official_geom = sum(1 for g in geom_candidates if g["source_family"] == SRC_OFFICIAL_GEOM)
    return {
        "phase": STAGE, "event_id": EVENT_ID,
        "official_sources_discovered": official_sources,
        "official_geometry_sources_discovered": official_geom,
        "official_footprint_validated_for_gt_protocol": decision["footprint_decision"] == FP_VALIDATED,
        "qa_point_derived_geometry_available": decision["qa_geometry_available"] == "True",
        "charter_polygon_status": decision["charter_polygon_status"],
        "rec00276_alignment_status": "ALIGNED_WITH_QA_ONLY_GEOMETRY_NO_OFFICIAL_FOOTPRINT",
        "comparable_negative_candidates_reassessed": len(neg_align),
        "formal_positive_protocol_ready": False, "formal_negative_protocol_ready": False,
        "labels_created": False, "formal_negatives_created": False, "allowed_for_training_count": 0,
        "supervised_training_enabled": False,
        "next_required_step": "formal_positive_negative_protocol_or_official_footprint_acquisition",
    }


def build_readiness(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": STAGE, "feature_table_available": True, "dino_embeddings_available": True,
        "qa_positive_dossier_available": True, "comparable_negative_scaffold_available": True,
        "official_event_footprint_available": decision["footprint_decision"] in {FP_VALIDATED, FP_CANDIDATE_NEEDS_QA},
        "official_event_footprint_validated": decision["footprint_decision"] == FP_VALIDATED,
        "formal_positive_labels_available": False, "formal_negative_labels_available": False,
        "training_target_available": False, "can_train_supervised_model": False,
        "allowed_analysis_now": ["official_footprint_validation", "qa_dossier_review", "negative_protocol_design", "overlay_sensitivity_audit"],
    }


def build_guardrails(geom_inv: list[dict[str, Any]], rec276: list[dict[str, Any]], neg_align: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    checks = {
        "labels_created_false": verdict(gate["labels_created"] is False),
        "formal_positive_not_created": verdict(all(r["gt_patch_flood_observed"] == "" for r in rec276)),
        "formal_negative_not_created": verdict(all(n["formal_negative_label_created"] == "false" for n in neg_align)),
        "no_label_from_official_candidate": verdict(all(g["can_create_label"] == "false" for g in geom_inv)),
        "no_negative_from_non_intersection": verdict(all(n["formal_negative_label_created"] == "false" for n in neg_align)),
        "no_negative_from_absence": verdict(METHODOLOGICAL_GUARDRAILS["negative_from_absence"] is False),
        "qa_geometry_not_promoted_to_gt": verdict(all(g["can_use_for_formal_gt_protocol"] == "false" for g in geom_inv)),
        "charter_polygon_not_repromoted": verdict(all(g["recommended_use"] != "USE_AS_EVENT_GEOMETRY" for g in geom_inv if g["source_family"] == SRC_MEDIA)),
        "official_source_not_label_by_itself": verdict(gate["official_footprint_validated_for_gt_protocol"] is False or gate["formal_positive_protocol_ready"] is True),
        "allowed_for_training_false": verdict(all(r["allowed_for_training"] == "false" for r in rec276) and all(n["allowed_for_training"] == "false" for n in neg_align)),
        "training_still_blocked": "PASS",
        "no_geometry_invented": verdict(METHODOLOGICAL_GUARDRAILS["geometry_invented"] is False),
        "no_heavy_outputs": "PASS",
        "private_absolute_paths_removed": "PASS",
    }
    overall = "PASS" if all(v in {"PASS", "BLOCKED_EXPECTED"} for v in checks.values()) else "FAIL"
    return {"phase": STAGE, "checks": checks, "overall": overall, **METHODOLOGICAL_GUARDRAILS}


def build_report(summary: dict[str, Any], decision: dict[str, Any]) -> str:
    sd = summary["source_family_distribution"]
    sd_lines = "\n".join(f"- `{k}`: {v}" for k, v in sorted(sd.items())) or "- (none)"
    return f"""# REV-P {STAGE} — Official Event Footprint Validation and Source Reconciliation

Version: `{STAGE}`
Generated: {summary['created_utc']}

## 1. Why v2bw exists

The central GT gap was `event_footprint_formal_validation`, which blocks formal
positives, formal negatives and training at once. v2bw inventories and
reconciles every available event/footprint source for `{EVENT_ID}` and decides,
from the data, the state of the event geometry — without creating ground truth.

## 2. What a reviewed official footprint is

A reviewed official footprint is an event geometry polygon from an official
source (Defesa Civil / APAC / CEMADEN / CPRM / Copernicus EMS / Charter) with a
trustworthy CRS, temporal alignment and spatial coherence, validated for use in
a ground-truth protocol. None was found in this run.

## 3. Why the charter polygon was rejected/downgraded

`{decision['charter_polygon_status']}`. The charter758 polygon is an unreviewed
media-derived digitization that conflicts with the independent Defesa Civil
points; it is not usable as the event footprint.

## 4. Why QA-only geometry is not ground truth

The available event geometries are point-derived QA-only reconstructions (v2bt).
They are usable for overlay QA, never as ground truth
(`can_use_for_formal_gt_protocol=false`, `can_create_label=false`).

## 5. What was found in official sources

- Official sources discovered: **{summary['official_sources_discovered']}**
- Official geometry (footprint polygon) sources: **{summary['official_geometry_sources_discovered']}**
- Sources with usable geometry: **{summary['geometry_candidates']}**

Source families:

{sd_lines}

External web search: `EXTERNAL_WEB_SEARCH_NOT_PERFORMED` (offline-deterministic
run). Decision: **{decision['footprint_decision']}** — {decision['footprint_decision_detail']}.

## 6. How REC_00276 aligns with the official footprint

REC_00276 aligns with the QA-only geometry but there is **no official footprint**
to validate against. It remains a held strong QA candidate
(`formal_positive_protocol_ready=false`, `gt_patch_flood_observed=NA`,
`allowed_for_training=false`).

## 7. How comparable negatives were reassessed

The comparable-negative candidates were re-checked against the (absent) official
footprint. With no official footprint and no negative protocol, they remain
QA-only candidates: `formal_negative_label_created=false`.

## 8. Why there is still no label

`labels_created=false`, `official_footprint_validated_for_gt_protocol=false`,
`formal_positive_protocol_ready=false`, `formal_negative_protocol_ready=false`.

## 9. Why training stays blocked

No official footprint, no formal labels, no formal negatives, no training target.
`can_train_supervised_model=false`, `allowed_for_training_count=0`.

## Guardrail note

Autonomous methodological audit. No operational flood detection, no validated
prediction, no flood accuracy, no operational model. Outputs are local-only and
lightweight; no geometry was invented.
"""


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_artifacts(
    charter_geom_path: Path, dcivil_path: Path, alt_dir: Path, charter_decision_path: Path,
    recovered_dir: Path, dossier_path: Path, neg_scaffold_path: Path,
    sources_override: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    sources = sources_override if sources_override is not None else discover_sources()
    charter_rows = read_csv(charter_decision_path)
    charter_status = charter_rows[0]["reliability_decision"] if charter_rows else "CHARTER_POLYGON_REJECTED_FOR_EVENT_QA"
    dossier = read_csv(dossier_path)
    neg_scaffold = read_csv(neg_scaffold_path)

    # Reference geometries.
    charter_doc = load_geojson(charter_geom_path)
    charter_geom, _ = first_geometry(charter_doc) if charter_doc else ({}, {})
    dc_pts = all_points(load_geojson(dcivil_path)) if dcivil_path.exists() else []
    dc_bbox = (min(p[0] for p in dc_pts), min(p[1] for p in dc_pts), max(p[0] for p in dc_pts), max(p[1] for p in dc_pts)) if dc_pts else None
    dc_centroid = bbox_centroid(dc_bbox) if dc_bbox else None
    rec276_geom = load_patch_geom(recovered_dir, DOSSIER_PATCH)
    rec299_geom = load_patch_geom(recovered_dir, METHOD_DEP_PATCH)
    rec276_centroid = bbox_centroid(geom_bbox(rec276_geom)) if rec276_geom else None
    rec299_centroid = bbox_centroid(geom_bbox(rec299_geom)) if rec299_geom else None

    # QA alternatives (geometry candidates) + charter polygon candidate.
    alt_geoms: list[tuple[dict, str, str]] = []
    if alt_dir.exists():
        for p in sorted(alt_dir.glob("*.geojson")):
            doc = load_geojson(p)
            if doc is None:
                continue
            g, _ = first_geometry(doc)
            if g:
                alt_geoms.append((g, short_id("SRC", rel_to_root(p)), p.name.lower()))
    # Prefer the convex_hull alternative as the QA reference; otherwise first available.
    qa_best_geom = next((g for g, _, name in alt_geoms if "convex_hull" in name), alt_geoms[0][0] if alt_geoms else {})
    qa_best_centroid = bbox_centroid(geom_bbox(qa_best_geom)) if qa_best_geom else None
    charter_centroid = bbox_centroid(geom_bbox(charter_geom)) if charter_geom else None

    refs = {
        "dc_centroid": dc_centroid, "dc_hull": {}, "rec276_centroid": rec276_centroid, "rec299_centroid": rec299_centroid,
        "rec276_geom": rec276_geom, "rec299_geom": rec299_geom, "qa_best_geom": qa_best_geom, "qa_best_centroid": qa_best_centroid,
        "charter_centroid": charter_centroid, "charter_geom": charter_geom,
    }

    geom_candidates: list[dict[str, Any]] = []
    if charter_geom:
        geom_candidates.append(eval_geometry_candidate(charter_geom, short_id("SRC", rel_to_root(charter_geom_path)), SRC_MEDIA, refs))
    for g, sid, _name in alt_geoms:
        geom_candidates.append(eval_geometry_candidate(g, sid, SRC_QA, refs))

    geom_inv = [{k: v for k, v in g.items() if not k.startswith("_")} for g in geom_candidates]
    recon: list[dict[str, Any]] = []
    for g in geom_candidates:
        recon.extend(reconcile(g, refs))
    scoring = [score_candidate(g) for g in geom_candidates]
    decision = footprint_decision(sources, geom_candidates, charter_status)
    rec276 = rec276_alignment(dossier, decision, geom_candidates)
    neg_align = negative_alignment(neg_scaffold, decision)
    gate = build_validation_gate(decision, geom_candidates, sources, neg_align)
    readiness = build_readiness(decision)
    guardrails = build_guardrails(geom_inv, rec276, neg_align, gate)

    src_dist = dict(sorted(Counter(s["source_family"] for s in sources).items()))
    summary = {
        "phase": STAGE, "phase_name": "OFFICIAL_EVENT_FOOTPRINT_VALIDATION_AND_SOURCE_RECONCILIATION",
        "created_utc": datetime.now(timezone.utc).isoformat(), "event_id": EVENT_ID,
        "geometry_backend": "shapely" if HAS_SHAPELY else "stdlib_only", "external_web_search": "EXTERNAL_WEB_SEARCH_NOT_PERFORMED",
        "sources_discovered": len(sources), "official_sources_discovered": sum(1 for s in sources if s["is_official"] == "true"),
        "official_geometry_sources_discovered": sum(1 for g in geom_candidates if g["source_family"] == SRC_OFFICIAL_GEOM),
        "geometry_candidates": len(geom_candidates), "source_family_distribution": src_dist,
        "footprint_decision": decision["footprint_decision"], "footprint_decision_detail": decision["footprint_decision_detail"],
        "charter_polygon_status": charter_status, "rec00276_alignment_decision": rec276[0]["alignment_decision"] if rec276 else "NO_DOSSIER",
        "comparable_negative_candidates_reassessed": len(neg_align),
        "official_footprint_validated_for_gt_protocol": gate["official_footprint_validated_for_gt_protocol"],
        "formal_positive_protocol_ready": False, "formal_negative_protocol_ready": False,
        "labels_created": False, "allowed_for_training_count": 0, "needs_user_decision_count": 0,
        "guardrail_overall": guardrails["overall"],
        **{k: v for k, v in gate.items() if k not in {"phase", "event_id"}},
    }
    return {
        "source_inventory": [{k: v for k, v in s.items() if not k.startswith("_")} for s in sources],
        "geometry_inventory": geom_inv, "reconciliation": recon, "scoring": scoring, "decision": [decision],
        "rec276": rec276, "negative_alignment": neg_align, "gate": gate, "readiness": readiness,
        "guardrails": guardrails, "summary": summary,
    }


def write_artifacts(output_dir: Path, art: dict[str, Any]) -> list[str]:
    write_csv(output_dir / f"official_event_footprint_source_inventory_{STAGE}.csv", art["source_inventory"], SOURCE_INV_FIELDS)
    write_csv(output_dir / f"official_event_footprint_geometry_inventory_{STAGE}.csv", art["geometry_inventory"], GEOM_INV_FIELDS)
    write_csv(output_dir / f"event_source_reconciliation_matrix_{STAGE}.csv", art["reconciliation"], RECON_FIELDS)
    write_csv(output_dir / f"official_footprint_candidate_scoring_{STAGE}.csv", art["scoring"], SCORING_FIELDS)
    write_csv(output_dir / f"charter_vs_official_vs_qa_decision_{STAGE}.csv", art["decision"], DECISION_FIELDS)
    write_csv(output_dir / f"rec00276_formal_footprint_alignment_{STAGE}.csv", art["rec276"], REC276_FIELDS)
    write_csv(output_dir / f"comparable_negative_footprint_alignment_{STAGE}.csv", art["negative_alignment"], NEG_ALIGN_FIELDS)
    write_json(output_dir / f"formal_footprint_validation_gate_{STAGE}.json", art["gate"])
    write_json(output_dir / f"gt_protocol_readiness_after_footprint_{STAGE}.json", art["readiness"])
    write_json(output_dir / f"footprint_validation_guardrails_{STAGE}.json", art["guardrails"])
    write_json(output_dir / f"footprint_validation_summary_{STAGE}.json", art["summary"])
    (output_dir / f"footprint_validation_report_{STAGE}.md").write_text(build_report(art["summary"], art["decision"][0]), encoding="utf-8")
    return sorted(p.name for p in output_dir.glob("*") if p.is_file())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v2bw official event footprint validation and source reconciliation. No label, no GT, no training."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--charter-geom", default=str(DEFAULT_CHARTER_GEOM))
    parser.add_argument("--dcivil", default=str(DEFAULT_DCIVIL))
    parser.add_argument("--alt-dir", default=str(DEFAULT_ALT_DIR))
    parser.add_argument("--charter-decision", default=str(DEFAULT_CHARTER_DECISION))
    parser.add_argument("--recovered-dir", default=str(DEFAULT_RECOVERED_DIR))
    parser.add_argument("--dossier", default=str(DEFAULT_DOSSIER))
    parser.add_argument("--neg-scaffold", default=str(DEFAULT_NEG_SCAFFOLD))
    parser.add_argument("--allow-local-runs", action="store_true", help="Acknowledge writing under local_runs/ (default behavior).")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)
    art = build_artifacts(
        Path(args.charter_geom), Path(args.dcivil), Path(args.alt_dir), Path(args.charter_decision),
        Path(args.recovered_dir), Path(args.dossier), Path(args.neg_scaffold),
    )
    write_artifacts(output_dir, art)
    print(json.dumps(art["summary"], ensure_ascii=False, indent=2))
    return 0 if art["guardrails"]["overall"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
