"""SUSC-17C5 Geometry-to-Patch Linkage Resolver.

Consumes the SUSC-17C strong-reference acquisition pack and resolves candidate
geometries into review-only event/footprint to patch links. It is deliberately
fail-closed: text locations, missing CRS, risk context, and alerts never become
strong links, and no output is ground truth, trainable, or score-v7 enabling.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

DATASETS = ROOT / "datasets"
DAT_SUSC = DATASETS / "suscetibilidade"
OUT_SUSC = ROOT / "outputs_public" / "suscetibilidade"
OUT_TABLES = ROOT / "outputs_public" / "tables"
OUT_PREV = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_17c_strong_reference_acquisition_canary"
OUT_DATA = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_17c5_geometry_to_patch_linkage_resolver"
OUT_REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

TARGET_PACK = OUT_PREV / "susc_17c_source_target_pack.csv"
PREV_SUMMARY = OUT_PREV / "susc_17c_strong_reference_acquisition_summary.json"
PREV_QA = OUT_PREV / "reference_canary_qa_queue.csv"
SAR_METADATA_17C31 = OUT_SUSC / "susc_17c31_sar_metadata_feasibility.csv"
CHARTER_GEOMETRIES_17C4 = OUT_SUSC / "susc_17c4_candidate_geometries.geojson"
CHARTER_REFS_17C4 = OUT_SUSC / "susc_17c4_extracted_reference_candidates.csv"
CANDIDATE_PATCH_GRID_17C6 = OUT_SUSC / "susc_17c6_candidate_patch_grid.csv"
CANDIDATE_PATCH_GRID_GEOJSON_17C6 = OUT_SUSC / "susc_17c6_candidate_patch_grid.geojson"
PATCH_FEATURES = DAT_SUSC / "susc_features_by_patch_v1.csv"
SCORE_V6 = DAT_SUSC / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT_SUSC / "susc_score_v7_candidate_by_patch_v1.csv"
SPATIAL_BINDING = OUT_TABLES / "revp_patch_event_spatial_binding_v2ec.csv"
EVENT_PATCH_LINKAGE = DATASETS / "event_patch_linkage_registry.csv"

PREFLIGHT = OUT_DATA / "susc_17c5_preflight_registry_inventory.csv"
GEOMETRIES = OUT_DATA / "susc_17c5_normalized_geometry.csv"
PATCH_LINKS = OUT_DATA / "susc_17c5_patch_links.csv"
BLOCKERS = OUT_DATA / "susc_17c5_patch_link_blockers.csv"
QA_QUEUE = OUT_DATA / "reference_patch_link_qa_queue.csv"
LINK_GEOJSON = OUT_DATA / "susc_17c5_candidate_patch_links.geojson"
SUMMARY = OUT_DATA / "susc_17c5_geometry_to_patch_linkage_summary.json"
REPORT = OUT_REPORTS / "SUSC_17C5_GEOMETRY_TO_PATCH_LINKAGE_RESOLVER_REPORT.md"

SCHEMA_GEOMETRY = SCHEMAS / "susc_17c5_geometry_resolver_schema_v1.json"
SCHEMA_LINK = SCHEMAS / "susc_17c5_patch_link_schema_v1.json"
SCHEMA_QA = SCHEMAS / "susc_17c5_patch_link_qa_schema_v1.json"

REQUIRED_INPUTS = [
    TARGET_PACK,
    PREV_SUMMARY,
    SAR_METADATA_17C31,
    CHARTER_GEOMETRIES_17C4,
    CHARTER_REFS_17C4,
    CANDIDATE_PATCH_GRID_17C6,
    CANDIDATE_PATCH_GRID_GEOJSON_17C6,
    PATCH_FEATURES,
    SCORE_V6,
    SPATIAL_BINDING,
    EVENT_PATCH_LINKAGE,
]

REQUIRED_OUTPUTS = [
    PREFLIGHT,
    GEOMETRIES,
    PATCH_LINKS,
    BLOCKERS,
    QA_QUEUE,
    LINK_GEOJSON,
    SUMMARY,
    REPORT,
    SCHEMA_GEOMETRY,
    SCHEMA_LINK,
    SCHEMA_QA,
]

NO_GT_REASON = "review-only geometry-to-patch candidate; not ground truth, not trainable, no score v7"
FORBIDDEN_TRUE_FIELDS = {"ground_truth", "eligible_for_training", "score_v7_allowed"}
STRONG_LINK_CLASSES = {"exact_polygon_overlap", "point_within_patch", "bbox_overlap"}
QA_LINK_CLASSES = STRONG_LINK_CLASSES | {"near_patch_buffer_10m", "near_patch_buffer_30m", "near_patch_buffer_50m"}
TEXT_ONLY_TYPES = {"official_address_resolved", "administrative_disaster_record", "documentary_context", "insufficient"}
BLOCKED_SOURCE_TYPES = {"alert_only", "risk_area_not_event"}

PREFLIGHT_FIELDS = ["artifact_role", "path", "exists", "row_count", "columns_found", "notes"]
GEOMETRY_FIELDS = [
    "candidate_event_id",
    "event_id",
    "region",
    "city",
    "source_id",
    "source_type",
    "geometry_id",
    "geometry_type",
    "geometry_source",
    "geometry_format",
    "crs",
    "has_geometry_object",
    "uncertainty_m",
    "geometry_resolution_status",
    "geometry_blocking_reason",
    "risk_context_only",
    "review_only",
    "eligible_for_training",
    "ground_truth",
    "score_v7_allowed",
]
PATCH_LINK_FIELDS = [
    "patch_link_id",
    "candidate_event_id",
    "event_id",
    "source_id",
    "source_type",
    "geometry_id",
    "patch_id",
    "patch_namespace",
    "official_patch",
    "region",
    "city",
    "link_class",
    "link_strength",
    "strong_link_candidate",
    "overlap_area_m2",
    "overlap_ratio",
    "distance_m",
    "uncertainty_m",
    "spatial_eligible",
    "eligible_for_evaluation",
    "eligible_for_calibration",
    "eligible_for_training",
    "ground_truth",
    "review_only",
    "score_v7_allowed",
    "not_ground_truth_reason",
    "spatial_blocking_reason",
]
BLOCKER_FIELDS = [
    "blocker_id",
    "candidate_event_id",
    "event_id",
    "source_id",
    "source_type",
    "geometry_id",
    "blocker_class",
    "blocking_reason",
    "next_action",
    "review_only",
    "ground_truth",
    "eligible_for_training",
    "score_v7_allowed",
]
QA_FIELDS = [
    "qa_item_id",
    "candidate_event_id",
    "patch_id",
    "geometry_id",
    "link_class",
    "qa_priority",
    "qa_status",
    "qa_question",
    "map_artifact_ref",
    "evidence_summary",
    "blocking_reason",
    "review_only",
    "ground_truth",
    "eligible_for_training",
    "score_v7_allowed",
]


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _gov() -> dict:
    return {
        "review_only": "true",
        "eligible_for_training": "false",
        "ground_truth": "false",
        "score_v7_allowed": "false",
    }


def _run_git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _require_inputs() -> None:
    missing = [path for path in REQUIRED_INPUTS if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(path) for path in missing))
    if SCORE_V7.exists():
        raise AssertionError("score_v7 exists and is forbidden for SUSC-17C5")


def _read(path: Path) -> list[dict]:
    return read_csv(path) if path.exists() else []


def _columns(path: Path) -> str:
    rows = _read(path) if path.suffix.lower() == ".csv" else []
    if rows:
        return ";".join(rows[0].keys())
    if path.exists() and path.suffix.lower() == ".csv":
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].replace(",", ";")
    if path.exists() and path.suffix.lower() in {".json", ".geojson"}:
        return "json_or_geojson"
    return ""


def _row_count(path: Path) -> str:
    if not path.exists():
        return "0"
    if path.suffix.lower() == ".csv":
        return str(len(_read(path)))
    if path.suffix.lower() == ".geojson":
        obj = read_json(path)
        if obj.get("type") == "FeatureCollection":
            return str(len(obj.get("features", [])))
        if obj.get("type") == "Feature":
            return "1"
    return "not_csv"


def preflight_rows() -> list[dict]:
    tracked = [
        ("17c_target_pack", TARGET_PACK, "candidatas 17C consumidas pelo resolver"),
        ("17c_summary", PREV_SUMMARY, "estado herdado da sprint 17C"),
        ("17c_qa_queue", PREV_QA, "fila canaria herdada"),
        ("charter_17c4_geometry", CHARTER_GEOMETRIES_17C4, "geometria real Charter 758"),
        ("charter_17c4_refs", CHARTER_REFS_17C4, "metadados da referencia Charter"),
        ("candidate_patch_grid_17c6", CANDIDATE_PATCH_GRID_17C6, "patches canario review-only"),
        ("candidate_patch_geojson_17c6", CANDIDATE_PATCH_GRID_GEOJSON_17C6, "geometrias dos patches canario"),
        ("official_patch_features", PATCH_FEATURES, "patch grid oficial usado pelo score v6"),
        ("spatial_binding_v2ec", SPATIAL_BINDING, "estado anterior de intersecao/CRS"),
        ("event_patch_linkage_registry", EVENT_PATCH_LINKAGE, "linkages herdados, sem promover label"),
        ("score_v6_official", SCORE_V6, "somente leitura"),
        ("score_v7_forbidden", SCORE_V7, "deve permanecer ausente"),
    ]
    rows = []
    for role, path, notes in tracked:
        rows.append({
            "artifact_role": role,
            "path": rel(path),
            "exists": _bool(path.exists()),
            "row_count": _row_count(path),
            "columns_found": _columns(path),
            "notes": notes,
        })
    rows.append({
        "artifact_role": "git_initial_state",
        "path": ".",
        "exists": "true",
        "row_count": "not_applicable",
        "columns_found": "branch;head;staged_count;dirty_status_lines",
        "notes": f"branch={_run_git(['branch', '--show-current'])}; head={_run_git(['rev-parse', '--short', 'HEAD'])}; staged={len(_run_git(['diff', '--cached', '--name-only']).splitlines())}; dirty={len(_run_git(['status', '--short']).splitlines())}",
    })
    return rows


def _canonical_event_id(event_id: str) -> str:
    event_id = (event_id or "").strip()
    if event_id == "REC_2022_05":
        return "REC_2022_05_24_30"
    return event_id


def _region_code(region: str) -> str:
    text = (region or "").strip()
    if text in {"REC", "Recife", "recife"}:
        return "REC"
    if text in {"PET", "Petropolis", "Petrópolis", "petropolis"}:
        return "PET"
    if text in {"CUR", "CTB", "Curitiba", "curitiba"}:
        return "CUR"
    return text or "UNKNOWN"


def _region_key(region: str) -> str:
    code = _region_code(region)
    return {"REC": "recife", "PET": "petropolis", "CUR": "curitiba"}.get(code, code.lower())


def _city(region: str, fallback: str = "") -> str:
    return {"REC": "Recife", "PET": "Petropolis", "CUR": "Curitiba"}.get(_region_code(region), fallback or "unknown")


def _iter_coords(obj):
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (int, float)) and len(obj) >= 2:
            yield (float(obj[0]), float(obj[1]))
        else:
            for item in obj:
                yield from _iter_coords(item)


def bbox_of_geometry(geometry: dict | None) -> tuple[float, float, float, float] | None:
    if not geometry:
        return None
    pts = list(_iter_coords(geometry.get("coordinates", [])))
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_area_m2(bbox: tuple[float, float, float, float] | list[float]) -> float:
    x0, y0, x1, y1 = bbox
    lat = (y0 + y1) / 2
    width = max(0.0, (x1 - x0) * 111320.0 * math.cos(math.radians(lat)))
    height = max(0.0, (y1 - y0) * 110540.0)
    return width * height


def bbox_intersection(a: tuple[float, float, float, float] | list[float], b: tuple[float, float, float, float] | list[float]) -> tuple[float, float, float, float] | None:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def bbox_distance_m(a: tuple[float, float, float, float] | list[float], b: tuple[float, float, float, float] | list[float]) -> float:
    inter = bbox_intersection(a, b)
    if inter:
        return 0.0
    dx = max(0.0, a[0] - b[2], b[0] - a[2])
    dy = max(0.0, a[1] - b[3], b[1] - a[3])
    lat = (a[1] + a[3] + b[1] + b[3]) / 4
    mx = dx * 111320.0 * math.cos(math.radians(lat))
    my = dy * 110540.0
    return math.hypot(mx, my)


def point_in_bbox(lon: float, lat: float, bbox: tuple[float, float, float, float] | list[float]) -> bool:
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


def _polygon_rings(geometry: dict) -> list[list[tuple[float, float]]]:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    rings: list[list[tuple[float, float]]] = []
    if gtype == "Polygon":
        for ring in coords[:1]:
            rings.append([(float(x), float(y)) for x, y, *_ in ring])
    elif gtype == "MultiPolygon":
        for polygon in coords:
            for ring in polygon[:1]:
                rings.append([(float(x), float(y)) for x, y, *_ in ring])
    return rings


def _clip_polygon_to_bbox(ring: list[tuple[float, float]], bbox: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    def clip(points, inside, intersect):
        if not points:
            return []
        out = []
        prev = points[-1]
        prev_inside = inside(prev)
        for cur in points:
            cur_inside = inside(cur)
            if cur_inside:
                if not prev_inside:
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif prev_inside:
                out.append(intersect(prev, cur))
            prev, prev_inside = cur, cur_inside
        return out

    xmin, ymin, xmax, ymax = bbox
    points = ring[:]
    points = clip(points, lambda p: p[0] >= xmin, lambda a, b: (xmin, a[1] + (b[1] - a[1]) * ((xmin - a[0]) / (b[0] - a[0])) if b[0] != a[0] else a[1]))
    points = clip(points, lambda p: p[0] <= xmax, lambda a, b: (xmax, a[1] + (b[1] - a[1]) * ((xmax - a[0]) / (b[0] - a[0])) if b[0] != a[0] else a[1]))
    points = clip(points, lambda p: p[1] >= ymin, lambda a, b: (a[0] + (b[0] - a[0]) * ((ymin - a[1]) / (b[1] - a[1])) if b[1] != a[1] else a[0], ymin))
    points = clip(points, lambda p: p[1] <= ymax, lambda a, b: (a[0] + (b[0] - a[0]) * ((ymax - a[1]) / (b[1] - a[1])) if b[1] != a[1] else a[0], ymax))
    return points


def _shoelace_area_m2(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    lat = sum(p[1] for p in points) / len(points)
    scaled = [(x * 111320.0 * math.cos(math.radians(lat)), y * 110540.0) for x, y in points]
    area = 0.0
    for (x0, y0), (x1, y1) in zip(scaled, scaled[1:] + scaled[:1]):
        area += x0 * y1 - x1 * y0
    return abs(area) / 2


def polygon_overlap_area_m2(geometry: dict, patch_bbox: tuple[float, float, float, float]) -> float:
    total = 0.0
    for ring in _polygon_rings(geometry):
        clipped = _clip_polygon_to_bbox(ring, patch_bbox)
        total += _shoelace_area_m2(clipped)
    return total


def _bbox_polygon(bbox: tuple[float, float, float, float] | list[float]) -> dict:
    x0, y0, x1, y1 = bbox
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


def _parse_bbox(text: str) -> tuple[float, float, float, float] | None:
    if not text:
        return None
    sep = ";" if ";" in text else ","
    try:
        vals = [float(part.strip()) for part in str(text).split(sep)]
    except ValueError:
        return None
    if len(vals) != 4 or vals[2] <= vals[0] or vals[3] <= vals[1]:
        return None
    return vals[0], vals[1], vals[2], vals[3]


def _point_from_row(row: dict) -> tuple[float, float] | None:
    for lon_key, lat_key in [("lon", "lat"), ("longitude", "latitude"), ("resolved_lon", "resolved_lat")]:
        if row.get(lon_key) and row.get(lat_key):
            try:
                return float(row[lon_key]), float(row[lat_key])
            except ValueError:
                return None
    return None


def _load_charter_geometry() -> tuple[dict | None, dict]:
    obj = read_json(CHARTER_GEOMETRIES_17C4)
    features = obj.get("features", []) if obj.get("type") == "FeatureCollection" else [obj]
    for feature in features:
        props = feature.get("properties", {})
        if _canonical_event_id(props.get("event_id", "")) == "REC_2022_05_24_30":
            return feature.get("geometry"), props
    return None, {}


def _sar_bbox_by_event() -> dict[str, tuple[float, float, float, float]]:
    out = {}
    for row in _read(SAR_METADATA_17C31):
        bbox = _parse_bbox(row.get("aoi_bbox", ""))
        if bbox:
            out[_canonical_event_id(row.get("event_id", ""))] = bbox
    return out


def _target_rows() -> list[dict]:
    return _read(TARGET_PACK)


def normalized_geometry_rows() -> list[dict]:
    charter_geom, charter_props = _load_charter_geometry()
    sar_bbox = _sar_bbox_by_event()
    rows = []
    for idx, target in enumerate(_target_rows(), start=1):
        event_id = _canonical_event_id(target["event_id"])
        source_type = target["source_type"]
        geometry_id = f"S17C5_GEOM_{idx:04d}"
        base = {
            "candidate_event_id": target["candidate_event_id"],
            "event_id": event_id,
            "region": _region_code(target.get("region", "")),
            "city": target.get("city") or _city(target.get("region", "")),
            "source_id": target["source_id"],
            "source_type": source_type,
            "geometry_id": geometry_id,
            **_gov(),
        }
        if source_type == "risk_area_not_event":
            rows.append({
                **base,
                "geometry_type": "unresolved_geometry",
                "geometry_source": target.get("artifact_ref", "not_available"),
                "geometry_format": "risk_context_text",
                "crs": "not_available",
                "has_geometry_object": "false",
                "uncertainty_m": "not_available",
                "geometry_resolution_status": "risk_context_only",
                "geometry_blocking_reason": "risk_area_not_event_never_becomes_event_geometry",
                "risk_context_only": "true",
            })
            continue
        if source_type == "alert_only":
            rows.append({
                **base,
                "geometry_type": "unresolved_geometry",
                "geometry_source": target.get("artifact_ref", "not_available"),
                "geometry_format": "alert_text",
                "crs": "not_available",
                "has_geometry_object": "false",
                "uncertainty_m": "not_available",
                "geometry_resolution_status": "blocked_alert_only",
                "geometry_blocking_reason": "alert_only_never_becomes_observed_event_geometry",
                "risk_context_only": "false",
            })
            continue
        if event_id == "REC_2022_05_24_30" and charter_geom and source_type in {"technical_remote_sensing_flood_footprint", "official_observed_event_polygon"}:
            rows.append({
                **base,
                "geometry_type": "technical_footprint" if source_type == "technical_remote_sensing_flood_footprint" else "polygon",
                "geometry_source": rel(CHARTER_GEOMETRIES_17C4),
                "geometry_format": f"geojson_{charter_geom.get('type', 'geometry')}",
                "crs": charter_props.get("crs", "EPSG:4326") or "EPSG:4326",
                "has_geometry_object": "true",
                "uncertainty_m": "requires_human_review",
                "geometry_resolution_status": "resolved_geometry_object",
                "geometry_blocking_reason": "not_applicable",
                "risk_context_only": "false",
            })
            continue
        if source_type == "technical_remote_sensing_flood_footprint" and event_id in sar_bbox:
            rows.append({
                **base,
                "geometry_type": "bbox",
                "geometry_source": rel(SAR_METADATA_17C31),
                "geometry_format": "metadata_bbox",
                "crs": "EPSG:4326",
                "has_geometry_object": "true",
                "uncertainty_m": "bbox_aoi_not_footprint",
                "geometry_resolution_status": "resolved_bbox_metadata_only",
                "geometry_blocking_reason": "bbox_is_AOI_metadata_not_observed_footprint",
                "risk_context_only": "false",
            })
            continue
        if source_type in TEXT_ONLY_TYPES or target.get("has_address") == "true":
            rows.append({
                **base,
                "geometry_type": "unresolved_geometry",
                "geometry_source": target.get("artifact_ref", "not_available"),
                "geometry_format": "text_only",
                "crs": "not_available",
                "has_geometry_object": "false",
                "uncertainty_m": "not_available",
                "geometry_resolution_status": "blocked_text_only",
                "geometry_blocking_reason": "bairro_rua_or_administrative_text_without_coordinate",
                "risk_context_only": "false",
            })
            continue
        if target.get("has_point") == "true":
            rows.append({
                **base,
                "geometry_type": "unresolved_geometry",
                "geometry_source": target.get("artifact_ref", "not_available"),
                "geometry_format": "point_metadata_without_coordinate",
                "crs": "not_available",
                "has_geometry_object": "false",
                "uncertainty_m": "not_available",
                "geometry_resolution_status": "blocked_missing_coordinate_or_crs",
                "geometry_blocking_reason": "official_point_candidate_has_no_coordinate_or_crs_in_repo",
                "risk_context_only": "false",
            })
            continue
        rows.append({
            **base,
            "geometry_type": "unresolved_geometry",
            "geometry_source": target.get("artifact_ref", "not_available"),
            "geometry_format": "not_available",
            "crs": "not_available",
            "has_geometry_object": "false",
            "uncertainty_m": "not_available",
            "geometry_resolution_status": "blocked_no_geometry_object",
            "geometry_blocking_reason": "no_real_geometry_object_found_in_repo",
            "risk_context_only": "false",
        })
    return rows


def _patches() -> list[dict]:
    rows = []
    for row in _read(PATCH_FEATURES):
        bbox = _parse_bbox(";".join([row.get("xmin", ""), row.get("ymin", ""), row.get("xmax", ""), row.get("ymax", "")]))
        if not bbox:
            continue
        rows.append({
            "patch_id": row["patch_id"],
            "patch_namespace": "official_susc_score_v6_patch",
            "official_patch": "true",
            "region": _region_code(row.get("regiao", "")),
            "city": _city(row.get("regiao", "")),
            "bbox": bbox,
            "crs": "EPSG:4326",
        })
    for row in _read(CANDIDATE_PATCH_GRID_17C6):
        bbox = _parse_bbox(";".join([row.get("xmin", ""), row.get("ymin", ""), row.get("xmax", ""), row.get("ymax", "")]))
        if not bbox:
            continue
        rows.append({
            "patch_id": row["candidate_patch_id"],
            "patch_namespace": "susc_17c6_canary_patch_review_only",
            "official_patch": "false",
            "region": _region_code(row.get("region", "")),
            "city": _city(row.get("region", "")),
            "bbox": bbox,
            "crs": row.get("crs", "EPSG:4326") or "EPSG:4326",
        })
    return rows


def _geometry_object_for_row(row: dict) -> tuple[dict | None, tuple[float, float] | None, tuple[float, float, float, float] | None]:
    if row["has_geometry_object"] != "true":
        return None, None, None
    if row["event_id"] == "REC_2022_05_24_30":
        geom, _ = _load_charter_geometry()
        if geom:
            return geom, None, bbox_of_geometry(geom)
    bbox = _sar_bbox_by_event().get(row["event_id"])
    if bbox:
        return _bbox_polygon(bbox), None, bbox
    return None, None, None


def classify_patch_link(geometry_type: str, geometry: dict | None, point: tuple[float, float] | None, geometry_bbox: tuple[float, float, float, float] | None, patch_bbox: tuple[float, float, float, float]) -> tuple[str, float, float, float]:
    """Return link_class, overlap_area_m2, overlap_ratio, distance_m."""
    patch_area = bbox_area_m2(patch_bbox)
    if geometry_type == "point" and point:
        distance = 0.0 if point_in_bbox(point[0], point[1], patch_bbox) else bbox_distance_m((point[0], point[1], point[0], point[1]), patch_bbox)
        if distance == 0.0:
            return "point_within_patch", 0.0, 0.0, 0.0
        if distance <= 10:
            return "near_patch_buffer_10m", 0.0, 0.0, distance
        if distance <= 30:
            return "near_patch_buffer_30m", 0.0, 0.0, distance
        if distance <= 50:
            return "near_patch_buffer_50m", 0.0, 0.0, distance
        return "insufficient_for_patch_link", 0.0, 0.0, distance
    if geometry_type in {"polygon", "technical_footprint"} and geometry and geometry_bbox:
        distance = bbox_distance_m(geometry_bbox, patch_bbox)
        if distance == 0.0:
            overlap = polygon_overlap_area_m2(geometry, patch_bbox)
            ratio = overlap / patch_area if patch_area > 0 else 0.0
            if overlap > 0:
                return "exact_polygon_overlap", overlap, ratio, 0.0
        if distance <= 10:
            return "near_patch_buffer_10m", 0.0, 0.0, distance
        if distance <= 30:
            return "near_patch_buffer_30m", 0.0, 0.0, distance
        if distance <= 50:
            return "near_patch_buffer_50m", 0.0, 0.0, distance
        return "insufficient_for_patch_link", 0.0, 0.0, distance
    if geometry_type == "bbox" and geometry_bbox:
        inter = bbox_intersection(geometry_bbox, patch_bbox)
        distance = bbox_distance_m(geometry_bbox, patch_bbox)
        if inter:
            overlap = bbox_area_m2(inter)
            ratio = overlap / patch_area if patch_area > 0 else 0.0
            return "bbox_overlap", overlap, ratio, 0.0
        if distance <= 10:
            return "near_patch_buffer_10m", 0.0, 0.0, distance
        if distance <= 30:
            return "near_patch_buffer_30m", 0.0, 0.0, distance
        if distance <= 50:
            return "near_patch_buffer_50m", 0.0, 0.0, distance
        return "insufficient_for_patch_link", 0.0, 0.0, distance
    return "unresolved_geometry", 0.0, 0.0, -1.0


def _format_float(value: float, digits: int = 6) -> str:
    if value < 0:
        return "not_available"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") or "0"


def _placeholder_link(idx: int, geom: dict, link_class: str, reason: str) -> dict:
    patch_id = "NO_PATCH_RESOLUTION" if link_class != "same_region_only" else f"REGION_ONLY_{geom.get('region', 'UNKNOWN')}"
    return {
        "patch_link_id": f"S17C5_LINK_{idx:05d}",
        "candidate_event_id": geom["candidate_event_id"],
        "event_id": geom["event_id"],
        "source_id": geom["source_id"],
        "source_type": geom["source_type"],
        "geometry_id": geom["geometry_id"],
        "patch_id": patch_id,
        "patch_namespace": "none",
        "official_patch": "false",
        "region": geom.get("region", "UNKNOWN"),
        "city": geom.get("city", "unknown"),
        "link_class": link_class,
        "link_strength": "none",
        "strong_link_candidate": "false",
        "overlap_area_m2": "0",
        "overlap_ratio": "0",
        "distance_m": "not_available",
        "uncertainty_m": geom["uncertainty_m"],
        "spatial_eligible": "false",
        "eligible_for_evaluation": "false",
        "eligible_for_calibration": "false",
        "eligible_for_training": "false",
        "ground_truth": "false",
        "review_only": "true",
        "score_v7_allowed": "false",
        "not_ground_truth_reason": NO_GT_REASON,
        "spatial_blocking_reason": reason,
    }


def patch_link_rows(geometries: list[dict] | None = None) -> list[dict]:
    geometries = geometries or normalized_geometry_rows()
    patches = _patches()
    links: list[dict] = []
    idx = 0
    for geom in geometries:
        if geom["has_geometry_object"] != "true":
            idx += 1
            link_class = "insufficient_for_patch_link" if geom["source_type"] in BLOCKED_SOURCE_TYPES else "unresolved_geometry"
            if geom["source_type"] not in BLOCKED_SOURCE_TYPES and geom.get("region", "UNKNOWN") != "UNKNOWN":
                link_class = "same_region_only"
            links.append(_placeholder_link(idx, geom, link_class, geom["geometry_blocking_reason"]))
            continue
        geometry, point, geometry_bbox = _geometry_object_for_row(geom)
        if not geometry_bbox and not point:
            idx += 1
            links.append(_placeholder_link(idx, geom, "unresolved_geometry", "geometry_object_missing_after_resolution"))
            continue
        region = _region_code("REC" if geom["event_id"] == "REC_2022_05_24_30" else geom.get("event_id", ""))
        candidate_links = []
        for patch in patches:
            if geom["event_id"] == "REC_2022_05_24_30" and patch["region"] != "REC":
                continue
            link_class, overlap_area, overlap_ratio, distance_m = classify_patch_link(
                "polygon" if geom["geometry_type"] == "technical_footprint" else geom["geometry_type"],
                geometry,
                point,
                geometry_bbox,
                patch["bbox"],
            )
            if link_class in {"insufficient_for_patch_link", "unresolved_geometry"}:
                continue
            strong = link_class in STRONG_LINK_CLASSES
            eligible_eval = strong and patch["official_patch"] == "false"
            strength = "strong_candidate_pending_QA" if strong else "qa_buffer_candidate"
            if patch["official_patch"] == "false" and strong:
                reason = "candidate_canary_patch_overlap_requires_17D_QA_not_17B_unlock"
            elif patch["official_patch"] == "true" and strong:
                reason = "official_patch_overlap_candidate_requires_QA"
            else:
                reason = "near_buffer_candidate_requires_QA"
            candidate_links.append({
                "patch": patch,
                "link_class": link_class,
                "overlap_area": overlap_area,
                "overlap_ratio": overlap_ratio,
                "distance_m": distance_m,
                "strong": strong,
                "eligible_eval": eligible_eval,
                "strength": strength,
                "reason": reason,
            })
        candidate_links.sort(key=lambda item: (item["patch"]["official_patch"] != "false", -item["overlap_ratio"], item["distance_m"], item["patch"]["patch_id"]))
        if not candidate_links:
            idx += 1
            links.append(_placeholder_link(idx, geom, "same_region_only", "real_geometry_does_not_intersect_existing_patch_grid"))
            continue
        for item in candidate_links:
            idx += 1
            patch = item["patch"]
            links.append({
                "patch_link_id": f"S17C5_LINK_{idx:05d}",
                "candidate_event_id": geom["candidate_event_id"],
                "event_id": geom["event_id"],
                "source_id": geom["source_id"],
                "source_type": geom["source_type"],
                "geometry_id": geom["geometry_id"],
                "patch_id": patch["patch_id"],
                "patch_namespace": patch["patch_namespace"],
                "official_patch": patch["official_patch"],
                "region": patch["region"],
                "city": patch["city"],
                "link_class": item["link_class"],
                "link_strength": item["strength"],
                "strong_link_candidate": _bool(item["strong"]),
                "overlap_area_m2": _format_float(item["overlap_area"], 3),
                "overlap_ratio": _format_float(item["overlap_ratio"], 6),
                "distance_m": _format_float(item["distance_m"], 2),
                "uncertainty_m": geom["uncertainty_m"],
                "spatial_eligible": _bool(item["link_class"] != "same_region_only"),
                "eligible_for_evaluation": _bool(item["eligible_eval"]),
                "eligible_for_calibration": "false",
                "eligible_for_training": "false",
                "ground_truth": "false",
                "review_only": "true",
                "score_v7_allowed": "false",
                "not_ground_truth_reason": NO_GT_REASON,
                "spatial_blocking_reason": item["reason"],
            })
    return links


def blocker_rows(geometries: list[dict] | None = None, links: list[dict] | None = None) -> list[dict]:
    geometries = geometries or normalized_geometry_rows()
    links = links or patch_link_rows(geometries)
    linked = {row["geometry_id"] for row in links if row["strong_link_candidate"] == "true"}
    rows = []
    for idx, geom in enumerate(geometries, start=1):
        if geom["geometry_id"] in linked:
            continue
        if geom["has_geometry_object"] == "true":
            blocker_class = "no_patch_intersection"
            reason = "geometry_resolved_but_no_strong_patch_link"
            next_action = "17D_human_QA_or_patch_grid_review"
        else:
            blocker_class = geom["geometry_resolution_status"]
            reason = geom["geometry_blocking_reason"]
            next_action = "acquire_real_geometry_with_crs_or_keep_blocked"
        rows.append({
            "blocker_id": f"S17C5_BLOCK_{len(rows) + 1:04d}",
            "candidate_event_id": geom["candidate_event_id"],
            "event_id": geom["event_id"],
            "source_id": geom["source_id"],
            "source_type": geom["source_type"],
            "geometry_id": geom["geometry_id"],
            "blocker_class": blocker_class,
            "blocking_reason": reason,
            "next_action": next_action,
            "review_only": "true",
            "ground_truth": "false",
            "eligible_for_training": "false",
            "score_v7_allowed": "false",
        })
    return rows


def qa_queue_rows(links: list[dict] | None = None) -> list[dict]:
    links = links or patch_link_rows()
    qa = []
    for link in links:
        if link["link_class"] not in QA_LINK_CLASSES:
            continue
        if link["strong_link_candidate"] != "true" and not link["link_class"].startswith("near_patch_buffer"):
            continue
        qa.append({
            "qa_item_id": f"S17C5_QA_{len(qa) + 1:04d}",
            "candidate_event_id": link["candidate_event_id"],
            "patch_id": link["patch_id"],
            "geometry_id": link["geometry_id"],
            "link_class": link["link_class"],
            "qa_priority": "P1" if link["strong_link_candidate"] == "true" else "P2",
            "qa_status": "pending",
            "qa_question": "Confirmar se a geometria observada e o patch representam o mesmo evento/footprint sem promover ground truth.",
            "map_artifact_ref": rel(LINK_GEOJSON),
            "evidence_summary": f"{link['link_class']} overlap_ratio={link['overlap_ratio']} patch_namespace={link['patch_namespace']}",
            "blocking_reason": link["spatial_blocking_reason"],
            "review_only": "true",
            "ground_truth": "false",
            "eligible_for_training": "false",
            "score_v7_allowed": "false",
        })
    return qa


def link_geojson(links: list[dict] | None = None) -> dict:
    links = links or patch_link_rows()
    patch_by_id = {p["patch_id"]: p for p in _patches()}
    features = []
    for link in links:
        if link["link_class"] not in QA_LINK_CLASSES:
            continue
        patch = patch_by_id.get(link["patch_id"])
        if not patch:
            continue
        features.append({
            "type": "Feature",
            "geometry": _bbox_polygon(patch["bbox"]),
            "properties": {
                "patch_link_id": link["patch_link_id"],
                "candidate_event_id": link["candidate_event_id"],
                "geometry_id": link["geometry_id"],
                "patch_id": link["patch_id"],
                "patch_namespace": link["patch_namespace"],
                "link_class": link["link_class"],
                "strong_link_candidate": link["strong_link_candidate"] == "true",
                "review_only": True,
                "ground_truth": False,
                "eligible_for_training": False,
                "score_v7_allowed": False,
            },
        })
    return {
        "type": "FeatureCollection",
        "metadata": {
            "milestone": "SUSC-17C5",
            "review_only": True,
            "ground_truth": False,
            "eligible_for_training": False,
            "score_v7_allowed": False,
            "feature_count": len(features),
        },
        "features": features,
    }


def _schema(title: str, fields: list[str], consts: dict[str, str] | None = None, enums: dict[str, list[str]] | None = None) -> dict:
    consts = consts or {}
    enums = enums or {}
    properties = {}
    for field in fields:
        if field in consts:
            properties[field] = {"const": consts[field]}
        elif field in enums:
            properties[field] = {"enum": enums[field]}
        else:
            properties[field] = {"type": "string", "minLength": 1}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "description": "SUSC-17C5 review-only schema. No ground truth, no training, no score v7.",
        "required": fields,
        "properties": properties,
        "additionalProperties": True,
    }


def write_schemas() -> None:
    bools = ["true", "false"]
    write_json(SCHEMA_GEOMETRY, _schema(
        "SUSC-17C5 normalized geometry",
        GEOMETRY_FIELDS,
        consts={"review_only": "true", "eligible_for_training": "false", "ground_truth": "false", "score_v7_allowed": "false"},
        enums={
            "geometry_type": ["polygon", "point", "bbox", "technical_footprint", "unresolved_geometry"],
            "has_geometry_object": bools,
            "risk_context_only": bools,
        },
    ))
    write_json(SCHEMA_LINK, _schema(
        "SUSC-17C5 patch links",
        PATCH_LINK_FIELDS,
        consts={"review_only": "true", "eligible_for_training": "false", "ground_truth": "false", "score_v7_allowed": "false"},
        enums={
            "link_class": [
                "exact_polygon_overlap",
                "point_within_patch",
                "bbox_overlap",
                "near_patch_buffer_10m",
                "near_patch_buffer_30m",
                "near_patch_buffer_50m",
                "same_region_only",
                "unresolved_geometry",
                "insufficient_for_patch_link",
            ],
            "strong_link_candidate": bools,
            "spatial_eligible": bools,
            "eligible_for_evaluation": bools,
            "eligible_for_calibration": bools,
        },
    ))
    write_json(SCHEMA_QA, _schema(
        "SUSC-17C5 patch link QA queue",
        QA_FIELDS,
        consts={"review_only": "true", "ground_truth": "false", "eligible_for_training": "false", "score_v7_allowed": "false"},
        enums={"qa_status": ["pending", "blocked", "accepted", "rejected"]},
    ))


def summary_obj(geometries: list[dict], links: list[dict], blockers: list[dict], qa: list[dict]) -> dict:
    prev = read_json(PREV_SUMMARY)
    status = _run_git(["status", "--short"]).splitlines()
    staged = _run_git(["diff", "--cached", "--name-only"]).splitlines()
    strong_links = [row for row in links if row["strong_link_candidate"] == "true"]
    official_strong = [row for row in strong_links if row["official_patch"] == "true"]
    accepted_qa = [row for row in qa if row["qa_status"] == "accepted"]
    eligible_eval = [row for row in links if row["eligible_for_evaluation"] == "true"]
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len([line for line in staged if line.strip()]),
        "dirty_status_lines_count": len([line for line in status if line.strip()]),
        "inherited_17c_target_candidates": prev.get("target_candidates_count", 0),
        "inherited_17c_dated_events": prev.get("temporal_resolved_unique_event_count", 0),
        "inherited_17c_strong_geometry_rows": prev.get("strong_geometry_rows_count", 0),
        "geometry_rows_count": len(geometries),
        "geometries_resolved_count": sum(1 for row in geometries if row["has_geometry_object"] == "true"),
        "geometries_unresolved_count": sum(1 for row in geometries if row["has_geometry_object"] != "true"),
        "patch_link_rows_count": len(links),
        "patch_links_by_class": dict(sorted(Counter(row["link_class"] for row in links).items())),
        "strong_link_candidate_count": len(strong_links),
        "official_strong_link_candidate_count": len(official_strong),
        "qa_queue_count": len(qa),
        "qa_accepted_count": len(accepted_qa),
        "eligible_for_evaluation_count": len(eligible_eval),
        "eligible_for_training_count": sum(1 for row in links if row["eligible_for_training"] == "true"),
        "ground_truth_true_count": sum(1 for row in links if row["ground_truth"] == "true"),
        "score_v7_allowed_true_count": sum(1 for row in links if row["score_v7_allowed"] == "true"),
        "score_v6_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
        "blocker_rows_count": len(blockers),
        "eligible_for_17b_now": False,
        "status_17b": "17B_BLOQUEADO_PARCIAL_17D_QA",
        "review_only": True,
        "ground_truth": False,
        "trainable": False,
    }


def _counts_table(counts: dict) -> str:
    lines = ["| item | count |", "|---|---|"]
    for key, value in counts.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines) + "\n"


def report_markdown(geometries: list[dict], links: list[dict], blockers: list[dict], qa: list[dict], summary: dict) -> str:
    sources = "\n".join(f"- `{row['path']}`: {row['row_count']} ({row['artifact_role']})" for row in preflight_rows() if row["exists"] == "true")
    return f"""# SUSC-17C5 Geometry-to-Patch Linkage Resolver

## Estado inicial herdado do 17C

- Branch: `{summary['branch']}`
- HEAD: `{summary['head']}`
- Area staged: {summary['staged_count']} arquivo(s)
- Candidatas 17C: {summary['inherited_17c_target_candidates']}
- Eventos 17C datados: {summary['inherited_17c_dated_events']}
- Linhas 17C com geometria forte candidata: {summary['inherited_17c_strong_geometry_rows']}
- `score_v6` alterado: {summary['score_v6_changed']}
- `score_v7` criado: {summary['score_v7_created']}

## Fontes e registries usados

{sources}

## Geometrias

- Geometrias normalizadas: {summary['geometry_rows_count']}
- Geometrias resolvidas com objeto real: {summary['geometries_resolved_count']}
- Geometrias unresolved/bloqueadas: {summary['geometries_unresolved_count']}
- Blockers registrados: {summary['blocker_rows_count']}

## Patch-links

- Patch-links gerados: {summary['patch_link_rows_count']}
- Strong link candidates: {summary['strong_link_candidate_count']}
- Strong link candidates em patch oficial: {summary['official_strong_link_candidate_count']}
- Entradas em QA: {summary['qa_queue_count']}
- Eligible for evaluation: {summary['eligible_for_evaluation_count']}
- Eligible for training: {summary['eligible_for_training_count']}
- Ground truth: {summary['ground_truth_true_count']}
- score_v7_allowed=true: {summary['score_v7_allowed_true_count']}

### Links por classe

{_counts_table(summary['patch_links_by_class'])}
## Decisao 17B

17B continua **bloqueado**. A sprint criou links auditaveis para QA humana, mas todos permanecem review-only; nao ha QA accepted automatico, nao ha patch-link oficial aceito, nao ha ground truth, nao ha treino e nao ha score_v7.

O desbloqueio e parcial apenas para o proximo passo operacional: `17D Human QA` pode revisar os links candidatos em `reference_patch_link_qa_queue.csv`.

## Proximos passos para SUSC-17D Human QA

1. Revisar visualmente cada item de `reference_patch_link_qa_queue.csv`.
2. Confirmar se o patch canario 17C6 e aceitavel como patch de revisao, sem mistura com patch oficial.
3. Para PET, adquirir coordenadas/CRS reais dos pontos oficiais antes de qualquer link forte.
4. Para qualquer `same_region_only`, manter bloqueado ate existir geometria real.
5. Manter `ground_truth=false`, `eligible_for_training=false` e `score_v7_allowed=false`.
"""


def build_all() -> dict:
    _require_inputs()
    ensure_dir(OUT_DATA)
    ensure_dir(OUT_REPORTS)
    write_schemas()
    geometries = normalized_geometry_rows()
    links = patch_link_rows(geometries)
    blockers = blocker_rows(geometries, links)
    qa = qa_queue_rows(links)
    summary = summary_obj(geometries, links, blockers, qa)
    write_csv(PREFLIGHT, preflight_rows(), PREFLIGHT_FIELDS)
    write_csv(GEOMETRIES, geometries, GEOMETRY_FIELDS)
    write_csv(PATCH_LINKS, links, PATCH_LINK_FIELDS)
    write_csv(BLOCKERS, blockers, BLOCKER_FIELDS)
    write_csv(QA_QUEUE, qa, QA_FIELDS)
    write_json(LINK_GEOJSON, link_geojson(links))
    write_json(SUMMARY, summary)
    write_markdown(REPORT, report_markdown(geometries, links, blockers, qa, summary))
    return summary


def _schema_violations(row: dict, schema: dict) -> list[str]:
    errors = []
    for field in schema.get("required", []):
        if field not in row or str(row.get(field, "")) == "":
            errors.append(f"missing_required:{field}")
    for field, rule in schema.get("properties", {}).items():
        if field not in row:
            continue
        value = row.get(field)
        if "const" in rule and value != rule["const"]:
            errors.append(f"{field}:expected_const:{rule['const']}:got:{value}")
        if "enum" in rule and value not in rule["enum"]:
            errors.append(f"{field}:not_in_enum:{value}")
    return errors


def validate_output_rows(geometries: list[dict], links: list[dict], qa: list[dict]) -> list[str]:
    errors: list[str] = []
    schemas = [
        ("geometry", geometries, read_json(SCHEMA_GEOMETRY)),
        ("link", links, read_json(SCHEMA_LINK)),
        ("qa", qa, read_json(SCHEMA_QA)),
    ]
    geometry_by_id = {row["geometry_id"]: row for row in geometries}
    for name, rows, schema in schemas:
        for idx, row in enumerate(rows, start=1):
            for err in _schema_violations(row, schema):
                errors.append(f"{name}[{idx}]:{err}")
    for rows_name, rows in [("geometry", geometries), ("link", links), ("qa", qa)]:
        for row in rows:
            row_id = row.get("patch_link_id") or row.get("geometry_id") or row.get("qa_item_id", "unknown")
            for field in FORBIDDEN_TRUE_FIELDS:
                if row.get(field) == "true":
                    errors.append(f"{rows_name}:{row_id}:{field}_true_forbidden")
    for geom in geometries:
        if geom["has_geometry_object"] == "true" and geom["crs"] in {"", "not_available"}:
            errors.append(f"geometry:{geom['geometry_id']}:resolved_geometry_without_crs")
    for link in links:
        geom = geometry_by_id.get(link["geometry_id"], {})
        if link["strong_link_candidate"] == "true":
            if not link.get("geometry_id"):
                errors.append(f"link:{link['patch_link_id']}:strong_without_geometry_id")
            if not link.get("patch_id") or link["patch_id"] in {"NO_PATCH_RESOLUTION", ""}:
                errors.append(f"link:{link['patch_link_id']}:strong_without_patch_id")
            if link.get("uncertainty_m") in {"", "not_available"}:
                errors.append(f"link:{link['patch_link_id']}:strong_without_uncertainty_m")
            if geom.get("geometry_type") == "unresolved_geometry" or geom.get("geometry_format") == "text_only":
                errors.append(f"link:{link['patch_link_id']}:text_or_unresolved_promoted_to_strong")
            if link.get("source_type") in BLOCKED_SOURCE_TYPES:
                errors.append(f"link:{link['patch_link_id']}:alert_or_risk_promoted_to_strong")
        if link["link_class"] == "same_region_only" and link["eligible_for_evaluation"] == "true":
            errors.append(f"link:{link['patch_link_id']}:same_region_only_eligible_for_evaluation")
        if link["link_class"].startswith("near_patch_buffer") and link["strong_link_candidate"] == "true":
            errors.append(f"link:{link['patch_link_id']}:near_buffer_marked_strong")
    for row in qa:
        if row["qa_status"] == "accepted":
            errors.append(f"qa:{row['qa_item_id']}:qa_accepted_automatic_forbidden")
    return errors


def validate() -> int:
    summary = build_all()
    geometries = _read(GEOMETRIES)
    links = _read(PATCH_LINKS)
    blockers = _read(BLOCKERS)
    qa = _read(QA_QUEUE)
    errors = validate_output_rows(geometries, links, qa)
    if summary["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if summary["score_v7_created"]:
        errors.append("score_v7_created_forbidden")
    if summary["eligible_for_training_count"] != 0:
        errors.append("eligible_for_training_count_nonzero")
    if summary["ground_truth_true_count"] != 0:
        errors.append("ground_truth_true_count_nonzero")
    if summary["score_v7_allowed_true_count"] != 0:
        errors.append("score_v7_allowed_true_count_nonzero")
    if not links:
        errors.append("empty_patch_links")
    if not blockers:
        errors.append("empty_blockers")
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(
        "17C5 geometry-to-patch validated: "
        f"resolved={summary['geometries_resolved_count']} "
        f"blocked={summary['geometries_unresolved_count']} "
        f"strong_links={summary['strong_link_candidate_count']} "
        f"qa={summary['qa_queue_count']} "
        f"eligible_eval={summary['eligible_for_evaluation_count']} "
        f"17B={summary['status_17b']}"
    )
    return 0


def run_all() -> int:
    summary = build_all()
    print(
        "17C5 geometry-to-patch built: "
        f"resolved={summary['geometries_resolved_count']} "
        f"blocked={summary['geometries_unresolved_count']} "
        f"strong_links={summary['strong_link_candidate_count']} "
        f"17B={summary['status_17b']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())
