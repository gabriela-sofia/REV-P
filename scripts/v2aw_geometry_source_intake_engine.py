#!/usr/bin/env python3
"""v2aw - Geometry Source Intake + Recife Patch Boundary Filling Scaffold.

Creates the auditable channel to receive, validate, normalise and audit REAL patch
boundary geometries (and, when applicable, observed event geometries), starting
with the 55 Recife P1 patches from the v2av recovery queue.

It produces fillable templates, validates provided geometry sources, audits the
already-existing real geometries (the CPRM event point anchors), and reports
Recife readiness for v2av/v2au — without ever inventing geometry.

Hard methodological line (never crossed):
  - no final ground truth, no binary/operational label, no model training;
  - geometry is never invented; no boundary is created by default;
  - a centroid/point is never a patch boundary;
  - a point event is never an overlay (stays an anchor);
  - context/risk geometry never promotes C4;
  - the v2at, v2au and v2av outputs are never overwritten (only v2aw artefacts).

This stage does not solve the lack of external data; it builds the auditable
intake so that data can be inserted later without contaminating the project. The
maximum downstream meaning of a validated source is READY_FOR_V2AV /
READY_FOR_V2AU (a candidate input), never a label.

Offline, deterministic; outputs sorted by stable keys; stable hashes for ids;
exit code 0 even with expected blockers, non-zero only on a structural error.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

STAGE = "v2aw"
METHODOLOGICAL_STATUS = "GEOMETRY_SOURCE_INTAKE_READY_NOT_FOR_TRAINING"

THIS_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))

UNKNOWN = "UNKNOWN"
NOT_AVAILABLE = "NOT_AVAILABLE"
EARTH_RADIUS_M = 6378137.0


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def resolve_dirs():
    dataset_dir = os.environ.get("DATASET_DIR") or project_path("datasets")
    output_dir = os.environ.get("OUTPUT_DIR") or project_path("outputs_public")
    config_dir = os.environ.get("CONFIG_DIR") or project_path("configs")
    return dataset_dir, output_dir, config_dir


CONFIG_NAME = "v2aw_geometry_source_intake_config.json"

DEFAULT_CONFIG = {
    "offline_mode": True, "strict_crs": True,
    "accepted_crs": ["EPSG:4326", "EPSG:3857", "EPSG:31982", "EPSG:31983"],
    "preferred_crs": "EPSG:4326", "target_crs_for_area": "EPSG:3857",
    "priority_region": "Recife", "priority_allowed_use": "candidate_reference",
    "allow_bbox": True, "allow_wkt": True, "allow_geojson_inline": True,
    "allow_geojson_file": True, "allow_point_as_patch_boundary": False,
    "allow_default_patch_size": False, "fail_on_missing_optional_geometry": False,
    "required_source_fields": ["geometry_source_id", "geometry_role", "linked_patch_id",
                               "source_type", "crs", "provenance_note"],
}

# Output filenames -----------------------------------------------------------

OUT_PATCH_TEMPLATE = "v2aw_patch_geometry_sources_template.csv"
OUT_EVENT_TEMPLATE = "v2aw_event_geometry_sources_template.csv"
OUT_VALIDATION = "v2aw_geometry_source_validation_registry.csv"
OUT_READINESS = "v2aw_recife_p1_geometry_readiness.csv"

DOC_REL = os.path.join("docs", "v2aw_geometry_source_intake_instructions.md")
EXAMPLES_SUBDIR = os.path.join("examples", "v2aw_geometry_intake")

REPORT_REL = os.path.join("execution_reports", "v2aw_geometry_source_intake_report.md")
SUMMARY_REL = os.path.join("execution_reports", "v2aw_geometry_source_intake_summary.json")
SUPPLEMENT_REL = os.path.join("execution_reports", "v2aw_artifact_index_supplement.md")
LOG_REL = os.path.join("logs_summary", "v2aw_geometry_source_intake.txt")

# Input filenames (relative to dataset_dir) ----------------------------------

IN_RECOVERY_QUEUE = "v2av_patch_boundary_recovery_queue.csv"
IN_PACKAGES = "v2at_event_patch_package_registry.csv"
IN_GROUND_EVENTS = "ground_reference_event_registry.csv"
# Optional user-provided filled intake files (absent until a human fills them).
IN_PROVIDED_PATCH = "v2aw_patch_geometry_sources.csv"
IN_PROVIDED_EVENT = "v2aw_event_geometry_sources.csv"

REGION_NAME = {"REC": "Recife", "PET": "Petropolis", "CUR": "Curitiba"}

SOURCE_TYPES = ("bbox", "wkt", "geojson_inline", "geojson_file", "missing")
PROVENANCE_TYPES = ("sentinel_footprint", "patch_generation_metadata", "manual_digitization",
                    "gis_export", "official_vector", "unknown")
REVIEW_STATUSES = ("not_started", "provided_unreviewed", "format_validated", "needs_fix",
                   "approved_for_v2av", "rejected")
EVENT_ROLES = ("observed_event_polygon", "observed_event_line", "observed_event_point_anchor",
               "context_area", "risk_area", "unknown")
GEOMETRY_KINDS = ("patch_boundary", "event_geometry", "unknown")

COLUMNS = {
    OUT_PATCH_TEMPLATE: [
        "geometry_source_id", "linked_patch_id", "region", "city", "priority_rank",
        "source_type", "geometry_value", "geometry_path", "crs", "provenance_type",
        "provenance_note", "digitized_by", "digitized_at", "source_document",
        "source_document_page", "source_confidence", "license_status", "review_status", "notes",
    ],
    OUT_EVENT_TEMPLATE: [
        "event_geometry_source_id", "linked_event_id", "region", "city", "hazard_type",
        "source_type", "geometry_value", "geometry_path", "crs", "event_geometry_role",
        "source_id", "source_name", "provenance_type", "provenance_note", "digitized_by",
        "digitized_at", "source_document", "source_document_page", "source_confidence",
        "license_status", "review_status", "notes",
    ],
    OUT_VALIDATION: [
        "validation_id", "geometry_source_id", "geometry_kind", "linked_patch_id",
        "linked_event_id", "region", "city", "source_type", "geometry_role", "crs",
        "crs_status", "geometry_present", "geometry_valid", "geometry_format_valid",
        "geometry_hash", "can_be_used_by_v2av", "can_be_used_by_v2au", "blocking_reason",
        "recommended_fix", "notes",
    ],
    OUT_READINESS: [
        "readiness_id", "patch_id", "region", "city", "priority_rank", "package_count",
        "has_patch_boundary_source", "patch_boundary_valid", "has_event_geometry_source",
        "event_geometry_valid", "ready_for_v2av", "ready_for_v2au_overlay",
        "remaining_blocker", "next_required_action", "notes",
    ],
}

# --------------------------------------------------------------------------- #
# Small IO / helpers.
# --------------------------------------------------------------------------- #


def clean(value):
    return str(value if value is not None else "").strip()


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def stable_id(prefix, *parts, length=12):
    digest = hashlib.sha1("|".join(clean(p) for p in parts).encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:length]}"


def region_from_code(code):
    return REGION_NAME.get(clean(code).upper(), clean(code) or UNKNOWN)


def _b(value):
    return "true" if value else "false"


# --------------------------------------------------------------------------- #
# Geometry kernel (pure Python, offline).
# --------------------------------------------------------------------------- #


def normalise_crs(raw):
    raw = clean(raw).upper().replace(" ", "")
    if not raw:
        return ""
    if raw.isdigit():
        return f"EPSG:{raw}"
    return raw


def _parse_floats(text):
    out = []
    for token in clean(text).replace(",", " ").split():
        try:
            out.append(float(token))
        except ValueError:
            return None
    return out


def parse_geometry(source_type, value, geometry_path):
    """Return (kind_hint, gtype, ok) where gtype in {point, line, polygon, None}."""
    source_type = clean(source_type).lower()
    value = clean(value)
    try:
        if source_type == "bbox":
            nums = _parse_floats(value)
            return ("polygon", True) if nums and len(nums) >= 4 else (None, False)
        if source_type == "wkt":
            upper = value.upper()
            if upper.startswith("POLYGON"):
                return "polygon", _valid_wkt_polygon(value)
            if upper.startswith("POINT"):
                return "point", bool(_parse_floats(value[value.find("(") + 1:value.find(")")]))
            if upper.startswith("LINESTRING"):
                return "line", True
            return None, False
        if source_type == "geojson_inline":
            return _geojson_type(json.loads(value))
        if source_type == "geojson_file":
            gpath = clean(geometry_path)
            abs_path = gpath if os.path.isabs(gpath) else os.path.join(PROJECT_ROOT, gpath)
            if not gpath or not os.path.exists(abs_path):
                return None, False
            with open(abs_path, encoding="utf-8") as handle:
                return _geojson_type(json.load(handle))
    except (ValueError, json.JSONDecodeError, OSError, TypeError):
        return None, False
    return None, False


def _valid_wkt_polygon(text):
    try:
        inner = text[text.index("((") + 2: text.index("))")]
    except ValueError:
        return False
    pts = []
    for pair in inner.split(","):
        nums = _parse_floats(pair)
        if not nums or len(nums) < 2:
            return False
        pts.append((nums[0], nums[1]))
    return len(pts) >= 3


def _geojson_type(obj):
    geom = obj.get("geometry", obj) if isinstance(obj, dict) else None
    if not isinstance(geom, dict):
        return None, False
    gtype = clean(geom.get("type")).lower()
    coords = geom.get("coordinates")
    if gtype == "point" and isinstance(coords, list) and len(coords) >= 2:
        return "point", True
    if gtype in ("polygon", "multipolygon") and isinstance(coords, list) and coords:
        return "polygon", True
    if gtype in ("linestring", "multilinestring") and isinstance(coords, list) and coords:
        return "line", True
    return None, False


def geometry_hash(source_type, value, geometry_path, crs):
    basis = (clean(source_type), clean(value), clean(geometry_path), clean(crs))
    return hashlib.sha1(repr(basis).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Discovery.
# --------------------------------------------------------------------------- #


def priority_patches(inputs, config):
    region = clean(config["priority_region"])
    rows = []
    for r in inputs["recovery_queue"]:
        if clean(r.get("region")) == region and clean(r.get("is_recife_priority")).lower() == "true" \
                and clean(r.get("priority_rank")) == "1":
            rows.append({
                "patch_id": clean(r.get("patch_id")), "region": clean(r.get("region")),
                "city": clean(r.get("city")) or region, "priority_rank": clean(r.get("priority_rank")),
                "package_count": clean(r.get("is_needed_by_packages_count")) or "0",
            })
    rows.sort(key=lambda x: x["patch_id"])
    return rows


def patch_events(inputs):
    """patch_id -> (event_id, hazard_type) from v2at packages (first match)."""
    out = {}
    for pkg in inputs["packages"]:
        pid = clean(pkg.get("patch_id"))
        if pid and pid not in out:
            out[pid] = (clean(pkg.get("event_id")), clean(pkg.get("hazard_type")))
    return out


def package_events(inputs):
    seen = {}
    for pkg in inputs["packages"]:
        eid = clean(pkg.get("event_id"))
        if eid and "MISSING" not in eid.upper() and eid != "UNKNOWN_EVENT" and eid not in seen:
            seen[eid] = {"event_id": eid, "region": clean(pkg.get("region")),
                         "city": clean(pkg.get("city")) or clean(pkg.get("region")),
                         "hazard_type": clean(pkg.get("hazard_type"))}
    return sorted(seen.values(), key=lambda x: x["event_id"])


def _map_cprm_event(region_code, raw_date, pkg_event_ids):
    region = clean(region_code).upper()
    year = ""
    for token in clean(raw_date).replace("-", "/").split("/"):
        if len(token) == 4 and token.isdigit():
            year = token
            break
    for ev in pkg_event_ids:
        if ev.upper().startswith(region) and (not year or year in ev):
            return ev
    return f"{region}_EVENT_{year or 'UNKNOWN'}"


def cprm_point_anchors(inputs):
    pkg_ids = sorted({clean(p.get("event_id")) for p in inputs["packages"] if clean(p.get("event_id"))})
    anchors = []
    for ev in inputs["ground_events"]:
        lat, lon = clean(ev.get("latitude")), clean(ev.get("longitude"))
        if not lat or not lon:
            continue
        anchors.append({
            "linked_event_id": _map_cprm_event(ev.get("region"), ev.get("event_or_survey_date"), pkg_ids),
            "region": region_from_code(ev.get("region")),
            "city": clean(ev.get("municipality")) or region_from_code(ev.get("region")),
            "lat": lat, "lon": lon, "source_id": "SGB_RISK_CARTOGRAPHY",
            "source_name": "SGB/CPRM field survey point", "raw_event": clean(ev.get("event_id")),
        })
    anchors.sort(key=lambda a: (a["linked_event_id"], a["lon"], a["lat"]))
    return anchors


# --------------------------------------------------------------------------- #
# Templates.
# --------------------------------------------------------------------------- #


def build_patch_template(patches):
    rows = []
    for p in patches:
        rows.append({
            "geometry_source_id": stable_id("PGS_", p["patch_id"]), "linked_patch_id": p["patch_id"],
            "region": p["region"], "city": p["city"], "priority_rank": p["priority_rank"],
            "source_type": "missing", "geometry_value": "", "geometry_path": "", "crs": "",
            "provenance_type": "unknown", "provenance_note": "FILL: provide a real patch boundary "
            "polygon (bbox/WKT/GeoJSON) with a verified CRS, derived from the patch Sentinel "
            "footprint or patch-generation metadata. Do NOT use a centroid/point as a boundary.",
            "digitized_by": "", "digitized_at": "", "source_document": "", "source_document_page": "",
            "source_confidence": "", "license_status": "", "review_status": "not_started",
            "notes": "Auto-generated empty intake row; geometry never invented.",
        })
    rows.sort(key=lambda r: r["linked_patch_id"])
    return rows


def build_event_template(events, anchors):
    rows = []
    # One "polygon needed" row per package event (digitize from Charter/EMS/VHR).
    for ev in events:
        rows.append({
            "event_geometry_source_id": stable_id("EGS_", ev["event_id"], "observed_event_polygon"),
            "linked_event_id": ev["event_id"], "region": ev["region"], "city": ev["city"],
            "hazard_type": ev["hazard_type"], "source_type": "missing", "geometry_value": "",
            "geometry_path": "", "crs": "", "event_geometry_role": "observed_event_polygon",
            "source_id": "", "source_name": "", "provenance_type": "unknown",
            "provenance_note": "FILL: digitize the observed event polygon from an official validated "
            "product (Charter/EMS/VHR) with a verified CRS. A point anchor is NOT an overlay.",
            "digitized_by": "", "digitized_at": "", "source_document": "", "source_document_page": "",
            "source_confidence": "", "license_status": "", "review_status": "not_started",
            "notes": "Observed event polygon needed for a patch-event overlay.",
        })
    # Seed the already-existing real CPRM point anchors (they ARE real geometry).
    for a in anchors:
        rows.append({
            "event_geometry_source_id": stable_id("EGS_", a["raw_event"], "point_anchor"),
            "linked_event_id": a["linked_event_id"], "region": a["region"], "city": a["city"],
            "hazard_type": "mass_movement", "source_type": "wkt",
            "geometry_value": f"POINT({a['lon']} {a['lat']})", "geometry_path": "", "crs": "EPSG:4326",
            "event_geometry_role": "observed_event_point_anchor", "source_id": a["source_id"],
            "source_name": a["source_name"], "provenance_type": "official_vector",
            "provenance_note": "Real official survey point. Stays an anchor; never an overlay/boundary.",
            "digitized_by": "SGB/CPRM", "digitized_at": "", "source_document": a["raw_event"],
            "source_document_page": "", "source_confidence": "OFFICIAL_EXPLICIT_COORDINATE",
            "license_status": "open_public", "review_status": "provided_unreviewed",
            "notes": "Point anchor is not a patch overlay; needs a real event polygon for v2au.",
        })
    rows.sort(key=lambda r: (r["linked_event_id"], r["event_geometry_role"], r["event_geometry_source_id"]))
    return rows


# --------------------------------------------------------------------------- #
# Validation of provided / existing geometry sources.
# --------------------------------------------------------------------------- #


def _validate_source(kind, src, config):
    accepted = set(config["accepted_crs"])
    allow_point_patch = bool(config["allow_point_as_patch_boundary"])
    source_type = clean(src.get("source_type")).lower() or "missing"
    crs = normalise_crs(src.get("crs"))
    role = clean(src.get("geometry_role")) or clean(src.get("event_geometry_role")) or (
        "patch_boundary" if kind == "patch_boundary" else "observed_event_polygon")

    present = source_type != "missing" and bool(
        clean(src.get("geometry_value")) or clean(src.get("geometry_path")))
    gtype, valid = (None, False)
    fmt_valid = source_type in ("bbox", "wkt", "geojson_inline", "geojson_file")
    if present and fmt_valid:
        gtype, valid = parse_geometry(source_type, src.get("geometry_value"), src.get("geometry_path"))
    crs_status = "KNOWN" if crs in accepted else "UNKNOWN"

    can_v2av = False
    can_v2au = False
    blocking = ""
    fix = ""

    if not present:
        blocking, fix = "BLOCKED_MISSING_GEOMETRY", "Provide bbox/WKT/GeoJSON geometry."
    elif not fmt_valid:
        blocking, fix = "BLOCKED_INVALID_GEOMETRY", "Use one of: bbox, wkt, geojson_inline, geojson_file."
    elif not valid:
        blocking, fix = "BLOCKED_INVALID_GEOMETRY", "Fix the geometry payload so it parses."
    elif not crs:
        blocking, fix = "BLOCKED_UNKNOWN_CRS", "Add a CRS (e.g. EPSG:4326)."
    elif crs_status != "KNOWN":
        blocking, fix = "BLOCKED_UNKNOWN_CRS", "Use an accepted CRS code."
    else:
        # Geometry present, valid and CRS known: apply role/type rules.
        if kind == "patch_boundary":
            if gtype == "point" and not allow_point_patch:
                blocking, fix = "BLOCKED_POINT_NOT_PATCH_BOUNDARY", \
                    "A point is not a patch boundary; provide a polygon."
            elif gtype == "polygon":
                can_v2av = True
            else:
                blocking, fix = "BLOCKED_INVALID_GEOMETRY", "Patch boundary must be a polygon."
        else:  # event_geometry
            if role in ("context_area", "risk_area"):
                blocking, fix = "CONTEXT_GEOMETRY_NOT_PROMOTABLE", \
                    "Context/risk geometry cannot promote C4."
            elif role == "observed_event_point_anchor" or gtype == "point":
                blocking, fix = "POINT_ANCHOR_NOT_OVERLAY", \
                    "A point event is an anchor, not an overlay; provide an observed event polygon."
            elif gtype == "polygon":
                can_v2au = True
            else:
                blocking, fix = "BLOCKED_INVALID_GEOMETRY", "Observed event geometry must be a polygon."

    return {
        "source_type": source_type, "geometry_role": role, "crs": crs or UNKNOWN,
        "crs_status": crs_status, "geometry_present": present, "geometry_valid": valid,
        "geometry_format_valid": fmt_valid,
        "geometry_hash": geometry_hash(source_type, src.get("geometry_value"),
                                       src.get("geometry_path"), crs) if present else "NONE",
        "can_v2av": can_v2av, "can_v2au": can_v2au, "blocking_reason": blocking, "recommended_fix": fix,
        "gtype": gtype,
    }


def build_validation_registry(inputs, event_template, config):
    rows = []

    # (A) provided patch boundary sources (user-filled file).
    for src in inputs["provided_patch"]:
        v = _validate_source("patch_boundary", src, config)
        rows.append(_validation_row("patch_boundary", src.get("geometry_source_id"),
                                    clean(src.get("linked_patch_id")), "", src, v))

    # (B) provided event geometry sources (user-filled file).
    for src in inputs["provided_event"]:
        v = _validate_source("event_geometry", src, config)
        rows.append(_validation_row("event_geometry", src.get("event_geometry_source_id"),
                                    "", clean(src.get("linked_event_id")), src, v))

    # (C) real existing event geometry already present in the repo: the CPRM point
    # anchors seeded into the event template (validated honestly: valid but not an overlay).
    if not inputs["provided_event"]:
        for src in event_template:
            if clean(src.get("event_geometry_role")) == "observed_event_point_anchor":
                v = _validate_source("event_geometry", src, config)
                rows.append(_validation_row("event_geometry", src.get("event_geometry_source_id"),
                                            "", clean(src.get("linked_event_id")), src, v))

    rows.sort(key=lambda r: (r["geometry_kind"], r["linked_patch_id"], r["linked_event_id"],
                             r["geometry_source_id"]))
    for idx, row in enumerate(rows):
        row["validation_id"] = stable_id("GSV_", row["geometry_source_id"],
                                         row["geometry_kind"], str(idx))
    return rows


def _validation_row(kind, source_id, linked_patch, linked_event, src, v):
    return {
        "geometry_source_id": clean(source_id) or stable_id("GS_", kind, linked_patch, linked_event),
        "geometry_kind": kind, "linked_patch_id": linked_patch, "linked_event_id": linked_event,
        "region": clean(src.get("region")) or UNKNOWN, "city": clean(src.get("city")) or UNKNOWN,
        "source_type": v["source_type"], "geometry_role": v["geometry_role"], "crs": v["crs"],
        "crs_status": v["crs_status"], "geometry_present": _b(v["geometry_present"]),
        "geometry_valid": _b(v["geometry_valid"]), "geometry_format_valid": _b(v["geometry_format_valid"]),
        "geometry_hash": v["geometry_hash"], "can_be_used_by_v2av": _b(v["can_v2av"]),
        "can_be_used_by_v2au": _b(v["can_v2au"]), "blocking_reason": v["blocking_reason"],
        "recommended_fix": v["recommended_fix"],
        "notes": "Validated geometry source; never a label or ground truth.",
    }


# --------------------------------------------------------------------------- #
# Recife readiness.
# --------------------------------------------------------------------------- #


def build_readiness(patches, validation, patch_event_map):
    by_patch = {}
    for v in validation:
        if v["geometry_kind"] == "patch_boundary" and v["linked_patch_id"]:
            by_patch.setdefault(v["linked_patch_id"], []).append(v)
    event_ok = {}
    for v in validation:
        if v["geometry_kind"] == "event_geometry" and v["can_be_used_by_v2au"] == "true":
            event_ok[v["linked_event_id"]] = True

    rows = []
    for p in patches:
        pid = p["patch_id"]
        patch_sources = by_patch.get(pid, [])
        has_patch_src = bool(patch_sources)
        patch_valid = any(s["can_be_used_by_v2av"] == "true" for s in patch_sources)
        event_id = patch_event_map.get(pid, ("", ""))[0]
        has_event_src = event_id in event_ok or any(
            v["linked_event_id"] == event_id and v["geometry_present"] == "true"
            for v in validation if v["geometry_kind"] == "event_geometry")
        event_valid = bool(event_ok.get(event_id))
        ready_v2av = patch_valid
        ready_v2au = patch_valid and event_valid

        if not has_patch_src:
            blocker = "NO_PATCH_BOUNDARY_SOURCE_PROVIDED"
            action = "Provide a real patch boundary polygon (bbox/WKT/GeoJSON) with a verified CRS"
        elif not patch_valid:
            blocker = "PATCH_BOUNDARY_SOURCE_INVALID_OR_UNUSABLE"
            action = "Fix the patch boundary geometry/CRS (no point as boundary)"
        elif not event_valid:
            blocker = "NO_VALID_OBSERVED_EVENT_POLYGON"
            action = "Digitize the observed event polygon (Charter/EMS/VHR) with a verified CRS"
        else:
            blocker = ""
            action = "Run v2av then v2au; review the C4 candidate manually"

        rows.append({
            "readiness_id": stable_id("RDY_", pid), "patch_id": pid, "region": p["region"],
            "city": p["city"], "priority_rank": p["priority_rank"], "package_count": p["package_count"],
            "has_patch_boundary_source": _b(has_patch_src), "patch_boundary_valid": _b(patch_valid),
            "has_event_geometry_source": _b(has_event_src), "event_geometry_valid": _b(event_valid),
            "ready_for_v2av": _b(ready_v2av), "ready_for_v2au_overlay": _b(ready_v2au),
            "remaining_blocker": blocker, "next_required_action": action,
            "notes": "No patch is declared ready without real geometry; geometry never invented.",
        })
    rows.sort(key=lambda r: r["patch_id"])
    return rows


# --------------------------------------------------------------------------- #
# Validation of outputs (structural).
# --------------------------------------------------------------------------- #


def validate_outputs(written, expected_priority):
    errors = []
    patch_template = written[OUT_PATCH_TEMPLATE]
    if len(patch_template) != expected_priority:
        errors.append(f"patch template has {len(patch_template)} rows, expected {expected_priority}")
    for name, col in ((OUT_PATCH_TEMPLATE, "geometry_source_id"),
                      (OUT_EVENT_TEMPLATE, "event_geometry_source_id"),
                      (OUT_VALIDATION, "validation_id"), (OUT_READINESS, "readiness_id")):
        for row in written[name]:
            if not clean(row.get(col)):
                errors.append(f"{name}: empty id in {col}")
                break
    # No invented geometry: every 'missing' template row must have empty geometry_value.
    for row in patch_template:
        if row["source_type"] == "missing" and clean(row["geometry_value"]):
            errors.append(f"patch template {row['geometry_source_id']} invented geometry")
            break
    # A point can never be flagged usable as a patch boundary.
    for row in written[OUT_VALIDATION]:
        if row["geometry_kind"] == "patch_boundary" and row["can_be_used_by_v2av"] == "true" \
                and "POINT" in row["blocking_reason"].upper():
            errors.append("point flagged usable as patch boundary")
            break
    # No patch declared ready without a valid boundary.
    for row in written[OUT_READINESS]:
        if row["ready_for_v2av"] == "true" and row["patch_boundary_valid"] != "true":
            errors.append(f"readiness {row['patch_id']} ready without a valid boundary")
            break
    return errors


# --------------------------------------------------------------------------- #
# Instructions doc + synthetic examples.
# --------------------------------------------------------------------------- #


def write_instructions(output_doc_path):
    text = """# v2aw - Geometry Source Intake instructions

## 1. Objetivo
A v2aw cria o canal auditavel para inserir geometrias reais de boundary de patch (e, quando
aplicavel, geometrias observadas de evento) sem contaminar o projeto. Ela comeca pelos 55
patches Recife P1 priorizados pela fila de recuperacao da v2av.

## 2. Por que o REV-P precisa de boundary vetorial de patch
A v2au mostrou que existem 172 pacotes evento-patch, mas 0 overlay, porque nao existe
geometria vetorial de patch. Sem um poligono de boundary (com CRS), nao ha como calcular
`patch ∩ evento`. A v2aw nao inventa geometria: ela pede a geometria real.

## 3. Formatos aceitos
- `bbox`: `minx,miny,maxx,maxy`
- `wkt`: `POLYGON((x y, x y, ...))`
- `geojson_inline`: um objeto GeoJSON Polygon em uma celula
- `geojson_file`: caminho relativo para um arquivo `.geojson`

### Exemplos
- bbox (EPSG:4326): `-34.95,-8.10,-34.90,-8.05`
- WKT (EPSG:3857): `POLYGON((-3888000 -893000, -3887000 -893000, -3887000 -892000, -3888000 -892000, -3888000 -893000))`
- GeoJSON inline: `{"type":"Polygon","coordinates":[[[-34.95,-8.10],[-34.90,-8.10],[-34.90,-8.05],[-34.95,-8.05],[-34.95,-8.10]]]}`

## 4. Campos obrigatorios (patch)
`geometry_source_id, linked_patch_id, source_type, crs, provenance_note` (alem de
`geometry_value` OU `geometry_path`). Sem CRS, a geometria e bloqueada.

## 5. Como preencher o template Recife
1. abra `datasets/v2aw_patch_geometry_sources_template.csv` (55 linhas Recife P1, `source_type=missing`);
2. para cada patch com geometria real, preencha `source_type`, `geometry_value`/`geometry_path`, `crs`,
   `provenance_type`, `provenance_note`, `digitized_by`, `digitized_at`, `source_document`,
   `source_confidence`, `license_status` e mude `review_status` para `provided_unreviewed`;
3. NAO invente geometria; deixe `missing` o que nao tiver dado real.

## 6. Como evitar erro metodologico
- Um ponto (centroide) NAO e boundary de patch. Boundary precisa ser poligono.
- Um ponto de evento (ex.: ponto CPRM) e `observed_event_point_anchor`, NAO um overlay.
- Geometria de contexto/risco NAO promove C4.
- CRS e obrigatorio; CRS desconhecido bloqueia.
- Geometria nunca e inventada; ausencia vira blocker, nunca negativo.

## 7. Por que ponto nao serve como boundary
Um overlay `patch ∩ evento` precisa de area. Um ponto tem area zero; usa-lo como boundary
fabricaria uma area falsa. Por isso `allow_point_as_patch_boundary=false`.

## 8. Por que CRS e obrigatorio
Sem CRS nao da para reprojetar nem calcular area/intersecao de forma confiavel. CRS aceitos:
EPSG:4326, EPSG:3857, EPSG:31982, EPSG:31983.

## 9. Fluxo depois do preenchimento
1. preencher `datasets/v2aw_patch_geometry_sources_template.csv`;
2. salvar como `datasets/v2av_patch_geometry_sources.csv` (mapeando `linked_patch_id` -> `patch_id`,
   `source_type`/`geometry_value`/`crs` iguais) ou alimentar o motor v2av conforme previsto;
3. rodar a v2av (`python scripts/run_v2av_patch_boundary_geometry_builder.py`) para gerar os GeoJSON;
4. apontar os GeoJSON gerados no manifesto da v2au (`datasets/v2au_geometry_sources.csv`) como
   `patch_boundary`, junto com a geometria observada de evento (poligono digitalizado);
5. rodar a v2au (`python scripts/run_v2au_patch_event_overlay_geometry.py`);
6. revisar o C4 candidate manualmente.

## 10. Isto NAO cria label automaticamente
Nenhum passo acima cria label operacional, ground truth final ou treina modelo. O resultado
maximo de um overlay confirmado e `C4_CANDIDATE_REQUIRES_HUMAN_REVIEW`, sempre sob revisao humana.
"""
    write_text(output_doc_path, text)


def write_examples(examples_dir):
    write_text(os.path.join(examples_dir, "example_patch_boundary_bbox.csv"),
               "geometry_source_id,linked_patch_id,region,city,priority_rank,source_type,"
               "geometry_value,geometry_path,crs,provenance_type,provenance_note,digitized_by,"
               "digitized_at,source_document,source_document_page,source_confidence,license_status,"
               "review_status,notes\n"
               "PGS_EXAMPLE_BBOX,EXAMPLE_PATCH_0001,ExampleCity,ExampleCity,1,bbox,"
               "\"-34.95,-8.10,-34.90,-8.05\",,EPSG:4326,sentinel_footprint,"
               "Synthetic example bbox; not real data,example,2026-01-01,example_doc,1,EXAMPLE,"
               "example_license,provided_unreviewed,SYNTHETIC EXAMPLE - do not use as real data\n")
    write_text(os.path.join(examples_dir, "example_patch_boundary_wkt.csv"),
               "geometry_source_id,linked_patch_id,region,city,priority_rank,source_type,"
               "geometry_value,geometry_path,crs,provenance_type,provenance_note,digitized_by,"
               "digitized_at,source_document,source_document_page,source_confidence,license_status,"
               "review_status,notes\n"
               "PGS_EXAMPLE_WKT,EXAMPLE_PATCH_0002,ExampleCity,ExampleCity,1,wkt,"
               "\"POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))\",,EPSG:3857,manual_digitization,"
               "Synthetic example WKT; not real data,example,2026-01-01,example_doc,1,EXAMPLE,"
               "example_license,provided_unreviewed,SYNTHETIC EXAMPLE - do not use as real data\n")
    write_text(os.path.join(examples_dir, "example_patch_boundary_geojson.geojson"),
               json.dumps({
                   "type": "Feature",
                   "properties": {"crs": "EPSG:4326", "example": True,
                                  "note": "SYNTHETIC EXAMPLE - not real patch geometry"},
                   "geometry": {"type": "Polygon",
                                "coordinates": [[[-34.95, -8.10], [-34.90, -8.10],
                                                 [-34.90, -8.05], [-34.95, -8.05], [-34.95, -8.10]]]},
               }, indent=2) + "\n")
    write_text(os.path.join(examples_dir, "example_event_geometry_wkt.csv"),
               "event_geometry_source_id,linked_event_id,region,city,hazard_type,source_type,"
               "geometry_value,geometry_path,crs,event_geometry_role,source_id,source_name,"
               "provenance_type,provenance_note,digitized_by,digitized_at,source_document,"
               "source_document_page,source_confidence,license_status,review_status,notes\n"
               "EGS_EXAMPLE_WKT,EXAMPLE_EVENT_0001,ExampleCity,ExampleCity,urban_flood,wkt,"
               "\"POLYGON((10 10, 60 10, 60 60, 10 60, 10 10))\",,EPSG:3857,observed_event_polygon,"
               "EXAMPLE_SOURCE,Example product,manual_digitization,"
               "Synthetic example observed event polygon; not real data,example,2026-01-01,"
               "example_doc,1,EXAMPLE,example_license,provided_unreviewed,"
               "SYNTHETIC EXAMPLE - do not use as real data\n")
    write_text(os.path.join(examples_dir, "README.md"),
               "# v2aw synthetic geometry intake examples\n\n"
               "These files are **synthetic** and exist only for documentation and tests.\n\n"
               "- They use placeholder ids (`EXAMPLE_PATCH_*`, `EXAMPLE_EVENT_*`) and never a real "
               "`patch_id` or `event_id`.\n"
               "- Do **not** mix them with real data and do **not** feed them to v2av/v2au as if real.\n"
               "- They show the accepted formats: bbox, WKT and GeoJSON.\n\n"
               "Nothing here is a label, ground truth or training target.\n")


# --------------------------------------------------------------------------- #
# Report / summary / log.
# --------------------------------------------------------------------------- #


def build_summary(inputs_found, outputs_written, written, config):
    validation = written[OUT_VALIDATION]
    patch_sources = [v for v in validation if v["geometry_kind"] == "patch_boundary"]
    event_sources = [v for v in validation if v["geometry_kind"] == "event_geometry"]
    readiness = written[OUT_READINESS]

    def count_block(reason):
        return sum(1 for v in validation if v["blocking_reason"] == reason)

    return {
        "stage": STAGE, "status": "OK_WITH_EXPECTED_BLOCKERS",
        "priority_region": clean(config["priority_region"]),
        "total_priority_patches": len(written[OUT_PATCH_TEMPLATE]),
        "patch_sources_provided": len(patch_sources),
        "patch_sources_valid": sum(1 for v in patch_sources if v["can_be_used_by_v2av"] == "true"),
        "event_sources_provided": len(event_sources),
        "event_sources_valid": sum(1 for v in event_sources if v["can_be_used_by_v2au"] == "true"),
        "ready_for_v2av_count": sum(1 for r in readiness if r["ready_for_v2av"] == "true"),
        "ready_for_v2au_count": sum(1 for r in readiness if r["ready_for_v2au_overlay"] == "true"),
        "blocked_missing_geometry_count": count_block("BLOCKED_MISSING_GEOMETRY"),
        "blocked_unknown_crs_count": count_block("BLOCKED_UNKNOWN_CRS"),
        "blocked_invalid_geometry_count": count_block("BLOCKED_INVALID_GEOMETRY"),
        "point_anchor_not_overlay_count": count_block("POINT_ANCHOR_NOT_OVERLAY"),
        "event_point_anchors_seeded": sum(
            1 for v in event_sources if v["geometry_role"] == "observed_event_point_anchor"),
        "total_recife_readiness_rows": len(readiness),
        "can_train_model": False, "can_create_operational_labels": False,
        "methodological_status": METHODOLOGICAL_STATUS,
        "inputs_found": inputs_found, "outputs_written": outputs_written,
    }


def build_report(summary):
    return f"""# v2aw - Geometry Source Intake + Recife Patch Boundary Filling Scaffold

## 1. Objetivo
Criar o canal auditavel para receber, validar, normalizar e auditar geometrias reais de
boundary de patch (e geometrias observadas de evento), comecando pelos {summary['total_priority_patches']}
patches Recife P1 da fila da v2av. A etapa nao inventa geometria; cria templates, validacao,
prontidao e instrucoes para inserir dados reais sem contaminar o projeto.

## 2. Entradas lidas
{os.linesep.join('- `' + x + '`' for x in summary['inputs_found']) or '- (none)'}

## 3. Arquivos criados
{os.linesep.join('- `' + x + '`' for x in summary['outputs_written'])}

## 4. Contagens
- Total de patches Recife P1 (template): **{summary['total_priority_patches']}**
- Fontes de patch fornecidas: **{summary['patch_sources_provided']}**
- Fontes de patch validas (prontas p/ v2av): **{summary['patch_sources_valid']}**
- Fontes de evento fornecidas/auditadas: **{summary['event_sources_provided']}**
- Fontes de evento validas como poligono (prontas p/ v2au): **{summary['event_sources_valid']}**
- Pontos-ancora de evento (CPRM) auditados: **{summary['event_point_anchors_seeded']}** (ancora, nao overlay)
- Patches prontos para v2av: **{summary['ready_for_v2av_count']}**
- Patches prontos para overlay v2au: **{summary['ready_for_v2au_count']}**

## 5. Principais blockers
- Geometria ausente: **{summary['blocked_missing_geometry_count']}**
- CRS desconhecido: **{summary['blocked_unknown_crs_count']}**
- Geometria invalida: **{summary['blocked_invalid_geometry_count']}**
- Ponto-ancora (nao overlay): **{summary['point_anchor_not_overlay_count']}**

## 6. O que precisa ser preenchido manualmente
Para cada patch Recife P1 em `datasets/v2aw_patch_geometry_sources_template.csv`:
`source_type` (bbox/wkt/geojson_inline/geojson_file), `geometry_value` ou `geometry_path`,
`crs` (obrigatorio), `provenance_type`, `provenance_note`, `digitized_by`, `digitized_at`,
`source_document`, `source_confidence`, `license_status`, `review_status`.
Instrucoes completas em `docs/v2aw_geometry_source_intake_instructions.md`.

## 7. Confirmacoes metodologicas explicitas
- Nenhum label operacional/binario foi criado (`can_create_operational_labels=false`).
- Nenhum modelo foi treinado (`can_train_model=false`).
- Nenhum ground truth final foi declarado; ausencia de geometria virou blocker, nunca negativo.
- Geometria nunca foi inventada; ponto nunca virou boundary; ponto de evento permaneceu ancora.
- v2at/v2au/v2av nao foram sobrescritos.

## 8. Interpretacao metodologica
{summary['methodological_status']}. Como ainda nao ha geometrias reais fornecidas, todos os
patches Recife P1 ficam bloqueados por geometria ausente. Isso e correto: a v2aw cria o canal
de entrada auditavel; ela nao resolve a falta de dado externo, mas garante que esse dado entre
de forma rastreavel, sem fabricar geometria nem criar label/ground truth/treino.
"""


def build_supplement(summary):
    return f"""# v2aw artifact index supplement

Additive supplement to `final_delivery_artifact_index.md`. Nothing existing was removed or
rewritten; only v2aw artefacts were added. v2at/v2au/v2av were NOT overwritten.

| Artifact | Path | Function |
|---|---|---|
| Patch geometry source template | `datasets/{OUT_PATCH_TEMPLATE}` | Fillable intake for {summary['total_priority_patches']} Recife P1 patch boundaries. |
| Event geometry source template | `datasets/{OUT_EVENT_TEMPLATE}` | Fillable intake for observed event geometries (+ CPRM point anchors). |
| Geometry source validation registry | `datasets/{OUT_VALIDATION}` | Validates provided/existing geometry sources. |
| Recife P1 readiness | `datasets/{OUT_READINESS}` | Per-patch readiness for v2av/v2au. |
| Intake instructions | `docs/v2aw_geometry_source_intake_instructions.md` | How to fill the templates safely. |
| Synthetic examples | `datasets/{EXAMPLES_SUBDIR.replace(os.sep, '/')}/` | Synthetic format examples (no real ids). |
| Report | `outputs_public/{REPORT_REL.replace(os.sep, '/')}` | v2aw methodological report. |
| Summary | `outputs_public/{SUMMARY_REL.replace(os.sep, '/')}` | v2aw machine-readable summary. |

Methodological status: **{summary['methodological_status']}**
(`can_train_model=false`, `can_create_operational_labels=false`).
"""


def log_lines(summary, errors):
    lines = [
        f"[{STAGE}] Geometry Source Intake + Recife Patch Boundary Filling Scaffold",
        f"[{STAGE}] inputs_found={len(summary['inputs_found'])} outputs_written={len(summary['outputs_written'])}",
        f"[{STAGE}] priority_region={summary['priority_region']} "
        f"priority_patches={summary['total_priority_patches']}",
        f"[{STAGE}] patch_sources_provided={summary['patch_sources_provided']} "
        f"patch_sources_valid={summary['patch_sources_valid']} "
        f"event_sources_provided={summary['event_sources_provided']} "
        f"event_sources_valid={summary['event_sources_valid']}",
        f"[{STAGE}] ready_for_v2av={summary['ready_for_v2av_count']} "
        f"ready_for_v2au={summary['ready_for_v2au_count']} "
        f"point_anchors={summary['event_point_anchors_seeded']}",
        f"[{STAGE}] can_train_model={summary['can_train_model']} "
        f"can_create_operational_labels={summary['can_create_operational_labels']}",
        f"[{STAGE}] methodological_status={summary['methodological_status']}",
        f"[{STAGE}] structural_errors={len(errors)}",
        f"[{STAGE}] status={'OK' if not errors else 'STRUCTURAL_ERROR'}",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #


def load_config(config_dir):
    path = os.path.join(config_dir, CONFIG_NAME)
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                loaded = json.load(handle)
            for key, value in loaded.items():
                config[key] = value
        except (json.JSONDecodeError, OSError):
            pass
    config["accepted_crs"] = [normalise_crs(c) for c in config.get("accepted_crs", [])]
    return config


def load_inputs(dataset_dir):
    found = []
    data = {}
    mapping = {"recovery_queue": IN_RECOVERY_QUEUE, "packages": IN_PACKAGES,
               "ground_events": IN_GROUND_EVENTS, "provided_patch": IN_PROVIDED_PATCH,
               "provided_event": IN_PROVIDED_EVENT}
    for key, rel in mapping.items():
        rows = load_csv(os.path.join(dataset_dir, rel))
        data[key] = rows
        if rows:
            found.append(rel)
    return data, found


def run(dataset_dir=None, output_dir=None, config_dir=None):
    """Runs the full engine. Returns (exit_code, summary)."""
    env_dataset, env_output, env_config = resolve_dirs()
    dataset_dir = dataset_dir or env_dataset
    output_dir = output_dir or env_output
    config_dir = config_dir or env_config

    config = load_config(config_dir)
    inputs, inputs_found = load_inputs(dataset_dir)

    patches = priority_patches(inputs, config)
    events = package_events(inputs)
    anchors = cprm_point_anchors(inputs)
    patch_event_map = patch_events(inputs)

    patch_template = build_patch_template(patches)
    event_template = build_event_template(events, anchors)
    validation = build_validation_registry(inputs, event_template, config)
    readiness = build_readiness(patches, validation, patch_event_map)

    written = {
        OUT_PATCH_TEMPLATE: patch_template, OUT_EVENT_TEMPLATE: event_template,
        OUT_VALIDATION: validation, OUT_READINESS: readiness,
    }

    errors = validate_outputs(written, len(patches))
    if errors:
        for err in errors:
            sys.stderr.write(f"[{STAGE}] STRUCTURAL ERROR: {err}\n")
        return 3, None

    outputs_written = []
    for name in (OUT_PATCH_TEMPLATE, OUT_EVENT_TEMPLATE, OUT_VALIDATION, OUT_READINESS):
        write_csv(os.path.join(dataset_dir, name), COLUMNS[name], written[name])
        outputs_written.append(f"datasets/{name}")

    # Docs + synthetic examples (deterministic, static).
    write_instructions(os.path.join(PROJECT_ROOT, DOC_REL))
    examples_dir = os.path.join(dataset_dir, EXAMPLES_SUBDIR)
    write_examples(examples_dir)
    outputs_written.append(f"docs/{os.path.basename(DOC_REL)}")
    outputs_written.append(f"datasets/{EXAMPLES_SUBDIR.replace(os.sep, '/')}/")

    summary = build_summary(inputs_found, outputs_written, written, config)
    summary["outputs_written"] = outputs_written + [
        f"outputs_public/{REPORT_REL.replace(os.sep, '/')}",
        f"outputs_public/{SUMMARY_REL.replace(os.sep, '/')}",
        f"outputs_public/{LOG_REL.replace(os.sep, '/')}",
        f"outputs_public/{SUPPLEMENT_REL.replace(os.sep, '/')}",
    ]

    write_text(os.path.join(output_dir, REPORT_REL), build_report(summary))
    write_text(os.path.join(output_dir, SUMMARY_REL), json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_text(os.path.join(output_dir, SUPPLEMENT_REL), build_supplement(summary))
    write_text(os.path.join(output_dir, LOG_REL), log_lines(summary, errors))

    sys.stdout.write(log_lines(summary, errors))
    return 0, summary


def main(_argv=None):
    code, _ = run()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
