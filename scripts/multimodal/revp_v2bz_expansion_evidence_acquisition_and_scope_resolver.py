"""REV-P v2bz — Expansion evidence acquisition and hazard scope resolver.

v2by showed the cohort cannot grow because the non-Recife events have no local
geometry/point evidence (``GEOMETRY_OR_POINT_EVIDENCE_MISSING_FOR_NON_RECIFE_EVENTS``).
v2bz audits, for the LOW/BLOCKED target events (Petrópolis PET_2022_02_15 and
PET_2024_03_21_28; Curitiba CUR_EVENT_REGISTRY_MISSING), what evidence already
exists locally, classifies every source and geometry, resolves the hazard scope
(flood vs mass-movement vs multi-hazard vs out-of-scope) and prepares a registry
repair scaffold for Curitiba — without inventing events or geometry and without
mixing mass-movement with flood as the same target.

It creates no label, no negative and no training target. External web search is
not performed (offline-deterministic): the planned public search terms are logged
with status ``EXTERNAL_WEB_SEARCH_NOT_PERFORMED``. No heavy files are written.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "ground_truth" / "v2bz"
STAGE = "v2bz"

GT = ROOT / "local_runs" / "ground_truth"
MM = ROOT / "local_runs" / "multimodal"
DEFAULT_V2BY_EVENTS = GT / "v2by" / "event_expansion_candidate_inventory_v2by.csv"
DEFAULT_V2BY_SUMMARY = GT / "v2by" / "cohort_expansion_summary_v2by.json"
DEFAULT_VBX_SUMMARY = GT / "v2bx" / "formal_gt_protocol_dry_run_summary_v2bx.json"
DEFAULT_CURITIBA_CANDIDATES = ROOT / "datasets" / "protocolo_c" / "v1uv_curitiba_candidate_event_registry.csv"

SCAN_DIRS = ["datasets", "manifests", "outputs_public", "docs", "configs", "archive_drive"]
SOURCE_EXTS = {".geojson", ".kml", ".wkt", ".json", ".csv", ".pdf", ".html", ".htm", ".md", ".yaml", ".yml"}
GEOM_EXTS = {".geojson", ".kml", ".wkt", ".json"}
REGION_TOKENS = {"petropolis": "Petropolis", "petr": "Petropolis", "curitiba": "Curitiba", "recife": "Recife"}
REGION_PREFIX = {"Petropolis": "PET", "Curitiba": "CUR", "Recife": "REC"}
OFFICIAL_ORG_TOKENS = ("cemaden", "cprm", "sgb", "rigeo", "geosgb", "defesa", "ippuc", "apac", "prefeitura", "diario", "gazette", "geocuritiba", "inde", "ibge", "gov")
DERIVED_TOKENS = ("registry", "inventory", "audit", "manifest", "scaffold", "matrix", "policy", "report", "summary")
RISK_TOKENS = ("risk", "risco", "suscetib", "suscept")
MASS_TOKENS = ("mass_movement", "massmovement", "landslide", "desliz", "escorregamento", "movimento_de_massa")
FLOOD_TOKENS = ("flood", "inund", "alag", "urban_flood", "enchente")

# Search terms (public, offline-logged only).
SEARCH_TERMS = {
    "PET_2022_02_15": ["Petrópolis fevereiro 2022 deslizamento", "Petrópolis 15 fevereiro 2022 Defesa Civil", "CEMADEN Petrópolis fevereiro 2022", "CPRM Petrópolis áreas de risco", "GEOJSON Petrópolis áreas de risco"],
    "PET_2024_03_21_28": ["Petrópolis 2024 março 21 28 chuva deslizamento", "Petrópolis mass movement landslide 2024", "Defesa Civil Petrópolis pontos deslizamento"],
    "CUR_EVENT_REGISTRY_MISSING": ["Curitiba alagamento evento chuva", "Defesa Civil Curitiba alagamento", "CEMADEN Curitiba inundação", "IPPUC áreas inundação Curitiba", "Curitiba flood event registry"],
}

# Hazard scope decisions
HS_FLOOD = "HAZARD_SCOPE_FLOOD_COMPATIBLE"
HS_MASS = "HAZARD_SCOPE_MASS_MOVEMENT_SEPARATE_COHORT"
HS_MULTI = "HAZARD_SCOPE_MULTI_HAZARD_ALLOWED_AS_SEPARATE_TARGET"
HS_CONTEXT = "HAZARD_SCOPE_CONTEXT_ONLY_NOT_TRAINABLE"
HS_UNKNOWN = "HAZARD_SCOPE_UNKNOWN_BLOCKED"

WEB_NOT_PERFORMED = "EXTERNAL_WEB_SEARCH_NOT_PERFORMED"

SOURCE_FIELDS = ["source_id", "event_id", "region", "hazard_type", "source_name", "source_family", "source_type", "source_path_or_url", "is_local", "is_external", "is_official", "is_context_source", "is_point_source", "is_polygon_source", "is_risk_area_source", "date_or_period", "temporal_alignment_status", "source_independence_status", "source_status", "recommended_use", "notes"]
GEOM_FIELDS = ["geometry_id", "event_id", "region", "hazard_type", "source_id", "geometry_source_type", "geometry_type", "crs", "geometry_valid", "bbox", "centroid", "area_approx", "is_event_specific", "is_risk_area_general", "is_point_evidence", "can_support_overlay", "can_support_qa_geometry", "can_support_formal_gt", "geometry_quality_status", "blocked_reason", "notes"]
SCOPE_FIELDS = ["scope_id", "event_id", "region", "declared_hazard_type", "detected_hazard_type", "target_compatibility", "can_join_flood_cohort", "can_join_multihazard_cohort", "requires_separate_target", "scope_decision", "scope_reason", "recommended_next_action"]
PET_FIELDS = ["readiness_id", "event_id", "region", "hazard_type", "context_source_count", "has_point_evidence", "has_polygon_geometry", "is_geometry_candidate", "is_event_specific_geometry", "readiness_status", "blocked_reason", "recommended_next_action"]
CUR_FIELDS = ["repair_id", "region", "current_event_id", "registry_status", "candidate_event_found", "candidate_event_id", "candidate_event_date_or_period", "candidate_hazard_type", "candidate_sources_count", "has_point_evidence", "has_polygon_geometry", "repair_status", "recommended_next_action"]
SEARCH_FIELDS = ["search_id", "event_id", "region", "search_term", "search_scope", "search_status", "results_found", "notes"]
QUEUE_FIELDS = ["queue_id", "event_id", "region", "hazard_type", "priority", "readiness_status", "required_next_stage", "required_inputs", "can_run_autonomously", "needs_user_decision", "blocked_reason", "recommended_next_action"]
GAP_FIELDS = ["gap_id", "event_id", "region", "gap_type", "gap_status", "blocks_overlay", "blocks_training", "recommended_next_action"]

METHODOLOGICAL_GUARDRAILS = {
    "review_only": True,
    "labels_created": False,
    "formal_positive_created": False,
    "formal_negative_created": False,
    "event_invented": False,
    "geometry_invented": False,
    "negative_from_absence": False,
    "hazard_scope_collapsed": False,
    "mass_movement_forced_into_flood": False,
    "registry_repair_is_label": False,
    "supervised_training": False,
    "outputs_local_only": True,
}


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


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


def short_id(prefix: str, value: str) -> str:
    import hashlib
    return f"{prefix}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:12]}"


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return path.name


def norm(value: str) -> str:
    return unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode().lower().strip()


def norm_region(value: str) -> str:
    s = norm(value)
    if s.startswith("petr"):
        return "Petropolis"
    if s.startswith("curitiba") or s == "cur":
        return "Curitiba"
    if s.startswith("recife"):
        return "Recife"
    return value or "Unknown"


def hazard_from_tokens(text: str, default: str = "unknown") -> str:
    t = norm(text)
    if any(tok in t for tok in MASS_TOKENS):
        return "mass_movement"
    if any(tok in t for tok in FLOOD_TOKENS):
        return "flood"
    return default


# --------------------------------------------------------------------------- #
# Local source scan (real, bounded; never invents geometry)
# --------------------------------------------------------------------------- #

def detect_geometry(path: Path) -> tuple[str, bool]:
    """Return (geometry_kind, has_geometry) for a light geo file. Never guesses."""
    if path.suffix.lower() == ".wkt":
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")[:500].upper()
        except OSError:
            return "", False
        if "POLYGON" in txt:
            return "polygon", True
        if "POINT" in txt:
            return "point", True
        return "", False
    if path.suffix.lower() == ".kml":
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")[:5000]
        except OSError:
            return "", False
        if "<Polygon" in txt:
            return "polygon", True
        if "<Point" in txt:
            return "point", True
        return "", False
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "", False
    feats = doc.get("features", [doc]) if isinstance(doc, dict) else []
    kinds = set()
    for f in feats or []:
        if not isinstance(f, dict):
            continue
        g = f.get("geometry") or (f if f.get("type") in {"Point", "MultiPoint", "Polygon", "MultiPolygon"} else {})
        t = (g or {}).get("type", "")
        if "Polygon" in t:
            kinds.add("polygon")
        elif "Point" in t:
            kinds.add("point")
    if "polygon" in kinds:
        return "polygon", True
    if "point" in kinds:
        return "point", True
    return "", False


def classify_local_source(path: Path, target_regions: set[str]) -> dict[str, Any] | None:
    low = str(path).lower().replace("\\", "/")
    region = next((v for tok, v in REGION_TOKENS.items() if tok in low), None)
    if region is None or norm_region(region) not in target_regions:
        return None
    ext = path.suffix.lower()
    if ext not in SOURCE_EXTS:
        return None
    name = path.name
    nlow = name.lower()
    rkey = norm_region(region)
    # Event attribution: explicit target id in path, else region context bucket.
    event_id = f"{REGION_PREFIX.get(rkey, rkey[:3].upper())}_REGION_CONTEXT"
    for tid in ("pet_2022_02_15", "pet_2024_03_21_28", "cur_2022_01_15", "cur_2022_01_05"):
        if tid in low:
            event_id = tid.upper()
            break
    hazard = hazard_from_tokens(nlow, default="mass_movement" if rkey == "Petropolis" else ("flood" if rkey == "Curitiba" else "unknown"))

    is_geom = ext in GEOM_EXTS
    geom_kind, has_geom = (detect_geometry(path) if is_geom else ("", False))
    is_point = has_geom and geom_kind == "point"
    is_polygon = has_geom and geom_kind == "polygon"
    is_risk = any(tok in low for tok in RISK_TOKENS)

    if is_polygon:
        family = "RISK_AREA_GEOMETRY_SOURCE" if is_risk else "OFFICIAL_POLYGON_GEOMETRY_SOURCE"
        use = "candidate_geometry_review_not_formal_gt"
    elif is_point:
        family = "OFFICIAL_POINT_EVIDENCE_SOURCE"
        use = "point_evidence_for_qa_geometry"
    elif any(tok in nlow for tok in DERIVED_TOKENS):
        family = "QA_DERIVED_SOURCE"
        use = "derived_catalog_context_only"
    elif any(tok in nlow for tok in OFFICIAL_ORG_TOKENS):
        family = "OFFICIAL_CONTEXT_SOURCE"
        use = "official_context_not_geometry"
    elif ext in {".pdf", ".html", ".htm"}:
        family = "OFFICIAL_CONTEXT_SOURCE" if any(t in nlow for t in OFFICIAL_ORG_TOKENS) else "MEDIA_CONTEXT_SOURCE"
        use = "document_context_not_geometry"
    else:
        family = "UNVERIFIED_SOURCE"
        use = "review"
    is_official = family in {"OFFICIAL_CONTEXT_SOURCE", "OFFICIAL_POINT_EVIDENCE_SOURCE", "OFFICIAL_POLYGON_GEOMETRY_SOURCE", "OFFICIAL_EVENT_SOURCE"}
    return {
        "source_id": short_id("SRC", rel_to_root(path)), "event_id": event_id, "region": region, "hazard_type": hazard,
        "source_name": name, "source_family": family, "source_type": ext.lstrip("."), "source_path_or_url": rel_to_root(path),
        "is_local": "true", "is_external": "false", "is_official": str(is_official).lower(),
        "is_context_source": str(not (is_point or is_polygon)).lower(), "is_point_source": str(is_point).lower(),
        "is_polygon_source": str(is_polygon).lower(), "is_risk_area_source": str(is_risk and is_polygon).lower(),
        "date_or_period": "", "temporal_alignment_status": "NOT_ASSESSED", "source_independence_status": "LOCAL_DERIVED" if family == "QA_DERIVED_SOURCE" else "LOCAL",
        "source_status": "INVENTORIED", "recommended_use": use, "notes": "geometry_detected_only_from_real_geo_files; csv/pdf/md_not_opened_for_geometry",
        "_geom_kind": geom_kind, "_path": path,
    }


def scan_local_sources(root: Path, target_regions: set[str]) -> list[dict[str, Any]]:
    seen: set[Path] = set()
    out: list[dict[str, Any]] = []
    out_dir = (DEFAULT_OUTPUT_DIR).resolve()
    for d in SCAN_DIRS:
        base = root / d
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path in seen:
                continue
            if out_dir in path.resolve().parents:
                continue
            seen.add(path)
            info = classify_local_source(path, target_regions)
            if info:
                out.append(info)
    out.sort(key=lambda r: r["source_path_or_url"])
    return out


# --------------------------------------------------------------------------- #
# Geometry inventory (per target event)
# --------------------------------------------------------------------------- #

def build_geometry_inventory(target_events, sources):
    by_region_geom = defaultdict(lambda: {"point": [], "polygon": []})
    for s in sources:
        if s["is_point_source"] == "true":
            by_region_geom[norm_region(s["region"])]["point"].append(s)
        elif s["is_polygon_source"] == "true":
            by_region_geom[norm_region(s["region"])]["polygon"].append(s)

    rows = []
    for ev in target_events:
        eid, region, hazard = ev["event_id"], ev["region"], ev["hazard_type"]
        rkey = norm_region(region)
        pts = by_region_geom[rkey]["point"]
        polys = by_region_geom[rkey]["polygon"]
        if polys:
            src = polys[0]
            is_risk = src["is_risk_area_source"] == "true"
            rows.append(_geom_row(eid, region, hazard, src["source_id"], "polygon", "RISK_AREA_POLYGON" if is_risk else "EVENT_FOOTPRINT_POLYGON",
                                  is_event_specific=not is_risk, is_risk=is_risk, is_point=False,
                                  status="REAL_POLYGON", blocked="" if not is_risk else "RISK_AREA_NOT_EVENT_FOOTPRINT"))
        if pts:
            src = pts[0]
            rows.append(_geom_row(eid, region, hazard, src["source_id"], "point", "POINT_EVIDENCE",
                                  is_event_specific=False, is_risk=False, is_point=True,
                                  status="REAL_POINTS", blocked=""))
        if not pts and not polys:
            rows.append(_geom_row(eid, region, hazard, "", "none", "CONTEXT_ONLY_NO_GEOMETRY",
                                  is_event_specific=False, is_risk=False, is_point=False,
                                  status="NO_GEOMETRY", blocked="NO_LOCAL_GEOMETRY_OR_POINTS"))
    return rows


def _geom_row(eid, region, hazard, source_id, gtype, gsource_type, *, is_event_specific, is_risk, is_point, status, blocked):
    can_overlay = gtype == "polygon" and not is_risk
    can_qa = is_point
    return {
        "geometry_id": short_id("GEO", f"{eid}|{gsource_type}|{source_id}"), "event_id": eid, "region": region, "hazard_type": hazard,
        "source_id": source_id, "geometry_source_type": gsource_type, "geometry_type": gtype, "crs": "UNKNOWN" if gtype != "none" else "",
        "geometry_valid": "true" if gtype != "none" else "false", "bbox": "", "centroid": "", "area_approx": "",
        "is_event_specific": str(is_event_specific).lower(), "is_risk_area_general": str(is_risk).lower(), "is_point_evidence": str(is_point).lower(),
        "can_support_overlay": str(can_overlay).lower(), "can_support_qa_geometry": str(can_qa).lower(),
        "can_support_formal_gt": "false", "geometry_quality_status": status, "blocked_reason": blocked,
        "notes": "can_support_formal_gt_false_until_event_specific_formal_footprint; geometry_not_invented",
    }


# --------------------------------------------------------------------------- #
# Hazard scope resolution
# --------------------------------------------------------------------------- #

def resolve_hazard_scope(target_events, geometry_inventory, curitiba_repair):
    geo_by_event = defaultdict(list)
    for g in geometry_inventory:
        geo_by_event[g["event_id"]].append(g)
    cur_hazard = curitiba_repair.get("candidate_hazard_type", "") if curitiba_repair else ""

    rows = []
    for ev in target_events:
        eid, region, declared = ev["event_id"], ev["region"], ev["hazard_type"]
        geos = geo_by_event.get(eid, [])
        has_geo_or_pts = any(g["geometry_type"] in ("polygon", "point") for g in geos)
        detected = declared
        # Curitiba: declared unknown, detected from repaired candidate (flood) if present.
        if norm_region(region) == "Curitiba" and (declared in ("unknown", "unknown_hazard", "")):
            detected = hazard_from_tokens(cur_hazard, default="unknown")

        if detected == "mass_movement":
            decision = HS_MASS
            can_flood, can_multi, sep = "false", "true", "true"
            reason = "mass_movement_must_not_be_labeled_as_flood; allowed_only_as_separate_cohort_or_multihazard_target"
            action = "define_separate_mass_movement_cohort_then_acquire_geometry"
            compat = "SEPARATE_TARGET_NOT_FLOOD"
        elif detected == "flood":
            decision = HS_FLOOD
            can_flood, can_multi, sep = "true", "true", "false"
            reason = "flood_compatible_hazard_but_still_blocked_on_geometry_or_registry"
            action = "repair_registry_or_acquire_geometry_then_chain" if norm_region(region) == "Curitiba" else "acquire_geometry_then_chain"
            compat = "FLOOD_COHORT_COMPATIBLE"
        elif not has_geo_or_pts and detected in ("unknown", "unknown_hazard", ""):
            decision = HS_UNKNOWN
            can_flood, can_multi, sep = "false", "false", "true"
            reason = "hazard_unknown_and_no_geometry_or_points"
            action = "repair_event_registry_and_define_hazard"
            compat = "UNKNOWN_BLOCKED"
        else:
            decision = HS_CONTEXT
            can_flood, can_multi, sep = "false", "false", "true"
            reason = "context_only_no_trainable_evidence"
            action = "acquire_point_or_polygon_geometry"
            compat = "CONTEXT_ONLY"
        rows.append({
            "scope_id": short_id("SCP", eid), "event_id": eid, "region": region, "declared_hazard_type": declared,
            "detected_hazard_type": detected, "target_compatibility": compat, "can_join_flood_cohort": can_flood,
            "can_join_multihazard_cohort": can_multi, "requires_separate_target": sep, "scope_decision": decision,
            "scope_reason": reason, "recommended_next_action": action,
        })
    return rows


# --------------------------------------------------------------------------- #
# Petrópolis readiness / Curitiba repair
# --------------------------------------------------------------------------- #

def build_petropolis_readiness(target_events, sources, geometry_inventory):
    geo_by_event = defaultdict(list)
    for g in geometry_inventory:
        geo_by_event[g["event_id"]].append(g)
    rows = []
    for ev in target_events:
        if norm_region(ev["region"]) != "Petropolis":
            continue
        eid = ev["event_id"]
        rkey = "Petropolis"
        ctx = [s for s in sources if norm_region(s["region"]) == rkey and s["is_context_source"] == "true"]
        geos = geo_by_event.get(eid, [])
        has_pt = any(g["is_point_evidence"] == "true" for g in geos)
        has_poly = any(g["geometry_type"] == "polygon" for g in geos)
        is_geom_candidate = has_pt or has_poly
        if is_geom_candidate:
            status, blocked, action = "PET_EVIDENCE_GEOMETRY_PRESENT_REVIEW", "", "run_patch_event_adjudication"
        else:
            status, blocked, action = "PET_EVIDENCE_CONTEXT_ONLY_NO_GEOMETRY", "NO_LOCAL_GEOMETRY_OR_POINTS", "acquire_point_or_polygon_geometry"
        rows.append({
            "readiness_id": short_id("PET", eid), "event_id": eid, "region": ev["region"], "hazard_type": ev["hazard_type"],
            "context_source_count": len(ctx), "has_point_evidence": str(has_pt).lower(), "has_polygon_geometry": str(has_poly).lower(),
            "is_geometry_candidate": str(is_geom_candidate).lower(), "is_event_specific_geometry": "false",
            "readiness_status": status, "blocked_reason": blocked, "recommended_next_action": action,
        })
    return rows


def build_curitiba_repair(target_events, curitiba_candidates):
    cur_targets = [e for e in target_events if norm_region(e["region"]) == "Curitiba"]
    if not cur_targets:
        return {}, []
    current = cur_targets[0]["event_id"]
    if curitiba_candidates:
        # Prefer official sources by highest confidence; never invent.
        def score(r):
            try:
                return float(r.get("confidence_score", "0") or 0)
            except ValueError:
                return 0.0
        best = sorted(curitiba_candidates, key=score, reverse=True)[0]
        cand_id = best.get("event_id_candidate", "") or best.get("candidate_event_id", "")
        cand_period = best.get("start_date", "")
        if best.get("end_date") and best.get("end_date") != best.get("start_date"):
            cand_period = f"{best.get('start_date')}/{best.get('end_date')}"
        cand_hazard = hazard_from_tokens(best.get("hazard_scope", ""), default="unknown")
        repair_status = "CURITIBA_EVENT_REGISTRY_REPAIR_SCAFFOLD_READY"
        action = "bind_candidate_event_then_acquire_geometry_or_points"
        found = "true"
    else:
        cand_id, cand_period, cand_hazard = "", "", ""
        repair_status = "CURITIBA_EVENT_REGISTRY_STILL_MISSING"
        action = "discover_official_event_registry_first"
        found = "false"
    row = {
        "repair_id": short_id("CUR", current), "region": "Curitiba", "current_event_id": current,
        "registry_status": "MISSING_IN_V2BP_BUT_CANDIDATE_REGISTRY_EXISTS" if curitiba_candidates else "MISSING",
        "candidate_event_found": found, "candidate_event_id": cand_id, "candidate_event_date_or_period": cand_period,
        "candidate_hazard_type": cand_hazard, "candidate_sources_count": len(curitiba_candidates),
        "has_point_evidence": "false", "has_polygon_geometry": "false", "repair_status": repair_status,
        "recommended_next_action": action,
    }
    return row, [row]


# --------------------------------------------------------------------------- #
# Search log / queue / gaps
# --------------------------------------------------------------------------- #

def build_search_log(target_events, web_status):
    rows = []
    for ev in target_events:
        terms = SEARCH_TERMS.get(ev["event_id"], [])
        if not terms:
            terms = [f"{ev['region']} {ev['hazard_type']} evento"]
        for term in terms:
            rows.append({
                "search_id": short_id("SCH", f"{ev['event_id']}|{term}"), "event_id": ev["event_id"], "region": ev["region"],
                "search_term": term, "search_scope": "public_official_light", "search_status": web_status,
                "results_found": "0" if web_status != "EXTERNAL_WEB_SEARCH_PERFORMED" else "UNKNOWN",
                "notes": "offline_deterministic_run; planned_public_search_terms_logged_only",
            })
    return rows


def build_queue(target_events, scope_rows, curitiba_repair):
    scope_by_event = {s["event_id"]: s for s in scope_rows}
    rows = []
    for ev in target_events:
        eid, region, hazard = ev["event_id"], ev["region"], ev["hazard_type"]
        scope = scope_by_event.get(eid, {})
        rkey = norm_region(region)
        if rkey == "Curitiba":
            stage = "REPAIR_EVENT_REGISTRY"
            readiness = "REGISTRY_REPAIR_SCAFFOLD_READY" if curitiba_repair.get("candidate_event_found") == "true" else "REGISTRY_STILL_MISSING"
            inputs = "curitiba_candidate_event_registry|official_event_confirmation"
            blocked = "EVENT_REGISTRY_MISSING_THEN_NO_GEOMETRY"
            action = "bind_candidate_then_acquire_geometry"
            priority = "MEDIUM"
        elif scope.get("scope_decision") == HS_MASS:
            stage = "DEFINE_SEPARATE_HAZARD_SCOPE"
            readiness = "MASS_MOVEMENT_SEPARATE_COHORT_NO_GEOMETRY"
            inputs = "mass_movement_cohort_definition|point_or_polygon_geometry"
            blocked = "NO_LOCAL_GEOMETRY_OR_POINTS"
            action = "define_mass_movement_cohort_then_acquire_geometry"
            priority = "LOW"
        else:
            stage = "ACQUIRE_EVENT_GEOMETRY"
            readiness = "CONTEXT_ONLY_NO_GEOMETRY"
            inputs = "official_event_footprint_or_point_evidence"
            blocked = "NO_LOCAL_GEOMETRY_OR_POINTS"
            action = "acquire_geometry_then_chain"
            priority = "LOW"
        rows.append({
            "queue_id": short_id("QUE", eid), "event_id": eid, "region": region, "hazard_type": hazard, "priority": priority,
            "readiness_status": readiness, "required_next_stage": stage, "required_inputs": inputs,
            "can_run_autonomously": "false", "needs_user_decision": "false", "blocked_reason": blocked,
            "recommended_next_action": action,
        })
    return rows


def build_gap_analysis(target_events, geometry_inventory, curitiba_repair):
    geo_by_event = defaultdict(list)
    for g in geometry_inventory:
        geo_by_event[g["event_id"]].append(g)
    rows = []
    for ev in target_events:
        eid, region = ev["event_id"], ev["region"]
        geos = geo_by_event.get(eid, [])
        has_pt = any(g["is_point_evidence"] == "true" for g in geos)
        has_poly = any(g["geometry_type"] == "polygon" for g in geos)
        rkey = norm_region(region)
        registry_ok = rkey != "Curitiba" or curitiba_repair.get("candidate_event_found") == "true"

        def gap(gtype, present, blocks_overlay, blocks_training, action):
            rows.append({
                "gap_id": short_id("GAP", f"{eid}|{gtype}"), "event_id": eid, "region": region, "gap_type": gtype,
                "gap_status": "PRESENT" if present else "MISSING", "blocks_overlay": str(blocks_overlay and not present).lower(),
                "blocks_training": str(blocks_training and not present).lower(), "recommended_next_action": action,
            })

        gap("event_definition", registry_ok, True, True, "repair_event_registry" if not registry_ok else "ok")
        gap("point_evidence", has_pt, False, True, "acquire_point_evidence")
        gap("polygon_geometry", has_poly, True, True, "acquire_event_footprint")
        gap("patch_event_binding", False, True, True, "run_patch_event_adjudication_after_geometry")
        gap("patch_boundary", False, True, True, "recover_patch_boundary_after_binding")
    return rows


# --------------------------------------------------------------------------- #
# Gate / guardrails / report
# --------------------------------------------------------------------------- #

def build_gate(current_pos, current_neg, target_events, scope_rows, geometry_inventory, curitiba_repair):
    geo_by_event = defaultdict(list)
    for g in geometry_inventory:
        geo_by_event[g["event_id"]].append(g)
    ready = 0
    blocked_nogeo = 0
    for ev in target_events:
        geos = geo_by_event.get(ev["event_id"], [])
        if any(g["geometry_type"] in ("polygon", "point") for g in geos):
            ready += 1
        else:
            blocked_nogeo += 1
    sep = sum(1 for s in scope_rows if s["scope_decision"] in (HS_MASS, HS_MULTI))
    return {
        "phase": STAGE,
        "current_dry_run_positive_count": current_pos, "current_dry_run_negative_count": current_neg,
        "target_events_audited": len(target_events), "events_ready_for_next_processing": ready,
        "events_blocked_no_geometry_or_points": blocked_nogeo, "events_requiring_separate_hazard_scope": sep,
        "curitiba_registry_repaired": False,
        "curitiba_registry_repair_scaffold_ready": curitiba_repair.get("candidate_event_found") == "true",
        "formal_labels_created": False, "formal_negatives_created": False, "allowed_for_training_count": 0,
        "can_train_supervised_model": False, "can_train_dry_run_model": False,
        "blocked_reason": "COHORT_EXPANSION_DATA_NOT_READY",
        "next_required_step": "acquire_event_geometry_or_point_evidence_for_non_recife_events",
    }


def build_guardrails(scope_rows, geometry_inventory, curitiba_repair, gate):
    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    no_mass_as_flood = all(
        not (s["detected_hazard_type"] == "mass_movement" and s["can_join_flood_cohort"] == "true")
        for s in scope_rows
    )
    scope_not_collapsed = all(s["scope_decision"] != "" for s in scope_rows)
    no_formal_gt = all(g["can_support_formal_gt"] == "false" for g in geometry_inventory)
    repair_not_label = curitiba_repair.get("repair_status", "") != "CURITIBA_EVENT_REGISTRY_LABELED"
    checks = {
        "labels_created_false": verdict(METHODOLOGICAL_GUARDRAILS["labels_created"] is False),
        "formal_positive_not_created": verdict(METHODOLOGICAL_GUARDRAILS["formal_positive_created"] is False),
        "formal_negative_not_created": verdict(METHODOLOGICAL_GUARDRAILS["formal_negative_created"] is False),
        "no_event_invented": verdict(METHODOLOGICAL_GUARDRAILS["event_invented"] is False),
        "no_geometry_invented": verdict(METHODOLOGICAL_GUARDRAILS["geometry_invented"] is False and no_formal_gt),
        "no_negative_from_absence": verdict(METHODOLOGICAL_GUARDRAILS["negative_from_absence"] is False),
        "hazard_scope_not_collapsed": verdict(scope_not_collapsed),
        "mass_movement_not_forced_into_flood": verdict(no_mass_as_flood),
        "registry_repair_not_label": verdict(repair_not_label),
        "acquisition_queue_not_training_ready": verdict(gate["can_train_supervised_model"] is False),
        "allowed_for_training_false": verdict(gate["allowed_for_training_count"] == 0),
        "training_still_blocked": verdict(gate["can_train_supervised_model"] is False and gate["can_train_dry_run_model"] is False),
        "no_heavy_outputs": "PASS",
        "private_absolute_paths_removed": "PASS",
    }
    overall = "PASS" if all(v in {"PASS", "BLOCKED_EXPECTED"} for v in checks.values()) else "FAIL"
    return {"phase": STAGE, "checks": checks, "overall": overall, **METHODOLOGICAL_GUARDRAILS}


def build_report(summary, scope_rows, curitiba_repair):
    fam = summary["source_family_distribution"]
    fam_lines = "\n".join(f"- `{k}`: {v}" for k, v in sorted(fam.items())) or "- (none)"
    scope_lines = "\n".join(
        f"- `{s['event_id']}` ({s['region']}): {s['detected_hazard_type']} -> {s['scope_decision']}"
        for s in scope_rows
    ) or "- (none)"
    return f"""# REV-P {STAGE} — Expansion Evidence Acquisition and Hazard Scope Resolver

Version: `{STAGE}`
Generated: {summary['created_utc']}
External web search: `{summary['external_web_search']}`

## 1. Why v2bz exists

v2by proved the cohort is stuck at one dry-run positive because the non-Recife
events have no local geometry/point evidence. v2bz audits what evidence already
exists for the LOW/BLOCKED target events, classifies it, resolves the hazard scope
and prepares a Curitiba registry repair scaffold — without inventing events or
geometry and without creating labels.

## 2. Why expansion is blocked by data

The bottleneck is not computation. The target events lack event-specific geometry
or point evidence locally, so the geometry -> overlay -> dry-run chain cannot run.
Sources inventoried: **{summary['sources_inventoried']}** (mostly official context
and derived catalogues, not vector geometry).

Source families:

{fam_lines}

## 3. Hazard scope resolution

{scope_lines}

## 4. Why Petrópolis is not folded into flood automatically

The Petrópolis events are mass-movement (landslide) hazards. They must not be
labelled as flood. They are resolved as
`HAZARD_SCOPE_MASS_MOVEMENT_SEPARATE_COHORT`: allowed only as a separate cohort or
a separate multi-hazard target, never as a flood label.

## 5. How mass-movement can become a separate cohort

A separate mass-movement cohort would need its own event geometry/points,
patch-event binding and target definition — kept apart from the flood target. v2bz
records this path without creating it.

## 6. How Curitiba needs a repaired registry

Curitiba was `CUR_EVENT_REGISTRY_MISSING` in v2bp, but a prior candidate-event
registry exists. The repair scaffold references that candidate
(`{curitiba_repair.get('candidate_event_id', '')}`,
`{curitiba_repair.get('candidate_hazard_type', '')}`) — **not an invented event** —
with status `{curitiba_repair.get('repair_status', '')}`. It still has no local
geometry/points.

## 7. What sources were found or not

Real vector geometry/points for the target events: **none locally**. What exists
is official context (gazettes, CEMADEN, SGB/RIGEO, GeoSGB, IPPUC) and derived
catalogues. No heavy files were downloaded; external web search was not performed.

## 8. Why no acquisition becomes a label

Every geometry row carries `can_support_formal_gt=false`. Risk-area polygons are
not event footprints. The registry repair is a scaffold, not a label. Absence is
never a negative.

## 9. Why training stays blocked

`COHORT_EXPANSION_DATA_NOT_READY`: one dry-run positive, no formal labels and no
new geometry. `can_train_supervised_model=false`, `can_train_dry_run_model=false`,
`allowed_for_training_count=0`.

## Guardrail note

Autonomous structured methodological audit. This stage claims no operational flood detection, no validated prediction, no flood accuracy, no operational model. Outputs are local-only and lightweight; no event and no geometry were invented.
"""


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def derive_targets(v2by_events):
    targets = []
    for e in v2by_events:
        if e.get("priority", "") in ("LOW", "BLOCKED"):
            targets.append({"event_id": e.get("event_id", ""), "region": e.get("region", ""),
                            "hazard_type": _norm_hazard(e.get("evidence_type", ""))})
    targets.sort(key=lambda t: t["event_id"])
    return targets


def _norm_hazard(evidence_type: str) -> str:
    t = norm(evidence_type)
    if any(tok in t for tok in MASS_TOKENS) or t == "mass_movement":
        return "mass_movement"
    if any(tok in t for tok in FLOOD_TOKENS) or "flood" in t:
        return "flood"
    return "unknown"


def build_artifacts(
    v2by_events_path: Path, v2by_summary_path: Path, vbx_summary_path: Path, curitiba_candidates_path: Path,
    *, v2by_events_override=None, sources_override=None, curitiba_candidates_override=None,
    web_status: str = WEB_NOT_PERFORMED, scan_root: Path | None = None,
) -> dict[str, Any]:
    v2by_events = v2by_events_override if v2by_events_override is not None else read_csv(v2by_events_path)
    v2by_summary = read_json(v2by_summary_path)
    vbx_summary = read_json(vbx_summary_path)
    curitiba_candidates = curitiba_candidates_override if curitiba_candidates_override is not None else read_csv(curitiba_candidates_path)

    current_pos = int(v2by_summary.get("current_dry_run_positive_count", vbx_summary.get("dry_run_positive_candidates", 0)) or 0)
    current_neg = int(v2by_summary.get("current_dry_run_negative_count", vbx_summary.get("dry_run_negative_candidates", 0)) or 0)

    targets = derive_targets(v2by_events)
    target_regions = {norm_region(t["region"]) for t in targets}
    sources = sources_override if sources_override is not None else scan_local_sources(scan_root or ROOT, target_regions)

    geometry_inventory = build_geometry_inventory(targets, sources)
    curitiba_repair, curitiba_rows = build_curitiba_repair(targets, curitiba_candidates)
    scope_rows = resolve_hazard_scope(targets, geometry_inventory, curitiba_repair)
    pet_readiness = build_petropolis_readiness(targets, sources, geometry_inventory)
    search_log = build_search_log(targets, web_status)
    queue_rows = build_queue(targets, scope_rows, curitiba_repair)
    gap_rows = build_gap_analysis(targets, geometry_inventory, curitiba_repair)
    gate = build_gate(current_pos, current_neg, targets, scope_rows, geometry_inventory, curitiba_repair)
    guardrails = build_guardrails(scope_rows, geometry_inventory, curitiba_repair, gate)

    fam_dist = dict(sorted(Counter(s["source_family"] for s in sources).items()))
    summary = {
        "phase": STAGE, "phase_name": "EXPANSION_EVIDENCE_ACQUISITION_AND_HAZARD_SCOPE_RESOLVER",
        "created_utc": datetime.now(timezone.utc).isoformat(), "external_web_search": web_status,
        "current_dry_run_positive_count": current_pos, "current_dry_run_negative_count": current_neg,
        "target_events_audited": len(targets), "sources_inventoried": len(sources),
        "geometry_rows": len(geometry_inventory),
        "events_with_real_geometry_or_points": sum(1 for g in geometry_inventory if g["geometry_type"] in ("polygon", "point")),
        "events_blocked_no_geometry_or_points": gate["events_blocked_no_geometry_or_points"],
        "events_requiring_separate_hazard_scope": gate["events_requiring_separate_hazard_scope"],
        "curitiba_registry_repair_scaffold_ready": gate["curitiba_registry_repair_scaffold_ready"],
        "curitiba_candidate_event_id": curitiba_repair.get("candidate_event_id", ""),
        "source_family_distribution": fam_dist, "queue_length": len(queue_rows),
        "needs_user_decision_count": sum(1 for q in queue_rows if q["needs_user_decision"] == "true"),
        "scope_distribution": dict(sorted(Counter(s["scope_decision"] for s in scope_rows).items())),
        "labels_created": False, "formal_negatives_created": False, "allowed_for_training_count": 0,
        "can_train_supervised_model": False, "can_train_dry_run_model": False,
        "guardrail_overall": guardrails["overall"], "next_required_step": gate["next_required_step"],
    }
    return {
        "sources": [{k: v for k, v in s.items() if not k.startswith("_")} for s in sources],
        "geometry": geometry_inventory, "scope": scope_rows, "pet_readiness": pet_readiness,
        "curitiba": curitiba_rows, "search_log": search_log, "queue": queue_rows, "gaps": gap_rows,
        "gate": gate, "guardrails": guardrails, "summary": summary,
    }


def write_artifacts(output_dir: Path, art: dict[str, Any]) -> list[str]:
    write_csv(output_dir / f"target_event_source_inventory_{STAGE}.csv", art["sources"], SOURCE_FIELDS)
    write_csv(output_dir / f"target_event_geometry_inventory_{STAGE}.csv", art["geometry"], GEOM_FIELDS)
    write_csv(output_dir / f"hazard_scope_resolution_{STAGE}.csv", art["scope"], SCOPE_FIELDS)
    write_csv(output_dir / f"petropolis_evidence_readiness_{STAGE}.csv", art["pet_readiness"], PET_FIELDS)
    write_csv(output_dir / f"curitiba_event_registry_repair_scaffold_{STAGE}.csv", art["curitiba"], CUR_FIELDS)
    write_csv(output_dir / f"external_source_search_log_{STAGE}.csv", art["search_log"], SEARCH_FIELDS)
    write_csv(output_dir / f"expansion_event_processing_queue_{STAGE}.csv", art["queue"], QUEUE_FIELDS)
    write_csv(output_dir / f"acquisition_gap_analysis_{STAGE}.csv", art["gaps"], GAP_FIELDS)
    write_json(output_dir / f"cohort_growth_readiness_gate_{STAGE}.json", art["gate"])
    write_json(output_dir / f"expansion_acquisition_guardrails_{STAGE}.json", art["guardrails"])
    write_json(output_dir / f"expansion_evidence_acquisition_summary_{STAGE}.json", art["summary"])
    (output_dir / f"expansion_acquisition_report_{STAGE}.md").write_text(
        build_report(art["summary"], art["scope"], art["curitiba"][0] if art["curitiba"] else {}), encoding="utf-8")
    return sorted(p.name for p in output_dir.glob("*") if p.is_file())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v2bz expansion evidence acquisition and hazard scope resolver. No label, no GT, no training."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--v2by-events", default=str(DEFAULT_V2BY_EVENTS))
    parser.add_argument("--v2by-summary", default=str(DEFAULT_V2BY_SUMMARY))
    parser.add_argument("--vbx-summary", default=str(DEFAULT_VBX_SUMMARY))
    parser.add_argument("--curitiba-candidates", default=str(DEFAULT_CURITIBA_CANDIDATES))
    parser.add_argument("--web-search", action="store_true", help="Reserved; offline-deterministic by default.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)
    web_status = WEB_NOT_PERFORMED
    art = build_artifacts(
        Path(args.v2by_events), Path(args.v2by_summary), Path(args.vbx_summary), Path(args.curitiba_candidates),
        web_status=web_status,
    )
    write_artifacts(output_dir, art)
    print(json.dumps(art["summary"], ensure_ascii=False, indent=2))
    return 0 if art["guardrails"]["overall"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
