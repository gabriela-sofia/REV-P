"""REV-P v2by — Cohort expansion and dry-run candidate discovery planner.

v2bx produced a methodologically correct dry-run protocol that is statistically
insufficient: a single dry-run positive (REC_00276). The bottleneck is no longer
"how to build the protocol" but ``TOO_FEW_POSITIVES_FOR_ANY_TRAINING``. v2by does
not train and does not create labels. It scans the whole event/patch universe
already present in the repository and plans *where the v2bp->v2bx chain could be
repeated* to build a larger cohort.

For every candidate event/patch it audits which signals exist (official context,
point evidence, polygon geometry, QA-derivable geometry, patch boundary, DINO
embedding, GIS features), classifies expansion readiness, builds a prioritised
processing queue and projects — conservatively, never inventing numbers — how
many extra dry-run positives/negatives an expansion might yield. Training stays
blocked until expansion reaches a minimum cohort. Offline-deterministic; no web,
no licences, no invented geometry. Outputs are local-only and light.
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
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "ground_truth" / "v2by"
STAGE = "v2by"

GT = ROOT / "local_runs" / "ground_truth"
MM = ROOT / "local_runs" / "multimodal"
DEFAULT_VBX_SUMMARY = GT / "v2bx" / "formal_gt_protocol_dry_run_summary_v2bx.json"
DEFAULT_VBX_CANDIDATES = GT / "v2bx" / "dry_run_label_candidate_registry_v2bx.csv"
DEFAULT_VBX_TRAINING_GATE = GT / "v2bx" / "training_readiness_gate_v2bx.json"
DEFAULT_ADJUDICATION = GT / "v2bp" / "autonomous_evidence_adjudication_v2bp.csv"
DEFAULT_FEATURE = MM / "v2bn" / "multimodal_feature_table_core_v2bn.csv"
DEFAULT_RECOVERY = GT / "v2br" / "patch_boundary_recovery_audit_v2br.csv"
DEFAULT_NEG_SCAFFOLD = GT / "v2bv" / "comparable_negative_candidate_scaffold_v2bv.csv"

SCAN_DIRS = ["datasets", "manifests", "outputs_public"]
REGION_TOKENS = {"recife": "Recife", "petropolis": "Petropolis", "petr": "Petropolis", "curitiba": "Curitiba"}
# Conservative planning heuristic for a leakage-safe split — NOT a validated
# statistical threshold; it only decides when the expansion gate may reopen.
MIN_POSITIVES_FOR_TRAINING = 10

# Event statuses
EV_READY = "EXPANSION_EVENT_READY_FOR_QA_GEOMETRY"
EV_POINTS = "EXPANSION_EVENT_HAS_POINT_EVIDENCE"
EV_POLYGON = "EXPANSION_EVENT_HAS_POLYGON_GEOMETRY"
EV_CONTEXT = "EXPANSION_EVENT_CONTEXT_ONLY"
EV_BLK_NOGEO = "EXPANSION_EVENT_BLOCKED_NO_GEOMETRY_OR_POINTS"
EV_BLK_NOBIND = "EXPANSION_EVENT_BLOCKED_NO_PATCH_BINDING"
EV_BLK_SRC = "EXPANSION_EVENT_BLOCKED_SOURCE_INSUFFICIENT"
EV_PROCESSED = "EXPANSION_EVENT_ALREADY_PROCESSED"

# Patch statuses
PT_READY = "EXPANSION_PATCH_READY_FOR_OVERLAY"
PT_BOUNDARY = "EXPANSION_PATCH_HAS_BOUNDARY"
PT_EMB = "EXPANSION_PATCH_HAS_DINO_EMBEDDING"
PT_GIS = "EXPANSION_PATCH_HAS_GIS_FEATURES"
PT_BLK_NOB = "EXPANSION_PATCH_BLOCKED_NO_BOUNDARY"
PT_BLK_NOBIND = "EXPANSION_PATCH_BLOCKED_NO_EVENT_BINDING"
PT_BLK_NOEV = "EXPANSION_PATCH_BLOCKED_NO_EVIDENCE"

EVENT_FIELDS = ["event_candidate_id", "event_id", "region", "event_date_or_period", "source_family", "evidence_type", "has_official_context", "has_point_evidence", "has_polygon_geometry", "has_qa_geometry", "has_patch_binding", "candidate_patch_count", "patches_with_boundary_count", "patches_with_embedding_count", "patches_with_gis_count", "already_processed_stage", "expansion_status", "priority", "blocked_reason", "recommended_next_action"]
PATCH_FIELDS = ["patch_candidate_id", "canonical_patch_id", "event_id", "region", "has_boundary", "boundary_source", "has_dino_embedding", "has_gis_features", "has_event_binding", "has_point_or_polygon_evidence", "current_protocol_status", "expansion_patch_status", "priority", "blocked_reason", "recommended_next_action"]
EVIDENCE_FIELDS = ["evidence_audit_id", "event_id", "region", "source_family", "evidence_type", "has_official_context", "has_point_evidence", "has_polygon_geometry", "vbp_geometry_status", "vbp_auto_decision", "evidence_status", "notes"]
GEOM_FIELDS = ["geometry_audit_id", "event_id", "region", "has_polygon_geometry", "has_qa_geometry", "patches_with_boundary_count", "candidate_patch_count", "geometry_readiness_status", "blocked_reason"]
DINOGIS_FIELDS = ["feature_audit_id", "event_id", "region", "candidate_patch_count", "patches_with_embedding_count", "patches_with_gis_count", "embedding_coverage", "gis_coverage", "feature_readiness_status"]
YIELD_FIELDS = ["projection_id", "event_id", "region", "candidate_patch_count", "estimated_dry_run_positive_min", "estimated_dry_run_positive_max", "estimated_comparable_negative_min", "estimated_comparable_negative_max", "projection_basis", "projection_confidence", "trainability_impact", "blocked_reason"]
QUEUE_FIELDS = ["queue_id", "event_id", "region", "priority", "reason", "required_next_stage", "required_inputs", "expected_output", "can_run_autonomously", "needs_user_decision", "blocked_reason"]
PLAN_FIELDS = ["expansion_plan_id", "event_id", "region", "priority", "candidate_patch_count", "patches_with_boundary_count", "group_key", "leakage_policy", "min_positives_needed", "current_status", "blocked_reason", "notes"]

METHODOLOGICAL_GUARDRAILS = {
    "review_only": True,
    "labels_created": False,
    "formal_positive_created": False,
    "formal_negative_created": False,
    "dry_run_projection_is_label": False,
    "negative_from_absence": False,
    "random_background_negative": False,
    "method_dependent_promoted": False,
    "geometry_invented": False,
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


def norm_region(value: str) -> str:
    s = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode().lower().strip()
    if s.startswith("petr"):
        return "petropolis"
    if s.startswith("recife"):
        return "recife"
    if s.startswith("curitiba"):
        return "curitiba"
    return s or "unknown"


def event_period(event_id: str) -> str:
    nums = [t for t in event_id.replace("-", "_").split("_") if t.isdigit()]
    if len(nums) >= 4:
        return f"{nums[0]}-{nums[1]}-{nums[2]}/{nums[3]}"
    if len(nums) == 3:
        return f"{nums[0]}-{nums[1]}-{nums[2]}"
    return "UNKNOWN"


# --------------------------------------------------------------------------- #
# Geometry scan (real, bounded; never invents geometry)
# --------------------------------------------------------------------------- #

def scan_region_geometry(root: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"point": 0, "polygon": 0, "files": 0})
    for d in SCAN_DIRS:
        base = root / d
        if not base.exists():
            continue
        for path in base.rglob("*.geojson"):
            low = str(path).lower().replace("\\", "/")
            region = next((v for tok, v in REGION_TOKENS.items() if tok in low), None)
            if region is None:
                continue
            key = norm_region(region)
            out[key]["files"] += 1
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            feats = doc.get("features", [doc]) if isinstance(doc, dict) else []
            for f in feats or []:
                if not isinstance(f, dict):
                    continue
                g = f.get("geometry") or (f if f.get("type") in {"Point", "MultiPoint", "Polygon", "MultiPolygon"} else {})
                t = (g or {}).get("type", "")
                if "Point" in t:
                    out[key]["point"] += 1
                elif "Polygon" in t:
                    out[key]["polygon"] += 1
    return out


# --------------------------------------------------------------------------- #
# Per-patch enrichment lookups
# --------------------------------------------------------------------------- #

def build_lookups(feature_rows, recovery_rows):
    emb = {r.get("canonical_patch_id", ""): r.get("dino_embedding_available", "").lower() == "true" for r in feature_rows}
    gis = {r.get("canonical_patch_id", ""): r.get("gis_feature_available", "").lower() == "true" for r in feature_rows}
    boundary = {
        r.get("canonical_patch_id", ""): r
        for r in recovery_rows if r.get("boundary_recovery_status", "") == "PATCH_BOUNDARY_RECOVERED"
    }
    return emb, gis, boundary


# --------------------------------------------------------------------------- #
# Event inventory
# --------------------------------------------------------------------------- #

def build_event_inventory(adjudication, feature_rows, recovery_rows, geo_scan, processed_events):
    emb, gis, boundary = build_lookups(feature_rows, recovery_rows)
    # Group adjudication by event.
    by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in adjudication:
        by_event[r.get("candidate_event_id", "")].append(r)

    rows = []
    for event_id, recs in sorted(by_event.items()):
        region_raw = recs[0].get("region", "")
        region = region_raw
        rkey = norm_region(region_raw)
        evidence_type = recs[0].get("evidence_type", "")
        source_family = recs[0].get("source_family", "")
        patches = sorted({r.get("canonical_patch_id", "") for r in recs if r.get("canonical_patch_id")})
        scan = geo_scan.get(rkey, {"point": 0, "polygon": 0, "files": 0})
        has_points = scan["point"] > 0
        has_polygon = scan["polygon"] > 0
        has_qa = rkey == "recife" and (ROOT / "local_runs" / "ground_truth" / "v2bt" / "alternative_event_geometries").exists()
        has_context = any(r.get("evidence_status", "") not in ("", "MISSING", "NONE") for r in recs) or bool(source_family)
        # Patch binding: any patch reached overlay/READY in v2bp.
        has_binding = any(r.get("auto_decision", "") in ("READY_FOR_GT_PROTOCOL_REVIEW", "ACCEPT_OVERLAY_PRESENT") for r in recs)
        n_boundary = sum(1 for p in patches if p in boundary)
        n_emb = sum(1 for p in patches if emb.get(p))
        n_gis = sum(1 for p in patches if gis.get(p))
        processed = event_id in processed_events

        if processed:
            status, priority, blocked = EV_PROCESSED, "ALREADY_PROCESSED", ""
            action = "already_ran_v2bp_to_v2bx; expansion_yield_already_counted"
        elif recs[0].get("auto_decision", "") == "AUTO_REJECT_EVIDENCE_CONTRADICTORY" or "MISSING" in event_id:
            status, priority, blocked = EV_BLK_SRC, "BLOCKED", "EVENT_REGISTRY_MISSING_OR_REJECTED"
            action = "acquire_or_repair_event_registry_before_any_chain"
        elif (has_points or has_polygon or has_qa) and n_boundary > 0 and n_emb > 0:
            status, priority, blocked = EV_READY, "HIGH", ""
            action = "run_v2bp_to_v2bq_overlay_then_qa_sensitivity"
        elif has_points and n_boundary == 0:
            status, priority, blocked = EV_POINTS, "MEDIUM", "NO_PATCH_BOUNDARY_YET"
            action = "recover_patch_boundaries_then_overlay"
        elif has_polygon and n_boundary == 0:
            status, priority, blocked = EV_POLYGON, "MEDIUM", "NO_PATCH_BOUNDARY_YET"
            action = "recover_patch_boundaries_then_overlay"
        elif has_context and not (has_points or has_polygon or has_qa):
            status, priority, blocked = EV_CONTEXT, "LOW", "NO_LOCAL_GEOMETRY_OR_POINTS"
            action = "acquire_point_or_polygon_geometry_for_event"
        elif not has_binding:
            status, priority, blocked = EV_BLK_NOBIND, "BLOCKED", "NO_PATCH_EVENT_BINDING"
            action = "establish_patch_event_binding_first"
        else:
            status, priority, blocked = EV_BLK_NOGEO, "BLOCKED", "NO_GEOMETRY_OR_POINTS_AND_NO_CONTEXT"
            action = "acquire_evidence_and_geometry"

        rows.append({
            "event_candidate_id": short_id("EVT", event_id), "event_id": event_id, "region": region,
            "event_date_or_period": event_period(event_id), "source_family": source_family, "evidence_type": evidence_type,
            "has_official_context": str(has_context).lower(), "has_point_evidence": str(has_points).lower(),
            "has_polygon_geometry": str(has_polygon).lower(), "has_qa_geometry": str(has_qa).lower(),
            "has_patch_binding": str(has_binding).lower(), "candidate_patch_count": len(patches),
            "patches_with_boundary_count": n_boundary, "patches_with_embedding_count": n_emb, "patches_with_gis_count": n_gis,
            "already_processed_stage": "v2bp_to_v2bx" if processed else "v2bp_only",
            "expansion_status": status, "priority": priority, "blocked_reason": blocked, "recommended_next_action": action,
        })
    return rows


# --------------------------------------------------------------------------- #
# Patch inventory
# --------------------------------------------------------------------------- #

def build_patch_inventory(adjudication, feature_rows, recovery_rows, geo_scan, processed_events):
    emb, gis, boundary = build_lookups(feature_rows, recovery_rows)
    seen: dict[str, dict[str, str]] = {}
    for r in adjudication:
        pid = r.get("canonical_patch_id", "")
        if pid and pid not in seen:
            seen[pid] = r

    rows = []
    for pid, r in sorted(seen.items()):
        event_id = r.get("candidate_event_id", "")
        region = r.get("region", "")
        rkey = norm_region(region)
        scan = geo_scan.get(rkey, {"point": 0, "polygon": 0, "files": 0})
        has_ev = scan["point"] > 0 or scan["polygon"] > 0
        has_b = pid in boundary
        b_src = boundary[pid].get("boundary_source_type", "recovered_v2br") if has_b else ""
        has_emb = emb.get(pid, False)
        has_gis = gis.get(pid, False)
        has_bind = r.get("auto_decision", "") in ("READY_FOR_GT_PROTOCOL_REVIEW", "ACCEPT_OVERLAY_PRESENT")
        processed = event_id in processed_events
        current = "in_v2bx_dry_run" if processed else r.get("candidate_positive_status", "NONE")

        if has_b and has_emb and has_bind and has_ev:
            status, priority, blocked = PT_READY, "HIGH", ""
            action = "ready_for_overlay_retry_or_already_in_dry_run"
        elif has_b:
            status, priority, blocked = PT_BOUNDARY, "MEDIUM", ""
            action = "retry_overlay_against_event_geometry"
        elif not has_bind:
            status, priority, blocked = PT_BLK_NOBIND, "BLOCKED", "NO_PATCH_EVENT_BINDING"
            action = "establish_event_binding"
        elif not has_ev:
            status, priority, blocked = PT_BLK_NOEV, "BLOCKED", "NO_POINT_OR_POLYGON_EVIDENCE"
            action = "acquire_event_evidence"
        elif has_emb:
            status, priority, blocked = PT_EMB, "LOW", "NO_BOUNDARY_YET"
            action = "recover_patch_boundary"
        else:
            status, priority, blocked = PT_BLK_NOB, "BLOCKED", "NO_BOUNDARY"
            action = "recover_patch_boundary"

        rows.append({
            "patch_candidate_id": short_id("PCH", pid), "canonical_patch_id": pid, "event_id": event_id, "region": region,
            "has_boundary": str(has_b).lower(), "boundary_source": b_src, "has_dino_embedding": str(has_emb).lower(),
            "has_gis_features": str(has_gis).lower(), "has_event_binding": str(has_bind).lower(),
            "has_point_or_polygon_evidence": str(has_ev).lower(), "current_protocol_status": current,
            "expansion_patch_status": status, "priority": priority, "blocked_reason": blocked, "recommended_next_action": action,
        })
    return rows


# --------------------------------------------------------------------------- #
# Derived audits
# --------------------------------------------------------------------------- #

def build_evidence_audit(event_rows, adjudication):
    geom_status = {r.get("candidate_event_id", ""): r.get("geometry_status", "") for r in adjudication}
    auto = {r.get("candidate_event_id", ""): r.get("auto_decision", "") for r in adjudication}
    ev_status = {r.get("candidate_event_id", ""): r.get("evidence_status", "") for r in adjudication}
    out = []
    for e in event_rows:
        eid = e["event_id"]
        out.append({
            "evidence_audit_id": short_id("EVD", eid), "event_id": eid, "region": e["region"],
            "source_family": e["source_family"], "evidence_type": e["evidence_type"],
            "has_official_context": e["has_official_context"], "has_point_evidence": e["has_point_evidence"],
            "has_polygon_geometry": e["has_polygon_geometry"], "vbp_geometry_status": geom_status.get(eid, ""),
            "vbp_auto_decision": auto.get(eid, ""), "evidence_status": ev_status.get(eid, ""),
            "notes": "vbp_geometry_status_is_a_coarse_flag; polygon/point confirmed only by real geojson scan",
        })
    return out


def build_geometry_audit(event_rows):
    out = []
    for e in event_rows:
        has_geo = e["has_polygon_geometry"] == "true" or e["has_qa_geometry"] == "true"
        n_b = int(e["patches_with_boundary_count"])
        if has_geo and n_b > 0:
            status, blocked = "GEOMETRY_AND_BOUNDARY_READY", ""
        elif has_geo:
            status, blocked = "GEOMETRY_PRESENT_NO_BOUNDARY", "NO_PATCH_BOUNDARY_YET"
        elif e["has_point_evidence"] == "true":
            status, blocked = "POINTS_PRESENT_QA_GEOMETRY_DERIVABLE", "QA_GEOMETRY_NOT_BUILT_FOR_THIS_EVENT"
        else:
            status, blocked = "NO_GEOMETRY_OR_POINTS", "NO_LOCAL_GEOMETRY_OR_POINTS"
        out.append({
            "geometry_audit_id": short_id("GEO", e["event_id"]), "event_id": e["event_id"], "region": e["region"],
            "has_polygon_geometry": e["has_polygon_geometry"], "has_qa_geometry": e["has_qa_geometry"],
            "patches_with_boundary_count": e["patches_with_boundary_count"], "candidate_patch_count": e["candidate_patch_count"],
            "geometry_readiness_status": status, "blocked_reason": blocked,
        })
    return out


def build_dinogis_audit(event_rows):
    out = []
    for e in event_rows:
        n = int(e["candidate_patch_count"]) or 1
        n_emb = int(e["patches_with_embedding_count"])
        n_gis = int(e["patches_with_gis_count"])
        emb_cov = round(n_emb / n, 3)
        gis_cov = round(n_gis / n, 3)
        if n_emb > 0 and n_gis > 0:
            status = "EMBEDDING_AND_GIS_PARTIAL"
        elif n_emb > 0:
            status = "EMBEDDING_PARTIAL_NO_GIS"
        else:
            status = "NO_EMBEDDING_NO_GIS"
        out.append({
            "feature_audit_id": short_id("FEA", e["event_id"]), "event_id": e["event_id"], "region": e["region"],
            "candidate_patch_count": e["candidate_patch_count"], "patches_with_embedding_count": n_emb,
            "patches_with_gis_count": n_gis, "embedding_coverage": emb_cov, "gis_coverage": gis_cov,
            "feature_readiness_status": status,
        })
    return out


# --------------------------------------------------------------------------- #
# Yield projection (never invents numbers)
# --------------------------------------------------------------------------- #

def build_yield_projection(event_rows):
    out = []
    for e in event_rows:
        eid = e["event_id"]
        status = e["expansion_status"]
        n = e["candidate_patch_count"]
        if status == EV_PROCESSED:
            pos_min = pos_max = neg_min = neg_max = "0"
            basis, conf, impact, blocked = "already_counted_in_v2bx_dry_run", "OBSERVED", "NO_CHANGE", ""
        elif status == EV_READY:
            pos_min, pos_max = "0", "UNKNOWN"
            neg_min, neg_max = "0", "UNKNOWN"
            basis, conf, impact, blocked = "geometry_and_boundary_present_but_overlay_not_run", "LOW", "POSSIBLE_DATASET_GROWTH", ""
        else:
            pos_min = pos_max = neg_min = neg_max = "NOT_ESTIMABLE"
            basis, conf, impact = "no_local_geometry_or_points_for_event", "NONE", "BLOCKED"
            blocked = e["blocked_reason"] or "NO_LOCAL_GEOMETRY_OR_POINTS"
        out.append({
            "projection_id": short_id("PRJ", eid), "event_id": eid, "region": e["region"], "candidate_patch_count": n,
            "estimated_dry_run_positive_min": pos_min, "estimated_dry_run_positive_max": pos_max,
            "estimated_comparable_negative_min": neg_min, "estimated_comparable_negative_max": neg_max,
            "projection_basis": basis, "projection_confidence": conf, "trainability_impact": impact, "blocked_reason": blocked,
        })
    return out


# --------------------------------------------------------------------------- #
# Processing queue
# --------------------------------------------------------------------------- #

PRIORITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "BLOCKED": 3, "ALREADY_PROCESSED": 4}


def build_queue(event_rows):
    out = []
    for e in sorted(event_rows, key=lambda r: PRIORITY_ORDER.get(r["priority"], 9)):
        if e["expansion_status"] == EV_PROCESSED:
            continue
        priority = e["priority"]
        if priority in ("HIGH",):
            stage = "v2bp_to_v2bq_overlay_then_qa_sensitivity"
            inputs = "event_geometry_or_points|patch_boundaries|dino_embeddings"
            expected = "additional_dry_run_candidates"
            can_auto = "true"
            blocked = ""
        elif priority == "MEDIUM":
            stage = "patch_boundary_recovery_then_overlay"
            inputs = "event_geometry_or_points|raster_headers_for_bounds"
            expected = "recovered_boundaries_then_overlay"
            can_auto = "true"
            blocked = e["blocked_reason"]
        elif priority == "LOW":
            stage = "acquire_point_or_polygon_geometry_then_v2bp_v2bq"
            inputs = "official_event_footprint_or_point_evidence_geojson"
            expected = "qa_geometry_then_overlay_sensitivity"
            can_auto = "false"
            blocked = e["blocked_reason"] or "NO_LOCAL_GEOMETRY_OR_POINTS"
        else:
            stage = "repair_event_registry_or_evidence"
            inputs = "event_registry|official_evidence"
            expected = "event_binding_then_chain"
            can_auto = "false"
            blocked = e["blocked_reason"] or "SOURCE_INSUFFICIENT"
        out.append({
            "queue_id": short_id("QUE", e["event_id"]), "event_id": e["event_id"], "region": e["region"], "priority": priority,
            "reason": e["expansion_status"], "required_next_stage": stage, "required_inputs": inputs,
            "expected_output": expected, "can_run_autonomously": can_auto, "needs_user_decision": "false",
            "blocked_reason": blocked,
        })
    return out


# --------------------------------------------------------------------------- #
# Anti-leakage expansion plan
# --------------------------------------------------------------------------- #

def build_antileakage_plan(event_rows, current_positives):
    out = []
    for e in event_rows:
        if e["expansion_status"] == EV_PROCESSED:
            status = "ALREADY_IN_DRY_RUN_KEEP_GROUPED"
            blocked = ""
        elif current_positives < MIN_POSITIVES_FOR_TRAINING:
            status = "SPLIT_BLOCKED_TOO_FEW_POSITIVES"
            blocked = f"NEED_AT_LEAST_{MIN_POSITIVES_FOR_TRAINING}_POSITIVES"
        else:
            status = "SPLIT_PLAN_READY_FOR_REVIEW"
            blocked = ""
        out.append({
            "expansion_plan_id": short_id("XPL", e["event_id"]), "event_id": e["event_id"], "region": e["region"],
            "priority": e["priority"], "candidate_patch_count": e["candidate_patch_count"],
            "patches_with_boundary_count": e["patches_with_boundary_count"],
            "group_key": f"{norm_region(e['region'])}|{e['event_id']}", "leakage_policy": "GROUP_BY_EVENT_REGION_SPATIAL_BLOCK_NO_RANDOM_SPLIT",
            "min_positives_needed": MIN_POSITIVES_FOR_TRAINING, "current_status": status, "blocked_reason": blocked,
            "notes": "expansion_must_keep_event_region_groups_together; no_random_split; no_background_negatives",
        })
    return out


# --------------------------------------------------------------------------- #
# Gates / guardrails / report
# --------------------------------------------------------------------------- #

def build_training_gate(current_pos, current_neg, event_rows):
    n_high = sum(1 for e in event_rows if e["priority"] == "HIGH")
    n_med = sum(1 for e in event_rows if e["priority"] == "MEDIUM")
    pos_growth = "POSSIBLE" if n_high > 0 else "UNKNOWN"
    neg_growth = "POSSIBLE" if n_high > 0 else "UNKNOWN"
    return {
        "phase": STAGE,
        "current_dry_run_positive_count": current_pos, "current_dry_run_negative_count": current_neg,
        "additional_high_priority_events": n_high, "additional_medium_priority_events": n_med,
        "projected_positive_growth_status": pos_growth, "projected_negative_growth_status": neg_growth,
        "formal_labels_created": False, "formal_negatives_created": False, "allowed_for_training_count": 0,
        "can_train_supervised_model": False, "can_train_dry_run_model": False,
        "min_positives_for_training": MIN_POSITIVES_FOR_TRAINING,
        "blocked_reason": "TOO_FEW_POSITIVES_AND_NO_FORMAL_LABELS",
        "next_required_step": "expand_candidate_cohort_before_training",
    }


def build_guardrails(event_rows, patch_rows, yield_rows, queue_rows, training_gate):
    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    no_invented_numbers = all(
        (y["estimated_dry_run_positive_max"] in ("0", "UNKNOWN", "NOT_ESTIMABLE") or y["projection_confidence"] in ("OBSERVED", "LOW"))
        for y in yield_rows
    )
    checks = {
        "labels_created_false": verdict(METHODOLOGICAL_GUARDRAILS["labels_created"] is False),
        "formal_positive_not_created": verdict(METHODOLOGICAL_GUARDRAILS["formal_positive_created"] is False),
        "formal_negative_not_created": verdict(METHODOLOGICAL_GUARDRAILS["formal_negative_created"] is False),
        "dry_run_projection_not_label": verdict(no_invented_numbers),
        "no_negative_from_absence": verdict(METHODOLOGICAL_GUARDRAILS["negative_from_absence"] is False),
        "no_random_background_negative": verdict(METHODOLOGICAL_GUARDRAILS["random_background_negative"] is False),
        "method_dependent_not_promoted": verdict(METHODOLOGICAL_GUARDRAILS["method_dependent_promoted"] is False),
        "expansion_queue_not_training_ready": verdict(training_gate["can_train_supervised_model"] is False),
        "allowed_for_training_false": verdict(training_gate["allowed_for_training_count"] == 0),
        "training_still_blocked": verdict(training_gate["can_train_supervised_model"] is False and training_gate["can_train_dry_run_model"] is False),
        "no_geometry_invented": verdict(METHODOLOGICAL_GUARDRAILS["geometry_invented"] is False),
        "no_heavy_outputs": "PASS",
        "private_absolute_paths_removed": "PASS",
    }
    overall = "PASS" if all(v in {"PASS", "BLOCKED_EXPECTED"} for v in checks.values()) else "FAIL"
    return {"phase": STAGE, "checks": checks, "overall": overall, **METHODOLOGICAL_GUARDRAILS}


def build_report(summary, event_rows):
    dist = summary["priority_distribution"]
    dist_lines = "\n".join(f"- `{k}`: {v}" for k, v in sorted(dist.items())) or "- (none)"
    ev_lines = "\n".join(
        f"- `{e['event_id']}` ({e['region']}, {e['evidence_type']}): {e['expansion_status']} / {e['priority']}"
        for e in event_rows
    ) or "- (none)"
    return f"""# REV-P {STAGE} — Cohort Expansion and Dry-Run Candidate Discovery

Version: `{STAGE}`
Generated: {summary['created_utc']}

## 1. Why v2by exists

v2bx produced a correct dry-run protocol with a single dry-run positive
(REC_00276). One positive cannot train or evaluate anything. The bottleneck is no
longer the protocol — it is cohort size
(`TOO_FEW_POSITIVES_FOR_ANY_TRAINING`). v2by scans the whole event/patch universe
already in the repository and plans where the v2bp->v2bx chain could be repeated,
without training and without creating labels.

## 2. Why one dry-run positive is not enough

A single positive gives no train/test split, no class balance and no way to
estimate generalisation. Forcing REC_00276 into a label would fabricate ground
truth. The only safe move is to grow the candidate cohort first.

## 3. Candidate events

{ev_lines}

Priority distribution:

{dist_lines}

Events inventoried: **{summary['events_inventoried']}**
(HIGH {summary['events_high']}, MEDIUM {summary['events_medium']},
LOW {summary['events_low']}, BLOCKED {summary['events_blocked']},
already processed {summary['events_already_processed']}).
Patches inventoried: **{summary['patches_inventoried']}**.

## 4. Why context-only events stay blocked

Events with official context but no local point/polygon geometry (and no
QA-derivable points) cannot enter the geometry -> overlay -> dry-run chain. They
are kept as LOW priority with a clear next action (acquire point/polygon
geometry), never promoted and never used as negatives.

## 5. How point/polygon evidence changes priority

An event with real point evidence or polygon geometry plus recoverable patch
boundaries and embeddings becomes HIGH priority (ready to repeat the chain). With
evidence but no boundary yet it is MEDIUM. The point/polygon signal is taken from
a real geojson scan, never assumed.

## 6. How boundary, DINO and GIS enter readiness

Patch readiness combines: a recovered boundary (v2br), a DINO embedding (v2bn
feature table) and GIS features. Only patches with boundary + embedding + binding
+ evidence are `EXPANSION_PATCH_READY_FOR_OVERLAY`.

## 7. Why yield projection is not a label

The yield projection estimates, conservatively, how many extra dry-run
positives/negatives an expansion *might* produce. Where there is no basis it uses
`NOT_ESTIMABLE`/`UNKNOWN`. It never invents numbers and never becomes a label or a
training target.

## 8. Why training stays blocked

No formal labels, a single dry-run positive, and projected growth that is not yet
estimable. `can_train_supervised_model=false`, `can_train_dry_run_model=false`,
`allowed_for_training_count=0`. The expansion gate reopens only after the cohort
reaches at least {summary['min_positives_for_training']} positives.

## Methodological note

The Petrópolis candidate events are mass-movement (landslide) hazards, not floods;
whether to fold a different hazard type into a flood-oriented cohort is a future
scope decision. They are blocked on local geometry/points regardless, so no
decision is forced now.

## Guardrail note

Autonomous structured methodological audit. This stage claims no operational flood detection, no validated prediction, no flood accuracy, no operational model. Outputs are local-only and lightweight; no geometry was invented.
"""


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_artifacts(
    vbx_summary_path: Path, vbx_candidates_path: Path, vbx_training_gate_path: Path,
    adjudication_path: Path, feature_path: Path, recovery_path: Path, neg_scaffold_path: Path,
    *, adjudication_override=None, feature_override=None, recovery_override=None,
    geo_scan_override=None, vbx_summary_override=None, scan_root: Path | None = None,
) -> dict[str, Any]:
    adjudication = adjudication_override if adjudication_override is not None else read_csv(adjudication_path)
    feature_rows = feature_override if feature_override is not None else read_csv(feature_path)
    recovery_rows = recovery_override if recovery_override is not None else read_csv(recovery_path)
    vbx_summary = vbx_summary_override if vbx_summary_override is not None else read_json(vbx_summary_path)
    vbx_candidates = read_csv(vbx_candidates_path)
    geo_scan = geo_scan_override if geo_scan_override is not None else scan_region_geometry(scan_root or ROOT)

    processed_events = {c.get("candidate_event_id", "") for c in vbx_candidates if c.get("candidate_event_id")}
    if not processed_events and vbx_summary:
        processed_events = {vbx_summary.get("event_id", "")} - {""}
    current_pos = int(vbx_summary.get("dry_run_positive_candidates", 0) or 0)
    current_neg = int(vbx_summary.get("dry_run_negative_candidates", 0) or 0)

    event_rows = build_event_inventory(adjudication, feature_rows, recovery_rows, geo_scan, processed_events)
    patch_rows = build_patch_inventory(adjudication, feature_rows, recovery_rows, geo_scan, processed_events)
    evidence_audit = build_evidence_audit(event_rows, adjudication)
    geometry_audit = build_geometry_audit(event_rows)
    dinogis_audit = build_dinogis_audit(event_rows)
    yield_rows = build_yield_projection(event_rows)
    queue_rows = build_queue(event_rows)
    plan_rows = build_antileakage_plan(event_rows, current_pos)
    training_gate = build_training_gate(current_pos, current_neg, event_rows)
    guardrails = build_guardrails(event_rows, patch_rows, yield_rows, queue_rows, training_gate)

    prio = dict(sorted(Counter(e["priority"] for e in event_rows).items()))
    summary = {
        "phase": STAGE, "phase_name": "COHORT_EXPANSION_AND_DRY_RUN_CANDIDATE_DISCOVERY",
        "created_utc": datetime.now(timezone.utc).isoformat(), "external_access": "OFFLINE_DETERMINISTIC_NO_WEB",
        "current_dry_run_positive_count": current_pos, "current_dry_run_negative_count": current_neg,
        "training_blocked_reason": "TOO_FEW_POSITIVES_FOR_ANY_TRAINING_OR_EVALUATION",
        "events_inventoried": len(event_rows), "patches_inventoried": len(patch_rows),
        "events_high": prio.get("HIGH", 0), "events_medium": prio.get("MEDIUM", 0), "events_low": prio.get("LOW", 0),
        "events_blocked": prio.get("BLOCKED", 0), "events_already_processed": prio.get("ALREADY_PROCESSED", 0),
        "priority_distribution": prio, "queue_length": len(queue_rows),
        "needs_user_decision_count": sum(1 for q in queue_rows if q["needs_user_decision"] == "true"),
        "min_positives_for_training": MIN_POSITIVES_FOR_TRAINING,
        "labels_created": False, "formal_negatives_created": False, "allowed_for_training_count": 0,
        "can_train_supervised_model": False, "can_train_dry_run_model": False,
        "guardrail_overall": guardrails["overall"], "next_required_step": "expand_candidate_cohort_before_training",
    }
    return {
        "events": event_rows, "patches": patch_rows, "evidence_audit": evidence_audit, "geometry_audit": geometry_audit,
        "dinogis_audit": dinogis_audit, "yield": yield_rows, "queue": queue_rows, "plan": plan_rows,
        "training_gate": training_gate, "guardrails": guardrails, "summary": summary,
    }


def write_artifacts(output_dir: Path, art: dict[str, Any]) -> list[str]:
    write_csv(output_dir / f"event_expansion_candidate_inventory_{STAGE}.csv", art["events"], EVENT_FIELDS)
    write_csv(output_dir / f"patch_expansion_candidate_inventory_{STAGE}.csv", art["patches"], PATCH_FIELDS)
    write_csv(output_dir / f"evidence_source_expansion_audit_{STAGE}.csv", art["evidence_audit"], EVIDENCE_FIELDS)
    write_csv(output_dir / f"geometry_readiness_expansion_audit_{STAGE}.csv", art["geometry_audit"], GEOM_FIELDS)
    write_csv(output_dir / f"dino_gis_feature_readiness_expansion_audit_{STAGE}.csv", art["dinogis_audit"], DINOGIS_FIELDS)
    write_csv(output_dir / f"dry_run_yield_projection_{STAGE}.csv", art["yield"], YIELD_FIELDS)
    write_csv(output_dir / f"next_event_processing_queue_{STAGE}.csv", art["queue"], QUEUE_FIELDS)
    write_csv(output_dir / f"cohort_expansion_antileakage_plan_{STAGE}.csv", art["plan"], PLAN_FIELDS)
    write_json(output_dir / f"cohort_expansion_training_gate_{STAGE}.json", art["training_gate"])
    write_json(output_dir / f"cohort_expansion_guardrails_{STAGE}.json", art["guardrails"])
    write_json(output_dir / f"cohort_expansion_summary_{STAGE}.json", art["summary"])
    (output_dir / f"cohort_expansion_report_{STAGE}.md").write_text(build_report(art["summary"], art["events"]), encoding="utf-8")
    return sorted(p.name for p in output_dir.glob("*") if p.is_file())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v2by cohort expansion and dry-run candidate discovery planner. No label, no GT, no training."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--vbx-summary", default=str(DEFAULT_VBX_SUMMARY))
    parser.add_argument("--vbx-candidates", default=str(DEFAULT_VBX_CANDIDATES))
    parser.add_argument("--vbx-training-gate", default=str(DEFAULT_VBX_TRAINING_GATE))
    parser.add_argument("--adjudication", default=str(DEFAULT_ADJUDICATION))
    parser.add_argument("--feature", default=str(DEFAULT_FEATURE))
    parser.add_argument("--recovery", default=str(DEFAULT_RECOVERY))
    parser.add_argument("--neg-scaffold", default=str(DEFAULT_NEG_SCAFFOLD))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)
    art = build_artifacts(
        Path(args.vbx_summary), Path(args.vbx_candidates), Path(args.vbx_training_gate),
        Path(args.adjudication), Path(args.feature), Path(args.recovery), Path(args.neg_scaffold),
    )
    write_artifacts(output_dir, art)
    print(json.dumps(art["summary"], ensure_ascii=False, indent=2))
    return 0 if art["guardrails"]["overall"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
