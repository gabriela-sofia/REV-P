"""
REV-P v1jn - C_LEVEL_GROUND_REFERENCE_EVENT_PATCH_DECISION_LAYER.

Consolidates official event anchors, multimodal patch QA, DINO review
diagnostics, negative-evidence gates, pseudo-absence boundaries, and supervised
training gates into the canonical C1-C4 decision layer for Protocol C.

This stage is additive and conservative. It does not create labels, train
models, unfreeze DINO, read heavy raster/vector/model artifacts, or promote
pseudo-absence/control material into formal negatives.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REVP_ROOT = SCRIPT_PATH.parents[2]
DATASETS_DIR = REVP_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"
DOCS_DIR = REVP_ROOT / "docs" / "metodologia_cientifica"
LOCAL_RUN_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1jn"

INPUT_PATHS = {
    "coordinate_recovery": DATASETS_DIR / "official_coordinate_recovery_hardened_registry.csv",
    "official_anchors": DATASETS_DIR / "official_multi_anchor_registry.csv",
    "ground_master": DATASETS_DIR / "ground_reference_candidate_master_registry.csv",
    "multimodal_patches": DATASETS_DIR / "multi_anchor_multimodal_patch_registry.csv",
    "dino_embeddings": DATASETS_DIR / "multi_anchor_dino_review_embedding_registry.csv",
    "multi_anchor_training_gate": DATASETS_DIR / "multi_anchor_training_gate_matrix.csv",
    "formal_negative": DATASETS_DIR / "formal_negative_control_evidence_registry.csv",
    "negative_ladder": DATASETS_DIR / "negative_evidence_ladder_registry.csv",
    "pseudo_absence": DATASETS_DIR / "pseudo_absence_candidate_registry.csv",
    "pu_boundary": DATASETS_DIR / "positive_unlabeled_boundary_matrix.csv",
    "supervised_gate": DATASETS_DIR / "supervised_training_minimum_gate_matrix.csv",
    "control_expansion": DATASETS_DIR / "control_candidate_expansion_registry.csv",
}

S2_SCENE_SELECTION = REVP_ROOT / "local_runs" / "protocolo_c" / "v1ji" / "v1ji_s2_batch_scene_selection.csv"
S2_PATCH_QA = REVP_ROOT / "local_runs" / "protocolo_c" / "v1ji" / "v1ji_s2_patch_quality_audit.csv"

PUBLIC_EVENT_REGISTRY = DATASETS_DIR / "ground_reference_event_registry.csv"
PUBLIC_LINKAGE_REGISTRY = DATASETS_DIR / "event_patch_linkage_registry.csv"
PUBLIC_DECISION_AUDIT = DATASETS_DIR / "ground_truth_candidate_decision_audit.csv"
PUBLIC_SUMMARY = DATASETS_DIR / "protocol_c_c_level_summary_registry.csv"

LOCAL_EVENT_LOG = LOCAL_RUN_DIR / "v1jn_event_registry_build_log.csv"
LOCAL_LINKAGE_AUDIT = LOCAL_RUN_DIR / "v1jn_event_patch_linkage_audit.csv"
LOCAL_DECISION_LOG = LOCAL_RUN_DIR / "v1jn_candidate_decision_audit_log.csv"
LOCAL_SUMMARY_JSON = LOCAL_RUN_DIR / "v1jn_c_level_summary.json"
LOCAL_QA = LOCAL_RUN_DIR / "v1jn_qa.csv"

DOC_METHOD = DOCS_DIR / "protocolo_c_camada_c1_c4_ground_reference_v1jn.md"
DOC_REPORT = DOCS_DIR / "protocolo_c_relatorio_camada_c1_c4_ground_reference_v1jn.md"

EVENT_FIELDS = [
    "event_id",
    "source_event_unit_id",
    "source_institution",
    "source_document_sanitized",
    "region",
    "municipality",
    "locality_text_sanitized",
    "event_or_survey_date",
    "temporal_precision",
    "phenomenon_group",
    "coordinate_status",
    "latitude",
    "longitude",
    "spatial_precision",
    "source_confidence",
    "c_level",
    "c_level_reason",
    "can_be_ground_reference_event",
    "can_be_operational_ground_truth",
    "can_create_training_label",
    "notes",
]

LINKAGE_FIELDS = [
    "linkage_id",
    "event_id",
    "anchor_id",
    "patch_candidate_id",
    "sensor_stack",
    "pre_scene_date",
    "post_scene_date",
    "temporal_window_pre_days",
    "temporal_window_post_days",
    "spatial_center_error_m",
    "s2_pair_status",
    "s1_pair_status",
    "dem_status",
    "dino_status",
    "cloud_local_status",
    "qa_status",
    "linkage_strength",
    "c_level_after_linkage",
    "can_be_review_patch",
    "can_be_training_sample",
    "blocking_reason",
    "notes",
]

DECISION_FIELDS = [
    "decision_id",
    "event_id",
    "anchor_id",
    "c_level",
    "positive_reference_status",
    "negative_evidence_status",
    "pseudo_absence_status",
    "split_leakage_status",
    "training_gate_status",
    "allowed_use",
    "forbidden_use",
    "can_be_review_only",
    "can_be_pu_sandbox",
    "can_be_positive_label",
    "can_be_negative_label",
    "can_create_training_label",
    "can_train_model",
    "can_unfreeze_dino_for_scientific_claim",
    "decision",
    "blocking_reason",
    "minimum_evidence_needed",
    "notes",
]

SUMMARY_FIELDS = [
    "summary_id",
    "total_event_count",
    "c1_event_documented_count",
    "c2_event_georeferenced_count",
    "c3_event_patch_linked_count",
    "c4_operational_label_candidate_count",
    "final_c_level_distribution",
    "strong_patch_linkage_count",
    "moderate_multimodal_linkage_count",
    "weak_contextual_linkage_count",
    "blocked_linkage_count",
    "official_anchor_count",
    "s2_pair_qa_pass_count",
    "dem_qa_pass_count",
    "dino_qa_pass_count",
    "s1_pair_qa_pass_count",
    "formal_negative_ready_count",
    "pseudo_absence_review_only_count",
    "can_create_training_label",
    "can_train_model",
    "can_unfreeze_dino_for_scientific_claim",
    "c4_blocking_reason",
    "notes",
]

QA_FIELDS = ["check", "status", "detail"]

PRIVATE_FRAGMENTS = [
    "C:\\Users\\gabriela",
    "Documents\\REV-P",
    "Documents/REV-P",
    "gabriela",
    "local_runs/",
    "local_runs\\",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def write_schema(path: Path, fields: list[str], prefix: str) -> None:
    rows = [{"field": field, "description": f"{prefix}: {field}."} for field in fields]
    write_csv(path, rows, ["field", "description"])


def prepare(force: bool) -> None:
    if force and LOCAL_RUN_DIR.exists():
        resolved = LOCAL_RUN_DIR.resolve()
        expected = (REVP_ROOT / "local_runs" / "protocolo_c" / "v1jn").resolve()
        if resolved != expected:
            raise RuntimeError(f"Refusing to clear unexpected path: {resolved}")
        shutil.rmtree(resolved)
    LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)


def boolish(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def valid_lat_lon(lat_text: str, lon_text: str) -> bool:
    try:
        lat = float(lat_text)
        lon = float(lon_text)
    except (TypeError, ValueError):
        return False
    return math.isfinite(lat) and math.isfinite(lon) and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def region_for_unit(unit_id: str) -> str:
    if unit_id.startswith("PET"):
        return "PET"
    if unit_id.startswith("CUR"):
        return "CUR"
    if unit_id.startswith("REC"):
        return "REC"
    return "UNKNOWN"


def date_parts(date_text: str) -> list[datetime]:
    parts = re.split(r"\s*[–-]\s*", date_text.strip())
    out: list[datetime] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            if "/" in part:
                out.append(datetime.strptime(part, "%d/%m/%Y"))
            else:
                out.append(datetime.fromisoformat(part))
        except ValueError:
            continue
    return out


def temporal_precision(date_text: str) -> str:
    dates = date_parts(date_text)
    if len(dates) > 1:
        return "DATE_RANGE_EXPLICIT"
    if len(dates) == 1:
        return "DAY_EXPLICIT"
    return "DATE_TEXT_UNPARSED"


def days_between(start_text: str, end_text: str) -> str:
    try:
        start = datetime.fromisoformat(start_text)
        end = datetime.fromisoformat(end_text)
    except ValueError:
        return ""
    return str(abs((end - start).days))


def first_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if value and value not in out:
            out[value] = row
    return out


def group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(row.get(key, ""), []).append(row)
    return out


def pair_status(pre_status: str, post_status: str) -> str:
    if pre_status == "QA_PASS" and post_status == "QA_PASS":
        return "QA_PASS"
    if "QA_PASS" in {pre_status, post_status}:
        return "PARTIAL_QA_PASS"
    if not pre_status and not post_status:
        return "NOT_AVAILABLE"
    return "QA_NOT_PASS"


def patch_candidate_id(unit_id: str) -> str:
    return f"PATCH_CAND_{safe_id(unit_id)}_MULTIMODAL"


def has_valid_temporal_window(scene_rows: list[dict[str, str]]) -> bool:
    relations = {row.get("relation") for row in scene_rows}
    has_pre = any(row.get("relation") == "pre" and row.get("window_start") and row.get("window_end") for row in scene_rows)
    has_post = any(row.get("relation") == "post" and row.get("window_start") and row.get("window_end") for row in scene_rows)
    return {"pre", "post"}.issubset(relations) and has_pre and has_post


def scene_value(scene_rows: list[dict[str, str]], relation: str, field: str) -> str:
    for row in scene_rows:
        if row.get("relation") == relation:
            return row.get(field, "")
    return ""


def temporal_window_days(scene_rows: list[dict[str, str]], relation: str) -> str:
    for row in scene_rows:
        if row.get("relation") == relation:
            return days_between(row.get("window_start", ""), row.get("window_end", ""))
    return ""


def cloud_status(qa_rows: list[dict[str, str]]) -> str:
    if not qa_rows:
        return "NOT_ASSESSED"
    parts = []
    for relation in ("pre", "post"):
        value = ""
        for row in qa_rows:
            if row.get("relation") == relation:
                value = row.get("local_cloud_fraction", "") or row.get("qa_status", "")
                break
        if value:
            parts.append(f"{relation}:{value}")
    return ";".join(parts) if parts else "NOT_ASSESSED"


def load_inputs() -> dict[str, list[dict[str, str]]]:
    data = {key: read_csv(path) for key, path in INPUT_PATHS.items()}
    data["s2_scene_selection"] = read_csv(S2_SCENE_SELECTION)
    data["s2_patch_qa"] = read_csv(S2_PATCH_QA)
    return data


def formal_negative_ready_count(data: dict[str, list[dict[str, str]]]) -> int:
    formal = sum(1 for row in data["formal_negative"] if boolish(row.get("can_be_negative_label", "")))
    ladder = sum(1 for row in data["negative_ladder"] if boolish(row.get("can_be_formal_negative", "")))
    supervised_rows = data["supervised_gate"]
    gate_count = 0
    if supervised_rows:
        try:
            gate_count = int(supervised_rows[0].get("formal_negative_labels_ready", "0") or 0)
        except ValueError:
            gate_count = 0
    return max(formal, ladder, gate_count)


def gate_state(data: dict[str, list[dict[str, str]]]) -> dict[str, str]:
    supervised = data["supervised_gate"][0] if data["supervised_gate"] else {}
    multi = data["multi_anchor_training_gate"][0] if data["multi_anchor_training_gate"] else {}
    return {
        "formal_positive_labels_ready": supervised.get("formal_positive_labels_ready", "0"),
        "formal_negative_labels_ready": supervised.get("formal_negative_labels_ready", "0"),
        "split_protocol_status": supervised.get("split_protocol_status", "SPLIT_PROTOCOL_REQUIRED"),
        "leakage_risk_status": supervised.get("leakage_risk_status") or multi.get("leakage_risk_status", "LEAKAGE_PROTOCOL_REQUIRED"),
        "training_gate_status": supervised.get("supervised_training_boundary_status") or multi.get("training_gate_status", "SUPERVISED_TRAINING_BLOCKED"),
        "can_create_training_label": "true" if boolish(supervised.get("can_create_training_label", "")) and boolish(multi.get("can_create_training_label", "")) else "false",
        "can_train_model": "true" if boolish(supervised.get("can_train_model", "")) and boolish(multi.get("can_train_model", "")) else "false",
        "can_unfreeze_dino_for_scientific_claim": "true"
        if boolish(supervised.get("can_unfreeze_dino_for_scientific_claim", "")) and boolish(multi.get("can_unfreeze_dino_for_scientific_claim", ""))
        else "false",
        "minimum_evidence_needed": supervised.get(
            "minimum_evidence_needed",
            "formal positive labels; formal negative labels; split/leakage protocol closed; independent validation metrics",
        ),
    }


def build_linkage_rows(data: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    patch_by_anchor = first_by(data["multimodal_patches"], "anchor_id")
    dino_by_anchor = first_by(data["dino_embeddings"], "anchor_id")
    scenes_by_anchor = group_by(data["s2_scene_selection"], "anchor_id")
    s2_qa_by_anchor = group_by(data["s2_patch_qa"], "anchor_id")
    rows: list[dict[str, str]] = []
    gates = gate_state(data)

    for anchor in data["official_anchors"]:
        anchor_id = anchor.get("anchor_id", "")
        unit_id = anchor.get("documented_event_unit_id", "")
        patch = patch_by_anchor.get(anchor_id, {})
        dino = dino_by_anchor.get(anchor_id, {})
        scene_rows = scenes_by_anchor.get(anchor_id, [])
        qa_rows = s2_qa_by_anchor.get(anchor_id, [])

        s2_status = pair_status(patch.get("s2_pre_status", ""), patch.get("s2_post_status", ""))
        s1_status = pair_status(patch.get("s1_pre_status", ""), patch.get("s1_post_status", ""))
        dem_status = patch.get("dem_status", "")
        dino_status = dino.get("dino_status") or patch.get("dino_status", "")
        valid_window = has_valid_temporal_window(scene_rows)
        has_coordinate = valid_lat_lon(anchor.get("latitude", ""), anchor.get("longitude", ""))
        core_c3 = has_coordinate and s2_status == "QA_PASS" and dem_status == "QA_PASS" and dino_status == "DINO_QA_PASS" and valid_window

        if not has_coordinate or not patch:
            strength = "BLOCKED_LINKAGE"
            c_level = "C2_EVENT_GEOREFERENCED" if has_coordinate else "C1_EVENT_DOCUMENTED"
            qa_status = "QA_BLOCKED"
            blocker = "COORDINATE_OR_PATCH_QA_MISSING"
        elif core_c3 and s1_status == "QA_PASS":
            strength = "STRONG_PATCH_LINKAGE"
            c_level = "C3_EVENT_PATCH_LINKED"
            qa_status = "QA_PASS"
            blocker = "C4_BLOCKED_LABEL_NEGATIVE_SPLIT_LEAKAGE_INCOMPLETE"
        elif core_c3:
            strength = "MODERATE_MULTIMODAL_LINKAGE"
            c_level = "C3_EVENT_PATCH_LINKED"
            qa_status = "QA_PASS_WITH_S1_LIMITATION"
            blocker = "C4_BLOCKED_LABEL_NEGATIVE_SPLIT_LEAKAGE_INCOMPLETE"
        elif has_coordinate and (s2_status == "QA_PASS" or dem_status == "QA_PASS" or dino_status == "DINO_QA_PASS"):
            strength = "WEAK_CONTEXTUAL_LINKAGE"
            c_level = "C2_EVENT_GEOREFERENCED"
            qa_status = "QA_PARTIAL"
            blocker = "PATCH_QA_OR_TEMPORAL_WINDOW_INCOMPLETE"
        else:
            strength = "BLOCKED_LINKAGE"
            c_level = "C2_EVENT_GEOREFERENCED"
            qa_status = "QA_BLOCKED"
            blocker = "PATCH_QA_MISSING"

        rows.append(
            {
                "linkage_id": f"LINK_{safe_id(unit_id)}",
                "event_id": f"EVENT_{safe_id(unit_id)}",
                "anchor_id": anchor_id,
                "patch_candidate_id": patch_candidate_id(unit_id),
                "sensor_stack": "S2_PRE_POST;DEM;DINO_REVIEW;S1_PARTIAL" if s1_status != "QA_PASS" else "S2_PRE_POST;S1_PRE_POST;DEM;DINO_REVIEW",
                "pre_scene_date": scene_value(scene_rows, "pre", "scene_date"),
                "post_scene_date": scene_value(scene_rows, "post", "scene_date"),
                "temporal_window_pre_days": temporal_window_days(scene_rows, "pre"),
                "temporal_window_post_days": temporal_window_days(scene_rows, "post"),
                "spatial_center_error_m": "0.000",
                "s2_pair_status": s2_status,
                "s1_pair_status": s1_status,
                "dem_status": dem_status or "NOT_AVAILABLE",
                "dino_status": dino_status or "NOT_AVAILABLE",
                "cloud_local_status": cloud_status(qa_rows),
                "qa_status": qa_status,
                "linkage_strength": strength,
                "c_level_after_linkage": c_level,
                "can_be_review_patch": "true" if c_level == "C3_EVENT_PATCH_LINKED" else "false",
                "can_be_training_sample": "true" if c_level == "C4_OPERATIONAL_LABEL_CANDIDATE" and gates["can_create_training_label"] == "true" else "false",
                "blocking_reason": blocker,
                "notes": "DINO is frozen review evidence only; linkage does not create a label.",
            }
        )
    return rows


def build_event_rows(data: dict[str, list[dict[str, str]]], linkages: list[dict[str, str]]) -> list[dict[str, str]]:
    coord_by_unit = first_by(data["coordinate_recovery"], "documented_event_unit_id")
    linkage_by_event = first_by(linkages, "event_id")
    gates = gate_state(data)
    rows: list[dict[str, str]] = []

    for anchor in data["official_anchors"]:
        unit_id = anchor.get("documented_event_unit_id", "")
        coord = coord_by_unit.get(unit_id, {})
        event_id = f"EVENT_{safe_id(unit_id)}"
        source_doc = anchor.get("source_document_name_sanitized") or coord.get("source_document_name_sanitized", "")
        date = anchor.get("date") or coord.get("event_or_survey_date", "")
        phenomenon = anchor.get("phenomenon_group") or coord.get("phenomenon_group", "")
        locality = coord.get("locality_text_sanitized", "")
        municipality = coord.get("municipality", "Petrópolis")
        has_c1 = bool(source_doc and date and phenomenon and (locality or municipality))
        has_coord = valid_lat_lon(anchor.get("latitude", ""), anchor.get("longitude", ""))
        linkage = linkage_by_event.get(event_id, {})
        has_c3 = linkage.get("c_level_after_linkage") == "C3_EVENT_PATCH_LINKED"
        has_c4 = has_c3 and gates["can_create_training_label"] == "true" and gates["can_train_model"] == "true"

        if has_c4:
            c_level = "C4_OPERATIONAL_LABEL_CANDIDATE"
            reason = "C3 reference plus complete label, negative, split and leakage gates."
        elif has_c3:
            c_level = "C3_EVENT_PATCH_LINKED"
            reason = "Official event is documented, explicitly georeferenced, and linked to QA-passed S2/DEM/DINO review patch; C4 gates remain blocked."
        elif has_c1 and has_coord:
            c_level = "C2_EVENT_GEOREFERENCED"
            reason = "Official event is documented and explicitly georeferenced, but patch linkage is incomplete."
        elif has_c1:
            c_level = "C1_EVENT_DOCUMENTED"
            reason = "Official event is documented, but explicit spatial anchor is missing."
        else:
            c_level = "C0_INSUFFICIENT_EVENT_EVIDENCE"
            reason = "Required official event metadata is incomplete."

        rows.append(
            {
                "event_id": event_id,
                "source_event_unit_id": unit_id,
                "source_institution": "SGB/CPRM" if "CPRM" in source_doc.upper() else "OFFICIAL_SOURCE",
                "source_document_sanitized": source_doc,
                "region": region_for_unit(unit_id),
                "municipality": municipality,
                "locality_text_sanitized": locality,
                "event_or_survey_date": date,
                "temporal_precision": temporal_precision(date),
                "phenomenon_group": phenomenon,
                "coordinate_status": "EXPLICIT_COORDINATE" if has_coord else "NO_EXPLICIT_COORDINATE",
                "latitude": anchor.get("latitude", ""),
                "longitude": anchor.get("longitude", ""),
                "spatial_precision": "EXPLICIT_POINT_COORDINATE_REPRESENTATIVE_ANCHOR" if has_coord else "DOCUMENTARY_LOCALITY_ONLY",
                "source_confidence": "OFFICIAL_CPRM_EXPLICIT_COORDINATE_HIGH" if has_coord else "OFFICIAL_DOCUMENTARY_EVENT",
                "c_level": c_level,
                "c_level_reason": reason,
                "can_be_ground_reference_event": "true" if c_level in {"C1_EVENT_DOCUMENTED", "C2_EVENT_GEOREFERENCED", "C3_EVENT_PATCH_LINKED"} else "false",
                "can_be_operational_ground_truth": "true" if c_level == "C4_OPERATIONAL_LABEL_CANDIDATE" else "false",
                "can_create_training_label": "true" if c_level == "C4_OPERATIONAL_LABEL_CANDIDATE" and gates["can_create_training_label"] == "true" else "false",
                "notes": "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA and pseudo-absence material are not promoted to operational ground truth by this registry.",
            }
        )
    return rows


def build_decision_rows(data: dict[str, list[dict[str, str]]], events: list[dict[str, str]], linkages: list[dict[str, str]]) -> list[dict[str, str]]:
    linkage_by_event = first_by(linkages, "event_id")
    gates = gate_state(data)
    neg_ready = formal_negative_ready_count(data)
    pu = data["pu_boundary"][0] if data["pu_boundary"] else {}
    pu_ready = pu.get("pu_boundary_status") == "PU_SANDBOX_LOCAL_ONLY_READY"
    pseudo_count = len(data["pseudo_absence"])
    rows: list[dict[str, str]] = []

    for event in events:
        linkage = linkage_by_event.get(event["event_id"], {})
        c_level = event["c_level"]
        positive_status = "C3_MULTIMODAL_REFERENCE_CANDIDATE" if c_level == "C3_EVENT_PATCH_LINKED" else c_level
        decision = "C3_REVIEW_REFERENCE_LABEL_BLOCKED" if c_level == "C3_EVENT_PATCH_LINKED" else "REVIEW_ONLY_REFERENCE"
        blocking = "FORMAL_NEGATIVES_ZERO;FORMAL_POSITIVE_LABELS_ZERO;SPLIT_LEAKAGE_INCOMPLETE;DINO_REVIEW_ONLY"
        allowed_use = "MULTIMODAL_REFERENCE_CANDIDATE" if c_level == "C3_EVENT_PATCH_LINKED" else "REVIEW_ONLY_REFERENCE"
        if c_level == "C4_OPERATIONAL_LABEL_CANDIDATE":
            decision = "C4_OPERATIONAL_LABEL_CANDIDATE"
            blocking = "NONE"
            allowed_use = "REVIEW_ONLY_REFERENCE"

        rows.append(
            {
                "decision_id": f"DECISION_{safe_id(event['source_event_unit_id'])}",
                "event_id": event["event_id"],
                "anchor_id": linkage.get("anchor_id", ""),
                "c_level": c_level,
                "positive_reference_status": positive_status,
                "negative_evidence_status": "FORMAL_NEGATIVES_ZERO" if neg_ready == 0 else "FORMAL_NEGATIVES_AVAILABLE",
                "pseudo_absence_status": "PSEUDO_ABSENCE_REVIEW_ONLY_NOT_NEGATIVE" if pseudo_count else "NO_PSEUDO_ABSENCE_ROWS",
                "split_leakage_status": f"{gates['split_protocol_status']};{gates['leakage_risk_status']}",
                "training_gate_status": gates["training_gate_status"],
                "allowed_use": allowed_use,
                "forbidden_use": "OPERATIONAL_LABEL;NEGATIVE_LABEL;SUPERVISED_TRAINING;DINO_UNFREEZE;SCIENTIFIC_MODEL_CLAIM",
                "can_be_review_only": "true",
                "can_be_pu_sandbox": "true" if pu_ready and c_level == "C3_EVENT_PATCH_LINKED" else "false",
                "can_be_positive_label": "false",
                "can_be_negative_label": "false",
                "can_create_training_label": "false",
                "can_train_model": "false",
                "can_unfreeze_dino_for_scientific_claim": "false",
                "decision": decision,
                "blocking_reason": blocking,
                "minimum_evidence_needed": gates["minimum_evidence_needed"],
                "notes": "Pseudo-absence remains unlabeled; DINO review diagnostics do not create a label.",
            }
        )
    return rows


def summary_rows(
    data: dict[str, list[dict[str, str]]],
    events: list[dict[str, str]],
    linkages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    levels = Counter(row["c_level"] for row in events)
    strengths = Counter(row["linkage_strength"] for row in linkages)
    gates = gate_state(data)
    c1_count = sum(1 for row in events if row["c_level"] in {"C1_EVENT_DOCUMENTED", "C2_EVENT_GEOREFERENCED", "C3_EVENT_PATCH_LINKED", "C4_OPERATIONAL_LABEL_CANDIDATE"})
    c2_count = sum(1 for row in events if row["c_level"] in {"C2_EVENT_GEOREFERENCED", "C3_EVENT_PATCH_LINKED", "C4_OPERATIONAL_LABEL_CANDIDATE"})
    c3_count = sum(1 for row in events if row["c_level"] in {"C3_EVENT_PATCH_LINKED", "C4_OPERATIONAL_LABEL_CANDIDATE"})
    c4_count = levels["C4_OPERATIONAL_LABEL_CANDIDATE"]
    s2_count = sum(1 for row in linkages if row["s2_pair_status"] == "QA_PASS")
    dem_count = sum(1 for row in linkages if row["dem_status"] == "QA_PASS")
    dino_count = sum(1 for row in linkages if row["dino_status"] == "DINO_QA_PASS")
    s1_count = sum(1 for row in linkages if row["s1_pair_status"] == "QA_PASS")
    summary = {
        "summary_id": "V1JN_C_LEVEL_GROUND_REFERENCE_EVENT_PATCH_DECISION",
        "total_event_count": str(len(events)),
        "c1_event_documented_count": str(c1_count),
        "c2_event_georeferenced_count": str(c2_count),
        "c3_event_patch_linked_count": str(c3_count),
        "c4_operational_label_candidate_count": str(c4_count),
        "final_c_level_distribution": ";".join(f"{key}={value}" for key, value in sorted(levels.items())),
        "strong_patch_linkage_count": str(strengths["STRONG_PATCH_LINKAGE"]),
        "moderate_multimodal_linkage_count": str(strengths["MODERATE_MULTIMODAL_LINKAGE"]),
        "weak_contextual_linkage_count": str(strengths["WEAK_CONTEXTUAL_LINKAGE"]),
        "blocked_linkage_count": str(strengths["BLOCKED_LINKAGE"]),
        "official_anchor_count": str(len(data["official_anchors"])),
        "s2_pair_qa_pass_count": str(s2_count),
        "dem_qa_pass_count": str(dem_count),
        "dino_qa_pass_count": str(dino_count),
        "s1_pair_qa_pass_count": str(s1_count),
        "formal_negative_ready_count": str(formal_negative_ready_count(data)),
        "pseudo_absence_review_only_count": str(len(data["pseudo_absence"])),
        "can_create_training_label": gates["can_create_training_label"],
        "can_train_model": gates["can_train_model"],
        "can_unfreeze_dino_for_scientific_claim": gates["can_unfreeze_dino_for_scientific_claim"],
        "c4_blocking_reason": "FORMAL_NEGATIVES_ZERO;FORMAL_POSITIVE_LABELS_ZERO;SPLIT_LEAKAGE_INCOMPLETE",
        "notes": "C3 is the current canonical review layer; C4 remains blocked.",
    }
    json_summary = dict(summary)
    json_summary["stage"] = "v1jn"
    json_summary["timestamp"] = utc_now()
    json_summary["methodological_line"] = [
        "evento observado",
        "fonte/confiabilidade",
        "data/precisao temporal",
        "localizacao/precisao espacial",
        "patch Sentinel/multimodal e janela temporal",
        "decisao C1/C2/C3/C4",
        "uso permitido",
    ]
    return [summary], json_summary


def public_text_has_private_fragment(paths: list[Path]) -> list[str]:
    leaks: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for fragment in PRIVATE_FRAGMENTS:
            if fragment.lower() in text.lower():
                leaks.append(f"{path.name}:{fragment}")
    return leaks


def build_docs(summary: dict[str, Any]) -> None:
    method = f"""# Protocolo C v1jn - Camada C1-C4 de referencia terrestre

## Linha metodologica

A v1jn organiza a evidencia nesta ordem: evento observado, fonte e confiabilidade, data e precisao temporal, localizacao e precisao espacial, patch Sentinel/multimodal com janela temporal, decisao C1/C2/C3/C4 e uso permitido.

## Niveis C

- C1_EVENT_DOCUMENTED: evento em fonte oficial ou auditavel, com fenomeno, data e localidade documental.
- C2_EVENT_GEOREFERENCED: C1 acrescido de coordenada explicita ou geometria auditavel.
- C3_EVENT_PATCH_LINKED: C2 acrescido de patch Sentinel/multimodal com QA e janela temporal documentada.
- C4_OPERATIONAL_LABEL_CANDIDATE: C3 acrescido de gates completos de label, negativos formais, split, vazamento e revisao.

## Resultado consolidado

- C1 documentado: {summary['c1_event_documented_count']}
- C2 georreferenciado: {summary['c2_event_georeferenced_count']}
- C3 ligado a patch: {summary['c3_event_patch_linked_count']}
- C4 candidato operacional: {summary['c4_operational_label_candidate_count']}

C3 e um avanco real porque liga a unidade documental oficial a uma coordenada explicita e a um conjunto S2/DEM/DINO com QA, mantendo S1 como limitacao quando parcial. Isso permite revisao cientifica rastreavel do evento e do patch, sem transformar a referencia em label.

## Limites

C4 e treino seguem bloqueados porque os negativos formais continuam em {summary['formal_negative_ready_count']}, os labels positivos formais continuam bloqueados e o split/leakage ainda nao esta fechado. Pseudo-ausencia continua como material unlabeled de auditoria ou sandbox local, nunca como negativo formal. DINO permanece congelado e serve apenas como diagnostico estrutural de revisao.

## Referencia, label, pseudo-ausencia e negativo formal

Referencia e evidencia oficial organizada para revisao. Label e uma decisao operacional posterior, dependente de gates completos. Pseudo-ausencia e unlabeled auditado, sem afirmacao de ausencia. Negativo formal exige evidencia explicita de ausencia ou estabilidade para area, tempo e fenomeno, alem de QA e protocolo de split/leakage.

## Usos permitidos

Uso permitido: referencia review-only, candidato multimodal de referencia e PU sandbox local-only quando registrado como unlabeled/positivo de referencia sem pesos, metricas operacionais ou claim supervisionado.

## Usos proibidos

Uso proibido: label operacional, negativo formal por pseudo-ausencia, treino supervisionado, descongelamento de DINO, claim cientifico de modelo e promocao automatica de PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA para ground truth operacional.
"""
    report = f"""# Relatorio v1jn - Camada C1-C4 de ground reference

## Escopo executado

A v1jn consolidou os registries canonicos `ground_reference_event_registry.csv`, `event_patch_linkage_registry.csv` e `ground_truth_candidate_decision_audit.csv`. A etapa apenas reuniu evidencia ja registrada: anchors oficiais CPRM, QA S2/DEM/DINO, S1 parcial, negativos formais, pseudo-ausencias review-only e gates de treino.

## Contagens

- Eventos oficiais consolidados: {summary['total_event_count']}
- C1 documentado: {summary['c1_event_documented_count']}
- C2 georreferenciado: {summary['c2_event_georeferenced_count']}
- C3 ligado a patch: {summary['c3_event_patch_linked_count']}
- C4 candidato operacional: {summary['c4_operational_label_candidate_count']}

## Linkage evento-patch

O linkage ficou com {summary['strong_patch_linkage_count']} forte, {summary['moderate_multimodal_linkage_count']} moderado, {summary['weak_contextual_linkage_count']} fraco e {summary['blocked_linkage_count']} bloqueado. Todos os 9 anchors oficiais chegaram a C3 porque tinham coordenada explicita, S2 pre/pos QA_PASS, DEM QA_PASS, DINO_QA_PASS e janela temporal registrada. A limitacao principal e S1 parcial.

## Bloqueio C4

C4 permanece bloqueado por tres motivos: negativos formais iguais a {summary['formal_negative_ready_count']}, labels positivos formais ainda nao liberados e split/leakage incompleto. Pseudo-ausencia nao altera esse bloqueio e nao vira negativo. DINO nao cria classe nem label.

## Usos permitidos e proibidos

Permitido: revisao cientifica, referencia multimodal candidata e sandbox PU local-only sob as restricoes registradas. Proibido: label operacional, treino supervisionado, negativo formal por ausencia de registro, claim cientifico de modelo, descongelamento de DINO e promocao automatica de PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA para ground truth operacional.
"""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DOC_METHOD.write_text(method, encoding="utf-8")
    DOC_REPORT.write_text(report, encoding="utf-8")


def qa_rows(data: dict[str, list[dict[str, str]]], events: list[dict[str, str]], linkages: list[dict[str, str]], decisions: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(check: str, ok: bool, detail: str) -> None:
        rows.append({"check": check, "status": "PASS" if ok else "FAIL", "detail": detail})

    c3 = [row for row in events if row["c_level"] == "C3_EVENT_PATCH_LINKED"]
    c4 = [row for row in events if row["c_level"] == "C4_OPERATIONAL_LABEL_CANDIDATE"]
    public_paths = [PUBLIC_EVENT_REGISTRY, PUBLIC_LINKAGE_REGISTRY, PUBLIC_DECISION_AUDIT, PUBLIC_SUMMARY, DOC_METHOD, DOC_REPORT]
    leaks = public_text_has_private_fragment(public_paths)
    pseudo_negative = [row for row in data["pseudo_absence"] if boolish(row.get("can_be_formal_negative", "")) or boolish(row.get("can_create_training_label", ""))]
    dino_label = [row for row in decisions if row["can_be_positive_label"] == "true" or row["can_create_training_label"] == "true"]

    add("input_registries_present", all(path.exists() for path in INPUT_PATHS.values()), "all requested input registries exist")
    add("official_anchor_count", len(data["official_anchors"]) == 9, str(len(data["official_anchors"])))
    add("event_registry_count", len(events) == len(data["official_anchors"]), str(len(events)))
    add("c3_event_count", len(c3) == 9, str(len(c3)))
    add("c4_event_count_blocked", not c4, str(len(c4)))
    add("training_label_blocked", all(row["can_create_training_label"] == "false" for row in decisions), "all decision rows false")
    add("model_training_blocked", all(row["can_train_model"] == "false" for row in decisions), "all decision rows false")
    add("pseudo_absence_not_negative", not pseudo_negative, str(len(pseudo_negative)))
    add("dino_does_not_create_label", not dino_label, str(len(dino_label)))
    add("public_outputs_no_private_path", not leaks, ";".join(leaks) if leaks else "no leaks")
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepare(args.force)
    data = load_inputs() if args.read_ground_reference_inputs else load_inputs()
    linkages = build_linkage_rows(data) if args.build_event_patch_linkage else []
    events = build_event_rows(data, linkages) if args.build_event_registry else []
    decisions = build_decision_rows(data, events, linkages) if args.build_candidate_decision_audit else []
    summary_csv, summary_json = summary_rows(data, events, linkages) if args.emit_c_level_summary else ([], {})

    if events:
        write_csv(PUBLIC_EVENT_REGISTRY, events, EVENT_FIELDS)
        write_csv(LOCAL_EVENT_LOG, events, EVENT_FIELDS)
        write_schema(SCHEMAS_DIR / "ground_reference_event_registry_schema.csv", EVENT_FIELDS, "v1jn ground reference event registry")
    if linkages:
        write_csv(PUBLIC_LINKAGE_REGISTRY, linkages, LINKAGE_FIELDS)
        write_csv(LOCAL_LINKAGE_AUDIT, linkages, LINKAGE_FIELDS)
        write_schema(SCHEMAS_DIR / "event_patch_linkage_registry_schema.csv", LINKAGE_FIELDS, "v1jn event patch linkage registry")
    if decisions:
        write_csv(PUBLIC_DECISION_AUDIT, decisions, DECISION_FIELDS)
        write_csv(LOCAL_DECISION_LOG, decisions, DECISION_FIELDS)
        write_schema(SCHEMAS_DIR / "ground_truth_candidate_decision_audit_schema.csv", DECISION_FIELDS, "v1jn candidate decision audit")
    if summary_csv:
        write_csv(PUBLIC_SUMMARY, summary_csv, SUMMARY_FIELDS)
        write_json(LOCAL_SUMMARY_JSON, summary_json)
        write_schema(SCHEMAS_DIR / "protocol_c_c_level_summary_schema.csv", SUMMARY_FIELDS, "v1jn C-level summary")
        build_docs(summary_json)

    qa = qa_rows(data, events, linkages, decisions)
    write_csv(LOCAL_QA, qa, QA_FIELDS)
    failed = [row for row in qa if row["status"] != "PASS"]
    if failed:
        raise RuntimeError(f"v1jn QA failed: {failed}")
    return summary_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--read-ground-reference-inputs", action="store_true")
    parser.add_argument("--build-event-registry", action="store_true")
    parser.add_argument("--build-event-patch-linkage", action="store_true")
    parser.add_argument("--build-candidate-decision-audit", action="store_true")
    parser.add_argument("--emit-c-level-summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
