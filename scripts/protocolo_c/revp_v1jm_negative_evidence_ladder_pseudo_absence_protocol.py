"""
REV-P v1jm - NEGATIVE_EVIDENCE_LADDER_AND_PSEUDO_ABSENCE_PROTOCOL.

Builds an explicit decision ladder for formal negatives, audited
pseudo-absence, background/unlabeled material, PU sandbox limits, and external
benchmark transfer options. The stage is additive and conservative: it does not
create labels, train models, save weights, unfreeze DINO, download large data,
or promote absence-of-record into event absence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REVP_ROOT = SCRIPT_PATH.parents[2]
LOCAL_RUN_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1jm"
DATASETS_DIR = REVP_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"
DOCS_DIR = REVP_ROOT / "docs" / "metodologia_cientifica"

ANCHORS = DATASETS_DIR / "official_multi_anchor_registry.csv"
PATCHES = DATASETS_DIR / "multi_anchor_multimodal_patch_registry.csv"
DINO = DATASETS_DIR / "multi_anchor_dino_review_embedding_registry.csv"
CONTROLS = DATASETS_DIR / "control_candidate_expansion_registry.csv"
FORMAL_NEGATIVE = DATASETS_DIR / "formal_negative_control_evidence_registry.csv"
NEGATIVE_READINESS = DATASETS_DIR / "negative_label_readiness_matrix.csv"
SUPERVISED_GATE = DATASETS_DIR / "supervised_training_minimum_gate_matrix.csv"
GROUND_MASTER = DATASETS_DIR / "ground_reference_candidate_master_registry.csv"
EXTERNAL_EVIDENCE = DATASETS_DIR / "external_evidence_registry.csv"
PATCH_TAXONOMY = DATASETS_DIR / "patch_corpus_taxonomy_registry.csv"
V1JE_COORDS = REVP_ROOT / "local_runs" / "protocolo_c" / "v1je" / "v1je_coordinate_recovery_hardened.csv"
V1JE_TEXT_QA = REVP_ROOT / "local_runs" / "protocolo_c" / "v1je" / "v1je_text_extraction_quality.csv"
V1JI_SUMMARY = REVP_ROOT / "local_runs" / "protocolo_c" / "v1ji" / "v1ji_summary.json"

PUBLIC_LADDER = DATASETS_DIR / "negative_evidence_ladder_registry.csv"
PUBLIC_PSEUDO = DATASETS_DIR / "pseudo_absence_candidate_registry.csv"
PUBLIC_BACKGROUND = DATASETS_DIR / "background_unlabeled_candidate_registry.csv"
PUBLIC_PU = DATASETS_DIR / "positive_unlabeled_boundary_matrix.csv"
PUBLIC_EXTERNAL = DATASETS_DIR / "external_benchmark_transfer_option_registry.csv"

EXPLICIT_ABSENCE_TERMS = [
    "sem indicios",
    "nao foram observados processos",
    "ausencia de instabilidade",
    "area estavel",
    "sem movimentacao",
    "sem cicatrizes",
    "vistoria sem ocorrencia",
    "ponto de controle",
    "area controle",
    "estabilidade",
]

INSUFFICIENT_ABSENCE_TERMS = [
    "baixo risco",
    "fora da area de risco",
    "sem registro",
    "sem ocorrencia registrada",
    "absence of record",
    "absence_of_record",
    "invalid negative absence assumption",
    "invalid_negative_absence_assumption",
    "no record",
]

LADDER_FIELDS = [
    "candidate_id",
    "candidate_type",
    "region",
    "source_evidence",
    "source_type",
    "coordinate_status",
    "temporal_status",
    "phenomenon_status",
    "explicit_absence_or_stability_evidence",
    "distance_to_nearest_positive_anchor_m",
    "within_positive_buffer",
    "environmental_stratum",
    "s2_status",
    "s1_status",
    "dem_status",
    "dino_status",
    "negative_evidence_status",
    "pseudo_absence_status",
    "background_status",
    "pu_status",
    "can_be_formal_negative",
    "can_be_pseudo_absence",
    "can_be_background_unlabeled",
    "can_create_training_label",
    "can_train_supervised_model",
    "can_train_pu_sandbox",
    "leakage_risk_status",
    "scientific_claim_status",
    "blocking_reason",
    "minimum_evidence_needed",
    "notes",
]

FORMAL_SCAN_FIELDS = [
    "scan_id",
    "source_path",
    "source_type",
    "candidate_id",
    "source_evidence",
    "matched_explicit_absence_terms",
    "matched_insufficient_terms",
    "coordinate_status",
    "temporal_status",
    "phenomenon_status",
    "patch_multimodal_qa_status",
    "leakage_risk_status",
    "formal_negative_status",
    "can_be_formal_negative",
    "blocking_reason",
    "notes",
]

PSEUDO_FIELDS = LADDER_FIELDS
BACKGROUND_FIELDS = LADDER_FIELDS

PU_FIELDS = [
    "boundary_id",
    "positive_count",
    "unlabeled_candidate_count",
    "pseudo_absence_review_only_count",
    "background_unlabeled_count",
    "strong_control_review_only_count",
    "formal_negative_ready_count",
    "pu_boundary_status",
    "supervised_training_status",
    "metrics_status",
    "model_artifact_status",
    "can_create_training_label",
    "can_train_supervised_model",
    "can_train_pu_sandbox",
    "can_save_model_weights",
    "scientific_claim_status",
    "blocking_reason",
    "notes",
]

EXTERNAL_FIELDS = [
    "option_id",
    "benchmark_name",
    "benchmark_family",
    "download_status",
    "local_ground_truth_status",
    "role_in_revp",
    "can_supply_local_negative_ground_truth",
    "can_enable_local_supervised_claim",
    "transfer_status",
    "compatibility_status",
    "minimum_evidence_needed",
    "notes",
]

QA_FIELDS = ["check", "status", "detail"]

PRIVATE_FRAGMENTS = ["C:\\Users\\gabriela", "Documents\\REV-P", "Documents/REV-P", "gabriela"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(text: Any) -> str:
    raw = "" if text is None else str(text)
    ascii_text = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.lower().split())


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
        expected = (REVP_ROOT / "local_runs" / "protocolo_c" / "v1jm").resolve()
        if resolved != expected:
            raise RuntimeError(f"Refusing to clear unexpected path: {resolved}")
        shutil.rmtree(resolved)
    LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)


def boolish(value: Any) -> bool:
    return normalize(value) == "true"


def public_path(path: Path) -> str:
    try:
        return path.relative_to(REVP_ROOT).as_posix()
    except ValueError:
        return path.name


def has_private_fragment(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    return any(fragment.lower() in text for fragment in PRIVATE_FRAGMENTS)


def row_text(row: dict[str, str]) -> str:
    return normalize(" ".join(row.values()))


def find_terms(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if normalize(term) in text]


def get_id(row: dict[str, str], fallback: str) -> str:
    for key in ("anchor_id", "candidate_id", "control_candidate_id", "recovery_id", "evidence_id", "taxonomy_id", "matrix_id", "gate_id"):
        if row.get(key):
            return row[key]
    return fallback


def source_type_for(path: Path) -> str:
    name = path.name.lower()
    if "official" in name or "cprm" in name or "coordinate_recovery" in name:
        return "OFFICIAL_OR_DERIVED_DOCUMENTARY_REGISTRY"
    if "formal_negative" in name or "control" in name:
        return "CONTROL_OR_NEGATIVE_GOVERNANCE_REGISTRY"
    if "external" in name:
        return "EXTERNAL_CONTEXT_REGISTRY"
    if "taxonomy" in name:
        return "PATCH_CORPUS_TAXONOMY_REGISTRY"
    return "PROJECT_REGISTRY"


def coordinate_status(row: dict[str, str]) -> str:
    if row.get("latitude") and row.get("longitude"):
        return row.get("coordinate_confidence") or row.get("coordinate_status") or "EXPLICIT_COORDINATE_PRESENT"
    if row.get("coordinate_available") == "true":
        return "COORDINATE_AVAILABLE"
    if row.get("nearest_anchor_id", "").startswith("ANCHOR_"):
        return "ANCHOR_REFERENCED_NO_INDEPENDENT_COORDINATE"
    return row.get("coordinate_status") or "NO_EXPLICIT_COORDINATE_FOR_NEGATIVE_EVIDENCE"


def temporal_status(row: dict[str, str]) -> str:
    for key in ("date", "event_or_survey_date", "temporal_status"):
        if row.get(key):
            return "TEMPORAL_WINDOW_EXPLICIT"
    return "NO_EXPLICIT_TEMPORAL_WINDOW_FOR_ABSENCE"


def phenomenon_status(row: dict[str, str]) -> str:
    for key in ("phenomenon_group", "phenomenon", "phenomenon_status"):
        if row.get(key):
            return row[key] or "PHENOMENON_EXPLICIT"
    return "NO_EXPLICIT_PHENOMENON_ABSENCE_SCOPE"


def anchor_maps() -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    anchors = read_csv(ANCHORS)
    patches = {row["anchor_id"]: row for row in read_csv(PATCHES) if row.get("anchor_id")}
    dino = {row["anchor_id"]: row for row in read_csv(DINO) if row.get("anchor_id")}
    return anchors, patches, dino


def patch_status(anchor_id: str, patches: dict[str, dict[str, str]], dino: dict[str, dict[str, str]]) -> tuple[str, str, str, str]:
    patch = patches.get(anchor_id, {})
    dino_row = dino.get(anchor_id, {})
    return (
        "S2_PRE_POST_QA_PASS" if patch.get("s2_pre_status") == "QA_PASS" and patch.get("s2_post_status") == "QA_PASS" else patch.get("s2_pre_status") or "S2_NOT_ASSESSED",
        "S1_PRE_POST_QA_PASS" if patch.get("s1_pre_status") == "QA_PASS" and patch.get("s1_post_status") == "QA_PASS" else "S1_PARTIAL_OR_NOT_AVAILABLE",
        patch.get("dem_status") or "DEM_NOT_ASSESSED",
        dino_row.get("dino_status") or patch.get("dino_status") or "DINO_NOT_ASSESSED",
    )


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2.0) ** 2
    return radius * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def nearest_anchor_distance(row: dict[str, str], anchors: list[dict[str, str]]) -> str:
    try:
        lat = float(row.get("latitude", ""))
        lon = float(row.get("longitude", ""))
    except ValueError:
        return row.get("distance_to_nearest_anchor_m", "")
    distances = []
    for anchor in anchors:
        try:
            distances.append(haversine_m(lat, lon, float(anchor["latitude"]), float(anchor["longitude"])))
        except Exception:
            continue
    if not distances:
        return ""
    return f"{min(distances):.3f}"


def minimum_formal_negative_evidence() -> str:
    return "official auditable source; explicit area or coordinate; explicit temporal window; compatible phenomenon; explicit absence or stability; multimodal patch QA; leakage buffer pass; split pass"


def minimum_pseudo_absence_evidence() -> str:
    return "sampled point or patch id; distance and buffer audit against anchors; same-region priority; environmental stratum; compatible temporal window; S2 QA; no absence claim"


def minimum_background_evidence() -> str:
    return "study-area sampling frame; environmental stratum; spatial balance; anchor buffer avoidance; unlabeled status preserved"


def scan_sources() -> list[dict[str, str]]:
    paths = [
        ANCHORS,
        PATCHES,
        DINO,
        CONTROLS,
        FORMAL_NEGATIVE,
        NEGATIVE_READINESS,
        SUPERVISED_GATE,
        GROUND_MASTER,
        EXTERNAL_EVIDENCE,
        PATCH_TAXONOMY,
        V1JE_COORDS,
        V1JE_TEXT_QA,
    ]
    anchors, patches, dino = anchor_maps()
    rows: list[dict[str, str]] = []
    scan_index = 1
    for path in paths:
        source_rows = read_csv(path)
        if not source_rows:
            rows.append(
                {
                    "scan_id": f"V1JM_SCAN_{scan_index:04d}",
                    "source_path": public_path(path),
                    "source_type": source_type_for(path),
                    "candidate_id": path.stem,
                    "source_evidence": "SOURCE_NOT_AVAILABLE_OR_EMPTY",
                    "matched_explicit_absence_terms": "",
                    "matched_insufficient_terms": "",
                    "coordinate_status": "NOT_ASSESSED",
                    "temporal_status": "NOT_ASSESSED",
                    "phenomenon_status": "NOT_ASSESSED",
                    "patch_multimodal_qa_status": "NOT_ASSESSED",
                    "leakage_risk_status": "NOT_ASSESSED",
                    "formal_negative_status": "INSUFFICIENT_EVIDENCE",
                    "can_be_formal_negative": "false",
                    "blocking_reason": "SOURCE_NOT_AVAILABLE_OR_EMPTY",
                    "notes": "No accessible rows were available for formal negative evidence scanning.",
                }
            )
            scan_index += 1
            continue
        for row in source_rows:
            text = row_text(row)
            explicit_terms = find_terms(text, EXPLICIT_ABSENCE_TERMS)
            insufficient_terms = find_terms(text, INSUFFICIENT_ABSENCE_TERMS)
            candidate_id = get_id(row, f"{path.stem}_{scan_index}")
            anchor_id = row.get("anchor_id", "")
            s2, s1, dem, dino_status = patch_status(anchor_id, patches, dino)
            coord_status = coordinate_status(row)
            temp_status = temporal_status(row)
            phen_status = phenomenon_status(row)
            source_type = source_type_for(path)
            official_source = source_type == "OFFICIAL_OR_DERIVED_DOCUMENTARY_REGISTRY"
            has_coord = coord_status not in {"NO_EXPLICIT_COORDINATE_FOR_NEGATIVE_EVIDENCE", "NOT_ASSESSED"}
            has_temp = temp_status == "TEMPORAL_WINDOW_EXPLICIT"
            has_phen = phen_status != "NO_EXPLICIT_PHENOMENON_ABSENCE_SCOPE"
            has_patch_qa = s2 == "S2_PRE_POST_QA_PASS" and dem == "QA_PASS" and dino_status == "DINO_QA_PASS"
            explicit_evidence = bool(explicit_terms)
            leakage_pass = row.get("leakage_risk_status") in {"LEAKAGE_LOW", "BUFFER_PASS", "SPLIT_READY"}
            can_formal = official_source and explicit_evidence and has_coord and has_temp and has_phen and has_patch_qa and leakage_pass
            if can_formal:
                status = "FORMAL_NEGATIVE_READY"
                blocker = ""
            elif explicit_evidence and official_source:
                status = "FORMAL_NEGATIVE_CANDIDATE_REVIEW"
                blocker = "FORMAL_NEGATIVE_GATES_INCOMPLETE"
            elif insufficient_terms:
                status = "INVALID_NEGATIVE_ABSENCE_ASSUMPTION"
                blocker = "INSUFFICIENT_ABSENCE_TERM_CANNOT_CREATE_NEGATIVE"
            else:
                status = "INSUFFICIENT_EVIDENCE"
                blocker = "NO_EXPLICIT_OFFICIAL_ABSENCE_OR_STABILITY_EVIDENCE"
            rows.append(
                {
                    "scan_id": f"V1JM_SCAN_{scan_index:04d}",
                    "source_path": public_path(path),
                    "source_type": source_type,
                    "candidate_id": candidate_id,
                    "source_evidence": ";".join(explicit_terms) if explicit_terms else "NO_EXPLICIT_ABSENCE_OR_STABILITY_TEXT_FOUND",
                    "matched_explicit_absence_terms": ";".join(explicit_terms),
                    "matched_insufficient_terms": ";".join(insufficient_terms),
                    "coordinate_status": coord_status,
                    "temporal_status": temp_status,
                    "phenomenon_status": phen_status,
                    "patch_multimodal_qa_status": f"{s2};{s1};{dem};{dino_status}",
                    "leakage_risk_status": row.get("leakage_risk_status") or "LEAKAGE_PROTOCOL_REQUIRED",
                    "formal_negative_status": status,
                    "can_be_formal_negative": str(can_formal).lower(),
                    "blocking_reason": blocker,
                    "notes": "Scanner result only; review is required before any formal negative decision.",
                }
            )
            scan_index += 1
    return rows


def ladder_row(**overrides: Any) -> dict[str, Any]:
    base = {
        "candidate_id": "",
        "candidate_type": "",
        "region": "",
        "source_evidence": "",
        "source_type": "",
        "coordinate_status": "NOT_ASSESSED",
        "temporal_status": "NOT_ASSESSED",
        "phenomenon_status": "NOT_ASSESSED",
        "explicit_absence_or_stability_evidence": "false",
        "distance_to_nearest_positive_anchor_m": "",
        "within_positive_buffer": "UNKNOWN",
        "environmental_stratum": "NOT_ASSESSED",
        "s2_status": "NOT_ASSESSED",
        "s1_status": "NOT_ASSESSED",
        "dem_status": "NOT_ASSESSED",
        "dino_status": "NOT_ASSESSED",
        "negative_evidence_status": "INSUFFICIENT_EVIDENCE",
        "pseudo_absence_status": "NOT_PSEUDO_ABSENCE",
        "background_status": "NOT_BACKGROUND",
        "pu_status": "NOT_USED_FOR_PU",
        "can_be_formal_negative": "false",
        "can_be_pseudo_absence": "false",
        "can_be_background_unlabeled": "false",
        "can_create_training_label": "false",
        "can_train_supervised_model": "false",
        "can_train_pu_sandbox": "false",
        "leakage_risk_status": "LEAKAGE_PROTOCOL_REQUIRED",
        "scientific_claim_status": "NO_SUPERVISED_CLAIM",
        "blocking_reason": "",
        "minimum_evidence_needed": minimum_formal_negative_evidence(),
        "notes": "",
    }
    base.update(overrides)
    return base


def ladder_from_formal_scan(scan_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scan in scan_rows:
        status = scan["formal_negative_status"]
        if status == "FORMAL_NEGATIVE_READY":
            candidate_type = "FORMAL_NEGATIVE_READY"
        elif status == "FORMAL_NEGATIVE_CANDIDATE_REVIEW":
            candidate_type = "FORMAL_NEGATIVE_CANDIDATE_REVIEW"
        elif status == "INVALID_NEGATIVE_ABSENCE_ASSUMPTION":
            candidate_type = "INVALID_NEGATIVE_ABSENCE_ASSUMPTION"
        else:
            candidate_type = "INSUFFICIENT_EVIDENCE"
        rows.append(
            ladder_row(
                candidate_id=scan["candidate_id"],
                candidate_type=candidate_type,
                region="PET" if "PET" in scan["candidate_id"].upper() or "petropolis" in normalize(scan["source_path"]) else "",
                source_evidence=scan["source_evidence"],
                source_type=scan["source_type"],
                coordinate_status=scan["coordinate_status"],
                temporal_status=scan["temporal_status"],
                phenomenon_status=scan["phenomenon_status"],
                explicit_absence_or_stability_evidence="true" if scan["matched_explicit_absence_terms"] else "false",
                negative_evidence_status=status,
                can_be_formal_negative=scan["can_be_formal_negative"],
                leakage_risk_status=scan["leakage_risk_status"],
                blocking_reason=scan["blocking_reason"],
                notes="Formal-negative scan row. It is not a label.",
            )
        )
    return rows


def pseudo_absence_candidates(controls: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for control in controls:
        control_type = control.get("control_type", "")
        if control_type not in {"EXISTING_PATCH_BACKGROUND_CANDIDATE", "SPATIAL_BACKGROUND_REVIEW_CANDIDATE", "SPATIAL_CONTEXT_CONTROL_CANDIDATE"}:
            continue
        same_region = control.get("region") == "PET"
        s2_status = "S2_QA_AVAILABLE" if boolish(control.get("s2_available")) else "S2_QA_REQUIRED"
        dem_status = "DEM_AVAILABLE" if boolish(control.get("dem_available")) else "DEM_REQUIRED"
        dino_status = "DINO_AVAILABLE" if boolish(control.get("dino_available")) else "DINO_OPTIONAL_NOT_AVAILABLE"
        distance = control.get("distance_to_nearest_anchor_m", "")
        buffer_status = control.get("buffer_status") or "BUFFER_NOT_ASSESSED"
        distance_ready = bool(distance)
        buffer_ready = buffer_status in {"OUTSIDE_POSITIVE_BUFFER", "BUFFER_PASS"}
        can_pseudo = same_region and boolish(control.get("s2_available")) and distance_ready and buffer_ready
        rows.append(
            ladder_row(
                candidate_id=control.get("control_candidate_id", ""),
                candidate_type="PSEUDO_ABSENCE_REVIEW_ONLY",
                region=control.get("region", ""),
                source_evidence=control.get("source_layer", "control_candidate_expansion_registry.csv"),
                source_type="AUDITED_CONTROL_CANDIDATE",
                coordinate_status="EXPLICIT_SAMPLE_POINT_REQUIRED" if not control.get("nearest_anchor_id", "").startswith("ANCHOR_") else "ANCHOR_REFERENCE_NOT_INDEPENDENT",
                temporal_status="TEMPORAL_WINDOW_COMPATIBLE_REQUIRED",
                phenomenon_status="NO_ABSENCE_PHENOMENON_CLAIM",
                explicit_absence_or_stability_evidence="false",
                distance_to_nearest_positive_anchor_m=distance,
                within_positive_buffer="false" if buffer_ready else "UNKNOWN_OR_NOT_ASSESSED",
                environmental_stratum="DEM_SLOPE_LANDCOVER_STRATIFICATION_REQUIRED",
                s2_status=s2_status,
                s1_status="S1_AVAILABLE" if boolish(control.get("s1_available")) else "S1_OPTIONAL_OR_PENDING",
                dem_status=dem_status,
                dino_status=dino_status,
                negative_evidence_status="INSUFFICIENT_EVIDENCE",
                pseudo_absence_status="PSEUDO_ABSENCE_REVIEW_ONLY",
                background_status="BACKGROUND_UNLABELED_COMPATIBLE" if control_type == "EXISTING_PATCH_BACKGROUND_CANDIDATE" else "NOT_BACKGROUND",
                pu_status="UNLABELED_FOR_PU_SANDBOX_ONLY",
                can_be_pseudo_absence=str(can_pseudo).lower(),
                can_be_background_unlabeled="true" if control_type == "EXISTING_PATCH_BACKGROUND_CANDIDATE" else "false",
                can_train_pu_sandbox="true",
                leakage_risk_status=control.get("leakage_risk_status") or "LEAKAGE_PROTOCOL_REQUIRED",
                scientific_claim_status="INVALID_FOR_SUPERVISED_CLAIM",
                blocking_reason="MISSING_DISTANCE_BUFFER_OR_EXPLICIT_SAMPLE_AUDIT" if not can_pseudo else "PSEUDO_ABSENCE_IS_NOT_FORMAL_NEGATIVE",
                minimum_evidence_needed=minimum_pseudo_absence_evidence(),
                notes="Pseudo-absence remains unlabeled and cannot be used as a formal negative.",
            )
        )
    return rows


def background_unlabeled_candidates(controls: list[dict[str, str]], taxonomy_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for control in controls:
        if control.get("control_type") not in {"EXISTING_PATCH_BACKGROUND_CANDIDATE", "CROSS_REGION_CONTEXT_CANDIDATE", "SPATIAL_BACKGROUND_REVIEW_CANDIDATE"}:
            continue
        rows.append(
            ladder_row(
                candidate_id=f"BG_{control.get('control_candidate_id', '')}",
                candidate_type="BACKGROUND_UNLABELED",
                region=control.get("region", ""),
                source_evidence=control.get("source_layer", "control_candidate_expansion_registry.csv"),
                source_type="BACKGROUND_OR_CONTEXT_CONTROL_CANDIDATE",
                coordinate_status="EXPLICIT_BACKGROUND_POINT_REQUIRED" if not control.get("nearest_anchor_id", "").startswith("ANCHOR_") else "ANCHOR_REFERENCE_NOT_BACKGROUND_POINT",
                temporal_status="UNLABELED_TEMPORAL_CONTEXT",
                phenomenon_status="NO_NEGATIVE_PHENOMENON_ASSERTION",
                explicit_absence_or_stability_evidence="false",
                distance_to_nearest_positive_anchor_m=control.get("distance_to_nearest_anchor_m", ""),
                within_positive_buffer="UNKNOWN_OR_NOT_ASSESSED",
                environmental_stratum="STRATIFICATION_REQUIRED",
                s2_status="S2_QA_AVAILABLE" if boolish(control.get("s2_available")) else "S2_QA_REQUIRED",
                s1_status="S1_AVAILABLE" if boolish(control.get("s1_available")) else "S1_OPTIONAL_OR_PENDING",
                dem_status="DEM_AVAILABLE" if boolish(control.get("dem_available")) else "DEM_REQUIRED",
                dino_status="DINO_AVAILABLE" if boolish(control.get("dino_available")) else "DINO_OPTIONAL_OR_PENDING",
                negative_evidence_status="INSUFFICIENT_EVIDENCE",
                background_status="BACKGROUND_UNLABELED",
                pu_status="UNLABELED_FOR_PU_SANDBOX_ONLY",
                can_be_background_unlabeled="true",
                can_train_pu_sandbox="true",
                leakage_risk_status=control.get("leakage_risk_status") or "LEAKAGE_PROTOCOL_REQUIRED",
                scientific_claim_status="INVALID_FOR_SUPERVISED_CLAIM",
                blocking_reason="BACKGROUND_UNLABELED_IS_NOT_NEGATIVE",
                minimum_evidence_needed=minimum_background_evidence(),
                notes="Background material can support sampling balance and PU sandbox, not negative labels.",
            )
        )
    for taxonomy in taxonomy_rows:
        rows.append(
            ladder_row(
                candidate_id=f"BG_{taxonomy.get('taxonomy_id', '')}",
                candidate_type="BACKGROUND_UNLABELED",
                region="MULTI_REGION",
                source_evidence=taxonomy.get("source_registry_or_manifest", "patch_corpus_taxonomy_registry.csv"),
                source_type="PATCH_CORPUS_TAXONOMY_REGISTRY",
                coordinate_status="CORPUS_LAYER_NO_INDIVIDUAL_BACKGROUND_POINT",
                temporal_status="CORPUS_LAYER_NO_ABSENCE_WINDOW",
                phenomenon_status="NO_NEGATIVE_PHENOMENON_ASSERTION",
                explicit_absence_or_stability_evidence="false",
                environmental_stratum=taxonomy.get("corpus_layer", "CORPUS_LAYER"),
                negative_evidence_status="INSUFFICIENT_EVIDENCE",
                background_status="BACKGROUND_UNLABELED",
                pu_status="UNLABELED_CONTEXT_ONLY",
                can_be_background_unlabeled="true",
                can_train_pu_sandbox="true",
                leakage_risk_status="LEAKAGE_PROTOCOL_REQUIRED",
                scientific_claim_status="INVALID_FOR_SUPERVISED_CLAIM",
                blocking_reason="CORPUS_TAXONOMY_DOES_NOT_ESTABLISH_ABSENCE",
                minimum_evidence_needed=minimum_background_evidence(),
                notes="Taxonomy records describe available corpus layers, not negative outcomes.",
            )
        )
    return rows


def strong_control_ladder_rows(formal_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in formal_rows:
        if row.get("control_strength_status") != "STRONG_CONTROL_CANDIDATE":
            continue
        rows.append(
            ladder_row(
                candidate_id=row.get("candidate_id", ""),
                candidate_type="STRONG_CONTROL_REVIEW_ONLY",
                region=row.get("region", ""),
                source_evidence=row.get("source_layer", "formal_negative_control_evidence_registry.csv"),
                source_type="TEMPORAL_SELF_CONTROL",
                coordinate_status=row.get("coordinate_available", ""),
                temporal_status="PRE_EVENT_SAME_ANCHOR",
                phenomenon_status="SAME_ANCHOR_CONTEXT_ONLY",
                explicit_absence_or_stability_evidence="false",
                distance_to_nearest_positive_anchor_m=row.get("distance_to_nearest_positive_anchor_m", ""),
                within_positive_buffer="true",
                environmental_stratum="SAME_ANCHOR_TEMPORAL_BASELINE",
                s2_status="S2_PRE_EVENT_AVAILABLE",
                s1_status="S1_PARTIAL_OR_NOT_AVAILABLE",
                dem_status="DEM_AVAILABLE",
                dino_status="DINO_AVAILABLE",
                negative_evidence_status="STRONG_CONTROL_REVIEW_ONLY",
                pu_status="UNLABELED_CONTROL_CONTEXT_ONLY",
                can_train_pu_sandbox="true",
                leakage_risk_status=row.get("leakage_risk_status", "LEAKAGE_RISK_HIGH"),
                scientific_claim_status="INVALID_FOR_SUPERVISED_CLAIM",
                blocking_reason="PRE_EVENT_SAME_ANCHOR_IS_NOT_INDEPENDENT_NEGATIVE",
                minimum_evidence_needed=minimum_formal_negative_evidence(),
                notes="Strong temporal review control, but split independence fails for negative labels.",
            )
        )
    return rows


def pu_boundary(ladder_rows: list[dict[str, Any]]) -> dict[str, Any]:
    anchors = read_csv(ANCHORS)
    formal_ready = sum(1 for row in ladder_rows if row["negative_evidence_status"] == "FORMAL_NEGATIVE_READY")
    pseudo_review = sum(1 for row in ladder_rows if row["pseudo_absence_status"] == "PSEUDO_ABSENCE_REVIEW_ONLY")
    background = sum(1 for row in ladder_rows if row["background_status"] == "BACKGROUND_UNLABELED")
    strong = sum(1 for row in ladder_rows if row["candidate_type"] == "STRONG_CONTROL_REVIEW_ONLY")
    unlabeled = pseudo_review + background + strong
    can_pu = len(anchors) > 0 and unlabeled > 0
    return {
        "boundary_id": "V1JM_POSITIVE_UNLABELED_BOUNDARY",
        "positive_count": str(len(anchors)),
        "unlabeled_candidate_count": str(unlabeled),
        "pseudo_absence_review_only_count": str(pseudo_review),
        "background_unlabeled_count": str(background),
        "strong_control_review_only_count": str(strong),
        "formal_negative_ready_count": str(formal_ready),
        "pu_boundary_status": "PU_SANDBOX_LOCAL_ONLY_READY" if can_pu else "PU_SANDBOX_BLOCKED_NO_UNLABELED",
        "supervised_training_status": "SUPERVISED_TRAINING_BLOCKED_NO_FORMAL_NEGATIVES" if formal_ready == 0 else "FORMAL_SUPERVISED_GATE_REVIEW_REQUIRED",
        "metrics_status": "EXPLORATORY_ONLY_INVALID_FOR_SUPERVISED_CLAIM",
        "model_artifact_status": "NO_MODEL_OR_WEIGHTS_SAVED",
        "can_create_training_label": "false",
        "can_train_supervised_model": "false",
        "can_train_pu_sandbox": str(can_pu).lower(),
        "can_save_model_weights": "false",
        "scientific_claim_status": "PU_SANDBOX_LOCAL_ONLY_NO_PERFORMANCE_CLAIM" if can_pu else "NO_MODEL_CLAIM",
        "blocking_reason": "FORMAL_NEGATIVES_ZERO; UNLABELED_IS_NOT_NEGATIVE",
        "notes": "Positives are official anchors; unlabeled material includes pseudo-absence/background/control candidates only.",
    }


def external_option() -> list[dict[str, Any]]:
    return [
        {
            "option_id": "V1JM_EXT_LANDSLIDE4SENSE_TRANSFER_OPTION",
            "benchmark_name": "Landslide4Sense",
            "benchmark_family": "EXTERNAL_LANDSLIDE_BENCHMARK",
            "download_status": "NOT_DOWNLOADED",
            "local_ground_truth_status": "EXTERNAL_NEGATIVE_NOT_LOCAL_GROUND_TRUTH",
            "role_in_revp": "EXTERNAL_SUPERVISED_PRETRAINING_OPTION",
            "can_supply_local_negative_ground_truth": "false",
            "can_enable_local_supervised_claim": "false",
            "transfer_status": "TRANSFER_REVIEW_ONLY",
            "compatibility_status": "COMPATIBLE_AS_OPTION_REQUIRES_SEPARATE_LICENSE_STORAGE_AND_DOMAIN_SHIFT_AUDIT",
            "minimum_evidence_needed": "dataset license; class semantics audit; sensor/region/domain-shift audit; no transfer of external negatives into local ground truth",
            "notes": "Benchmark can be evaluated as external pretraining or comparison option only; it does not solve local formal negatives.",
        }
    ]


def decision_status(formal_ready: int, pseudo_count: int, background_count: int, can_pu: bool) -> str:
    if formal_ready > 0:
        return "FORMAL_SUPERVISED_GATE_READY"
    if can_pu:
        return "PU_SANDBOX_LOCAL_ONLY_READY"
    if pseudo_count > 0 or background_count > 0:
        return "REVIEW_ONLY_PLUS_PSEUDO_ABSENCE_READY"
    return "SUPERVISED_TRAINING_BLOCKED_NO_FORMAL_NEGATIVES"


def build_qa(
    formal_scan: list[dict[str, str]],
    ladder_rows: list[dict[str, Any]],
    pseudo_rows: list[dict[str, Any]],
    background_rows: list[dict[str, Any]],
    pu_row: dict[str, Any],
    external_rows: list[dict[str, Any]],
    public_files: list[Path],
) -> list[dict[str, str]]:
    formal_ready = [row for row in ladder_rows if row["negative_evidence_status"] == "FORMAL_NEGATIVE_READY"]
    absence_record_invalid = [row for row in ladder_rows if row["negative_evidence_status"] == "INVALID_NEGATIVE_ABSENCE_ASSUMPTION"]
    distance_only_negatives = [
        row for row in ladder_rows if row["distance_to_nearest_positive_anchor_m"] and row["can_be_formal_negative"] == "true" and row["explicit_absence_or_stability_evidence"] != "true"
    ]
    qa = [
        {"check": "formal_negative_requires_explicit_absence_or_stability", "status": "PASS" if not formal_ready else "FAIL", "detail": str(len(formal_ready))},
        {"check": "absence_of_record_not_negative", "status": "PASS" if absence_record_invalid and all(row["can_be_formal_negative"] == "false" for row in absence_record_invalid) else "FAIL", "detail": str(len(absence_record_invalid))},
        {"check": "distance_from_anchor_not_negative", "status": "PASS" if not distance_only_negatives else "FAIL", "detail": str(len(distance_only_negatives))},
        {"check": "pseudo_absence_not_label", "status": "PASS" if pseudo_rows and all(row["can_create_training_label"] == "false" and row["can_be_formal_negative"] == "false" for row in pseudo_rows) else "FAIL", "detail": str(len(pseudo_rows))},
        {"check": "background_not_label", "status": "PASS" if background_rows and all(row["can_create_training_label"] == "false" and row["can_be_formal_negative"] == "false" for row in background_rows) else "FAIL", "detail": str(len(background_rows))},
        {"check": "pu_sandbox_not_supervised_claim", "status": "PASS" if pu_row["can_train_pu_sandbox"] == "true" and pu_row["can_train_supervised_model"] == "false" else "FAIL", "detail": pu_row["pu_boundary_status"]},
        {"check": "can_train_supervised_model_false_if_formal_negatives_zero", "status": "PASS" if pu_row["formal_negative_ready_count"] == "0" and pu_row["can_train_supervised_model"] == "false" else "FAIL", "detail": pu_row["formal_negative_ready_count"]},
        {"check": "can_create_training_label_false_if_gates_incomplete", "status": "PASS" if all(row["can_create_training_label"] == "false" for row in ladder_rows) and pu_row["can_create_training_label"] == "false" else "FAIL", "detail": "ladder_and_pu_checked"},
        {"check": "external_benchmark_not_local_ground_truth", "status": "PASS" if all(row["can_supply_local_negative_ground_truth"] == "false" for row in external_rows) else "FAIL", "detail": external_rows[0]["local_ground_truth_status"]},
        {"check": "formal_scan_executed", "status": "PASS" if formal_scan else "FAIL", "detail": str(len(formal_scan))},
        {"check": "no_private_path_in_public_outputs", "status": "PASS" if not any(has_private_fragment(path) for path in public_files if path.exists()) else "FAIL", "detail": "public registries checked"},
    ]
    return qa


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepare(args.force)
    controls = read_csv(CONTROLS)
    formal_control_rows = read_csv(FORMAL_NEGATIVE)
    taxonomy = read_csv(PATCH_TAXONOMY)

    formal_scan = scan_sources() if args.audit_formal_negative_evidence else []
    pseudo_rows = pseudo_absence_candidates(controls) if args.build_pseudo_absence_candidates else []
    background_rows = background_unlabeled_candidates(controls, taxonomy) if args.build_background_unlabeled_candidates else []
    strong_rows = strong_control_ladder_rows(formal_control_rows)
    formal_ladder_rows = ladder_from_formal_scan(formal_scan)
    ladder_rows = formal_ladder_rows + strong_rows + pseudo_rows + background_rows
    external_rows = external_option() if args.evaluate_external_benchmark_option else []
    pu_row = pu_boundary(ladder_rows) if args.evaluate_pu_boundary else {
        field: "" for field in PU_FIELDS
    }

    formal_ready = sum(1 for row in ladder_rows if row["negative_evidence_status"] == "FORMAL_NEGATIVE_READY")
    formal_review = sum(1 for row in ladder_rows if row["negative_evidence_status"] == "FORMAL_NEGATIVE_CANDIDATE_REVIEW")
    pseudo_count = len(pseudo_rows)
    background_count = len(background_rows)
    can_pu = pu_row.get("can_train_pu_sandbox") == "true"
    ladder_decision = decision_status(formal_ready, pseudo_count, background_count, can_pu)

    if args.emit_negative_ladder:
        write_csv(LOCAL_RUN_DIR / "v1jm_formal_negative_evidence_scan.csv", formal_scan, FORMAL_SCAN_FIELDS)
        write_csv(LOCAL_RUN_DIR / "v1jm_pseudo_absence_candidate_audit.csv", pseudo_rows, PSEUDO_FIELDS)
        write_csv(LOCAL_RUN_DIR / "v1jm_background_unlabeled_candidate_audit.csv", background_rows, BACKGROUND_FIELDS)
        write_csv(LOCAL_RUN_DIR / "v1jm_negative_ladder_decision.csv", ladder_rows, LADDER_FIELDS)
        write_csv(LOCAL_RUN_DIR / "v1jm_pu_boundary_matrix.csv", [pu_row], PU_FIELDS)
        write_csv(LOCAL_RUN_DIR / "v1jm_external_benchmark_option.csv", external_rows, EXTERNAL_FIELDS)

        write_csv(PUBLIC_LADDER, ladder_rows, LADDER_FIELDS)
        write_csv(PUBLIC_PSEUDO, pseudo_rows, PSEUDO_FIELDS)
        write_csv(PUBLIC_BACKGROUND, background_rows, BACKGROUND_FIELDS)
        write_csv(PUBLIC_PU, [pu_row], PU_FIELDS)
        write_csv(PUBLIC_EXTERNAL, external_rows, EXTERNAL_FIELDS)
        write_schema(SCHEMAS_DIR / "negative_evidence_ladder_schema.csv", LADDER_FIELDS, "REV-P v1jm negative evidence ladder field")
        write_schema(SCHEMAS_DIR / "pseudo_absence_candidate_schema.csv", PSEUDO_FIELDS, "REV-P v1jm pseudo-absence candidate field")
        write_schema(SCHEMAS_DIR / "background_unlabeled_candidate_schema.csv", BACKGROUND_FIELDS, "REV-P v1jm background unlabeled field")
        write_schema(SCHEMAS_DIR / "positive_unlabeled_boundary_schema.csv", PU_FIELDS, "REV-P v1jm positive-unlabeled boundary field")
        write_schema(SCHEMAS_DIR / "external_benchmark_transfer_option_schema.csv", EXTERNAL_FIELDS, "REV-P v1jm external benchmark transfer option field")

    public_files = [
        PUBLIC_LADDER,
        PUBLIC_PSEUDO,
        PUBLIC_BACKGROUND,
        PUBLIC_PU,
        PUBLIC_EXTERNAL,
        SCHEMAS_DIR / "negative_evidence_ladder_schema.csv",
        SCHEMAS_DIR / "pseudo_absence_candidate_schema.csv",
        SCHEMAS_DIR / "background_unlabeled_candidate_schema.csv",
        SCHEMAS_DIR / "positive_unlabeled_boundary_schema.csv",
        SCHEMAS_DIR / "external_benchmark_transfer_option_schema.csv",
    ]
    qa_rows = build_qa(formal_scan, ladder_rows, pseudo_rows, background_rows, pu_row, external_rows, public_files)
    write_csv(LOCAL_RUN_DIR / "v1jm_qa.csv", qa_rows, QA_FIELDS)

    summary = {
        "stage": "v1jm",
        "timestamp": utc_now(),
        "formal_negative_ready_count": formal_ready,
        "formal_negative_candidate_review_count": formal_review,
        "pseudo_absence_candidate_rows": pseudo_count,
        "pseudo_absence_ready_for_review_count": sum(1 for row in pseudo_rows if row["can_be_pseudo_absence"] == "true"),
        "background_unlabeled_candidate_rows": background_count,
        "strong_control_review_only_count": len(strong_rows),
        "pu_boundary_status": pu_row.get("pu_boundary_status", ""),
        "can_train_pu_sandbox": can_pu,
        "can_create_training_label": False,
        "can_train_supervised_model": False,
        "can_unfreeze_dino_for_scientific_claim": False,
        "external_benchmark_option_count": len(external_rows),
        "negative_ladder_decision_status": ladder_decision,
        "qa_status": "PASS" if all(row["status"] == "PASS" for row in qa_rows) else "FAIL",
    }
    write_json(LOCAL_RUN_DIR / "v1jm_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--audit-formal-negative-evidence", action="store_true")
    parser.add_argument("--build-pseudo-absence-candidates", action="store_true")
    parser.add_argument("--build-background-unlabeled-candidates", action="store_true")
    parser.add_argument("--evaluate-pu-boundary", action="store_true")
    parser.add_argument("--evaluate-external-benchmark-option", action="store_true")
    parser.add_argument("--emit-negative-ladder", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print("REV-P v1jm NEGATIVE EVIDENCE LADDER AND PSEUDO-ABSENCE PROTOCOL")
    print(f"Formal negatives ready: {summary['formal_negative_ready_count']}")
    print(f"Formal negative candidates in review: {summary['formal_negative_candidate_review_count']}")
    print(f"Pseudo-absence audit rows: {summary['pseudo_absence_candidate_rows']}")
    print(f"Background unlabeled rows: {summary['background_unlabeled_candidate_rows']}")
    print(f"PU boundary: {summary['pu_boundary_status']}")
    print(f"Supervised training enabled: {summary['can_train_supervised_model']}")
    print(f"QA status: {summary['qa_status']}")
    print("No git add, commit, push, training, model export, or heavy data download was performed.")
    return 0 if summary["qa_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
