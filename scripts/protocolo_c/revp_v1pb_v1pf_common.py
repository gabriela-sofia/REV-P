"""Shared helpers for REV-P Protocol C v1pb-v1pf.

Finalization block: orchestration, invariant audit, TCC export,
methodological report, and final bundle. No scientific decisions
are changed — this block only audits, exports, and documents.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
SCHEMAS = DATASETS / "schemas"
DOCS = ROOT / "docs" / "metodologia_cientifica"

# ---------------------------------------------------------------------------
# Expected statuses from v1og-v1pa
# ---------------------------------------------------------------------------

EXPECTED_TEMPORAL_STATUS = "TEMPORAL_RECOVERY_FAIL_CLOSED"
EXPECTED_OBSERVATIONAL_STATUS = "OBSERVED_EVIDENCE_REVIEW_ONLY_FAIL_CLOSED"

# ---------------------------------------------------------------------------
# Forbidden patterns — values that must NEVER appear as "true" in outputs
# ---------------------------------------------------------------------------

FORBIDDEN_TRUE_FIELDS = [
    "ground_truth",
    "can_train_model",
    "can_create_operational_label",
    "can_be_used_as_ground_truth",
    "can_promote_to_label",
    "dino_can_create_label",
    "dino_can_train_model",
    "dino_target_field_created",
    "can_be_used_for_training",
    "can_create_label",
]

FORBIDDEN_CSV_PATTERNS = [
    ("ground_truth,true", "GROUND_TRUTH_TRUE"),
    ("can_train_model,true", "CAN_TRAIN_MODEL_TRUE"),
    ("can_create_operational_label,true", "CAN_CREATE_OP_LABEL_TRUE"),
    ("can_be_used_as_ground_truth,true", "CAN_BE_GROUND_TRUTH_TRUE"),
    ("can_promote_to_label,true", "CAN_PROMOTE_TO_LABEL_TRUE"),
    ("dino_can_create_label,true", "DINO_CAN_CREATE_LABEL_TRUE"),
    ("dino_can_train_model,true", "DINO_CAN_TRAIN_MODEL_TRUE"),
    ("dino_target_field_created,true", "DINO_TARGET_FIELD_CREATED_TRUE"),
]

ABS_PATH_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")

# ---------------------------------------------------------------------------
# Expected outputs from v1og-v1pa
# ---------------------------------------------------------------------------

V1OG_V1OT_OUTPUTS = [
    "recife_patch_provenance_graph_registry.csv",
    "recife_sentinel_sidecar_discovery_v1om.csv",
    "recife_sentinel_product_date_candidates_v1on.csv",
    "recife_patch_scene_date_resolved_v3_v1oo.csv",
    "recife_event_patch_temporal_adjudication_v3_v1op.csv",
    "recife_c3_plus_recheck_after_scene_date_v3_v1oq.csv",
    "recife_scene_date_recovery_v3_master_summary_v1or.csv",
    "recife_fixture_contamination_audit_v1os.csv",
    "recife_scene_date_recovery_final_manifest_v1ot.csv",
    "recife_scene_date_recovery_final_quality_checks_v1ot.csv",
    "recife_scene_date_recovery_final_scientific_summary_v1ot.csv",
]

V1OU_V1PA_OUTPUTS = [
    "recife_external_evidence_source_inventory_v1ou.csv",
    "recife_external_evidence_source_inventory_summary_v1ou.csv",
    "recife_ground_reference_observed_event_registry_v1ov.csv",
    "recife_ground_reference_observed_event_summary_v1ov.csv",
    "recife_ground_reference_evidence_scoring_v1ow.csv",
    "recife_ground_reference_evidence_scoring_summary_v1ow.csv",
    "recife_event_patch_linkage_registry_v1ox.csv",
    "recife_event_patch_linkage_summary_v1ox.csv",
    "recife_ground_truth_candidate_decision_audit_v1oy.csv",
    "recife_ground_truth_candidate_decision_summary_v1oy.csv",
    "recife_dino_review_only_representation_queue_v1oz.csv",
    "recife_dino_review_only_representation_summary_v1oz.csv",
    "recife_protocol_c_observed_evidence_manifest_v1pa.csv",
    "recife_protocol_c_observed_evidence_quality_checks_v1pa.csv",
    "recife_protocol_c_observed_evidence_scientific_summary_v1pa.csv",
]

ALL_EXPECTED_OUTPUTS = V1OG_V1OT_OUTPUTS + V1OU_V1PA_OUTPUTS

# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _p(env: str, default: Path) -> Path:
    return Path(os.environ[env]) if env in os.environ else default


def read_csv_safe(path: Path, required_columns: list[str] | None = None) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            fields = list(reader.fieldnames or [])
            if required_columns:
                missing = [c for c in required_columns if c not in fields]
                if missing:
                    return []
            return list(reader)
    except Exception:
        return []


def write_csv_with_header(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fieldnames})


def write_schema(path: Path, fields: list[str], prefix: str) -> None:
    write_csv_with_header(
        path,
        [{"field": f, "description": f"{prefix}: {f}."} for f in fields],
        ["field", "description"],
    )


def sha256_16(path: Path) -> str:
    if not path.exists() or path.stat().st_size > 20_000_000:
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def relative_safe_path(path: Path, root: Path | None = None) -> str:
    root = root or ROOT
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def has_forbidden_pattern(text: str) -> list[str]:
    """Return list of forbidden pattern labels found in text."""
    violations: list[str] = []
    text_lower = text.lower()
    for pattern, label in FORBIDDEN_CSV_PATTERNS:
        if pattern.lower() in text_lower:
            violations.append(label)
    return violations


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return -1
    try:
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            return sum(1 for _ in reader)
    except Exception:
        return -1


def load_metric_from_summary(path: Path, metric_name: str) -> str:
    rows = read_csv_safe(path, ["metric", "value"])
    for r in rows:
        if r.get("metric") == metric_name:
            return r.get("value", "N/A")
    # Try stat_key/stat_value format
    rows2 = read_csv_safe(path, ["stat_key", "stat_value"])
    for r in rows2:
        if r.get("stat_key") == metric_name:
            return r.get("stat_value", "N/A")
    return "N/A"


def ensure_no_absolute_windows_path(value: str) -> bool:
    return not bool(ABS_PATH_RE.search(value))


def ensure_review_only_guardrails(row: dict[str, str]) -> list[str]:
    """Return list of violation descriptions for a row."""
    violations: list[str] = []
    for field in FORBIDDEN_TRUE_FIELDS:
        val = str(row.get(field, "")).strip().lower()
        if val == "true":
            violations.append(f"{field}=true")
    return violations


def status_pass_fail(condition: bool, fail_reason: str) -> tuple[str, str]:
    if condition:
        return ("PASS", "")
    return ("FAIL", fail_reason)


def emit_doc(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
