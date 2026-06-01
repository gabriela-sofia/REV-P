"""REV-P v1rc — External document intake validator.

Validates a manually-filled intake CSV referenced by
REVP_PROTOCOL_C_EXTERNAL_INTAKE_PATH. When absent, fail-closed with headers.
Emits one validation row PER CHECK per document. URLs are never fetched.
Review-only; never creates confirmed events, labels, targets or ground truth.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from revp_v1ra_v1rf_external_intake_common import (
    DATASETS,
    DOCS,
    SCHEMAS,
    _p,
    assert_clean_rows,
    classify_source_family,
    detect_absolute_path,
    detect_local_runs_exposure,
    guardrail_row,
    normalize_region,
    normalize_source_name,
    read_csv_safe,
    validate_license_access,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_VALIDATION = _p("REVP_V1RC_OUT_VALIDATION", DATASETS / "protocol_c_external_document_intake_validation_v1rc.csv")
OUT_SUMMARY = _p("REVP_V1RC_OUT_SUMMARY", DATASETS / "protocol_c_external_document_intake_validation_summary_v1rc.csv")
SCHEMA_VALIDATION = _p("REVP_V1RC_SCHEMA_VALIDATION", SCHEMAS / "protocol_c_external_document_intake_validation_v1rc_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1RC_SCHEMA_SUMMARY", SCHEMAS / "protocol_c_external_document_intake_validation_summary_v1rc_schema.csv")
DOC = _p("REVP_V1RC_DOC", DOCS / "revp_v1rc_external_document_intake_validator.md")

VALIDATION_FIELDS = [
    "validation_id", "document_id", "source_name", "source_family", "region",
    "hazard_type", "check_name", "status", "severity", "observed_value",
    "expected_value", "blocks_c3", "blocks_c4", "review_only",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative", "blocked_reason", "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]

WAITING = "EXTERNAL_INTAKE_WAITING_MANUAL_DOCUMENTS"
PASS = "EXTERNAL_INTAKE_VALIDATION_PASS_REVIEW_ONLY"
FAIL_CLOSED = "EXTERNAL_INTAKE_VALIDATION_FAIL_CLOSED"

REQUIRED_FIELDS = [
    "document_id", "source_name", "region", "hazard_type",
    "event_date_text", "event_location_text", "url_or_reference",
    "license_note", "evidence_type", "temporal_precision_claim",
    "spatial_precision_claim", "intake_status",
]

_TEMPORAL_OK = {"DAY", "MONTH", "YEAR", "DAY_EXPLICIT", "MONTH_PERIOD", "YEAR_PERIOD"}
_SPATIAL_OK = {"POINT", "ADDRESS", "ADMINISTRATIVE", "POINT_EXPLICIT", "ADDRESS_LEVEL"}


def _intake_path() -> Path | None:
    env = os.environ.get("REVP_PROTOCOL_C_EXTERNAL_INTAKE_PATH")
    if env and Path(env).exists():
        return Path(env)
    return None


def _check(base: dict[str, str], idx: int, check_name: str, ok: bool,
           severity: str, observed: str, expected: str,
           blocks_c3: bool = False, reason: str = "") -> dict[str, Any]:
    row = dict(base)
    row.update({
        "validation_id": f"V1RC_VAL_{idx:05d}",
        "check_name": check_name,
        "status": "PASS" if ok else "FAIL",
        "severity": severity,
        "observed_value": observed[:80],
        "expected_value": expected,
        "blocks_c3": "true" if (blocks_c3 and not ok) else "false",
        "blocks_c4": "false",
        "blocked_reason": "" if ok else (reason or check_name),
        "notes": "",
    })
    row.update(guardrail_row())
    return row


def validate_document(rows_out: list[dict[str, Any]], counter: list[int], doc: dict[str, str]) -> None:
    source_name = normalize_source_name(doc.get("source_name", ""))
    family = classify_source_family(source_name, doc.get("evidence_type", ""))
    region = normalize_region(doc.get("region", ""))
    base = {
        "document_id": doc.get("document_id", ""),
        "source_name": source_name,
        "source_family": family,
        "region": region,
        "hazard_type": doc.get("hazard_type", ""),
    }

    def emit(name, ok, sev, obs, exp, b3=False, reason=""):
        rows_out.append(_check(base, counter[0], name, ok, sev, str(obs), exp, b3, reason))
        counter[0] += 1

    # Required fields
    for f in REQUIRED_FIELDS:
        val = str(doc.get(f, "")).strip()
        emit(f"required_field_{f}", bool(val), "critical", val, "non_empty",
             b3=True, reason=f"MISSING_REQUIRED_{f.upper()}")

    # Source family strength
    emit("weak_source_family", family != "UNKNOWN_SOURCE", "high", family,
         "resolved_source_family", b3=True, reason="SOURCE_FAMILY_UNRESOLVED")

    # Temporal precision
    temporal = str(doc.get("temporal_precision_claim", "")).strip().upper()
    emit("missing_temporal_precision", temporal in _TEMPORAL_OK, "high", temporal,
         "DAY|MONTH|YEAR", b3=True, reason="TEMPORAL_PRECISION_MISSING")

    # Spatial precision
    spatial = str(doc.get("spatial_precision_claim", "")).strip().upper()
    emit("missing_spatial_precision", spatial in _SPATIAL_OK, "high", spatial,
         "POINT|ADDRESS|ADMINISTRATIVE", b3=True, reason="SPATIAL_PRECISION_MISSING")

    # Provenance
    provenance = str(doc.get("url_or_reference", "")).strip() or str(doc.get("local_document_hash", "")).strip()
    emit("missing_provenance", bool(provenance), "high", provenance,
         "url_or_reference_or_hash", b3=True, reason="PROVENANCE_MISSING")

    # License / access
    license_status, usable = validate_license_access(doc.get("license_note", ""))
    emit("license_access_unknown", usable, "medium", license_status,
         "license_declared_or_open", reason="LICENSE_ACCESS_UNKNOWN")

    # Path / URL safety (no fetch, just inspect string)
    ref = str(doc.get("url_or_reference", "")) + " " + str(doc.get("local_document_hash", ""))
    unsafe = detect_absolute_path(ref) or detect_local_runs_exposure(ref)
    emit("path_url_unsafe", not unsafe, "critical", "unsafe" if unsafe else "safe",
         "no_absolute_path_no_localrun", b3=True, reason="UNSAFE_PATH_OR_URL")


def run(datasets: Path | None = None) -> dict[str, Any]:
    intake_path = _intake_path()
    rows: list[dict[str, Any]] = []
    counter = [0]
    status = WAITING

    if intake_path is not None:
        intake = read_csv_safe(intake_path)
        data_rows = [r for r in intake if any(str(v).strip() for v in r.values())]
        if data_rows:
            for doc in data_rows:
                validate_document(rows, counter, doc)
            any_fail = any(r["status"] == "FAIL" and r["severity"] in ("critical", "high") for r in rows)
            status = FAIL_CLOSED if any_fail else PASS

    assert_clean_rows(rows, "v1rc_validation")
    write_csv_with_header(OUT_VALIDATION, rows, VALIDATION_FIELDS)
    write_schema_safe(SCHEMA_VALIDATION, VALIDATION_FIELDS, "v1rc_validation")

    docs = {r["document_id"] for r in rows}
    failed_checks = sum(1 for r in rows if r["status"] == "FAIL")
    passed_checks = sum(1 for r in rows if r["status"] == "PASS")
    blocked_docs = {r["document_id"] for r in rows if r["status"] == "FAIL" and r["blocks_c3"] == "true"}

    summary = [
        {"stat_key": "validation_status", "stat_value": status},
        {"stat_key": "documents_examined", "stat_value": str(len(docs))},
        {"stat_key": "checks_total", "stat_value": str(len(rows))},
        {"stat_key": "checks_passed", "stat_value": str(passed_checks)},
        {"stat_key": "checks_failed", "stat_value": str(failed_checks)},
        {"stat_key": "documents_blocked_for_c3", "stat_value": str(len(blocked_docs))},
        {"stat_key": "intake_path_present", "stat_value": "true" if intake_path else "false"},
        {"stat_key": "urls_fetched", "stat_value": "0"},
        {"stat_key": "stage", "stat_value": "v1rc"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1rc_summary")

    write_doc(
        DOC,
        "v1rc — External Document Intake Validator",
        [
            "## Objetivo",
            "Validar um CSV de intake preenchido manualmente "
            "(REVP_PROTOCOL_C_EXTERNAL_INTAKE_PATH). Sem o arquivo, fail-closed "
            "(EXTERNAL_INTAKE_WAITING_MANUAL_DOCUMENTS). Uma linha por checagem por documento.",
            "## Checagens",
            "required_field_*, weak_source_family, missing_temporal_precision, "
            "missing_spatial_precision, missing_provenance, license_access_unknown, "
            "path_url_unsafe. URLs nunca sao acessadas.",
            "## Resultado",
            f"Status: {status}. Documentos: {len(docs)}. Checagens: {len(rows)} "
            f"(passou {passed_checks}, falhou {failed_checks}).",
            "## Guardrails",
            "Nenhuma URL e baixada. Nenhum evento confirmado. review_only=true. "
            "formal_negative=false. Nenhum label/target/ground truth operacional.",
        ],
    )
    print(f"[v1rc] status={status} docs={len(docs)} checks={len(rows)} failed={failed_checks}")
    return {"status": status, "docs": len(docs), "checks": len(rows), "failed": failed_checks}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1rc intake validator").parse_args()
    run()
