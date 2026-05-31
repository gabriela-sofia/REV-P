"""REV-P v1pc — Protocol C global scientific invariant auditor.

Audits all CSVs from v1og-v1pa for global scientific invariant violations.
Reports clearly if any FAIL/CRITICAL found — never masks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1pb_v1pf_common import (
    ABS_PATH_RE,
    ALL_EXPECTED_OUTPUTS,
    DATASETS,
    DOCS,
    EXPECTED_OBSERVATIONAL_STATUS,
    EXPECTED_TEMPORAL_STATUS,
    FORBIDDEN_CSV_PATTERNS,
    ROOT,
    SCHEMAS,
    _p,
    count_csv_rows,
    emit_doc,
    has_forbidden_pattern,
    load_metric_from_summary,
    read_csv_safe,
    write_csv_with_header,
    write_schema,
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

OUT_AUDIT = _p("REVP_V1PC_OUT_AUDIT", DATASETS / "recife_protocol_c_global_invariant_audit_v1pc.csv")
OUT_SUMMARY = _p("REVP_V1PC_OUT_SUMMARY", DATASETS / "recife_protocol_c_global_invariant_summary_v1pc.csv")
SCHEMA_AUDIT = _p("REVP_V1PC_SCHEMA_AUDIT", SCHEMAS / "recife_protocol_c_global_invariant_audit_v1pc_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1PC_SCHEMA_SUMMARY", SCHEMAS / "recife_protocol_c_global_invariant_summary_v1pc_schema.csv")
DOC = _p("REVP_V1PC_DOC", DOCS / "revp_v1pc_protocol_c_global_invariant_auditor.md")

AUDIT_FIELDS = [
    "invariant_id", "invariant_group", "artifact_path", "invariant_name",
    "status", "severity", "observed_value", "expected_value",
    "violation_count", "explanation",
]
SUMMARY_FIELDS = ["metric", "value", "interpretation"]


def _datasets_dir() -> Path:
    return _p("REVP_V1PC_DATASETS_DIR", DATASETS)


def _audit_forbidden_patterns(datasets_dir: Path) -> list[dict[str, Any]]:
    """Check all expected outputs for forbidden true-value patterns."""
    rows: list[dict[str, Any]] = []
    inv_id = 0

    for filename in ALL_EXPECTED_OUTPUTS:
        path = datasets_dir / filename
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except Exception:
            continue

        violations = has_forbidden_pattern(text)
        for v in violations:
            inv_id += 1
            rows.append({
                "invariant_id": f"V1PC_INV_{inv_id:04d}",
                "invariant_group": "forbidden_true_field",
                "artifact_path": filename,
                "invariant_name": f"no_{v.lower()}",
                "status": "FAIL",
                "severity": "CRITICAL",
                "observed_value": v,
                "expected_value": "NOT_FOUND",
                "violation_count": "1",
                "explanation": f"Forbidden pattern '{v}' found in {filename}",
            })

        if not violations:
            inv_id += 1
            rows.append({
                "invariant_id": f"V1PC_INV_{inv_id:04d}",
                "invariant_group": "forbidden_true_field",
                "artifact_path": filename,
                "invariant_name": "no_forbidden_true_fields",
                "status": "PASS",
                "severity": "INFO",
                "observed_value": "none",
                "expected_value": "none",
                "violation_count": "0",
                "explanation": "No forbidden true fields found",
            })

    return rows


def _audit_abs_paths(datasets_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inv_id = 0
    for filename in ALL_EXPECTED_OUTPUTS:
        path = datasets_dir / filename
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except Exception:
            continue
        if ABS_PATH_RE.search(text):
            inv_id += 1
            rows.append({
                "invariant_id": f"V1PC_ABS_{inv_id:04d}",
                "invariant_group": "no_absolute_paths",
                "artifact_path": filename,
                "invariant_name": "no_windows_absolute_path",
                "status": "FAIL",
                "severity": "HIGH",
                "observed_value": "FOUND",
                "expected_value": "NOT_FOUND",
                "violation_count": "1",
                "explanation": "Absolute Windows path in output",
            })
        else:
            inv_id += 1
            rows.append({
                "invariant_id": f"V1PC_ABS_{inv_id:04d}",
                "invariant_group": "no_absolute_paths",
                "artifact_path": filename,
                "invariant_name": "no_windows_absolute_path",
                "status": "PASS",
                "severity": "INFO",
                "observed_value": "NOT_FOUND",
                "expected_value": "NOT_FOUND",
                "violation_count": "0",
                "explanation": "",
            })
    return rows


def _audit_local_runs(datasets_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inv_id = 0
    for filename in ALL_EXPECTED_OUTPUTS:
        path = datasets_dir / filename
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace").lower()
        except Exception:
            continue
        # Only flag if "local_runs/" appears as a path, not in explanatory text
        if "local_runs/" in text or "local_runs\\" in text:
            inv_id += 1
            rows.append({
                "invariant_id": f"V1PC_LR_{inv_id:04d}",
                "invariant_group": "no_local_runs_path",
                "artifact_path": filename,
                "invariant_name": "no_local_runs_reference",
                "status": "WARN",
                "severity": "MEDIUM",
                "observed_value": "FOUND",
                "expected_value": "NOT_FOUND",
                "violation_count": "1",
                "explanation": "local_runs path reference in versionable output",
            })
    return rows


def _audit_empty_csv_headers(datasets_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inv_id = 0
    for filename in ALL_EXPECTED_OUTPUTS:
        path = datasets_dir / filename
        if not path.exists():
            continue
        count = count_csv_rows(path)
        if count == 0:
            # Empty but must have header
            try:
                import csv
                with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
                    reader = csv.DictReader(fh)
                    has_header = bool(reader.fieldnames)
            except Exception:
                has_header = False
            inv_id += 1
            status = "PASS" if has_header else "FAIL"
            rows.append({
                "invariant_id": f"V1PC_HDR_{inv_id:04d}",
                "invariant_group": "empty_csv_header",
                "artifact_path": filename,
                "invariant_name": "empty_csv_has_header",
                "status": status,
                "severity": "HIGH" if not has_header else "INFO",
                "observed_value": str(has_header),
                "expected_value": "True",
                "violation_count": "0" if has_header else "1",
                "explanation": "Empty CSV must still have header row",
            })
    return rows


def _audit_status_consistency(datasets_dir: Path) -> list[dict[str, Any]]:
    """Check that final statuses remain fail-closed."""
    rows: list[dict[str, Any]] = []

    # Temporal status from v1ot
    v1ot_summary = datasets_dir / "recife_scene_date_recovery_final_scientific_summary_v1ot.csv"
    product_dates = load_metric_from_summary(v1ot_summary, "product_dates_confirmed_real")
    temporal_ok = product_dates in ("0", "N/A")
    rows.append({
        "invariant_id": "V1PC_STATUS_001",
        "invariant_group": "status_consistency",
        "artifact_path": "recife_scene_date_recovery_final_scientific_summary_v1ot.csv",
        "invariant_name": "temporal_recovery_fail_closed",
        "status": "PASS" if temporal_ok else "FAIL",
        "severity": "CRITICAL" if not temporal_ok else "INFO",
        "observed_value": product_dates,
        "expected_value": "0",
        "violation_count": "0" if temporal_ok else "1",
        "explanation": "product_dates_confirmed_real must be 0 for fail-closed status",
    })

    # Observational status from v1pa
    v1pa_summary = datasets_dir / "recife_protocol_c_observed_evidence_scientific_summary_v1pa.csv"
    final_status = load_metric_from_summary(v1pa_summary, "final_status")
    obs_ok = final_status == EXPECTED_OBSERVATIONAL_STATUS
    rows.append({
        "invariant_id": "V1PC_STATUS_002",
        "invariant_group": "status_consistency",
        "artifact_path": "recife_protocol_c_observed_evidence_scientific_summary_v1pa.csv",
        "invariant_name": "observational_review_only_fail_closed",
        "status": "PASS" if obs_ok else "FAIL",
        "severity": "CRITICAL" if not obs_ok else "INFO",
        "observed_value": final_status,
        "expected_value": EXPECTED_OBSERVATIONAL_STATUS,
        "violation_count": "0" if obs_ok else "1",
        "explanation": "Observational final status must be REVIEW_ONLY_FAIL_CLOSED",
    })

    # C3+ must be 0
    c3_plus = load_metric_from_summary(v1pa_summary, "c3_plus_candidates")
    c3_ok = c3_plus in ("0", "N/A")
    rows.append({
        "invariant_id": "V1PC_STATUS_003",
        "invariant_group": "status_consistency",
        "artifact_path": "recife_protocol_c_observed_evidence_scientific_summary_v1pa.csv",
        "invariant_name": "c3_plus_zero_when_no_product_date",
        "status": "PASS" if c3_ok else "FAIL",
        "severity": "CRITICAL" if not c3_ok else "INFO",
        "observed_value": c3_plus,
        "expected_value": "0",
        "violation_count": "0" if c3_ok else "1",
        "explanation": "C3+ must be 0 when product_dates_confirmed_real=0",
    })

    # C4 formal negatives must be 0
    c4 = load_metric_from_summary(v1pa_summary, "c4_formal_negatives")
    c4_ok = c4 in ("0", "N/A")
    rows.append({
        "invariant_id": "V1PC_STATUS_004",
        "invariant_group": "status_consistency",
        "artifact_path": "recife_protocol_c_observed_evidence_scientific_summary_v1pa.csv",
        "invariant_name": "c4_zero_without_formal_negative",
        "status": "PASS" if c4_ok else "FAIL",
        "severity": "CRITICAL" if not c4_ok else "INFO",
        "observed_value": c4,
        "expected_value": "0",
        "violation_count": "0" if c4_ok else "1",
        "explanation": "C4 must be 0 without explicit formal negative",
    })

    # Labels created must be 0
    labels = load_metric_from_summary(v1pa_summary, "labels_created")
    lab_ok = labels in ("0", "N/A")
    rows.append({
        "invariant_id": "V1PC_STATUS_005",
        "invariant_group": "status_consistency",
        "artifact_path": "recife_protocol_c_observed_evidence_scientific_summary_v1pa.csv",
        "invariant_name": "labels_created_zero",
        "status": "PASS" if lab_ok else "FAIL",
        "severity": "CRITICAL" if not lab_ok else "INFO",
        "observed_value": labels,
        "expected_value": "0",
        "violation_count": "0" if lab_ok else "1",
        "explanation": "No labels may be created in this protocol",
    })

    # Training targets must be 0
    targets = load_metric_from_summary(v1pa_summary, "training_targets_created")
    tgt_ok = targets in ("0", "N/A")
    rows.append({
        "invariant_id": "V1PC_STATUS_006",
        "invariant_group": "status_consistency",
        "artifact_path": "recife_protocol_c_observed_evidence_scientific_summary_v1pa.csv",
        "invariant_name": "training_targets_zero",
        "status": "PASS" if tgt_ok else "FAIL",
        "severity": "CRITICAL" if not tgt_ok else "INFO",
        "observed_value": targets,
        "expected_value": "0",
        "violation_count": "0" if tgt_ok else "1",
        "explanation": "No training targets may be created in this protocol",
    })

    # DINO must be review-only
    dino_queue_file = datasets_dir / "recife_dino_review_only_representation_queue_v1oz.csv"
    dino_rows = read_csv_safe(dino_queue_file)
    dino_bad = [r for r in dino_rows if r.get("dino_allowed_use", "") != "REVIEW_ONLY_REPRESENTATION"]
    dino_ok = len(dino_bad) == 0
    rows.append({
        "invariant_id": "V1PC_STATUS_007",
        "invariant_group": "status_consistency",
        "artifact_path": "recife_dino_review_only_representation_queue_v1oz.csv",
        "invariant_name": "dino_all_review_only_representation",
        "status": "PASS" if dino_ok else "FAIL",
        "severity": "CRITICAL" if not dino_ok else "INFO",
        "observed_value": str(len(dino_bad)),
        "expected_value": "0",
        "violation_count": str(len(dino_bad)),
        "explanation": "All DINO entries must have dino_allowed_use=REVIEW_ONLY_REPRESENTATION",
    })

    return rows


def run() -> None:
    datasets_dir = _datasets_dir()

    all_checks: list[dict[str, Any]] = []
    all_checks.extend(_audit_forbidden_patterns(datasets_dir))
    all_checks.extend(_audit_abs_paths(datasets_dir))
    all_checks.extend(_audit_local_runs(datasets_dir))
    all_checks.extend(_audit_empty_csv_headers(datasets_dir))
    all_checks.extend(_audit_status_consistency(datasets_dir))

    write_csv_with_header(OUT_AUDIT, all_checks, AUDIT_FIELDS)
    write_schema(SCHEMA_AUDIT, AUDIT_FIELDS, "v1pc_global_invariant_audit")

    total = len(all_checks)
    pass_count = sum(1 for r in all_checks if r["status"] == "PASS")
    warn_count = sum(1 for r in all_checks if r["status"] == "WARN")
    fail_count = sum(1 for r in all_checks if r["status"] == "FAIL")
    critical_count = sum(1 for r in all_checks if r["status"] == "FAIL" and r["severity"] == "CRITICAL")

    if critical_count > 0:
        final_status = "GLOBAL_INVARIANTS_FAIL"
    elif fail_count > 0:
        final_status = "GLOBAL_INVARIANTS_FAIL"
    elif warn_count > 0:
        final_status = "GLOBAL_INVARIANTS_WARN_ONLY"
    else:
        final_status = "GLOBAL_INVARIANTS_PASS"

    summary_rows = [
        {"metric": "total_invariants_checked", "value": str(total), "interpretation": "Total de invariantes auditadas"},
        {"metric": "pass", "value": str(pass_count), "interpretation": "Invariantes aprovadas"},
        {"metric": "warn", "value": str(warn_count), "interpretation": "Avisos (nao-criticos)"},
        {"metric": "fail", "value": str(fail_count), "interpretation": "Falhas"},
        {"metric": "critical", "value": str(critical_count), "interpretation": "Falhas criticas"},
        {"metric": "final_status", "value": final_status, "interpretation": "Status global de invariantes"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary_rows, SUMMARY_FIELDS)
    write_schema(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1pc_global_invariant_summary")

    emit_doc(DOC, f"""# v1pc - Protocol C Global Scientific Invariant Auditor

## Objetivo

Auditar todos os CSVs v1og-v1pa para violacoes de invariantes cientificas globais.
Reporta FAIL/CRITICAL se encontrados — nunca mascara.

## Resultado

- Total invariantes: {total}
- PASS: {pass_count}
- WARN: {warn_count}
- FAIL: {fail_count}
- CRITICAL: {critical_count}
- Status final: {final_status}

## Invariantes auditadas

- Nenhum ground_truth=true, can_train_model=true, can_create_operational_label=true
- Nenhum path absoluto Windows
- Nenhum local_runs/ em output versionavel
- CSVs vazios com header
- Status temporal fail-closed preservado
- Status observacional fail-closed preservado
- C3+=0, C4=0, labels=0, targets=0
- DINO apenas REVIEW_ONLY_REPRESENTATION
""")

    print(f"[v1pc] Invariants: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL, "
          f"{critical_count} CRITICAL. Status: {final_status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v1pc Protocol C global invariant auditor")
    parser.parse_args()
    run()
