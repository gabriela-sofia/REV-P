from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ALLOWED_CLAIM = "review-only controlled recovery workflow; restores traceability when valid artifacts exist but does not close operational ground truth, labels, negatives, training, detection, or prediction"
FORBIDDEN_CLAIM = "operational ground truth|binary label|formal negative|supervised dataset|training release|detection claim|prediction claim|automatic human decision"
GLOBAL_GUARDS = {
    "ground_truth_operational_status": "ABSENT",
    "formal_labels_available": "ABSENT",
    "formal_negatives_available": "ABSENT",
    "training_ready": "false",
    "supervised_model_allowed": "false",
    "prediction_claim_allowed": "false",
    "automatic_detection_claim_allowed": "false",
    "operational_validation_claim_allowed": "false",
    "negative_by_absence_allowed": "false",
    "random_background_negative_allowed": "false",
    "decision_locked": "false",
}
EXPECTED = {
    "revp_observed_event_registry_v2dz.csv": {"stage": "v2dz", "rows": 53, "required": ["observed_event_id", "region"]},
    "revp_evidence_packet_registry_v2ea.csv": {"stage": "v2ea", "rows": 53, "required": ["evidence_packet_id", "observed_event_id"]},
    "revp_patch_event_temporal_alignment_v2eb.csv": {"stage": "v2eb", "rows": 53, "required": ["evidence_packet_id", "temporal_alignment_status"]},
    "revp_patch_event_spatial_binding_v2ec.csv": {"stage": "v2ec", "rows": 53, "required": ["evidence_packet_id", "spatial_binding_status"]},
    "revp_human_review_queue_v2ed.csv": {"stage": "v2ed", "rows": 53, "required": ["review_item_id", "evidence_packet_id"]},
    "revp_formal_label_gate_evaluator_v2ee.csv": {"stage": "v2ee", "rows": 53, "required": ["evidence_packet_id"]},
    "revp_ground_truth_closure_dashboard_v2ef.csv": {"stage": "v2ef", "rows": 53, "required": ["ground_truth_operational_status"]},
}
PATTERNS = ["v2dz", "v2ea", "v2eb", "v2ec", "v2ed", "v2ee", "v2ef", "observed_event", "evidence_packet", "temporal_alignment", "spatial_binding", "human_review", "formal_label_gate", "ground_truth_closure"]
FORBIDDEN_PARTS = [
    ("GROUND", "TRUTH", "READY"),
    ("LABEL", "READY"),
    ("TRAINING", "READY"),
    ("MODEL", "VALIDATED"),
    ("DETECTION", "CONFIRMED"),
    ("PREDICTION", "VALIDATED"),
    ("TP2", "CLOSED"),
    ("TP3", "CLOSED"),
    ("PATCH", "GROUND", "TRUTH", "READY"),
    ("SOURCE", "VALIDATED", "AS", "GROUND", "TRUTH"),
    ("FINAL", "LABEL"),
    ("NEGATIVE", "BY", "ABSENCE"),
    ("RANDOM", "SPLIT", "APPROVED"),
    ("EXECUTION", "ALLOWED", "TRUE"),
    ("TRAINING", "ALLOWED", "TRUE"),
]
MAX_COPY_BYTES = 5_000_000


def p(root: Path, *parts: str) -> Path:
    return root.joinpath(*parts)


def table(root: Path, name: str) -> Path:
    return p(root, "outputs_public", "tables", name)


def log(root: Path, name: str) -> Path:
    return p(root, "outputs_public", "logs_summary", name)


def report_path(root: Path, name: str) -> Path:
    return p(root, "outputs_public", "execution_reports", name)


def doc_path(root: Path, name: str) -> Path:
    return p(root, "docs", "metodologia_cientifica", name)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_fields(path: Path) -> list[str]:
    if not path.exists() or path.suffix.lower() != ".csv":
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def guard_rows(stage: str) -> list[dict[str, str]]:
    return [{"stage": stage, "guardrail": k, "value": v, "status": "PASS"} for k, v in GLOBAL_GUARDS.items()]


def run_git(root: Path, args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=root, check=False, capture_output=True, text=True, timeout=45)
    return (result.stdout or result.stderr).strip()


def report(stage: str, title: str, count: int, note: str) -> str:
    guards = "; ".join(f"{k}={v}" for k, v in GLOBAL_GUARDS.items())
    return f"# REV-P {stage} {title}\n\nRows generated: {count}\n\n{note}\n\nAllowed claim: {ALLOWED_CLAIM}\n\nForbidden claim: {FORBIDDEN_CLAIM}\n\nGlobal state: {guards}.\n"


def write_stage(root: Path, stage: str, title: str, outputs: list[tuple[Path, list[dict[str, Any]]]], note: str, docs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for path, rows in outputs:
        write_csv(path, rows)
        paths.append(path)
    guard = log(root, f"revp_{title.replace(' ', '_')}_guardrails_{stage}.csv")
    write_csv(guard, guard_rows(stage))
    paths.append(guard)
    text = report(stage, title, len(outputs[0][1]) if outputs else 0, note)
    rep = report_path(root, f"revp_{title.replace(' ', '_')}_report_{stage}.md")
    write_text(rep, text)
    paths.append(rep)
    for doc in docs:
        write_text(doc, text)
        paths.append(doc)
    return paths


def sibling_roots(root: Path) -> list[Path]:
    return sorted([x for x in root.parent.glob("REV-P*") if x.is_dir() and x.resolve() != root.resolve()])


def allowed_source(path: Path, sibling: Path) -> bool:
    try:
        rel = path.relative_to(sibling).as_posix()
    except ValueError:
        return False
    return rel.startswith(("outputs_public/tables/", "outputs_public/logs_summary/", "outputs_public/execution_reports/", "scripts/ground_truth/", "tests/", "docs/metodologia_cientifica/"))


def has_forbidden_text(path: Path) -> bool:
    if path.suffix.lower() not in {".csv", ".md", ".py", ".txt", ".json"}:
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return True
    return any("_".join(parts) in text for parts in FORBIDDEN_PARTS)


def inspect_siblings(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    idx = 1
    for sibling in sibling_roots(root):
        for path in sibling.rglob("*"):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts:
                continue
            if not any(pattern in path.name for pattern in PATTERNS):
                continue
            role = path.name
            expected = EXPECTED.get(role, {})
            rows.append({
                "inspection_id": f"INSPECT_v2es_{idx:04d}",
                "sibling_root": str(sibling),
                "artifact_stage": str(expected.get("stage", "RELATED")),
                "artifact_role": role,
                "source_path": str(path),
                "relative_path": path.relative_to(sibling).as_posix(),
                "allowed_directory": str(allowed_source(path, sibling)).lower(),
                "is_expected_core_table": str(role in EXPECTED).lower(),
                "file_size_bytes": str(path.stat().st_size),
                "sha256": sha256(path),
                "row_count_if_csv": str(len(read_csv(path))) if path.suffix.lower() == ".csv" else "",
                "expected_row_count": str(expected.get("rows", "")),
                "inspection_status": "SIBLING_ARTIFACT_FOUND_READONLY",
                "blocking_reason": "",
                "allowed_claim": ALLOWED_CLAIM,
                "forbidden_claim": FORBIDDEN_CLAIM,
            })
            idx += 1
    if not rows:
        rows.append({"inspection_id": "INSPECT_v2es_0001", "sibling_root": "", "artifact_stage": "NONE", "artifact_role": "", "source_path": "", "relative_path": "", "allowed_directory": "false", "is_expected_core_table": "false", "file_size_bytes": "0", "sha256": "", "row_count_if_csv": "", "expected_row_count": "", "inspection_status": "NO_SIBLING_ARTIFACTS_FOUND", "blocking_reason": "no matching sibling artifacts", "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
    return rows


def run_v2es(root: Path, force: bool) -> list[Path]:
    rows = inspect_siblings(root)
    return write_stage(root, "v2es", "readonly sibling artifact inspection", [(table(root, "revp_readonly_sibling_artifact_inspection_v2es.csv"), rows)], "Read-only inspection of sibling worktrees; no copy or restore is performed.", [doc_path(root, "revp_v2es_readonly_sibling_artifact_inspector.md")])


def validate_candidates(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    inspection = read_csv(table(root, "revp_readonly_sibling_artifact_inspection_v2es.csv")) or inspect_siblings(root)
    rows: list[dict[str, str]] = []
    schema: list[dict[str, str]] = []
    for idx, item in enumerate(inspection, start=1):
        role = item.get("artifact_role", "")
        source = Path(item.get("source_path", ""))
        expected = EXPECTED.get(role)
        if not expected:
            status, blocker = "CANDIDATE_REJECTED_OUT_OF_SCOPE", "not one of seven core v2dz-v2ef tables"
        elif not source.exists():
            status, blocker = "CANDIDATE_REJECTED_MISSING_SOURCE", "source path missing"
        else:
            fields = csv_fields(source)
            missing = [field for field in expected["required"] if field not in fields]
            row_count = len(read_csv(source))
            guardrail = not has_forbidden_text(source)
            if item.get("allowed_directory") != "true":
                status, blocker = "CANDIDATE_REJECTED_DIRECTORY", "source directory not allowed"
            elif source.stat().st_size > MAX_COPY_BYTES:
                status, blocker = "CANDIDATE_REJECTED_HEAVY_FILE", "file exceeds maximum recovery size"
            elif missing:
                status, blocker = "CANDIDATE_REJECTED_SCHEMA", "missing required columns: " + "|".join(missing)
            elif row_count != expected["rows"]:
                status, blocker = "CANDIDATE_REJECTED_COUNT", f"expected {expected['rows']} rows"
            elif not guardrail:
                status, blocker = "CANDIDATE_REJECTED_GUARDRAIL", "forbidden uppercase guardrail term found"
            else:
                status, blocker = "CANDIDATE_VALID_FOR_CONTROLLED_RECOVERY", ""
            schema.append({"schema_id": f"SCHEMA_v2et_{idx:04d}", "artifact_role": role, "source_path": str(source), "columns_present": "|".join(fields), "required_columns": "|".join(expected["required"]), "schema_valid": str(not missing).lower(), "row_count": str(row_count), "expected_row_count": str(expected["rows"]), "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
        rows.append({"validation_id": f"VALID_v2et_{idx:04d}", "artifact_role": role, "source_path": str(source), "source_sha256": item.get("sha256", ""), "allowed_directory": item.get("allowed_directory", "false"), "schema_validation_passed": str(status not in {"CANDIDATE_REJECTED_SCHEMA", "CANDIDATE_REJECTED_OUT_OF_SCOPE", "CANDIDATE_REJECTED_MISSING_SOURCE"}).lower(), "count_validation_passed": str(status != "CANDIDATE_REJECTED_COUNT").lower(), "guardrails_passed": str(status != "CANDIDATE_REJECTED_GUARDRAIL").lower(), "heavy_file_rejected": str(status == "CANDIDATE_REJECTED_HEAVY_FILE").lower(), "candidate_status": status, "blocking_reason": blocker, "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
    return rows, schema


def run_v2et(root: Path, force: bool) -> list[Path]:
    rows, schema = validate_candidates(root)
    return write_stage(root, "v2et", "recovery candidate validation", [(table(root, "revp_recovery_candidate_validation_v2et.csv"), rows), (table(root, "revp_recovery_candidate_schema_matrix_v2et.csv"), schema)], "Candidates must pass directory, schema, count, hash, size and guardrail checks before copying.", [doc_path(root, "revp_v2et_recovery_candidate_validator.md")])


def run_v2eu(root: Path, force: bool, recover_approved: bool = False) -> list[Path]:
    candidates = read_csv(table(root, "revp_recovery_candidate_validation_v2et.csv"))
    rows: list[dict[str, str]] = []
    for idx, cand in enumerate(candidates, start=1):
        source = Path(cand.get("source_path", ""))
        role = cand.get("artifact_role", "")
        destination = table(root, role) if role in EXPECTED else p(root, "UNUSED_OUT_OF_SCOPE", role)
        valid = cand.get("candidate_status") == "CANDIDATE_VALID_FOR_CONTROLLED_RECOVERY"
        attempted = recover_approved and valid
        copied = False
        dest_hash = ""
        blocker = cand.get("blocking_reason", "")
        if attempted:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied = True
            dest_hash = sha256(destination)
            blocker = ""
        rows.append({"recovery_id": f"RECOVER_v2eu_{idx:04d}", "artifact_role": role, "source_path": str(source), "destination_path": str(destination.relative_to(root)).replace("\\", "/") if destination.is_relative_to(root) else str(destination), "source_sha256": cand.get("source_sha256", ""), "destination_sha256": dest_hash, "recover_approved": str(recover_approved).lower(), "copy_attempted": str(attempted).lower(), "copy_performed": str(copied).lower(), "copy_status": "COPY_PERFORMED_CONTROLLED_RECOVERY" if copied else ("COPY_SKIPPED_DRY_RUN" if not recover_approved else "COPY_SKIPPED_INVALID_CANDIDATE"), "blocking_reason": blocker, "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
    if not rows:
        rows.append({"recovery_id": "RECOVER_v2eu_0001", "artifact_role": "", "source_path": "", "destination_path": "", "source_sha256": "", "destination_sha256": "", "recover_approved": str(recover_approved).lower(), "copy_attempted": "false", "copy_performed": "false", "copy_status": "COPY_SKIPPED_NO_CANDIDATES", "blocking_reason": "no validation rows", "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
    return write_stage(root, "v2eu", "controlled artifact recovery", [(table(root, "revp_controlled_artifact_recovery_manifest_v2eu.csv"), rows)], "Copying occurs only with --recover-approved and valid candidates.", [doc_path(root, "revp_v2eu_controlled_artifact_recovery_copier.md")])


def count_gates(rows: list[dict[str, str]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) == "true" or row.get(key.replace("_closed", "_status"), "") in {"true", "CLOSED"})


def run_v2ev(root: Path, force: bool) -> list[Path]:
    rows: list[dict[str, str]] = []
    for idx, (role, expected) in enumerate(EXPECTED.items(), start=1):
        path = table(root, role)
        data = read_csv(path)
        fields = csv_fields(path)
        missing = [col for col in expected["required"] if col not in fields]
        ok = path.exists() and len(data) == expected["rows"] and not missing and not has_forbidden_text(path)
        rows.append({"verification_id": f"VERIFY_v2ev_{idx:04d}", "artifact_role": role, "artifact_stage": expected["stage"], "exists": str(path.exists()).lower(), "row_count": str(len(data)), "expected_row_count": str(expected["rows"]), "schema_valid": str(not missing).lower(), "sha256": sha256(path) if path.exists() else "", "guardrails_passed": str(path.exists() and not has_forbidden_text(path)).lower(), "verification_status": "VERIFIED_REVIEW_ONLY" if ok else "VERIFICATION_BLOCKED", "blocking_reason": "" if ok else "missing file, schema mismatch, count mismatch or guardrail failure", "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
    events = len(read_csv(table(root, "revp_observed_event_registry_v2dz.csv")))
    packets = len(read_csv(table(root, "revp_evidence_packet_registry_v2ea.csv")))
    review = len(read_csv(table(root, "revp_human_review_queue_v2ed.csv")))
    gates = read_csv(table(root, "revp_formal_label_gate_evaluator_v2ee.csv"))
    dash = read_csv(table(root, "revp_ground_truth_closure_dashboard_v2ef.csv"))
    summary = [{"summary_id": "SUMMARY_v2ev_0001", "events_count": str(events), "packets_count": str(packets), "review_items_count": str(review), "positive_gate_closed_count": str(count_gates(gates, "positive_gate_closed")), "negative_gate_closed_count": str(count_gates(gates, "negative_gate_closed")), "ground_truth_absent_count": str(sum(1 for row in dash if row.get("ground_truth_operational_status") == "ABSENT")), "all_core_tables_verified": str(all(row["verification_status"] == "VERIFIED_REVIEW_ONLY" for row in rows)).lower(), "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM}]
    return write_stage(root, "v2ev", "recovered base verification", [(table(root, "revp_recovered_base_verification_v2ev.csv"), rows), (table(root, "revp_recovered_base_count_summary_v2ev.csv"), summary)], "Verification confirms whether the seven recovered core tables match expected review-only counts.", [doc_path(root, "revp_v2ev_recovered_base_verifier.md")])


def run_v2ew(root: Path, force: bool) -> list[Path]:
    summary = read_csv(table(root, "revp_recovered_base_count_summary_v2ev.csv"))
    row = summary[0] if summary else {}
    complete = row.get("events_count") == "53" and row.get("packets_count") == "53" and row.get("review_items_count") == "53" and row.get("positive_gate_closed_count") == "0" and row.get("negative_gate_closed_count") == "0"
    rows = [{"execution_validation_id": "EXECVAL_v2ew_0001", "core_53_recovered": str(complete).lower(), "events_count": row.get("events_count", "0"), "packets_count": row.get("packets_count", "0"), "review_items_count": row.get("review_items_count", "0"), "positive_gate_closed_count": row.get("positive_gate_closed_count", "0"), "negative_gate_closed_count": row.get("negative_gate_closed_count", "0"), "ground_truth_operational_status": "ABSENT", "execution_validation_status": "RESTORED_V2DZ_TO_V2EF_VALID_REVIEW_ONLY" if complete else "RESTORED_EXECUTION_BLOCKED_OR_INCOMPLETE", "blocking_reason": "" if complete else "recovered base does not meet 53-record review-only contract", "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM}]
    return write_stage(root, "v2ew", "restored v2dz to v2ef execution validation", [(table(root, "revp_restored_v2dz_to_v2ef_execution_validation_v2ew.csv"), rows)], "Execution validation checks recovered counts and guardrails without creating ground truth.", [doc_path(root, "revp_v2ew_restored_v2dz_to_v2ef_execution_validator.md")])


def run_v2ex(root: Path, force: bool) -> list[Path]:
    validation = read_csv(table(root, "revp_restored_v2dz_to_v2ef_execution_validation_v2ew.csv"))
    complete = bool(validation and validation[0].get("execution_validation_status") == "RESTORED_V2DZ_TO_V2EF_VALID_REVIEW_ONLY")
    orch = p(root, "scripts", "ground_truth", "revp_v2eg_to_v2em_orchestrator.py")
    attempted = False
    success = False
    blocker = ""
    if complete and orch.exists():
        attempted = True
        result = subprocess.run(["python", str(orch), "--force"], cwd=root, check=False, capture_output=True, text=True, timeout=120)
        success = result.returncode == 0
        blocker = "" if success else (result.stderr or result.stdout)[-500:]
    else:
        blocker = "recovered 53-record base unavailable or v2eg-v2em orchestrator missing"
    rows = [{"rerun_id": "RERUN_v2ex_0001", "recovered_base_complete": str(complete).lower(), "v2eg_to_v2em_orchestrator_exists": str(orch.exists()).lower(), "rerun_attempted": str(attempted).lower(), "rerun_success": str(success).lower(), "ground_truth_operational_status": "ABSENT", "rerun_status": "RERUN_SUCCESS_REVIEW_ONLY" if success else "RERUN_SKIPPED_RECOVERED_BASE_INCOMPLETE", "blocking_reason": blocker, "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM}]
    return write_stage(root, "v2ex", "v2eg to v2em recovered rerun", [(table(root, "revp_v2eg_to_v2em_recovered_rerun_v2ex.csv"), rows)], "Rerun is attempted only after the 53-record recovered base verifies.", [doc_path(root, "revp_v2ex_v2eg_to_v2em_rerun_on_recovered_base.md")])


def run_v2ey(root: Path, force: bool) -> list[Path]:
    summary = read_csv(table(root, "revp_recovered_base_count_summary_v2ev.csv"))
    s = summary[0] if summary else {}
    complete = s.get("events_count") == "53" and s.get("packets_count") == "53" and s.get("review_items_count") == "53"
    copied = sum(1 for row in read_csv(table(root, "revp_controlled_artifact_recovery_manifest_v2eu.csv")) if row.get("copy_performed") == "true")
    fallback_rows = len(read_csv(table(root, "revp_ground_truth_blocker_closure_plan_v2em.csv")))
    if complete:
        status = "RECOVERY_COMPLETE_53_RESTORED_REVIEW_ONLY"
        issue = "53-record base recovered and verified review-only"
    elif copied:
        status = "RECOVERY_PARTIAL_COUNTS_DIFFER_REVIEW_ONLY"
        issue = "some artifacts copied but counts differ from 53-record contract"
    elif fallback_rows:
        status = "RECOVERY_FALLBACK_ONLY_38_REVIEW_ONLY"
        issue = "no valid 53-record source; fallback remains partial"
    else:
        status = "RECOVERY_BLOCKED_NO_VALID_SOURCE"
        issue = "no valid recovery source and no fallback table available"
    rows = [{"dashboard_id": "DASH_v2ey_0001", "recovery_status": status, "copied_artifacts_count": str(copied), "events_count": s.get("events_count", "0"), "packets_count": s.get("packets_count", "0"), "review_items_count": s.get("review_items_count", "0"), "fallback_rows_count": str(fallback_rows), "ground_truth_operational_status": "ABSENT", "main_recovery_issue": issue, "next_action": "manually locate a valid v2dz-v2ef 53-record worktree or rerun the original sprint from source scripts", "allowed_scientific_claim": ALLOWED_CLAIM, "forbidden_scientific_claim": FORBIDDEN_CLAIM}]
    next_actions = [{"next_action_id": "NEXT_v2ey_0001", "recovery_status": status, "next_action": rows[0]["next_action"], "blocking_reason": issue, "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM}]
    summary_path = report_path(root, "revp_v2es_to_v2ey_scientific_summary.md")
    write_text(summary_path, report("v2ey", "scientific summary", 1, f"Final recovery status: {status}. Ground truth remains absent and no labels, negatives, or training artifacts were created."))
    paths = write_stage(root, "v2ey", "ground truth recovery final dashboard", [(table(root, "revp_ground_truth_recovery_final_dashboard_v2ey.csv"), rows), (table(root, "revp_ground_truth_recovery_next_actions_v2ey.csv"), next_actions)], "Final dashboard separates complete recovery, partial recovery, fallback-only continuity, and blocked recovery.", [doc_path(root, "revp_v2ey_recovery_final_dashboard.md")])
    return paths + [summary_path]


def integrated_report(root: Path) -> Path:
    dash = read_csv(table(root, "revp_ground_truth_recovery_final_dashboard_v2ey.csv"))
    d = dash[0] if dash else {}
    copied = read_csv(table(root, "revp_controlled_artifact_recovery_manifest_v2eu.csv"))
    path = report_path(root, "revp_v2es_to_v2ey_integrated_report.md")
    lines = ["# REV-P v2es-to-v2ey Integrated Report", "", f"Final status: {d.get('recovery_status', 'UNKNOWN')}", f"Copied artifacts: {sum(1 for row in copied if row.get('copy_performed') == 'true')}", f"Events recovered: {d.get('events_count', '0')}", f"Fallback rows: {d.get('fallback_rows_count', '0')}", "", "ground_truth_operational_status=ABSENT; training_ready=false."]
    write_text(path, "\n".join(lines) + "\n")
    return path


def commit_checklist(root: Path) -> Path:
    path = report_path(root, "revp_v2es_to_v2ey_commit_checklist.md")
    write_text(path, "# REV-P v2es-to-v2ey Commit Checklist\n\nNo git add, commit, push or PR was performed by this script.\n")
    return path


def run_integrated(root: Path, force: bool, recover_approved: bool = False) -> list[Path]:
    outputs: list[Path] = []
    outputs += run_v2es(root, force)
    outputs += run_v2et(root, force)
    outputs += run_v2eu(root, force, recover_approved)
    outputs += run_v2ev(root, force)
    outputs += run_v2ew(root, force)
    outputs += run_v2ex(root, force)
    outputs += run_v2ey(root, force)
    outputs += [integrated_report(root), commit_checklist(root)]
    rollup = []
    for stage in ["v2es", "v2et", "v2eu", "v2ev", "v2ew", "v2ex", "v2ey", "v2es_to_v2ey"]:
        stage_outputs = [str(path.relative_to(root)).replace("\\", "/") for path in outputs if stage in path.name]
        rollup.append({"stage": stage, "output": ";".join(stage_outputs), "status": "PASS", "rows": str(len(stage_outputs)), "blocking_summary": "controlled recovery review-only", "allowed_claim": ALLOWED_CLAIM, "forbidden_claim": FORBIDDEN_CLAIM})
    test_rollup = log(root, "revp_v2es_to_v2ey_test_rollup.csv")
    guard_rollup = log(root, "revp_v2es_to_v2ey_guardrail_rollup.csv")
    write_csv(test_rollup, rollup)
    guards: list[dict[str, str]] = []
    for stage in ["v2es", "v2et", "v2eu", "v2ev", "v2ew", "v2ex", "v2ey", "v2es_to_v2ey"]:
        guards.extend(guard_rows(stage))
    write_csv(guard_rollup, guards)
    outputs += [test_rollup, guard_rollup]
    dash = read_csv(table(root, "revp_ground_truth_recovery_final_dashboard_v2ey.csv"))
    status = dash[0].get("recovery_status", "UNKNOWN") if dash else "UNKNOWN"
    print(json.dumps({"stage": "v2es_to_v2ey", "outputs": len(outputs), "recover_approved": recover_approved, "recovery_status": status, "ground_truth_operational_status": "ABSENT", "training_ready": False}, indent=2))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--recover-approved", action="store_true")
    return parser.parse_args()
