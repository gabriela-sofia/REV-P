"""Tests for SUSC-17C Strong Reference Acquisition Canary."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "data" / "susc_17c_strong_reference_acquisition_canary"
REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

EXPECTED = [
    OUT / "susc_17c_preflight_registry_inventory.csv",
    OUT / "susc_17c_source_target_pack.csv",
    OUT / "susc_17c_event_date_resolver.csv",
    OUT / "susc_17c_sar_feasibility_pack.csv",
    OUT / "susc_17c_canary_priority_events.csv",
    OUT / "reference_strong_acquisition_registry.csv",
    OUT / "reference_sar_feasibility_registry.csv",
    OUT / "reference_canary_qa_queue.csv",
    OUT / "susc_17c_strong_reference_acquisition_summary.json",
    REPORTS / "SUSC_17C_STRONG_REFERENCE_ACQUISITION_CANARY_REPORT.md",
    SCHEMAS / "susc_17c_source_target_pack_schema_v1.json",
    SCHEMAS / "susc_17c_event_date_resolver_schema_v1.json",
    SCHEMAS / "susc_17c_sar_feasibility_pack_schema_v1.json",
    SCHEMAS / "susc_17c_canary_priority_schema_v1.json",
    SCHEMAS / "susc_17c_reference_registry_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c_strong_reference_acquisition_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c_strong_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )


def test_build_deterministico_e_validator_passa():
    result = run_script("build_susc_17c_strong_reference_acquisition_canary.py")
    assert result.returncode == 0, result.stderr + result.stdout
    first = {path: path.read_bytes() for path in EXPECTED}
    result = run_script("build_susc_17c_strong_reference_acquisition_canary.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in EXPECTED} == first
    result = run_script("validate_susc_17c_strong_reference_acquisition_canary.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_e_outputs_review_only_sem_score_v7_ou_treino():
    s17c = _load_common()
    targets = read_csv(OUT / "susc_17c_source_target_pack.csv")
    dates = read_csv(OUT / "susc_17c_event_date_resolver.csv")
    sar_rows = read_csv(OUT / "susc_17c_sar_feasibility_pack.csv")
    registry = read_csv(OUT / "reference_strong_acquisition_registry.csv")
    sar_registry = read_csv(OUT / "reference_sar_feasibility_registry.csv")
    qa_rows = read_csv(OUT / "reference_canary_qa_queue.csv")
    assert targets
    assert qa_rows
    assert s17c.validate_output_rows(targets, dates, sar_rows, registry, sar_registry, qa_rows) == []
    all_rows = targets + dates + sar_rows + registry + sar_registry + qa_rows
    assert {row.get("ground_truth") for row in all_rows if "ground_truth" in row} == {"false"}
    assert {row.get("score_v7_allowed") for row in all_rows if "score_v7_allowed" in row} == {"false"}
    assert {row.get("eligible_for_training") for row in registry + sar_registry + qa_rows} == {"false"}
    for schema_path in EXPECTED[-5:]:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert schema["type"] == "object"
        assert schema["required"]


def test_regras_bloqueadas_em_casos_sinteticos():
    s17c = _load_common()
    targets = read_csv(OUT / "susc_17c_source_target_pack.csv")
    dates = read_csv(OUT / "susc_17c_event_date_resolver.csv")
    sar_rows = read_csv(OUT / "susc_17c_sar_feasibility_pack.csv")
    registry = read_csv(OUT / "reference_strong_acquisition_registry.csv")
    sar_registry = read_csv(OUT / "reference_sar_feasibility_registry.csv")
    qa_rows = read_csv(OUT / "reference_canary_qa_queue.csv")

    bad_targets = [dict(row) for row in targets]
    bad_targets[0]["ground_truth"] = "true"
    assert any("ground_truth_true_forbidden" in err for err in s17c.validate_output_rows(bad_targets, dates, sar_rows, registry, sar_registry, qa_rows))

    bad_targets = [dict(row) for row in targets]
    bad_targets[0]["source_type"] = "alert_only"
    bad_targets[0]["strong_candidate"] = "true"
    assert any("blocked_source_type_marked_strong" in err for err in s17c.validate_output_rows(bad_targets, dates, sar_rows, registry, sar_registry, qa_rows))

    bad_dates = [dict(row) for row in dates]
    bad_dates[0]["date_precision"] = "month_only"
    bad_dates[0]["temporal_eligible"] = "true"
    assert any("month_or_unknown_passed_to_sar" in err for err in s17c.validate_output_rows(targets, bad_dates, sar_rows, registry, sar_registry, qa_rows))

    bad_registry = [dict(row) for row in registry]
    bad_registry[0]["eligible_for_training"] = "true"
    assert any("eligible_for_training_true_forbidden" in err for err in s17c.validate_output_rows(targets, dates, sar_rows, bad_registry, sar_registry, qa_rows))

    weak_qa = [dict(row) for row in qa_rows]
    weak_targets = [dict(row) for row in targets]
    weak_sar = [dict(row) for row in sar_rows]
    weak_qa[0]["qa_status"] = "accepted"
    cid = weak_qa[0]["candidate_event_id"]
    for row in weak_targets:
        if row["candidate_event_id"] == cid:
            row["has_official_geometry"] = "false"
            row["has_bbox"] = "false"
    for row in weak_sar:
        if row["candidate_event_id"] == cid:
            row["sar_feasible"] = "false"
    assert any("accepted_without_strong_spatial_evidence" in err for err in s17c.validate_output_rows(weak_targets, dates, weak_sar, registry, sar_registry, weak_qa))


def test_relatorio_declara_17b_bloqueado_e_fila_auditavel():
    summary = json.loads((OUT / "susc_17c_strong_reference_acquisition_summary.json").read_text(encoding="utf-8"))
    report = (REPORTS / "SUSC_17C_STRONG_REFERENCE_ACQUISITION_CANARY_REPORT.md").read_text(encoding="utf-8")
    assert summary["decision"] == "17B_BLOQUEADO_FAIL_CLOSED"
    assert summary["eligible_for_training_count"] == 0
    assert summary["ground_truth_true_count"] == 0
    assert summary["score_v7_allowed_true_count"] == 0
    assert summary["canary_priority_count"] >= 3
    assert "17B permanece **bloqueado**" in report
    assert "reference_canary_qa_queue.csv" in report
