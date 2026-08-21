from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file()) / "scripts" / "ground_truth"))

from revp_v2dz_to_v2ef_common import GLOBAL_GUARDS, read_csv, run_integrated


def make_repo(tmp_path: Path) -> Path:
    data = tmp_path / "datasets"
    data.mkdir()
    with (data / "ground_reference_event_registry.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "source_event_unit_id", "source_document_sanitized", "region", "event_or_survey_date", "coordinate_status", "crs", "hash_available"])
        writer.writeheader()
        writer.writerow({"event_id": "EVT_1", "source_event_unit_id": "DOC_1", "source_document_sanitized": "doc.pdf", "region": "PET", "event_or_survey_date": "2022-02-20", "coordinate_status": "EXPLICIT_COORDINATE", "crs": "EPSG:32723", "hash_available": "true"})
    with (data / "official_anchor_sentinel_patch_registry.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reference_patch_id", "source_documented_event_unit_id", "region", "scene_date", "patch_generated_locally", "crs"])
        writer.writeheader()
        writer.writerow({"reference_patch_id": "PATCH_1", "source_documented_event_unit_id": "DOC_1", "region": "PET", "scene_date": "2022-03-04", "patch_generated_locally": "true", "crs": "EPSG:32723"})
    return tmp_path


def test_orchestrator_runs_complete(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    run_integrated(root, True)
    assert (root / "outputs_public" / "execution_reports" / "revp_v2dz_to_v2ef_integrated_report.md").exists()


def test_orchestrator_generates_guardrail_rollup(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    run_integrated(root, True)
    rows = read_csv(root / "outputs_public" / "logs_summary" / "revp_v2dz_to_v2ef_guardrail_rollup.csv")
    assert rows
    assert {row["status"] for row in rows} == {"PASS"}


def test_orchestrator_generates_test_rollup(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    run_integrated(root, True)
    rows = read_csv(root / "outputs_public" / "logs_summary" / "revp_v2dz_to_v2ef_test_rollup.csv")
    assert {row["stage"] for row in rows} == {"v2dz", "v2ea", "v2eb", "v2ec", "v2ed", "v2ee", "v2ef"}


@pytest.mark.parametrize("guardrail,value", sorted(GLOBAL_GUARDS.items()))
def test_orchestrator_preserves_global_guardrails(tmp_path: Path, guardrail: str, value: str) -> None:
    root = make_repo(tmp_path)
    run_integrated(root, True)
    rows = read_csv(root / "outputs_public" / "logs_summary" / "revp_v2dz_to_v2ef_guardrail_rollup.csv")
    matches = [row for row in rows if row["guardrail"] == guardrail]
    assert matches
    assert {row["value"] for row in matches} == {value}


def test_orchestrator_does_not_create_labels(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    run_integrated(root, True)
    rows = read_csv(root / "outputs_public" / "tables" / "revp_formal_label_gate_evaluator_v2ee.csv")
    assert {row["formal_label_allowed"] for row in rows} == {"false"}


def test_orchestrator_does_not_create_negatives(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    run_integrated(root, True)
    rows = read_csv(root / "outputs_public" / "tables" / "revp_negative_gate_protocol_v2ee.csv")
    assert {row["explicit_non_occurrence_evidence"] for row in rows} == {"false"}


def test_orchestrator_does_not_train(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    run_integrated(root, True)
    rows = read_csv(root / "outputs_public" / "logs_summary" / "revp_v2dz_to_v2ef_guardrail_rollup.csv")
    assert any(row["guardrail"] == "training_ready" and row["value"] == "false" for row in rows)


def test_commit_checklist_reference_only(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    run_integrated(root, True)
    text = (root / "outputs_public" / "execution_reports" / "revp_v2dz_to_v2ef_commit_checklist.md").read_text(encoding="utf-8")
    assert "No git add, commit, push or PR was performed" in text
