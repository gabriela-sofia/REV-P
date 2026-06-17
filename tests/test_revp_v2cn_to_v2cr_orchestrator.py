from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

from revp_v2cn_to_v2cr_orchestrator import main  # noqa: E402


FORBIDDEN = ["GROUND_TRUTH_READY", "LABEL_READY", "TRAINING_READY", "MODEL_VALIDATED", "DETECTION_CONFIRMED", "PREDICTION_VALIDATED"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_offline_orchestrator_runs_complete(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    assert main(["--repo-root", str(root), "--offline", "--force"]) == 0


def test_integrated_report_generated(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    assert (root / "outputs_public/execution_reports/revp_v2cn_to_v2cr_integrated_report.md").exists()


def test_test_rollup_generated(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    rows = read_csv(root / "outputs_public/logs_summary/revp_v2cn_to_v2cr_test_rollup.csv")
    assert [row["stage"] for row in rows] == ["v2cn", "v2co", "v2cp", "v2cq", "v2cr"]


def test_guardrail_rollup_generated(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    rows = read_csv(root / "outputs_public/logs_summary/revp_v2cn_to_v2cr_guardrail_rollup.csv")
    assert any(row["guardrail"] == "binary_labels" and row["observed_value"] == "absent" for row in rows)


def test_no_forbidden_status_tokens_in_outputs(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in (root / "outputs_public").rglob("*") if path.is_file())
    assert all(token not in text for token in FORBIDDEN)


def test_no_raw_external_files_in_outputs_public(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    assert not list((root / "outputs_public").rglob("*.geojson"))
    assert not list((root / "outputs_public").rglob("*.tif"))


def test_all_stage_tables_are_created(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    for path in [
        "outputs_public/tables/revp_external_evidence_gap_matrix_v2cn.csv",
        "outputs_public/tables/revp_external_source_registry_v2co.csv",
        "datasets/external_evidence/external_evidence_manifest_v2cp.csv",
        "outputs_public/tables/revp_external_geospatial_qa_v2cq.csv",
        "outputs_public/tables/revp_external_patch_pairing_v2cr.csv",
    ]:
        assert (root / path).exists()


def test_allow_downloads_with_missing_registry_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    assert main(["--repo-root", str(root), "--allow-downloads", "--force"]) == 1


def test_unrelated_file_is_not_changed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    sentinel = root / "unrelated.txt"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("keep", encoding="utf-8")
    main(["--repo-root", str(root), "--offline", "--force"])
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_commit_checklist_contains_requested_message(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    text = (root / "outputs_public/execution_reports/revp_v2cn_to_v2cr_commit_checklist.md").read_text(encoding="utf-8")
    assert "data: prepara aquisicao e QA geoespacial de evidencias externas" in text
