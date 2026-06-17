from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

from revp_v2cx_to_v2dd_orchestrator import main  # noqa: E402


FORBIDDEN = ["GROUND_TRUTH_READY", "LABEL_READY", "TRAINING_READY", "MODEL_VALIDATED", "DETECTION_CONFIRMED", "PREDICTION_VALIDATED", "TP2_CLOSED", "TP3_CLOSED", "PATCH_GROUND_TRUTH_READY", "SOURCE_VALIDATED_AS_GROUND_TRUTH"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_offline_orchestrator_runs_complete(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    assert main(["--repo-root", str(root), "--offline", "--force"]) == 0


def test_integrated_report_generated(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    assert (root / "outputs_public/execution_reports/revp_v2cx_to_v2dd_integrated_report.md").exists()


def test_commit_checklist_generated(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    assert "analysis: consolida prontidao cientifica para evidencia externa TP2" in (root / "outputs_public/execution_reports/revp_v2cx_to_v2dd_commit_checklist.md").read_text(encoding="utf-8")


def test_rollups_generated(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    assert (root / "outputs_public/logs_summary/revp_v2cx_to_v2dd_test_rollup.csv").exists()
    assert (root / "outputs_public/logs_summary/revp_v2cx_to_v2dd_guardrail_rollup.csv").exists()


def test_all_stage_outputs_generated(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    for rel in [
        "outputs_public/tables/revp_real_source_availability_v2cx.csv",
        "outputs_public/tables/revp_controlled_product_link_discovery_v2cy.csv",
        "outputs_public/tables/revp_product_license_audit_v2cz.csv",
        "datasets/external_evidence/download_plan_v2da.csv",
        "outputs_public/tables/revp_patch_boundary_readiness_v2db.csv",
        "outputs_public/tables/revp_integrated_readiness_matrix_v2dc.csv",
        "outputs_public/tables/revp_scientific_readiness_dashboard_v2dd.csv",
    ]:
        assert (root / rel).exists()


def test_no_forbidden_uppercase_tokens_in_outputs(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in (root / "outputs_public").rglob("*") if path.is_file())
    assert all(token not in text for token in FORBIDDEN)


def test_no_raw_external_files_in_outputs_public(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    assert not list((root / "outputs_public").rglob("*.geojson"))
    assert not list((root / "outputs_public").rglob("*.tif"))
    assert not list((root / "outputs_public").rglob("*.zip"))


def test_guardrails_preserved(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    rows = read_csv(root / "outputs_public/logs_summary/revp_v2cx_to_v2dd_guardrail_rollup.csv")
    assert any(row["guardrail"] == "download_default" and row["observed_value"] == "false" for row in rows)


def test_unrelated_file_not_changed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    sentinel = root / "keep.txt"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("keep", encoding="utf-8")
    main(["--repo-root", str(root), "--offline", "--force"])
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("rel", [
    "docs/metodologia_cientifica/revp_v2cx_real_source_availability_verifier.md",
    "docs/metodologia_cientifica/revp_v2cy_controlled_product_link_discovery.md",
    "docs/metodologia_cientifica/revp_v2cz_product_license_audit.md",
    "docs/metodologia_cientifica/revp_v2da_controlled_download_plan.md",
    "docs/metodologia_cientifica/revp_v2db_patch_boundary_readiness_audit.md",
    "docs/metodologia_cientifica/revp_v2dc_integrated_readiness_matrix.md",
    "docs/metodologia_cientifica/revp_v2dd_scientific_readiness_dashboard.md",
])
def test_method_docs_generated(tmp_path: Path, rel: str) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    assert (root / rel).exists()
