from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

from revp_v2cs_to_v2cw_orchestrator import main  # noqa: E402


FORBIDDEN = ["GROUND_TRUTH_READY", "LABEL_READY", "TRAINING_READY", "MODEL_VALIDATED", "DETECTION_CONFIRMED", "PREDICTION_VALIDATED", "TP2_CLOSED"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_orchestrator_runs_offline(tmp_path: Path) -> None:
    assert main(["--repo-root", str(tmp_path / "repo"), "--offline", "--force"]) == 0


def test_orchestrator_generates_integrated_report(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    assert (root / "outputs_public/execution_reports/revp_v2cs_to_v2cw_integrated_report.md").exists()


def test_orchestrator_generates_commit_checklist(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    text = (root / "outputs_public/execution_reports/revp_v2cs_to_v2cw_commit_checklist.md").read_text(encoding="utf-8")
    assert "data: registra fontes externas reais e triagem conservadora de evidencias" in text


def test_orchestrator_generates_rollups(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    assert (root / "outputs_public/logs_summary/revp_v2cs_to_v2cw_test_rollup.csv").exists()
    assert (root / "outputs_public/logs_summary/revp_v2cs_to_v2cw_guardrail_rollup.csv").exists()


def test_rollup_has_five_stages(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    rows = read_csv(root / "outputs_public/logs_summary/revp_v2cs_to_v2cw_test_rollup.csv")
    assert [row["stage"] for row in rows] == ["v2cs", "v2ct", "v2cu", "v2cv", "v2cw"]


def test_guardrail_rollup_blocks_labels_and_tp2(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    rows = read_csv(root / "outputs_public/logs_summary/revp_v2cs_to_v2cw_guardrail_rollup.csv")
    observed = {row["guardrail"]: row["observed_value"] for row in rows}
    assert observed["binary_labels"] == "absent"
    assert observed["tp2_not_closed"] == "true"


def test_all_requested_outputs_exist(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    for rel in [
        "datasets/external_evidence/real_sources_registry_v2cs.csv",
        "outputs_public/tables/revp_source_license_triage_v2ct.csv",
        "datasets/external_evidence/sources_registry_v2cu.csv",
        "outputs_public/tables/revp_external_product_discovery_checklist_v2cv.csv",
        "outputs_public/tables/revp_external_evidence_regional_readiness_v2cw.csv",
    ]:
        assert (root / rel).exists()


def test_no_forbidden_tokens_in_non_test_outputs(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in (root / "outputs_public").rglob("*") if path.is_file())
    text += "\n" + (root / "datasets/external_evidence/real_sources_registry_v2cs.csv").read_text(encoding="utf-8")
    assert all(token not in text for token in FORBIDDEN)


def test_orchestrator_does_not_create_raw_downloads(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    main(["--repo-root", str(root), "--offline", "--force"])
    assert not (root / "datasets/external_evidence/raw").exists()


def test_orchestrator_keeps_unrelated_file(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    sentinel = root / "unrelated.txt"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("keep", encoding="utf-8")
    main(["--repo-root", str(root), "--offline", "--force"])
    assert sentinel.read_text(encoding="utf-8") == "keep"
