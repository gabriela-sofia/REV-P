from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

import revp_v2cx_to_v2dd_common as common  # noqa: E402
from revp_v2dc_integrated_readiness_matrix import main  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prepared(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    common.write_csv(root / "datasets/external_evidence/real_sources_registry_v2cs.csv", [
        {"source_id": "SRC_REC", "source_family": "CHARTER", "region": "Recife", "source_url": "https://example.org/recife"}
    ], ["source_id", "source_family", "region", "source_url"])
    common.run_availability(root, force=True)
    common.run_discovery(root, force=True)
    common.run_license(root, force=True)
    common.run_download_plan(root, force=True)
    common.run_boundary(root, force=True)
    return root


def test_matrix_exists_after_cli(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    assert main(["--repo-root", str(root), "--offline", "--force"]) == 0
    assert common.readiness_path(root).exists()


def test_missing_license_blocks_download_readiness(tmp_path: Path) -> None:
    row = common.build_readiness(prepared(tmp_path))[0]
    assert row["license_ready"] == "false"
    assert row["download_ready"] == "false"


def test_missing_local_file_blocks_qa(tmp_path: Path) -> None:
    row = common.build_readiness(prepared(tmp_path))[0]
    assert row["local_file_available"] == "false"
    assert row["geospatial_qa_ready"] == "false"


def test_missing_patch_boundary_blocks_replay(tmp_path: Path) -> None:
    row = common.build_readiness(prepared(tmp_path))[0]
    assert row["patch_boundary_ready"] == "false"
    assert row["replay_ready"] == "false"


def test_no_status_closes_tp2(tmp_path: Path) -> None:
    assert all(row["tp2_candidate_readiness"] != "TP2_CLOSED" for row in common.build_readiness(prepared(tmp_path)))


def test_guardrail_generated(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    main(["--repo-root", str(root), "--offline", "--force"])
    rows = read_csv(root / "outputs_public/logs_summary/revp_integrated_readiness_guardrails_v2dc.csv")
    assert any(row["guardrail"] == "tp2_not_closed" for row in rows)


@pytest.mark.parametrize("field", common.READINESS_FIELDS)
def test_readiness_has_required_fields(tmp_path: Path, field: str) -> None:
    row = common.build_readiness(prepared(tmp_path))[0]
    assert field in row


def test_recommended_next_action_is_explicit(tmp_path: Path) -> None:
    row = common.build_readiness(prepared(tmp_path))[0]
    assert row["recommended_next_action"]
