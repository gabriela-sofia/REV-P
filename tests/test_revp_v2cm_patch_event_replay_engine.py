from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

from revp_v2cm_patch_event_replay_engine import build_replay, main  # noqa: E402


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def repo(tmp_path: Path, patch: str = "false", status: str = "NO_OBSERVED_VECTOR_GEOMETRY", crs: str = "false") -> Path:
    root = tmp_path / "repo"
    write_csv(root / "outputs_public/tables/revp_tp2_patch_pair_candidates_v2ci.csv",
              ["pair_id", "candidate_id", "patch_id", "region", "patch_boundary_available"],
              [{"pair_id": "P", "candidate_id": "A", "patch_id": "PATCH", "region": "Recife",
                "patch_boundary_available": patch}])
    write_csv(root / "outputs_public/tables/revp_observed_geometry_validation_v2cl.csv",
              ["candidate_id", "validation_status", "crs_known", "provenance_available", "hash_available", "blocking_reason"],
              [{"candidate_id": "A", "validation_status": status, "crs_known": crs,
                "provenance_available": "true", "hash_available": "true", "blocking_reason": "blocked"}])
    return root


def test_blocks_without_patch_boundary(tmp_path: Path) -> None:
    rows = build_replay(repo(tmp_path, patch="false"))
    assert rows[0]["replay_status"] == "REPLAY_BLOCKED_NO_PATCH_BOUNDARY"


def test_blocks_without_observed_geometry(tmp_path: Path) -> None:
    rows = build_replay(repo(tmp_path, patch="true"))
    assert rows[0]["replay_status"] == "REPLAY_BLOCKED_NO_OBSERVED_GEOMETRY"


def test_blocks_without_crs(tmp_path: Path) -> None:
    rows = build_replay(repo(tmp_path, patch="true", status="VALIDATED_OBSERVED_GEOMETRY_CANDIDATE", crs="false"))
    assert rows[0]["replay_status"] == "REPLAY_BLOCKED_MISSING_CRS"


def test_no_intersection_with_invalid_inputs(tmp_path: Path) -> None:
    rows = build_replay(repo(tmp_path, patch="true"))
    assert rows[0]["intersection_test_executed"] == "false"


def test_area_fields_empty_when_blocked(tmp_path: Path) -> None:
    row = build_replay(repo(tmp_path, patch="false"))[0]
    assert row["intersection_area"] == ""
    assert row["intersection_ratio_patch"] == ""
    assert row["intersection_ratio_event"] == ""


def test_ready_not_executed_is_candidate_language(tmp_path: Path) -> None:
    row = build_replay(repo(tmp_path, patch="true", status="VALIDATED_OBSERVED_GEOMETRY_CANDIDATE", crs="true"))[0]
    assert row["replay_status"] == "REPLAY_READY_NOT_EXECUTED"
    assert row["candidate_intersection_computed"] == "false"


def test_outputs_and_guardrails_written(tmp_path: Path) -> None:
    root = repo(tmp_path)
    assert main(["--repo-root", str(root), "--force"]) == 0
    assert (root / "outputs_public/tables/revp_patch_event_replay_v2cm.csv").exists()
    assert (root / "outputs_public/logs_summary/revp_patch_event_replay_guardrails_v2cm.csv").exists()


def test_report_says_blockable(tmp_path: Path) -> None:
    root = repo(tmp_path)
    main(["--repo-root", str(root), "--force"])
    text = (root / "outputs_public/execution_reports/revp_patch_event_replay_report_v2cm.md").read_text()
    assert "replay patch-evento bloqueavel" in text

