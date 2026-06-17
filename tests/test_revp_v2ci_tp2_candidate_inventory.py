"""Testes para o inventario TP2-ready v2ci."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

from revp_v2ci_tp2_candidate_inventory import (  # noqa: E402
    DISALLOWED_PROMOTION_STATUS,
    INVENTORY_FIELDS,
    PAIR_FIELDS,
    main,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    _write_csv(
        repo / "datasets/protocolo_c/v2bc_ground_truth_seed_registry.csv",
        [
            "seed_id",
            "candidate_id",
            "patch_id",
            "city",
            "region",
            "seed_status",
            "geometry_status",
            "can_create_ground_truth",
            "can_create_label",
            "can_train_model",
            "blocking_reason",
        ],
        [
            {
                "seed_id": "SEED_001",
                "candidate_id": "CTB_TEST_001",
                "patch_id": "",
                "city": "Curitiba",
                "region": "Curitiba",
                "seed_status": "CANDIDATE_GROUND_TRUTH_SEED",
                "geometry_status": "GEOMETRY_MISSING",
                "can_create_ground_truth": "false",
                "can_create_label": "false",
                "can_train_model": "false",
                "blocking_reason": "GEOMETRY_MISSING",
            }
        ],
    )
    _write_csv(
        repo / "datasets/protocolo_c/v2ap_patch_truth_boundary_update.csv",
        [
            "candidate_id",
            "patch_id",
            "region",
            "event_geometry_status",
            "patch_geometry_status",
            "source_reference",
            "forbidden_use",
        ],
        [
            {
                "candidate_id": "PET_READY_BUT_NO_CRS_HASH",
                "patch_id": "PET_PATCH_001",
                "region": "Petropolis",
                "event_geometry_status": "READY",
                "patch_geometry_status": "READY",
                "source_reference": "local_registry",
                "forbidden_use": "ground_truth|label|training|prediction",
            }
        ],
    )
    _write_csv(
        repo / "outputs_public/tables/protocol_c_cross_region_candidate_registry.csv",
        [
            "reference_id",
            "region",
            "city",
            "package_id",
            "reference_status",
            "evidence_basis",
            "allowed_use",
            "forbidden_use",
            "can_create_operational_label",
            "can_create_negative",
            "can_train_model",
        ],
        [
            {
                "reference_id": "XREF_001",
                "region": "Recife",
                "city": "Recife",
                "package_id": "ARP_TEST",
                "reference_status": "PROTOCOL_VALIDATED_CANDIDATE_REFERENCE",
                "evidence_basis": "Charter candidate",
                "allowed_use": "PROTOCOL_C_REFERENCE_REVIEW",
                "forbidden_use": "SUPERVISED_LABEL|NEGATIVE_LABEL|TRAINING_TARGET",
                "can_create_operational_label": "false",
                "can_create_negative": "false",
                "can_train_model": "false",
            }
        ],
    )
    return repo


def test_generates_expected_files(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    assert main(["--repo-root", str(repo), "--force"]) == 0
    expected = [
        "outputs_public/tables/revp_tp2_candidate_inventory_v2ci.csv",
        "outputs_public/tables/revp_tp2_patch_pair_candidates_v2ci.csv",
        "outputs_public/logs_summary/revp_tp2_candidate_guardrail_rollup_v2ci.csv",
        "outputs_public/execution_reports/revp_tp2_candidate_inventory_report_v2ci.md",
        "outputs_public/execution_reports/revp_tp2_candidate_commit_checklist_v2ci.md",
    ]
    for rel in expected:
        assert (repo / rel).exists(), rel


def test_inventory_required_columns(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    main(["--repo-root", str(repo), "--force"])
    rows = _read_csv(repo / "outputs_public/tables/revp_tp2_candidate_inventory_v2ci.csv")
    assert rows
    assert set(INVENTORY_FIELDS).issubset(rows[0].keys())


def test_pair_required_columns(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    main(["--repo-root", str(repo), "--force"])
    rows = _read_csv(repo / "outputs_public/tables/revp_tp2_patch_pair_candidates_v2ci.csv")
    assert rows
    assert set(PAIR_FIELDS).issubset(rows[0].keys())


def test_no_ground_truth_ready_status(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    main(["--repo-root", str(repo), "--force"])
    inv = _read_csv(repo / "outputs_public/tables/revp_tp2_candidate_inventory_v2ci.csv")
    pairs = _read_csv(repo / "outputs_public/tables/revp_tp2_patch_pair_candidates_v2ci.csv")
    assert all(row["candidate_status"] != DISALLOWED_PROMOTION_STATUS for row in inv)
    assert all(row["tp2_status"] != DISALLOWED_PROMOTION_STATUS for row in pairs)


def test_training_ready_remains_blocked(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    main(["--repo-root", str(repo), "--force"])
    guardrails = _read_csv(repo / "outputs_public/logs_summary/revp_tp2_candidate_guardrail_rollup_v2ci.csv")
    training = next(row for row in guardrails if row["guardrail"] == "training_ready")
    assert training["observed_value"] == "BLOCKED"
    assert training["status"] == "PASS"


def test_missing_observed_geometry_blocks_tp2(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    main(["--repo-root", str(repo), "--force"])
    inv = _read_csv(repo / "outputs_public/tables/revp_tp2_candidate_inventory_v2ci.csv")
    missing = [row for row in inv if row["has_observed_geometry"] == "false"]
    assert missing
    assert all(row["candidate_status"] in {"TP2_BLOCKED", "TP2_CANDIDATE_ONLY"} for row in missing)


def test_missing_crs_blocks_tp2(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    main(["--repo-root", str(repo), "--force"])
    inv = _read_csv(repo / "outputs_public/tables/revp_tp2_candidate_inventory_v2ci.csv")
    without_crs = [row for row in inv if row["crs_known"] == "false"]
    assert without_crs
    assert all(row["candidate_status"] != "TP2_READY_FOR_REPLAY" for row in without_crs)


def test_missing_hash_or_provenance_blocks_tp2(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    main(["--repo-root", str(repo), "--force"])
    inv = _read_csv(repo / "outputs_public/tables/revp_tp2_candidate_inventory_v2ci.csv")
    blocked = [row for row in inv if row["hash_available"] == "false" or row["provenance_available"] == "false"]
    assert blocked
    assert all(row["candidate_status"] != "TP2_READY_FOR_REPLAY" for row in blocked)


def test_tp3_never_ready_when_tp2_blocked(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    main(["--repo-root", str(repo), "--force"])
    pairs = _read_csv(repo / "outputs_public/tables/revp_tp2_patch_pair_candidates_v2ci.csv")
    assert all(not (row["tp2_status"] == "TP2_BLOCKED" and row["tp3_ready"] == "true") for row in pairs)


def test_forbidden_claims_explicit(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    main(["--repo-root", str(repo), "--force"])
    inv = _read_csv(repo / "outputs_public/tables/revp_tp2_candidate_inventory_v2ci.csv")
    for row in inv:
        assert "ground_truth_operacional" in row["forbidden_claim"]
        assert "label_binario" in row["forbidden_claim"]
        assert "treino" in row["forbidden_claim"]


def test_report_says_inventory_not_tp2_closure(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    main(["--repo-root", str(repo), "--force"])
    report = (repo / "outputs_public/execution_reports/revp_tp2_candidate_inventory_report_v2ci.md").read_text(
        encoding="utf-8"
    )
    assert "inventario TP2-ready" in report
    assert "nao um fechamento de TP2" in report
