from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

import revp_v2cx_to_v2dd_common as common  # noqa: E402
from revp_v2cx_real_source_availability_verifier import main  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prepared(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    common.write_csv(root / "datasets/external_evidence/real_sources_registry_v2cs.csv", [
        {"source_id": "SRC_REC", "source_family": "INTERNATIONAL_CHARTER", "region": "Recife", "source_url": "https://example.org/recife", "requires_manual_access": "true", "license_status": "UNKNOWN"},
        {"source_id": "SRC_PET", "source_family": "COPERNICUS_EMS", "region": "Petropolis", "source_url": "https://example.org/petropolis", "requires_manual_access": "false", "license_status": "UNKNOWN"},
        {"source_id": "SRC_CTB", "source_family": "CURITIBA_IPPUC", "region": "Curitiba", "source_url": "", "requires_manual_access": "true", "license_status": "UNKNOWN"},
    ], ["source_id", "source_family", "region", "source_url", "requires_manual_access", "license_status"])
    return root


def test_offline_marks_not_checked_without_network(tmp_path: Path) -> None:
    rows = common.build_availability(prepared(tmp_path), allow_network=False)
    assert rows[0]["availability_status"] == "NOT_CHECKED_OFFLINE"
    assert rows[0]["network_checked"] == "false"


def test_source_without_url_requires_manual_access(tmp_path: Path) -> None:
    rows = common.build_availability(prepared(tmp_path), allow_network=False)
    assert rows[2]["availability_status"] == "SOURCE_REQUIRES_MANUAL_ACCESS"
    assert rows[2]["blocking_reason"] == "SOURCE_URL_MISSING"


def test_cli_writes_availability_table(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    assert main(["--repo-root", str(root), "--offline", "--force"]) == 0
    assert common.availability_path(root).exists()


def test_guardrail_log_generated(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    main(["--repo-root", str(root), "--offline", "--force"])
    rows = read_csv(root / "outputs_public/logs_summary/revp_real_source_availability_guardrails_v2cx.csv")
    assert any(row["guardrail"] == "network_default" for row in rows)


def test_report_generated(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    main(["--repo-root", str(root), "--offline", "--force"])
    text = (root / "outputs_public/execution_reports/revp_real_source_availability_report_v2cx.md").read_text(encoding="utf-8")
    assert "v2cx real source availability" in text


def test_no_ground_truth_promotion_status(tmp_path: Path) -> None:
    rows = common.build_availability(prepared(tmp_path), allow_network=False)
    assert all("SOURCE_VALIDATED_AS_GROUND_TRUTH" not in row["availability_status"] for row in rows)


def test_registered_urls_are_registry_only(tmp_path: Path) -> None:
    urls = common.registered_urls(prepared(tmp_path))
    assert urls == {"https://example.org/recife", "https://example.org/petropolis"}


@pytest.mark.parametrize("field", common.AVAILABILITY_FIELDS)
def test_availability_has_required_fields(tmp_path: Path, field: str) -> None:
    row = common.build_availability(prepared(tmp_path), allow_network=False)[0]
    assert field in row


def test_allowed_claim_is_review_only(tmp_path: Path) -> None:
    row = common.build_availability(prepared(tmp_path), allow_network=False)[0]
    assert "review-only" in row["allowed_claim"]
