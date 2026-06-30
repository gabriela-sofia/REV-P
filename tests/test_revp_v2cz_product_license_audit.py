from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

import revp_v2cx_to_v2dd_common as common  # noqa: E402
from revp_v2cz_product_license_audit import main  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prepared(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    common.write_csv(common.discovery_path(root), [
        {"product_candidate_id": "PROD1", "source_id": "SRC1", "region": "Recife", "source_family": "CHARTER", "parent_url": "https://example.org", "candidate_url": "https://example.org/recife.zip", "candidate_label": "Recife", "candidate_extension": ".zip", "candidate_product_type": "candidate_vector_or_package_unvalidated", "relation_to_event": "term", "requires_manual_review": "true", "download_allowed": "false", "license_status": "UNKNOWN", "discovery_status": "PRODUCT_LINK_CANDIDATE_FOUND", "blocking_reason": "LICENSE_UNKNOWN", "allowed_claim": common.ALLOWED_CLAIM, "forbidden_claim": common.FORBIDDEN_CLAIM}
    ], common.DISCOVERY_FIELDS)
    common.write_csv(root / "outputs_public/tables/revp_source_license_triage_v2ct.csv", [
        {"source_id": "SRC1", "license_status": "UNKNOWN", "redistribution_allowed": "false", "raw_download_allowed": "false", "raw_public_output_allowed": "false", "license_reference": ""}
    ], ["source_id", "license_status", "redistribution_allowed", "raw_download_allowed", "raw_public_output_allowed", "license_reference"])
    return root


def test_unknown_license_blocks_download(tmp_path: Path) -> None:
    row = common.build_license_audit(prepared(tmp_path))[0]
    assert row["license_audit_status"] == "DOWNLOAD_BLOCKED_LICENSE_UNKNOWN"
    assert row["raw_download_allowed"] == "false"


def test_redistribution_unknown_blocks_public_output(tmp_path: Path) -> None:
    row = common.build_license_audit(prepared(tmp_path))[0]
    assert row["public_output_allowed"] == "false"


def test_metadata_only_allowed_even_when_raw_blocked(tmp_path: Path) -> None:
    row = common.build_license_audit(prepared(tmp_path))[0]
    assert row["metadata_only_allowed"] == "true"


def test_manual_license_review_required_for_unknown(tmp_path: Path) -> None:
    row = common.build_license_audit(prepared(tmp_path))[0]
    assert row["manual_license_review_required"] == "true"


def test_cli_writes_license_table(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    assert main(["--repo-root", str(root), "--offline", "--force"]) == 0
    assert common.license_path(root).exists()


def test_license_guardrail_generated(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    main(["--repo-root", str(root), "--offline", "--force"])
    rows = read_csv(root / "outputs_public/logs_summary/revp_product_license_guardrails_v2cz.csv")
    assert any(row["guardrail"] == "unknown_license_blocks_download" for row in rows)


@pytest.mark.parametrize("field", common.LICENSE_FIELDS)
def test_license_has_required_fields(tmp_path: Path, field: str) -> None:
    row = common.build_license_audit(prepared(tmp_path))[0]
    assert field in row


def test_license_is_not_inferred_by_domain(tmp_path: Path) -> None:
    row = common.build_license_audit(prepared(tmp_path))[0]
    assert row["license_status"] == "UNKNOWN"
