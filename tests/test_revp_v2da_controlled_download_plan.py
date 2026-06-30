from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

import revp_v2cx_to_v2dd_common as common  # noqa: E402
from revp_v2da_controlled_download_plan import main  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prepared(tmp_path: Path, url: str = "https://example.org/recife.zip", raw: str = "false") -> Path:
    root = tmp_path / "repo"
    common.write_csv(common.license_path(root), [
        {"license_audit_id": "LIC1", "product_candidate_id": "PROD1", "source_id": "SRC1", "source_family": "CHARTER", "candidate_url": url, "license_status": "UNKNOWN", "license_reference": "", "redistribution_allowed": "false", "raw_download_allowed": raw, "public_output_allowed": "false", "metadata_only_allowed": "true", "manual_license_review_required": "true", "license_audit_status": "DOWNLOAD_BLOCKED_LICENSE_UNKNOWN", "blocking_reason": "LICENSE_UNKNOWN", "allowed_claim": common.ALLOWED_CLAIM, "forbidden_claim": common.FORBIDDEN_CLAIM}
    ], common.LICENSE_FIELDS)
    return root


def test_plan_offline_does_not_download(tmp_path: Path) -> None:
    row = common.build_download_plan(prepared(tmp_path), allow_downloads=False)[0]
    assert row["download_executed"] == "false"


def test_license_blocks_download(tmp_path: Path) -> None:
    row = common.build_download_plan(prepared(tmp_path), allow_downloads=True)[0]
    assert row["download_status"] == "DOWNLOAD_BLOCKED_LICENSE"


def test_raw_never_planned_to_outputs_public(tmp_path: Path) -> None:
    row = common.build_download_plan(prepared(tmp_path), allow_downloads=False)[0]
    assert "outputs_public" not in row["planned_local_path"]


def test_hash_absent_when_no_file(tmp_path: Path) -> None:
    row = common.build_download_plan(prepared(tmp_path), allow_downloads=False)[0]
    assert row["sha256"] == ""


def test_forbidden_extension_blocks(tmp_path: Path) -> None:
    row = common.build_download_plan(prepared(tmp_path, url="https://example.org/file.exe", raw="true"), allow_downloads=True)[0]
    assert row["download_status"] == "DOWNLOAD_BLOCKED_EXTENSION"


def test_cli_writes_private_and_public_plan(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    assert main(["--repo-root", str(root), "--offline", "--force"]) == 0
    assert common.download_private_path(root).exists()
    assert common.download_public_path(root).exists()


def test_download_report_generated(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    main(["--repo-root", str(root), "--offline", "--force"])
    assert "v2da controlled download plan" in (root / "outputs_public/execution_reports/revp_controlled_download_plan_report_v2da.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("field", common.DOWNLOAD_FIELDS)
def test_download_has_required_fields(tmp_path: Path, field: str) -> None:
    row = common.build_download_plan(prepared(tmp_path), allow_downloads=False)[0]
    assert field in row


def test_no_candidate_url_blocks_offline(tmp_path: Path) -> None:
    row = common.build_download_plan(prepared(tmp_path, url=""), allow_downloads=False)[0]
    assert row["download_status"] == "DOWNLOAD_BLOCKED_OFFLINE"
