from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

import revp_v2cx_to_v2dd_common as common  # noqa: E402
from revp_v2cy_controlled_product_link_discovery import main  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prepared(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    common.write_csv(root / "datasets/external_evidence/real_sources_registry_v2cs.csv", [
        {"source_id": "SRC_REC", "source_family": "INTERNATIONAL_CHARTER", "region": "Recife", "source_url": "https://example.org/recife"}
    ], ["source_id", "source_family", "region", "source_url"])
    return root


def test_offline_discovery_blocks_without_local_html(tmp_path: Path) -> None:
    rows = common.build_discovery(prepared(tmp_path), allow_network=False)
    assert rows[0]["discovery_status"] == "DISCOVERY_BLOCKED_OFFLINE"


def test_offline_discovery_does_not_allow_download(tmp_path: Path) -> None:
    rows = common.build_discovery(prepared(tmp_path), allow_network=False)
    assert rows[0]["download_allowed"] == "false"


def test_extracts_related_zip_link() -> None:
    rows = common.extract_candidate_links('<a href="recife_flood_map.zip">Map product</a>', "https://example.org/base/", {"region": "Recife", "source_family": "CHARTER"})
    assert rows[0]["candidate_extension"] == ".zip"
    assert rows[0]["discovery_status"] == "PRODUCT_LINK_CANDIDATE_FOUND"


def test_rejects_unrelated_link() -> None:
    rows = common.extract_candidate_links('<a href="file.zip">unrelated</a>', "https://example.org/base/", {"region": "Nowhere", "source_family": "OTHER"})
    assert rows == []


@pytest.mark.parametrize("ext,expected", [(".png", "visual_documentary_not_geometry"), (".jpg", "visual_documentary_not_geometry"), (".pdf", "candidate_documentary_product"), (".geojson", "candidate_vector_or_package_unvalidated")])
def test_product_type_stays_unvalidated(ext: str, expected: str) -> None:
    assert common.product_type(ext) == expected


def test_cli_writes_discovery_table(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    assert main(["--repo-root", str(root), "--offline", "--force"]) == 0
    assert common.discovery_path(root).exists()


def test_discovery_report_generated(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    main(["--repo-root", str(root), "--offline", "--force"])
    text = (root / "outputs_public/execution_reports/revp_controlled_product_link_discovery_report_v2cy.md").read_text(encoding="utf-8")
    assert "v2cy controlled product link discovery" in text


@pytest.mark.parametrize("field", common.DISCOVERY_FIELDS)
def test_discovery_has_required_fields(tmp_path: Path, field: str) -> None:
    row = common.build_discovery(prepared(tmp_path), allow_network=False)[0]
    assert field in row


def test_candidate_link_requires_manual_review() -> None:
    rows = common.extract_candidate_links('<a href="curitiba_map.geojson">Curitiba map</a>', "https://example.org/", {"region": "Curitiba", "source_family": "IPPUC"})
    assert rows[0]["requires_manual_review"] == "true"
