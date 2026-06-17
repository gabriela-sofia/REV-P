from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

import revp_v2cs_to_v2cw_common as common  # noqa: E402


FORBIDDEN = ["GROUND_TRUTH_READY", "LABEL_READY", "TRAINING_READY", "MODEL_VALIDATED", "DETECTION_CONFIRMED", "PREDICTION_VALIDATED", "TP2_CLOSED"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_seed_contains_seven_real_sources() -> None:
    rows = common.seed_rows()
    assert len(rows) == 7


def test_seed_contains_charter_751_petropolis() -> None:
    row = next(row for row in common.seed_rows() if "CHARTER_751" in row["source_id"])
    assert row["region"] == "Petropolis"
    assert "activation-751" in row["source_url"]
    assert row["initial_status"] == "DOCUMENTARY_SOURCE_REQUIRES_REVIEW"


def test_seed_contains_charter_758_recife_without_flood_inference() -> None:
    row = next(row for row in common.seed_rows() if "CHARTER_758" in row["source_id"])
    assert row["region"] == "Recife"
    assert "LANDSLIDE_NOT_AUTOMATIC_FLOOD_EXTENT" in row["blocking_reason"]


def test_seed_contains_copernicus_ems_and_gfm() -> None:
    families = {row["source_family"] for row in common.seed_rows()}
    assert {"COPERNICUS_EMS", "COPERNICUS_GFM"}.issubset(families)


def test_seed_contains_sgb_drm_and_curitiba() -> None:
    ids = {row["source_id"] for row in common.seed_rows()}
    assert "REAL_v2cs_SGB_CPRM_PREVENCAO_DESASTRES" in ids
    assert "REAL_v2cs_DRM_RJ_CARTA_RISCO_PETROPOLIS" in ids
    assert "REAL_v2cs_CURITIBA_DADOS_ABERTOS_IPPUC" in ids


def test_all_sources_block_downloads_initially() -> None:
    assert all(row["download_allowed"] == "false" for row in common.seed_rows())


def test_all_sources_block_public_repo_raw_files_initially() -> None:
    assert all(row["public_repo_allowed"] == "false" for row in common.seed_rows())


def test_allowed_and_forbidden_claims_explicit() -> None:
    rows = common.seed_rows()
    assert all(row["allowed_claim"] for row in rows)
    assert all("ground_truth_operacional" in row["forbidden_claim"] for row in rows)


def test_run_seeding_writes_private_and_public_registries(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    assert common.run_seeding(root, force=True) == 0
    assert common.real_sources_path(root).exists()
    assert (root / "outputs_public/tables/revp_real_external_sources_public_v2cs.csv").exists()


def test_seeding_outputs_do_not_use_forbidden_status_tokens(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    common.run_seeding(root, force=True)
    text = common.real_sources_path(root).read_text(encoding="utf-8")
    assert all(token not in text for token in FORBIDDEN)
