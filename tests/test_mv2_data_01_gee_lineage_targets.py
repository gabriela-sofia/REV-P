"""Testes do marco MV2-DATA-01 — GEE Lineage Target Pack (Vertente B).

Cobrem as travas fail-closed: targets existem com colunas obrigatórias; template manual não
inventa scene_id/datetime/tile/cloud; planos GEE/STAC não executam; nenhum download/crop/
raster/feature/label/silver/gold/negativo; T23KPR não auto-vincula; cidade/região não basta;
matriz de risco completa; summary coerente; outputs leves sem caminho absoluto privado.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_gee_lineage_targets"
SCHEMA_DIR = PROJECT_ROOT / "datasets" / "schemas"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mv2_data_01_common as common  # noqa: E402
import mv2_data_01_run_gee_lineage_targets as runner  # noqa: E402
import mv2_data_01_validate_lineage_targets as validator  # noqa: E402

TARGET_REQUIRED_COLS = [
    "target_id", "binding_id", "anchor_id", "asset_id", "patch_id", "region", "sensor",
    "platform", "bbox", "crs", "has_asset_id", "has_patch_id", "has_bbox", "has_crs",
    "has_scene_id", "has_datetime", "has_tile", "has_cloud_cover", "lineage_status",
    "priority", "blocking_reason", "next_action",
]
TEMPLATE_INVENTED = ["product_id", "scene_id", "acquisition_datetime", "mgrs_tile",
                     "cloudy_pixel_percentage", "datatake_identifier", "gee_collection"]


def _read(name: str) -> list[dict[str, str]]:
    with (OUTPUT_DIR / name).open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


@pytest.fixture(scope="module", autouse=True)
def build_all() -> None:
    rc = runner.main()
    assert rc == 0, "orquestrador MV2-DATA-01 falhou"


# 1. targets existem
def test_targets_exist() -> None:
    rows = _read("mv2_data_01_lineage_targets.csv")
    assert rows, "targets vazio"
    # 128 ou divergência registrada vs. entrada
    src = common.read_csv_rows(common.resolve_data_input("medium_binding_index")[0])
    assert len(rows) == len(src)
    if len(src) == 128:
        assert len(rows) == 128


# 2. targets têm colunas obrigatórias
def test_targets_required_columns() -> None:
    rows = _read("mv2_data_01_lineage_targets.csv")
    cols = set(rows[0].keys())
    for c in TARGET_REQUIRED_COLS:
        assert c in cols, c


# 3. template manual existe
def test_template_exists() -> None:
    rows = _read("mv2_data_01_manual_gee_lineage_template.csv")
    assert len(rows) == len(_read("mv2_data_01_lineage_targets.csv"))


# 4. template manual não inventa scene_id
def test_template_no_invented_scene_id() -> None:
    for r in _read("mv2_data_01_manual_gee_lineage_template.csv"):
        assert not r["scene_id"].strip()
        assert not r["product_id"].strip()


# 5. template manual não inventa datetime
def test_template_no_invented_datetime() -> None:
    for r in _read("mv2_data_01_manual_gee_lineage_template.csv"):
        assert not r["acquisition_datetime"].strip()


# 6. query plan GEE não executa GEE
def test_gee_plan_does_not_execute() -> None:
    rows = _read("mv2_data_01_gee_metadata_query_plan.csv")
    assert rows
    for r in rows:
        assert r["would_call_gee"] == "false"
        assert r["would_export_raster"] == "false"
        assert r["execution_status"] == "NOT_EXECUTED_PLAN_ONLY"
    # campos mínimos de metadados
    for mf in common.METADATA_FIELDS:
        assert mf in rows[0]["metadata_fields_to_collect"]


# 7. query plan STAC/OData não executa HTTP
def test_stac_plan_does_not_execute_http() -> None:
    rows = _read("mv2_data_01_stac_odata_metadata_plan.csv")
    assert rows
    providers = {r["provider"] for r in rows}
    assert set(common.STAC_PROVIDERS) <= providers
    for r in rows:
        assert r["would_execute_http"] == "false"
        assert r["would_download"] == "false"
        assert r["execution_status"] == "NOT_EXECUTED_PLAN_ONLY"


# 8. nenhum download é marcado
def test_no_download_marked() -> None:
    for r in _read("mv2_data_01_stac_odata_metadata_plan.csv"):
        assert r["would_download"] == "false"
    s = json.loads((OUTPUT_DIR / "mv2_data_01_summary.json").read_text(encoding="utf-8"))
    assert s["downloads_executed"] == 0


# 9. nenhum crop é criado
def test_no_crop_created() -> None:
    s = json.loads((OUTPUT_DIR / "mv2_data_01_summary.json").read_text(encoding="utf-8"))
    assert s["crops_created"] == 0


# 10. nenhum raster nativo é criado
def test_no_native_raster_created() -> None:
    s = json.loads((OUTPUT_DIR / "mv2_data_01_summary.json").read_text(encoding="utf-8"))
    assert s["native_rasters_created"] == 0
    assert s["spectral_features_created"] == 0


# 11. nenhum label/silver/gold/negativo é criado
def test_no_label_silver_gold_negative() -> None:
    s = json.loads((OUTPUT_DIR / "mv2_data_01_summary.json").read_text(encoding="utf-8"))
    for k in ["labels_created", "silver_created", "gold_created", "negatives_created"]:
        assert s[k] == 0
    assert s["can_train"] is False
    assert s["sandbox_status"] == "bloqueado"


# 12. T23KPR não auto-vincula
def test_no_tile_auto_join() -> None:
    # template: nenhum tile preenchido
    for r in _read("mv2_data_01_manual_gee_lineage_template.csv"):
        assert not r["mgrs_tile"].strip()
    # targets: nenhum tile presente
    for r in _read("mv2_data_01_lineage_targets.csv"):
        assert r["has_tile"] == "false"
    # risco de auto-join registrado e bloqueante
    risks = _read("mv2_data_01_risk_matrix.csv")
    tile_risks = [r for r in risks if r["risk_type"] == "RISK_TILE_AUTO_JOIN"]
    assert tile_risks
    assert all(r["blocks_next_step"] == "true" for r in tile_risks)


# 13. cidade/região não basta para lineage
def test_region_not_enough_for_lineage() -> None:
    rows = _read("mv2_data_01_lineage_targets.csv")
    # todos têm região, mas nenhum tem scene_id derivado dela
    assert all(r["region"] for r in rows)
    assert all(r["has_scene_id"] == "false" for r in rows)
    risks = _read("mv2_data_01_risk_matrix.csv")
    assert any(r["risk_type"] == "RISK_CITY_REGION_PROXY" for r in risks)


# 14. risk matrix contém riscos obrigatórios
def test_risk_matrix_has_all_required() -> None:
    present = {r["risk_type"] for r in _read("mv2_data_01_risk_matrix.csv")}
    for rt in common.RISK_TYPES:
        assert rt in present, rt


# 15. summary JSON é coerente
def test_summary_coherent() -> None:
    s = json.loads((OUTPUT_DIR / "mv2_data_01_summary.json").read_text(encoding="utf-8"))
    targets = _read("mv2_data_01_lineage_targets.csv")
    assert s["total_targets"] == len(targets)
    assert s["gee_metadata_query_plans"] == len(_read("mv2_data_01_gee_metadata_query_plan.csv"))
    assert s["stac_odata_query_plans"] == len(_read("mv2_data_01_stac_odata_metadata_plan.csv"))
    assert s["manual_template_rows"] == len(_read("mv2_data_01_manual_gee_lineage_template.csv"))
    assert s["targets_with_asset_id"] == sum(1 for t in targets if t["has_asset_id"] == "true")
    assert s["targets_with_scene_id"] == 0
    assert s["can_unlock_day10_now"] is False


# 16. outputs públicos são leves
def test_outputs_are_light() -> None:
    heavy = {".tif", ".tiff", ".jp2", ".safe", ".npy", ".npz", ".png", ".jpg", ".zip"}
    for p in OUTPUT_DIR.rglob("*"):
        if p.is_file():
            assert p.suffix.lower() not in heavy, p.name
            assert p.stat().st_size <= 2_000_000, p.name


# 17. sem caminho absoluto privado
def test_no_private_absolute_path() -> None:
    for p in OUTPUT_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            assert "C:/Users" not in txt and "C:\\Users" not in txt, p.name


# 18. validador passa
def test_validator_passes() -> None:
    assert validator.main() == 0


# extra: schemas carregam com required_fields
def test_schemas_load() -> None:
    for n in [
        "schema_mv2_data_01_lineage_targets.json",
        "schema_mv2_data_01_manual_gee_template.json",
        "schema_mv2_data_01_metadata_query_plan.json",
    ]:
        d = json.loads((SCHEMA_DIR / n).read_text(encoding="utf-8"))
        assert d["schema_id"]
        assert d.get("required_fields") or d.get("required_fields_gee")
