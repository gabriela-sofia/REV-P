"""Testes do SUSC-17C18 - estatistica zonal CHIRPS e consulta STAC Sentinel-2 por AOI."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"

EXPECTED = [
    OUT / "SUSC_17C18_ESTATISTICA_ZONAL_CHIRPS_E_CONSULTA_STAC_SENTINEL2_REPORT.md",
    OUT / "susc_17c18_sensor_aoi_execution_plan.csv",
    OUT / "susc_17c18_chirps_source_resolution.csv",
    OUT / "susc_17c18_chirps_lightweight_artifact_manifest.csv",
    OUT / "susc_17c18_chirps_zonal_stats.csv",
    OUT / "susc_17c18_rainfall_trigger_features.csv",
    OUT / "susc_17c18_sentinel2_stac_query_plan.csv",
    OUT / "susc_17c18_sentinel2_stac_query_results.csv",
    OUT / "susc_17c18_sentinel2_candidate_scene_registry.csv",
    OUT / "susc_17c18_embedding_tile_blockers.csv",
    OUT / "susc_17c18_sensor_feature_provenance.csv",
    OUT / "susc_17c18_no_leakage_audit.csv",
    OUT / "susc_17c18_gate_evaluation_matrix.csv",
    OUT / "susc_17c18_readiness_summary.json",
    OUT / "susc_17c18_promotion_blockers.csv",
    SCHEMAS / "susc_17c18_chirps_zonal_schema_v1.json",
    SCHEMAS / "susc_17c18_sentinel2_scene_schema_v1.json",
    SCHEMAS / "susc_17c18_sensor_feature_provenance_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c18_chirps_stac_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c18_chirps_stac_common", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str, *args: str):
    env = os.environ.copy()
    env.pop("SUSC_17C18_ALLOW_NETWORK", None)
    env.pop("SUSC_17C18_ALLOW_LIGHTWEIGHT_DOWNLOAD", None)
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )


def _scene_snapshots(s17c18) -> list[Path]:
    return [
        s17c18._scene_snapshot_path(ctx["candidate_patch_id"])
        for ctx in s17c18._patch_context()
        if s17c18._load_scene_snapshot(ctx["candidate_patch_id"]) is not None
    ]


def test_build_offline_gera_todos_artefatos_e_valida():
    result = run_script("build_susc_17c18_chirps_stac_aoi.py")
    assert result.returncode == 0, result.stderr + result.stdout
    s17c18 = _load_common()
    tracked = EXPECTED + _scene_snapshots(s17c18)
    first = {path: path.read_bytes() for path in tracked}
    result = run_script("build_susc_17c18_chirps_stac_aoi.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in tracked} == first
    result = run_script("validate_susc_17c18_chirps_stac_aoi.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_validos_e_ids_deterministicos():
    s17c18 = _load_common()
    zonal_schema = json.loads((SCHEMAS / "susc_17c18_chirps_zonal_schema_v1.json").read_text(encoding="utf-8"))
    scene_schema = json.loads((SCHEMAS / "susc_17c18_sentinel2_scene_schema_v1.json").read_text(encoding="utf-8"))
    prov_schema = json.loads((SCHEMAS / "susc_17c18_sensor_feature_provenance_schema_v1.json").read_text(encoding="utf-8"))
    for row in read_csv(OUT / "susc_17c18_chirps_zonal_stats.csv"):
        assert s17c18._schema_violations(row, zonal_schema) == []
    for row in read_csv(OUT / "susc_17c18_sentinel2_candidate_scene_registry.csv"):
        assert s17c18._schema_violations(row, scene_schema) == []
    for row in read_csv(OUT / "susc_17c18_sensor_feature_provenance.csv"):
        assert s17c18._schema_violations(row, prov_schema) == []
    assert [row["sensor_aoi_plan_id"] for row in read_csv(OUT / "susc_17c18_sensor_aoi_execution_plan.csv")] == [
        f"S17C18_AOIEXEC_{idx:04d}" for idx in range(1, 6)
    ]


def test_query_rede_exige_opt_in():
    for mode in ["query-chirps", "query-stac-sentinel2"]:
        result = run_script("susc_17c18_chirps_stac_aoi_runner.py", mode)
        assert result.returncode == 2
        assert "blocked_network_not_enabled" in result.stdout


def test_download_leve_exige_opt_in():
    s17c18 = _load_common()
    env = os.environ.copy()
    env.pop("SUSC_17C18_ALLOW_LIGHTWEIGHT_DOWNLOAD", None)
    assert s17c18._lightweight_download_enabled() is False


def test_chirps_sem_artefato_nao_cria_feature_real_e_exige_hash():
    zonal = read_csv(OUT / "susc_17c18_chirps_zonal_stats.csv")
    rainfall = read_csv(OUT / "susc_17c18_rainfall_trigger_features.csv")
    manifest = read_csv(OUT / "susc_17c18_chirps_lightweight_artifact_manifest.csv")
    assert manifest == []
    assert {row["feature_value_real"] for row in zonal} == {"false"}
    assert {row["feature_value_real"] for row in rainfall} == {"false"}
    assert {row["stat_value"] for row in zonal} == {""}
    for row in zonal + rainfall:
        if row["feature_value_real"] == "true":
            assert row["source_artifact_manifest_id"] != "not_available" or row.get("source_manifest_id") != "not_available"


def test_sentinel2_stac_nao_baixa_produto_nem_cria_tile():
    scenes = read_csv(OUT / "susc_17c18_sentinel2_candidate_scene_registry.csv")
    plan = read_csv(OUT / "susc_17c18_sentinel2_stac_query_plan.csv")
    assert len(scenes) > 0
    assert {row["downloaded"] for row in scenes} == {"false"}
    assert {row["tile_created"] for row in scenes} == {"false"}
    assert {row["download_product"] for row in plan} == {"false"}
    assert {row["create_tile"] for row in plan} == {"false"}
    summary = json.loads((OUT / "susc_17c18_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["sentinel2_products_downloaded_count"] == 0
    assert summary["sentinel2_tiles_created_count"] == 0
    assert summary["embedding_tiles_created_count"] == 0


def test_cena_pos_evento_nao_vira_feature_pre_evento():
    scenes = read_csv(OUT / "susc_17c18_sentinel2_candidate_scene_registry.csv")
    for row in scenes:
        if row["is_post_event"] == "true":
            assert row["usable_for_pre_event_feature"] == "false"
    leakage = read_csv(OUT / "susc_17c18_no_leakage_audit.csv")
    assert {row["uses_post_event_as_pre_event_feature"] for row in leakage} == {"false"}
    assert {row["passes_no_leakage"] for row in leakage} == {"true"}


def test_cena_e_chirps_nao_viram_evento_nem_ground_reference():
    gates = read_csv(OUT / "susc_17c18_gate_evaluation_matrix.csv")
    assert {row["G4_vinculo_espacial_evento"] for row in gates} == {"false"}
    assert {row["G5_separacao_fenomeno"] for row in gates} == {"false"}
    assert {row["all_gates_passed_for_ground_reference"] for row in gates} == {"false"}
    summary = json.loads((OUT / "susc_17c18_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["ground_reference_candidates_created_count"] == 0
    leakage = read_csv(OUT / "susc_17c18_no_leakage_audit.csv")
    assert {row["uses_catalog_metadata_as_event"] for row in leakage} == {"false"}


def test_score_guardrails_e_17b_inelegivel():
    summary = json.loads((OUT / "susc_17c18_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
    assert summary["review_only"] is True
    assert summary["trainable"] is False
    assert summary["ground_truth"] is False
    assert summary["features_extracted_this_sprint"] == 0
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    for target in [
        DAT / "susc_score_v6_candidate_by_patch_v1.csv",
        DAT / "susc_patches_official_v1.csv",
        DAT / "susc_patch_links_official_v1.csv",
    ]:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--", str(target.relative_to(ROOT))],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.stdout.strip() == ""
    blockers = read_csv(OUT / "susc_17c18_promotion_blockers.csv")
    assert any(row["blocker_type"] == "17b_blocked_until_event_reference_and_policy" for row in blockers)
    assert {row["blocks_17b"] for row in blockers} == {"true"}
