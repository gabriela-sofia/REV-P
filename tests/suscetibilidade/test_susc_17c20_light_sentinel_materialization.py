"""Testes do SUSC-17C20 - materializacao real de recorte leve Sentinel-2 por AOI."""

from __future__ import annotations

import csv
import hashlib
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
LIGHT_DIR = OUT / "susc_17c20_light_artifacts"

MAX_PUBLIC = 1_000_000

EXPECTED = [
    OUT / "SUSC_17C20_MATERIALIZACAO_REAL_RECORTE_LEVE_SENTINEL2_REPORT.md",
    OUT / "susc_17c20_materialization_execution_policy.csv",
    OUT / "susc_17c20_sentinel2_source_resolution.csv",
    OUT / "susc_17c20_asset_resolution_attempts.csv",
    OUT / "susc_17c20_light_materialization_attempts.csv",
    OUT / "susc_17c20_light_artifact_manifest.csv",
    OUT / "susc_17c20_band_statistics.csv",
    OUT / "susc_17c20_index_statistics.csv",
    OUT / "susc_17c20_materialized_feature_registry.csv",
    OUT / "susc_17c20_fallback_source_audit.csv",
    OUT / "susc_17c20_replay_audit.csv",
    OUT / "susc_17c20_no_leakage_audit.csv",
    OUT / "susc_17c20_gate_evaluation_matrix.csv",
    OUT / "susc_17c20_readiness_summary.json",
    OUT / "susc_17c20_promotion_blockers.csv",
    SCHEMAS / "susc_17c20_light_artifact_manifest_schema_v1.json",
    SCHEMAS / "susc_17c20_materialized_feature_schema_v1.json",
    SCHEMAS / "susc_17c20_source_resolution_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c20_light_sentinel_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c20_light_sentinel_common", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str, *args: str):
    env = os.environ.copy()
    for var in ["SUSC_17C20_ALLOW_NETWORK", "SUSC_17C20_ALLOW_LIGHTWEIGHT_RASTER", "SUSC_17C20_ALLOW_STAC_COG_FALLBACK"]:
        env.pop(var, None)
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )


def _committed_artifacts(s17c20) -> list[Path]:
    paths = []
    for canonical in s17c20._canonical_rows():
        if s17c20._load_stats(canonical["candidate_patch_id"]) is not None:
            for _, path in s17c20._committed_artifact_files(canonical["candidate_patch_id"]):
                if path.exists():
                    paths.append(path)
    return paths


def test_build_offline_gera_todos_artefatos_e_valida():
    result = run_script("build_susc_17c20_light_sentinel_materialization.py")
    assert result.returncode == 0, result.stderr + result.stdout
    s17c20 = _load_common()
    tracked = EXPECTED + _committed_artifacts(s17c20)
    first = {path: path.read_bytes() for path in tracked}
    result = run_script("build_susc_17c20_light_sentinel_materialization.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in tracked} == first
    result = run_script("validate_susc_17c20_light_sentinel_materialization.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_validos_e_ids_deterministicos():
    s17c20 = _load_common()
    manifest_schema = json.loads((SCHEMAS / "susc_17c20_light_artifact_manifest_schema_v1.json").read_text(encoding="utf-8"))
    feature_schema = json.loads((SCHEMAS / "susc_17c20_materialized_feature_schema_v1.json").read_text(encoding="utf-8"))
    source_schema = json.loads((SCHEMAS / "susc_17c20_source_resolution_schema_v1.json").read_text(encoding="utf-8"))
    for row in read_csv(OUT / "susc_17c20_light_artifact_manifest.csv"):
        assert s17c20._schema_violations(row, manifest_schema) == []
    for row in read_csv(OUT / "susc_17c20_materialized_feature_registry.csv"):
        assert s17c20._schema_violations(row, feature_schema) == []
    for row in read_csv(OUT / "susc_17c20_sentinel2_source_resolution.csv"):
        assert s17c20._schema_violations(row, source_schema) == []
    assert [row["policy_id"] for row in read_csv(OUT / "susc_17c20_materialization_execution_policy.csv")] == [
        f"S17C20_POLICY_{idx:04d}" for idx in range(1, 6)
    ]


def test_modos_de_rede_exigem_opt_in():
    for mode in ["resolve-assets", "materialize-light"]:
        result = run_script("susc_17c20_light_sentinel_materializer.py", mode)
        assert result.returncode == 2
        assert "blocked_network_not_enabled" in result.stdout


def test_materialize_exige_lightweight_e_fallback_opt_in():
    env = os.environ.copy()
    env["SUSC_17C20_ALLOW_NETWORK"] = "1"
    env.pop("SUSC_17C20_ALLOW_LIGHTWEIGHT_RASTER", None)
    env.pop("SUSC_17C20_ALLOW_STAC_COG_FALLBACK", None)
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "susc_17c20_light_sentinel_materializer.py"), "materialize-light"],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=120, check=False,
    )
    assert result.returncode == 2
    assert "blocked_lightweight_not_enabled" in result.stdout


def test_cdse_product_level_only_nao_encerra_sem_fallback():
    source = read_csv(OUT / "susc_17c20_sentinel2_source_resolution.csv")
    assert {row["primary_source_status"] for row in source} == {"cdse_product_level_only"}
    assert {row["fallback_used"] for row in source} == {"true"}
    attempts = read_csv(OUT / "susc_17c20_asset_resolution_attempts.csv")
    cdse = [r for r in attempts if r["source_name"] == "CDSE/OData"]
    earth = [r for r in attempts if r["source_name"].startswith("earth_search")]
    assert all(r["asset_level_access_available"] == "false" for r in cdse)
    assert all(r["asset_level_access_available"] == "true" for r in earth)


def test_fallback_marcado_e_nao_e_ground_reference():
    manifests = read_csv(OUT / "susc_17c20_light_artifact_manifest.csv")
    assert len(manifests) >= 1
    assert {row["is_fallback_source"] for row in manifests} == {"true"}
    assert {row["source_role"] for row in manifests} == {"materialization_fallback"}
    fallback = read_csv(OUT / "susc_17c20_fallback_source_audit.csv")
    assert {row["can_be_used_for_ground_reference"] for row in fallback} == {"false"}
    gates = read_csv(OUT / "susc_17c20_gate_evaluation_matrix.csv")
    assert {row["all_gates_passed_for_ground_reference"] for row in gates} == {"false"}


def test_pelo_menos_um_artefato_real_com_hash_e_sob_limite():
    manifests = read_csv(OUT / "susc_17c20_light_artifact_manifest.csv")
    assert len(manifests) >= 1
    for row in manifests:
        path = ROOT / row["artifact_local_path"]
        assert path.exists(), row["artifact_local_path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == row["sha256"]
        assert str(path.stat().st_size) == row["size_bytes"]
        assert path.stat().st_size <= MAX_PUBLIC
        assert row["format"] in ("csv", "png", "json")
    summary = json.loads((OUT / "susc_17c20_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["minimum_success_achieved"] is True
    assert summary["light_artifacts_created_count"] >= 1


def test_nenhum_produto_completo_safe_zip_ou_raster_pesado():
    summary = json.loads((OUT / "susc_17c20_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["sentinel2_full_products_downloaded_count"] == 0
    assert summary["heavy_rasters_downloaded_count"] == 0
    assert summary["tiles_created_count"] == 0
    assert summary["embeddings_created_count"] == 0
    if LIGHT_DIR.exists():
        for path in LIGHT_DIR.glob("**/*"):
            if path.is_file():
                assert path.suffix.lower() not in (".tif", ".tiff", ".npz", ".npy", ".zip", ".safe", ".geotiff")


def test_cenas_sao_pre_evento_e_features_reais_exigem_hash():
    manifests = read_csv(OUT / "susc_17c20_light_artifact_manifest.csv")
    scenes = read_csv(OUT / "susc_17c18_sentinel2_candidate_scene_registry.csv")
    used_scene_ids = {row["scene_or_item_id"] for row in manifests}
    for scene_id in used_scene_ids:
        matches = [s for s in scenes if s["scene_or_item_id"] == scene_id]
        assert matches
        assert all(s["is_pre_event"] == "true" and s["is_post_event"] == "false" for s in matches)
    manifest_ids = {row["light_artifact_manifest_id"] for row in manifests}
    features = read_csv(OUT / "susc_17c20_materialized_feature_registry.csv")
    band = read_csv(OUT / "susc_17c20_band_statistics.csv")
    for row in features + band:
        if row["feature_value_real"] == "true":
            assert row["source_artifact_manifest_id"] in manifest_ids
    leakage = read_csv(OUT / "susc_17c20_no_leakage_audit.csv")
    assert {row["uses_synthetic_as_real"] for row in leakage} == {"false"}
    assert {row["uses_post_event_as_pre_event"] for row in leakage} == {"false"}


def test_score_guardrails_e_17b_inelegivel():
    summary = json.loads((OUT / "susc_17c20_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
    assert summary["review_only"] is True
    assert summary["trainable"] is False
    assert summary["ground_truth"] is False
    assert summary["ground_reference_candidates_created_count"] == 0
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    for target in [
        DAT / "susc_score_v6_candidate_by_patch_v1.csv",
        DAT / "susc_patches_official_v1.csv",
        DAT / "susc_patch_links_official_v1.csv",
    ]:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--", str(target.relative_to(ROOT))],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        assert result.stdout.strip() == ""
    features = read_csv(OUT / "susc_17c20_materialized_feature_registry.csv")
    assert {row["usable_for_training"] for row in features} == {"false"}
    assert {row["eligible_for_score_v7"] for row in features} == {"false"}
    blockers = read_csv(OUT / "susc_17c20_promotion_blockers.csv")
    assert any(row["blocker_type"] == "17b_blocked_until_event_reference_and_policy" for row in blockers)
    assert {row["blocks_17b"] for row in blockers} == {"true"}
