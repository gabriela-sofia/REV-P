"""Testes do SUSC-17C24 - serie temporal pre-evento multi-cena e reducao de nuvem Sentinel-2."""

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
LIGHT_DIR = OUT / "susc_17c24_light_artifacts"
MAX_PUBLIC = 1_000_000

EXPECTED = [
    OUT / "SUSC_17C24_SERIE_TEMPORAL_MULTICENA_REDUCAO_NUVEM_SENTINEL2_REPORT.md",
    OUT / "susc_17c24_multiscene_selection_policy.csv",
    OUT / "susc_17c24_pre_event_scene_selection.csv",
    OUT / "susc_17c24_multiscene_materialization_attempts.csv",
    OUT / "susc_17c24_light_artifact_manifest.csv",
    OUT / "susc_17c24_scene_band_statistics.csv",
    OUT / "susc_17c24_scene_index_statistics.csv",
    OUT / "susc_17c24_multitemporal_feature_registry.csv",
    OUT / "susc_17c24_cloud_robust_feature_matrix.csv",
    OUT / "susc_17c24_cloud_reduction_audit.csv",
    OUT / "susc_17c24_fallback_equivalence_reuse_audit.csv",
    OUT / "susc_17c24_replay_audit.csv",
    OUT / "susc_17c24_no_leakage_audit.csv",
    OUT / "susc_17c24_gate_evaluation_matrix.csv",
    OUT / "susc_17c24_readiness_summary.json",
    OUT / "susc_17c24_promotion_blockers.csv",
    SCHEMAS / "susc_17c24_scene_selection_schema_v1.json",
    SCHEMAS / "susc_17c24_multitemporal_feature_schema_v1.json",
    SCHEMAS / "susc_17c24_cloud_robust_matrix_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c24_multiscene_sentinel_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c24_multiscene_sentinel_common", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str, *args: str):
    env = os.environ.copy()
    for var in ["SUSC_17C24_ALLOW_NETWORK", "SUSC_17C24_ALLOW_LIGHTWEIGHT_RASTER", "SUSC_17C24_ALLOW_STAC_COG_FALLBACK"]:
        env.pop(var, None)
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=600, check=False,
    )


def _committed(s17c24) -> list[Path]:
    paths = []
    for patch_id in s17c24._patch_ids():
        for date in s17c24._materialized_scene_dates(patch_id):
            for _, path in s17c24._scene_committed_files(patch_id, date):
                if path.exists():
                    paths.append(path)
    return paths


def test_build_offline_gera_todos_artefatos_e_valida():
    result = run_script("build_susc_17c24_multiscene_sentinel.py")
    assert result.returncode == 0, result.stderr + result.stdout
    s17c24 = _load_common()
    tracked = EXPECTED + _committed(s17c24)
    first = {path: path.read_bytes() for path in tracked}
    result = run_script("build_susc_17c24_multiscene_sentinel.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in tracked} == first
    result = run_script("validate_susc_17c24_multiscene_sentinel.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_validos_e_ids_deterministicos():
    s17c24 = _load_common()
    sel_schema = json.loads((SCHEMAS / "susc_17c24_scene_selection_schema_v1.json").read_text(encoding="utf-8"))
    mt_schema = json.loads((SCHEMAS / "susc_17c24_multitemporal_feature_schema_v1.json").read_text(encoding="utf-8"))
    matrix_schema = json.loads((SCHEMAS / "susc_17c24_cloud_robust_matrix_schema_v1.json").read_text(encoding="utf-8"))
    for row in read_csv(OUT / "susc_17c24_pre_event_scene_selection.csv"):
        assert s17c24._schema_violations(row, sel_schema) == []
    for row in read_csv(OUT / "susc_17c24_multitemporal_feature_registry.csv"):
        assert s17c24._schema_violations(row, mt_schema) == []
    for row in read_csv(OUT / "susc_17c24_cloud_robust_feature_matrix.csv"):
        assert s17c24._schema_violations(row, matrix_schema) == []


def test_5_patches_com_pelo_menos_3_cenas():
    selection = read_csv(OUT / "susc_17c24_pre_event_scene_selection.csv")
    by_patch = {}
    for r in selection:
        by_patch.setdefault(r["candidate_patch_id"], set()).add(r["datetime"][:10])
    assert len(by_patch) == 5
    for patch, dates in by_patch.items():
        assert len(dates) >= 3, patch
    assert len(selection) >= 15


def test_modo_rede_exige_opt_in():
    result = run_script("susc_17c24_multiscene_sentinel_runner.py", "materialize-multiscene")
    assert result.returncode == 2
    assert "blocked_network_not_enabled" in result.stdout


def test_pelo_menos_15_artefatos_com_hash_e_sob_limite():
    manifests = read_csv(OUT / "susc_17c24_light_artifact_manifest.csv")
    assert len(manifests) >= 15
    for row in manifests:
        path = ROOT / row["artifact_local_path"]
        assert path.exists(), row["artifact_local_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert str(path.stat().st_size) == row["size_bytes"]
        assert path.stat().st_size <= MAX_PUBLIC
        assert row["format"] in ("csv", "png", "json")


def test_matriz_robusta_5_linhas_sem_deltas():
    matrix = read_csv(OUT / "susc_17c24_cloud_robust_feature_matrix.csv")
    assert len(matrix) == 5
    assert all("delta" not in col for col in matrix[0].keys())
    for col in ["scene_count", "cloud_cover_min", "cloud_cover_median", "temporal_span_days",
                "B03_mean_temporal_median", "NDVI_mean_temporal_median", "NDVI_mean_temporal_std"]:
        assert col in matrix[0]
    assert {r["usable_for_training"] for r in matrix} == {"false"}
    assert {r["eligible_for_score_v7"] for r in matrix} == {"false"}


def test_nenhuma_cena_pos_ou_durante_usada_e_sem_produto_completo():
    selection = read_csv(OUT / "susc_17c24_pre_event_scene_selection.csv")
    assert {r["is_pre_event"] for r in selection} == {"true"}
    assert {r["is_during_event"] for r in selection} == {"false"}
    assert {r["is_post_event"] for r in selection} == {"false"}
    scenes = read_csv(OUT / "susc_17c18_sentinel2_candidate_scene_registry.csv")
    for scene_id in {r["scene_or_item_id"] for r in selection}:
        matches = [s for s in scenes if s["scene_or_item_id"] == scene_id]
        assert matches and all(s["is_pre_event"] == "true" and s["is_post_event"] == "false" for s in matches)
    summary = json.loads((OUT / "susc_17c24_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["sentinel2_full_products_downloaded_count"] == 0
    assert summary["heavy_rasters_committed_count"] == 0
    assert summary["post_event_used_as_pre_event_feature_count"] == 0
    if LIGHT_DIR.exists():
        for path in LIGHT_DIR.glob("**/*"):
            if path.is_file():
                assert path.suffix.lower() not in (".tif", ".tiff", ".npz", ".npy", ".zip", ".safe", ".geotiff")


def test_fallback_review_only_e_multitemporal_features():
    fb = read_csv(OUT / "susc_17c24_fallback_equivalence_reuse_audit.csv")
    assert {r["reused_as_scientific_review_only"] for r in fb} == {"true"}
    assert {r["accepted_for_ground_reference"] for r in fb} == {"false"}
    assert {r["accepted_for_training"] for r in fb} == {"false"}
    mt = read_csv(OUT / "susc_17c24_multitemporal_feature_registry.csv")
    assert len(mt) >= 40
    assert {r["usable_for_training"] for r in mt} == {"false"}
    assert {r["pre_event_only"] for r in mt} == {"true"}


def test_score_guardrails_e_17b_inelegivel():
    summary = json.loads((OUT / "susc_17c24_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["minimum_success_achieved"] is True
    assert summary["features_promoted_to_training_count"] == 0
    assert summary["ground_reference_candidates_created_count"] == 0
    assert summary["ground_truth_created"] is False
    assert summary["training_labels_created"] is False
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
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
    gates = read_csv(OUT / "susc_17c24_gate_evaluation_matrix.csv")
    assert {r["G4_vinculo_espacial_evento"] for r in gates} == {"false"}
    assert {r["all_gates_passed_for_ground_reference"] for r in gates} == {"false"}
    blockers = read_csv(OUT / "susc_17c24_promotion_blockers.csv")
    assert any(row["blocker_type"] == "17b_blocked_until_event_reference_and_policy" for row in blockers)
