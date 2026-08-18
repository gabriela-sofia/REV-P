"""Testes do SUSC-17C21 - QA sensorial e comparacao pre/pos-evento Sentinel-2."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"
POST_DIR = OUT / "susc_17c21_post_event_light_artifacts"

MAX_PUBLIC = 1_000_000

EXPECTED = [
    OUT / "SUSC_17C21_QA_SENSORIAL_COMPARACAO_PRE_POS_SENTINEL2_REPORT.md",
    OUT / "susc_17c21_pre_event_artifact_qa.csv",
    OUT / "susc_17c21_post_event_scene_selection.csv",
    OUT / "susc_17c21_post_event_materialization_attempts.csv",
    OUT / "susc_17c21_post_event_light_artifact_manifest.csv",
    OUT / "susc_17c21_post_event_band_statistics.csv",
    OUT / "susc_17c21_post_event_index_statistics.csv",
    OUT / "susc_17c21_pre_post_delta_features.csv",
    OUT / "susc_17c21_observational_change_registry.csv",
    OUT / "susc_17c21_fallback_source_audit.csv",
    OUT / "susc_17c21_replay_audit.csv",
    OUT / "susc_17c21_no_leakage_audit.csv",
    OUT / "susc_17c21_gate_evaluation_matrix.csv",
    OUT / "susc_17c21_readiness_summary.json",
    OUT / "susc_17c21_promotion_blockers.csv",
    SCHEMAS / "susc_17c21_artifact_qa_schema_v1.json",
    SCHEMAS / "susc_17c21_temporal_delta_schema_v1.json",
    SCHEMAS / "susc_17c21_observational_change_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c21_sensor_qa_compare_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c21_sensor_qa_compare_common", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str, *args: str):
    env = os.environ.copy()
    for var in ["SUSC_17C21_ALLOW_NETWORK", "SUSC_17C21_ALLOW_LIGHTWEIGHT_RASTER", "SUSC_17C21_ALLOW_STAC_COG_FALLBACK"]:
        env.pop(var, None)
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=600, check=False,
    )


def _post_artifacts(s17c21) -> list[Path]:
    paths = []
    for patch_id in s17c21._patch_ids():
        if s17c21._load_post_stats(patch_id) is not None:
            for _, path in s17c21._post_committed_files(patch_id):
                if path.exists():
                    paths.append(path)
    return paths


def test_build_offline_gera_todos_artefatos_e_valida():
    result = run_script("build_susc_17c21_sensor_qa_temporal_compare.py")
    assert result.returncode == 0, result.stderr + result.stdout
    s17c21 = _load_common()
    tracked = EXPECTED + _post_artifacts(s17c21)
    first = {path: path.read_bytes() for path in tracked}
    result = run_script("build_susc_17c21_sensor_qa_temporal_compare.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in tracked} == first
    result = run_script("validate_susc_17c21_sensor_qa_temporal_compare.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_validos_e_ids_deterministicos():
    s17c21 = _load_common()
    qa_schema = json.loads((SCHEMAS / "susc_17c21_artifact_qa_schema_v1.json").read_text(encoding="utf-8"))
    delta_schema = json.loads((SCHEMAS / "susc_17c21_temporal_delta_schema_v1.json").read_text(encoding="utf-8"))
    obs_schema = json.loads((SCHEMAS / "susc_17c21_observational_change_schema_v1.json").read_text(encoding="utf-8"))
    for row in read_csv(OUT / "susc_17c21_pre_event_artifact_qa.csv"):
        assert s17c21._schema_violations(row, qa_schema) == []
    for row in read_csv(OUT / "susc_17c21_pre_post_delta_features.csv"):
        assert s17c21._schema_violations(row, delta_schema) == []
    for row in read_csv(OUT / "susc_17c21_observational_change_registry.csv"):
        assert s17c21._schema_violations(row, obs_schema) == []
    assert [row["observational_change_id"] for row in read_csv(OUT / "susc_17c21_observational_change_registry.csv")] == [
        f"S17C21_OBSCHANGE_{idx:04d}" for idx in range(1, 6)
    ]


def test_qa_pre_event_cobre_5_patches_e_hashes_17c20_conferem():
    qa = read_csv(OUT / "susc_17c21_pre_event_artifact_qa.csv")
    assert len({r["candidate_patch_id"] for r in qa}) == 5
    assert {r["passes_pre_event_qa"] for r in qa} == {"true"}
    manifest = read_csv(OUT / "susc_17c20_light_artifact_manifest.csv")
    for row in manifest:
        path = ROOT / row["artifact_local_path"]
        assert path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


def test_modos_de_rede_exigem_opt_in():
    result = run_script("susc_17c21_sensor_qa_temporal_compare.py", "materialize-post-event")
    assert result.returncode == 2
    assert "blocked_network_not_enabled" in result.stdout
    env = os.environ.copy()
    env["SUSC_17C21_ALLOW_NETWORK"] = "1"
    env.pop("SUSC_17C21_ALLOW_LIGHTWEIGHT_RASTER", None)
    env.pop("SUSC_17C21_ALLOW_STAC_COG_FALLBACK", None)
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "susc_17c21_sensor_qa_temporal_compare.py"), "materialize-post-event"],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=120, check=False,
    )
    assert result.returncode == 2
    assert "blocked_lightweight_not_enabled" in result.stdout


def test_selecao_pos_evento_cobre_5_patches_e_sao_pos_evento():
    selection = read_csv(OUT / "susc_17c21_post_event_scene_selection.csv")
    rank1 = [r for r in selection if r["selection_rank"] == "1"]
    assert len(rank1) == 5
    scenes = read_csv(OUT / "susc_17c18_sentinel2_candidate_scene_registry.csv")
    for r in rank1:
        matches = [s for s in scenes if s["scene_or_item_id"] == r["selected_scene_or_item_id"]]
        assert matches and all(s["is_post_event"] == "true" for s in matches)
        assert r["usable_as_pre_event_feature"] == "false"


def test_materializacao_pos_evento_cria_artefatos_reais_com_hash():
    manifests = read_csv(OUT / "susc_17c21_post_event_light_artifact_manifest.csv")
    assert len({r["candidate_patch_id"] for r in manifests}) == 5
    assert len(manifests) >= 5
    for row in manifests:
        path = ROOT / row["artifact_local_path"]
        assert path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert str(path.stat().st_size) == row["size_bytes"]
        assert path.stat().st_size <= MAX_PUBLIC
        assert row["format"] in ("csv", "png", "json")
        assert row["is_fallback_source"] == "true"


def test_deltas_existem_e_nao_viram_pre_evento_nem_label():
    deltas = read_csv(OUT / "susc_17c21_pre_post_delta_features.csv")
    assert len({r["candidate_patch_id"] for r in deltas}) == 5
    assert len(deltas) == 40
    assert {r["usable_as_pre_event_feature"] for r in deltas} == {"false"}
    assert {r["usable_for_training"] for r in deltas} == {"false"}
    assert {r["ground_truth"] for r in deltas} == {"false"}
    assert {r["usable_as_observational_change_candidate"] for r in deltas} == {"true"}
    band = read_csv(OUT / "susc_17c21_post_event_band_statistics.csv")
    index = read_csv(OUT / "susc_17c21_post_event_index_statistics.csv")
    assert {r["usable_as_pre_event_feature"] for r in band + index} == {"false"}
    leakage = read_csv(OUT / "susc_17c21_no_leakage_audit.csv")
    assert {r["uses_post_event_as_pre_event_feature"] for r in leakage} == {"false"}
    assert {r["uses_delta_as_training_label"] for r in leakage} == {"false"}


def test_fallback_nao_vira_ground_reference_e_sem_produto_completo():
    fallback = read_csv(OUT / "susc_17c21_fallback_source_audit.csv")
    assert {r["can_be_used_for_ground_reference"] for r in fallback} == {"false"}
    obs = read_csv(OUT / "susc_17c21_observational_change_registry.csv")
    assert {r["can_support_ground_reference"] for r in obs} == {"false"}
    summary = json.loads((OUT / "susc_17c21_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["sentinel2_full_products_downloaded_count"] == 0
    assert summary["heavy_rasters_downloaded_count"] == 0
    if POST_DIR.exists():
        for path in POST_DIR.glob("**/*"):
            if path.is_file():
                assert path.suffix.lower() not in (".tif", ".tiff", ".npz", ".npy", ".zip", ".safe", ".geotiff")


def test_score_guardrails_e_17b_inelegivel():
    summary = json.loads((OUT / "susc_17c21_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["minimum_success_achieved"] is True
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
    assert summary["ground_truth_created"] is False
    assert summary["training_labels_created"] is False
    assert summary["ground_reference_candidates_created_count"] == 0
    assert summary["post_event_used_as_pre_event_feature_count"] == 0
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
    blockers = read_csv(OUT / "susc_17c21_promotion_blockers.csv")
    assert any(row["blocker_type"] == "17b_blocked_until_event_reference_and_policy" for row in blockers)
    assert {row["blocks_17b"] for row in blockers} == {"true"}
