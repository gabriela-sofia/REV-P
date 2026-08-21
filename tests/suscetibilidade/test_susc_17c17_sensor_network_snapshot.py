"""Testes do SUSC-17C17 - snapshot de rede para metadados sensoriais e politica de raster leve."""

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
    OUT / "SUSC_17C17_SNAPSHOT_REDE_METADADOS_SENSORIAIS_REPORT.md",
    OUT / "susc_17c17_sensor_source_plan.csv",
    OUT / "susc_17c17_network_snapshot_registry.csv",
    OUT / "susc_17c17_chirps_snapshot_metadata.csv",
    OUT / "susc_17c17_sentinel2_cdse_snapshot_metadata.csv",
    OUT / "susc_17c17_snapshot_manifest.csv",
    OUT / "susc_17c17_snapshot_replay_audit.csv",
    OUT / "susc_17c17_candidate_patch_aoi_plan.csv",
    OUT / "susc_17c17_light_raster_policy.csv",
    OUT / "susc_17c17_light_raster_execution_plan.csv",
    OUT / "susc_17c17_blocked_heavy_raster_registry.csv",
    OUT / "susc_17c17_gate_evaluation_matrix.csv",
    OUT / "susc_17c17_anti_leakage_audit.csv",
    OUT / "susc_17c17_readiness_summary.json",
    OUT / "susc_17c17_promotion_blockers.csv",
    SCHEMAS / "susc_17c17_network_snapshot_schema_v1.json",
    SCHEMAS / "susc_17c17_light_raster_policy_schema_v1.json",
    SCHEMAS / "susc_17c17_snapshot_manifest_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c17_sensor_snapshot_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c17_sensor_snapshot_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str, *args: str):
    env = os.environ.copy()
    env.pop("SUSC_17C17_ALLOW_NETWORK", None)
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )


def _snapshot_outputs(s17c17) -> list[Path]:
    return [s17c17._snapshot_path(source) for source in s17c17.SOURCES if s17c17._load_snapshot(source) is not None]


def test_build_offline_gera_todos_artefatos_e_valida():
    result = run_script("build_susc_17c17_sensor_network_snapshot.py")
    assert result.returncode == 0, result.stderr + result.stdout
    s17c17 = _load_common()
    tracked = EXPECTED + _snapshot_outputs(s17c17)
    first = {path: path.read_bytes() for path in tracked}
    result = run_script("build_susc_17c17_sensor_network_snapshot.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in tracked} == first
    result = run_script("validate_susc_17c17_sensor_network_snapshot.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_validos_e_ids_deterministicos():
    s17c17 = _load_common()
    net_schema = json.loads((SCHEMAS / "susc_17c17_network_snapshot_schema_v1.json").read_text(encoding="utf-8"))
    policy_schema = json.loads((SCHEMAS / "susc_17c17_light_raster_policy_schema_v1.json").read_text(encoding="utf-8"))
    manifest_schema = json.loads((SCHEMAS / "susc_17c17_snapshot_manifest_schema_v1.json").read_text(encoding="utf-8"))
    for row in read_csv(OUT / "susc_17c17_network_snapshot_registry.csv"):
        assert s17c17._schema_violations(row, net_schema) == []
    for row in read_csv(OUT / "susc_17c17_light_raster_policy.csv"):
        assert s17c17._schema_violations(row, policy_schema) == []
    for row in read_csv(OUT / "susc_17c17_snapshot_manifest.csv"):
        assert s17c17._schema_violations(row, manifest_schema) == []
    assert set(manifest_schema["required"]) == set(s17c17.MANIFEST_FIELDS)
    assert [row["network_snapshot_id"] for row in read_csv(OUT / "susc_17c17_network_snapshot_registry.csv")] == [
        f"S17C17_SNAP_{idx:04d}" for idx in range(1, len(s17c17.SOURCES) + 1)
    ]


def test_plan_nao_acessa_rede():
    result = run_script("susc_17c17_sensor_snapshot_collector.py", "plan")
    assert result.returncode == 0
    assert "sensor_source_plan" in result.stdout
    assert "candidate_patch_aoi_plan" in result.stdout


def test_capture_snapshot_exige_opt_in():
    for mode in ["capture-snapshot"]:
        result = run_script("susc_17c17_sensor_snapshot_collector.py", mode)
        assert result.returncode == 2
        assert "blocked_network_not_enabled" in result.stdout


def test_replay_funciona_sem_rede():
    result = run_script("susc_17c17_sensor_snapshot_collector.py", "replay-snapshot")
    assert result.returncode == 0
    assert "replay-snapshot" in result.stdout
    registry = read_csv(OUT / "susc_17c17_network_snapshot_registry.csv")
    assert {row["snapshot_created"] for row in registry} <= {"true", "false"}


def test_metadado_sensorial_nao_vira_evento_observado():
    chirps = read_csv(OUT / "susc_17c17_chirps_snapshot_metadata.csv")
    sentinel = read_csv(OUT / "susc_17c17_sentinel2_cdse_snapshot_metadata.csv")
    gates = read_csv(OUT / "susc_17c17_gate_evaluation_matrix.csv")
    assert {row["raster_downloaded"] for row in chirps} == {"false"}
    assert {row["feature_extracted"] for row in chirps} == {"false"}
    assert {row["raster_downloaded"] for row in sentinel} == {"false"}
    assert {row["tile_created"] for row in sentinel} == {"false"}
    assert {row["embedding_created"] for row in sentinel} == {"false"}
    assert {row["G4_vinculo_espacial"] for row in gates} == {"false"}
    assert {row["G5_separacao_fenomeno"] for row in gates} == {"false"}
    assert {row["all_gates_passed"] for row in gates} == {"false"}


def test_raster_pesado_tile_e_embedding_bloqueados():
    heavy = read_csv(OUT / "susc_17c17_blocked_heavy_raster_registry.csv")
    execution_plan = read_csv(OUT / "susc_17c17_light_raster_execution_plan.csv")
    assert len(heavy) > 0
    assert {row["can_execute_now"] for row in execution_plan} == {"false"}
    summary = json.loads((OUT / "susc_17c17_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["heavy_raster_downloads_count"] == 0
    assert summary["raster_tiles_created_count"] == 0
    assert summary["dino_satmae_embeddings_created_count"] == 0
    assert summary["ground_reference_candidates_created_count"] == 0


def test_score_v6_intacto_e_score_v7_inexistente():
    summary = json.loads((OUT / "susc_17c17_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
    assert summary["review_only"] is True
    assert summary["trainable"] is False
    assert summary["ground_truth"] is False
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


def test_17b_inelegivel_e_sem_features_extraidas():
    summary = json.loads((OUT / "susc_17c17_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["features_extracted_this_sprint"] == 0
    assert summary["official_patch_created"] is False
    assert summary["official_patch_link_created"] is False
    assert summary["raw_raster_committed"] is False
    blockers = read_csv(OUT / "susc_17c17_promotion_blockers.csv")
    assert any(row["blocker_type"] == "17b_blocked_until_real_event_artifacts_and_policy" for row in blockers)
    assert {row["blocks_17b"] for row in blockers} == {"true"}
