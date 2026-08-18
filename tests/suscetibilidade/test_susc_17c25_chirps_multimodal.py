"""Testes do SUSC-17C25 - CHIRPS real por AOI e matriz multimodal cientifica review-only."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"
CHIRPS_DIR = OUT / "susc_17c25_chirps_artifacts"
MAX_PUBLIC = 1_000_000

EXPECTED = [
    OUT / "SUSC_17C25_CHIRPS_REAL_AOI_MATRIZ_MULTIMODAL_REVIEW_ONLY_REPORT.md",
    OUT / "susc_17c25_chirps_source_strategy.csv",
    OUT / "susc_17c25_chirps_source_resolution.csv",
    OUT / "susc_17c25_chirps_fetch_attempts.csv",
    OUT / "susc_17c25_chirps_artifact_manifest.csv",
    OUT / "susc_17c25_chirps_daily_values.csv",
    OUT / "susc_17c25_chirps_window_aggregates.csv",
    OUT / "susc_17c25_rainfall_trigger_feature_registry.csv",
    OUT / "susc_17c25_multimodal_feature_matrix.csv",
    OUT / "susc_17c25_feature_provenance_trace.csv",
    OUT / "susc_17c25_replay_audit.csv",
    OUT / "susc_17c25_no_leakage_audit.csv",
    OUT / "susc_17c25_gate_evaluation_matrix.csv",
    OUT / "susc_17c25_readiness_summary.json",
    OUT / "susc_17c25_promotion_blockers.csv",
    SCHEMAS / "susc_17c25_chirps_feature_schema_v1.json",
    SCHEMAS / "susc_17c25_multimodal_matrix_schema_v1.json",
    SCHEMAS / "susc_17c25_chirps_source_resolution_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c25_chirps_multimodal_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c25_chirps_multimodal_common", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str, *args: str):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=ROOT, text=True, capture_output=True, timeout=600, check=False,
    )


def _committed(s17c25) -> list[Path]:
    paths = []
    for patch_id in s17c25._patch_ids():
        if s17c25._load_provenance(patch_id) is not None:
            for _, path in s17c25._committed_files(patch_id):
                if path.exists():
                    paths.append(path)
    return paths


def test_build_offline_gera_todos_artefatos_e_valida():
    result = run_script("build_susc_17c25_chirps_multimodal.py")
    assert result.returncode == 0, result.stderr + result.stdout
    s17c25 = _load_common()
    tracked = EXPECTED + _committed(s17c25)
    first = {path: path.read_bytes() for path in tracked}
    result = run_script("build_susc_17c25_chirps_multimodal.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in tracked} == first
    result = run_script("validate_susc_17c25_chirps_multimodal.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_validos_e_ids_deterministicos():
    s17c25 = _load_common()
    feat_schema = json.loads((SCHEMAS / "susc_17c25_chirps_feature_schema_v1.json").read_text(encoding="utf-8"))
    matrix_schema = json.loads((SCHEMAS / "susc_17c25_multimodal_matrix_schema_v1.json").read_text(encoding="utf-8"))
    source_schema = json.loads((SCHEMAS / "susc_17c25_chirps_source_resolution_schema_v1.json").read_text(encoding="utf-8"))
    for row in read_csv(OUT / "susc_17c25_rainfall_trigger_feature_registry.csv"):
        assert s17c25._schema_violations(row, feat_schema) == []
    for row in read_csv(OUT / "susc_17c25_multimodal_feature_matrix.csv"):
        assert s17c25._schema_violations(row, matrix_schema) == []
    for row in read_csv(OUT / "susc_17c25_chirps_source_resolution.csv"):
        assert s17c25._schema_violations(row, source_schema) == []


def test_5_patches_15_janelas_150_diarios_15_features():
    summary = json.loads((OUT / "susc_17c25_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["patches_processed_count"] == 5
    assert summary["chirps_windows_processed_count"] == 15
    assert summary["chirps_daily_values_count"] >= 150
    assert summary["real_chirps_features_count"] >= 15
    assert summary["patches_with_real_chirps_count"] == 5
    daily = read_csv(OUT / "susc_17c25_chirps_daily_values.csv")
    assert len([r for r in daily if r["value_real"] == "true"]) >= 150


def test_cada_patch_tem_chirps_3d_7d_30d_e_manifesto_hash():
    agg = read_csv(OUT / "susc_17c25_chirps_window_aggregates.csv")
    s17c25 = _load_common()
    for patch_id in s17c25._patch_ids():
        windows = {a["feature_name"] for a in agg if a["candidate_patch_id"] == patch_id}
        assert windows == {"CHIRPS_3d", "CHIRPS_7d", "CHIRPS_30d"}, patch_id
    manifests = read_csv(OUT / "susc_17c25_chirps_artifact_manifest.csv")
    assert len(manifests) >= 5
    for row in manifests:
        path = ROOT / row["artifact_local_path"]
        assert path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert path.stat().st_size <= MAX_PUBLIC
        assert row["format"] in ("csv", "json")


def test_modo_fetch_exige_opt_in():
    result = run_script("susc_17c25_chirps_multimodal_runner.py", "fetch-chirps-real")
    assert result.returncode == 2
    assert "blocked_network_not_enabled" in result.stdout


def test_matriz_multimodal_5_linhas_sentinel2_e_chirps_sem_delta():
    matrix = read_csv(OUT / "susc_17c25_multimodal_feature_matrix.csv")
    assert len(matrix) == 5
    cols = matrix[0].keys()
    assert any(c.startswith("B03_mean") for c in cols)
    assert "CHIRPS_3d_sum_mm" in cols and "CHIRPS_30d_sum_mm" in cols
    assert all("delta" not in c for c in cols)
    assert {r["usable_for_training"] for r in matrix} == {"false"}
    assert {r["eligible_for_score_v7"] for r in matrix} == {"false"}


def test_chirps_pre_evento_e_nao_vira_evento():
    daily = read_csv(OUT / "susc_17c25_chirps_daily_values.csv")
    assert {r["pre_event_only"] for r in daily} == {"true"}
    assert all(r["date"] < "2022-05-24" for r in daily)
    leakage = read_csv(OUT / "susc_17c25_no_leakage_audit.csv")
    assert {r["uses_chirps_as_event_reference"] for r in leakage} == {"false"}
    assert {r["passes_no_leakage"] for r in leakage} == {"true"}
    gates = read_csv(OUT / "susc_17c25_gate_evaluation_matrix.csv")
    assert {r["G4_vinculo_espacial_evento"] for r in gates} == {"false"}
    assert {r["all_gates_passed_for_ground_reference"] for r in gates} == {"false"}


def test_score_guardrails_17b_inelegivel_e_sem_raster_pesado():
    summary = json.loads((OUT / "susc_17c25_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["minimum_success_achieved"] is True
    assert summary["features_promoted_to_training_count"] == 0
    assert summary["chirps_used_as_event_reference_count"] == 0
    assert summary["ground_reference_candidates_created_count"] == 0
    assert summary["ground_truth_created"] is False
    assert summary["training_labels_created"] is False
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["eligible_for_17b_now"] is False
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    if CHIRPS_DIR.exists():
        for path in CHIRPS_DIR.glob("**/*"):
            if path.is_file():
                assert path.suffix.lower() not in (".tif", ".tiff", ".nc", ".zip", ".gz", ".npz", ".npy")
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
    blockers = read_csv(OUT / "susc_17c25_promotion_blockers.csv")
    assert any(row["blocker_type"] == "17b_blocked_until_event_reference_and_policy" for row in blockers)
