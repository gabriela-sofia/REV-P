"""Testes do SUSC-17C36 - pipeline hidrologico local para TWI/TPI/flow accumulation/HAND full."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"
LIGHT = OUT / "susc_17c36_light_artifacts"
FORBIDDEN = (".tif", ".tiff", ".safe", ".zip", ".nc", ".jp2", ".gz")

METHOD = OUT / "susc_17c36_original_topography_method_inventory.csv"
DEM_ATT = OUT / "susc_17c36_dem_acquisition_attempts.csv"
DEM_MAN = OUT / "susc_17c36_dem_artifact_manifest.csv"
TPI = OUT / "susc_17c36_tpi_250m_features.csv"
FLOW = OUT / "susc_17c36_flow_accumulation_features.csv"
TWI = OUT / "susc_17c36_twi_features.csv"
HAND = OUT / "susc_17c36_hand_full_features.csv"
MATRIX = OUT / "susc_17c36_topography_hydrology_feature_matrix.csv"
PARITY = OUT / "susc_17c36_topography_hydrology_parity_matrix.csv"
READY = OUT / "susc_17c36_score_v6_replay_readiness_update.csv"
INTEGRITY = OUT / "susc_17c36_score_integrity_audit.csv"
LEAKAGE = OUT / "susc_17c36_no_leakage_audit.csv"
SUMMARY = OUT / "susc_17c36_readiness_summary.json"

EXPECTED = [
    OUT / "SUSC_17C36_HYDROLOGIC_TOPOGRAPHY_PIPELINE_REPORT.md", METHOD, DEM_ATT, DEM_MAN,
    OUT / "susc_17c36_dem_grid_metadata.csv", OUT / "susc_17c36_hydrologic_grid_metadata.csv",
    TPI, FLOW, TWI, HAND, MATRIX, PARITY, READY, INTEGRITY, LEAKAGE, SUMMARY,
    OUT / "susc_17c36_promotion_blockers.csv",
]


def rc(path):
    with path.open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def run_script(name, *args):
    env = os.environ.copy()
    for v in ["SUSC_17C36_ALLOW_NETWORK", "SUSC_17C36_ALLOW_PUBLIC_DOWNLOAD",
              "SUSC_17C36_ALLOW_LIGHTWEIGHT_RASTER", "SUSC_17C36_ALLOW_DEM",
              "SUSC_17C36_ALLOW_HYDROLOGIC_PROCESSING", "SUSC_17C36_ALLOW_HAND_PIPELINE"]:
        env.pop(v, None)
    return subprocess.run([sys.executable, str(SCRIPTS / name), *args],
                          cwd=ROOT, env=env, text=True, capture_output=True, timeout=600, check=False)


def test_build_offline_byte_identico_e_valida():
    r = run_script("build_susc_17c36_hydrologic_topography_pipeline.py")
    assert r.returncode == 0, r.stderr + r.stdout
    first = {p: p.read_bytes() for p in EXPECTED}
    r = run_script("build_susc_17c36_hydrologic_topography_pipeline.py")
    assert r.returncode == 0, r.stderr + r.stdout
    assert {p: p.read_bytes() for p in EXPECTED} == first
    r = run_script("validate_susc_17c36_hydrologic_topography_pipeline.py")
    assert r.returncode == 0, r.stderr + r.stdout


def test_th_ids_namespace_17c36():
    for r in rc(MATRIX):
        assert r["topography_hydrology_feature_id"].startswith("S17C36_TH_")


def test_canarios_e_metodo_inventariado():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["canary_patch_rows_consumed_count"] == 11
    assert len(rc(METHOD)) >= 6
    for r in rc(METHOD):
        assert r["used_in_score_v6"] == "true"


def test_dem_attempts_e_manifest():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["dem_acquisition_attempts_count"] >= 3
    assert s["dem_local_artifacts_count"] >= 1
    for r in rc(DEM_MAN):
        if r["source_name"] != "none":
            assert r["stored_in_outputs_public"] == "false"
            assert r["raw_heavy"] == "true"


def test_features_topograficas_11_linhas():
    for path in (TPI, FLOW, TWI, HAND, MATRIX):
        assert len(rc(path)) == 11
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["dem_grid_created"] is True
    assert s["hydrologic_grid_rows_count"] >= 1


def test_proxy_nao_vira_full_sem_flag():
    for r in rc(HAND):
        if r["HAND_full"] == "true":
            assert r["HAND_proxy"] == "false"
    for r in rc(LEAKAGE):
        assert r["uses_proxy_as_full_without_flag"] == "false"
        assert r["uses_osm_as_official_hydrography"] == "false"


def test_score_final_bloqueado_por_equivalencia_parcial():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["method_equivalence_status"] == "method_reconstructed_not_proven_equivalent"
    assert s["score_v6_full_replay_computable"] is False
    assert s["score_v6_review_only_final_score_computed_count"] == 0
    for r in rc(PARITY):
        assert r["sufficient_for_score_v6_full_replay"] == "false"
    for r in rc(INTEGRITY):
        assert r["final_score_computed"] == "false"


def test_score_v6_intacto_v7_inexistente_sem_gt_treino():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["score_v6_original_file_changed"] is False
    assert s["score_v7_created"] is False
    assert s["ground_truth_created"] is False
    assert s["training_labels_created"] is False
    assert s["eligible_for_17b_now"] is False
    r = subprocess.run(["git", "diff", "--name-only", "--", "datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv"],
                       cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.stdout.strip() == ""
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    for r in rc(MATRIX):
        assert r["ground_truth"] == "false"
        assert r["training_label"] == "false"


def test_ocorrencia_nao_feature_controle_nao_negativo():
    for r in rc(LEAKAGE):
        assert r["uses_occurrence_as_feature"] == "false"
        assert r["uses_control_as_negative_truth"] == "false"
        assert r["passes_no_leakage"] == "true"


def test_raster_pesado_nao_commitado():
    if LIGHT.exists():
        for p in LIGHT.glob("**/*"):
            if p.is_file():
                assert p.suffix.lower() not in FORBIDDEN, p.name
                assert p.stat().st_size <= 5_000_000, p.name


def test_extracao_bloqueia_sem_env():
    r = run_script("susc_17c36_hydrologic_topography_pipeline.py", "run-dem-acquisition")
    assert "BLOCKED" in (r.stdout + r.stderr)
