"""Testes do SUSC-17C34 - extracao pre-evento de features locais para patches canarios.

Offline e deterministico: usa os ledgers de extracao ja commitados.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"
LIGHT = OUT / "susc_17c34_light_artifacts"
FORBIDDEN = (".tif", ".tiff", ".safe", ".zip", ".nc", ".jp2", ".npz", ".npy", ".gz")

MATRIX = OUT / "susc_17c34_canary_pre_event_feature_matrix.csv"
S2 = OUT / "susc_17c34_sentinel2_pre_event_features.csv"
SPECTRAL = OUT / "susc_17c34_spectral_indices_features.csv"
CHIRPS = OUT / "susc_17c34_chirps_features.csv"
TERRAIN = OUT / "susc_17c34_terrain_features.csv"
DRAINAGE = OUT / "susc_17c34_drainage_distance_features.csv"
LANDCOVER = OUT / "susc_17c34_landcover_urban_features.csv"
AVAIL = OUT / "susc_17c34_local_feature_availability_matrix.csv"
REPLAY_POLICY = OUT / "susc_17c34_score_v6_replay_policy.json"
SCORE_REVIEW = OUT / "susc_17c34_score_v6_review_only_by_canary_patch.csv"
CONTRAST = OUT / "susc_17c34_original_vs_canary_feature_contrast.csv"
LEAKAGE = OUT / "susc_17c34_no_leakage_audit.csv"
INTEGRITY = OUT / "susc_17c34_score_integrity_audit.csv"
SUMMARY = OUT / "susc_17c34_readiness_summary.json"

EXPECTED = [
    OUT / "SUSC_17C34_EXTRACTION_FEATURES_PRE_EVENT_CANARY_REPORT.md",
    OUT / "susc_17c34_canary_geometry_audit.csv", OUT / "susc_17c34_input_feature_inventory.csv",
    OUT / "susc_17c34_pre_event_window_binding.csv", OUT / "susc_17c34_sentinel2_scene_selection.csv",
    S2, SPECTRAL, CHIRPS, DRAINAGE, TERRAIN, LANDCOVER, AVAIL, MATRIX, REPLAY_POLICY,
    SCORE_REVIEW, CONTRAST, OUT / "susc_17c34_observational_adherence_metrics.csv",
    INTEGRITY, LEAKAGE, SUMMARY, OUT / "susc_17c34_promotion_blockers.csv",
]


def rc(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def run_script(name: str, *args: str):
    env = os.environ.copy()
    for v in ["SUSC_17C34_ALLOW_NETWORK", "SUSC_17C34_ALLOW_PUBLIC_DOWNLOAD",
              "SUSC_17C34_ALLOW_LIGHTWEIGHT_RASTER", "SUSC_17C34_ALLOW_LIGHTWEIGHT_VECTOR",
              "SUSC_17C34_ALLOW_SENTINEL2_COG", "SUSC_17C34_ALLOW_TERRAIN_FEATURES",
              "SUSC_17C34_ALLOW_DRAINAGE_FEATURES"]:
        env.pop(v, None)
    return subprocess.run([sys.executable, str(SCRIPTS / name), *args],
                          cwd=ROOT, env=env, text=True, capture_output=True, timeout=600, check=False)


def test_build_offline_byte_identico_e_valida():
    r = run_script("build_susc_17c34_canary_pre_event_features.py")
    assert r.returncode == 0, r.stderr + r.stdout
    first = {p: p.read_bytes() for p in EXPECTED}
    r = run_script("build_susc_17c34_canary_pre_event_features.py")
    assert r.returncode == 0, r.stderr + r.stdout
    assert {p: p.read_bytes() for p in EXPECTED} == first
    r = run_script("validate_susc_17c34_canary_pre_event_features.py")
    assert r.returncode == 0, r.stderr + r.stdout


def test_canarios_e_controles_consumidos():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["canary_patch_rows_consumed_count"] >= 11
    assert s["original_control_patch_rows_consumed_count"] >= 5
    assert len(rc(MATRIX)) >= 11


def test_sentinel2_e_spectral_extraidos():
    s2 = rc(S2)
    assert len([r for r in s2 if r["sentinel2_available"] == "true"]) >= 11
    sp = rc(SPECTRAL)
    assert len([r for r in sp if r["spectral_available"] == "true"]) >= 11


def test_chirps_features_existem_como_trigger():
    ch = rc(CHIRPS)
    assert len(ch) >= 11
    for r in ch:
        assert r["chirps_is_trigger_not_observation"] == "true"
        assert r["chirps_is_post_event"] == "false"


def test_pelo_menos_3_familias_de_features():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["feature_family_available_count"] >= 3


def test_score_v6_replay_policy_existe_e_full_bloqueado():
    pol = json.loads(REPLAY_POLICY.read_text(encoding="utf-8"))
    assert pol["can_compute_full_replay"] is False
    assert pol["not_official_score"] is True
    assert pol["does_not_replace_score_v6"] is True
    for r in rc(SCORE_REVIEW):
        assert r["not_official_score"] == "true"
        assert r["does_not_replace_score_v6"] == "true"
        assert r["score_v6_review_only"] == "not_available"


def test_score_v6_oficial_intacto_e_v7_inexistente():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["score_v6_original_file_changed"] is False
    assert s["score_v7_created"] is False
    r = subprocess.run(["git", "diff", "--name-only", "--", "datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv"],
                       cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.stdout.strip() == ""
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    for r in rc(INTEGRITY):
        assert r["official_score_v6_changed"] == "false"


def test_ocorrencia_nao_e_feature_e_sem_pos_evento():
    for r in rc(LEAKAGE):
        assert r["uses_occurrence_as_feature"] == "false"
        assert r["uses_post_event_feature_as_pre_event"] == "false"
        assert r["uses_control_as_negative_truth"] == "false"
        assert r["passes_no_leakage"] == "true"
    for r in rc(MATRIX):
        assert r["pre_event_window_start"] == "2022-04-24"
        assert r["pre_event_window_end"] == "2022-05-23"


def test_controle_nao_vira_negativo_verdadeiro():
    for r in rc(CONTRAST):
        assert r["absence_of_occurrence_is_true_negative"] == "false"
        assert r["canary_has_official_hydrological_occurrence"] == "true"


def test_ground_truth_e_treino_nao_criados():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["ground_truth_created"] is False
    assert s["training_labels_created"] is False
    for r in rc(MATRIX):
        assert r["ground_truth"] == "false"
        assert r["training_label"] == "false"


def test_raster_pesado_nao_commitado():
    if LIGHT.exists():
        for p in LIGHT.glob("**/*"):
            if p.is_file():
                assert p.suffix.lower() not in FORBIDDEN, p.name


def test_contrast_matrix_existe():
    assert len(rc(CONTRAST)) >= 1


def test_extracao_bloqueia_sem_env():
    r = run_script("susc_17c34_canary_pre_event_features.py", "run-sentinel2-feature-extraction")
    assert "BLOCKED" in (r.stdout + r.stderr)
