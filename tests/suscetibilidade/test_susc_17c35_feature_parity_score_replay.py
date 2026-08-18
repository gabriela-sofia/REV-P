"""Testes do SUSC-17C35 - feature parity, HAND/hidrografia/landcover e readiness de replay score v6."""

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
LIGHT = OUT / "susc_17c35_light_artifacts"
FORBIDDEN = (".tif", ".tiff", ".safe", ".zip", ".nc", ".jp2", ".npz", ".npy", ".gz")

CONTRACT = OUT / "susc_17c35_score_v6_formula_contract.json"
POP = OUT / "susc_17c35_official_score_population_inventory.csv"
NORM = OUT / "susc_17c35_normalization_anchor_evaluation.csv"
DRAINAGE = OUT / "susc_17c35_official_drainage_features.csv"
HAND = OUT / "susc_17c35_hand_feature_candidates.csv"
LC = OUT / "susc_17c35_landcover_urban_features.csv"
PARITY = OUT / "susc_17c35_feature_parity_matrix.csv"
READY = OUT / "susc_17c35_score_v6_replay_readiness.csv"
REPLAY = OUT / "susc_17c35_score_v6_review_only_replay_by_canary_patch.csv"
CONTRAST = OUT / "susc_17c35_original_vs_canary_comparable_contrast.csv"
INTEGRITY = OUT / "susc_17c35_score_integrity_audit.csv"
LEAKAGE = OUT / "susc_17c35_no_leakage_audit.csv"
SUMMARY = OUT / "susc_17c35_readiness_summary.json"

EXPECTED = [
    OUT / "SUSC_17C35_FEATURE_PARITY_SCORE_REPLAY_REPORT.md", CONTRACT, POP, NORM,
    OUT / "susc_17c35_hydrography_acquisition_attempts.csv", OUT / "susc_17c35_hydrography_artifact_manifest.csv",
    DRAINAGE, HAND, OUT / "susc_17c35_landcover_acquisition_attempts.csv",
    OUT / "susc_17c35_landcover_artifact_manifest.csv", LC, PARITY, READY, REPLAY, CONTRAST,
    INTEGRITY, LEAKAGE, SUMMARY, OUT / "susc_17c35_promotion_blockers.csv",
]


def rc(path):
    with path.open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def run_script(name, *args):
    env = os.environ.copy()
    for v in ["SUSC_17C35_ALLOW_NETWORK", "SUSC_17C35_ALLOW_PUBLIC_DOWNLOAD",
              "SUSC_17C35_ALLOW_LIGHTWEIGHT_VECTOR", "SUSC_17C35_ALLOW_LIGHTWEIGHT_RASTER",
              "SUSC_17C35_ALLOW_HYDROGRAPHY", "SUSC_17C35_ALLOW_HAND_PIPELINE", "SUSC_17C35_ALLOW_LANDCOVER"]:
        env.pop(v, None)
    return subprocess.run([sys.executable, str(SCRIPTS / name), *args],
                          cwd=ROOT, env=env, text=True, capture_output=True, timeout=600, check=False)


def test_build_offline_byte_identico_e_valida():
    r = run_script("build_susc_17c35_feature_parity_score_replay.py")
    assert r.returncode == 0, r.stderr + r.stdout
    first = {p: p.read_bytes() for p in EXPECTED}
    r = run_script("build_susc_17c35_feature_parity_score_replay.py")
    assert r.returncode == 0, r.stderr + r.stdout
    assert {p: p.read_bytes() for p in EXPECTED} == first
    r = run_script("validate_susc_17c35_feature_parity_score_replay.py")
    assert r.returncode == 0, r.stderr + r.stdout


def test_canarios_consumidos_e_contrato():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["canary_patch_rows_consumed_count"] >= 11
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert c["canary_only_normalization_allowed"] is False
    assert c["can_replay_without_population"] is False


def test_hash_score_v6_before_after_igual_e_intacto():
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert c["score_v6_hash_before"] == c["score_v6_hash_after"]
    assert c["score_v6_original_file_changed"] is False
    r = subprocess.run(["git", "diff", "--name-only", "--", "datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv"],
                       cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.stdout.strip() == ""
    for row in rc(INTEGRITY):
        assert row["official_score_v6_changed"] == "false"


def test_populacao_oficial_inventariada_e_norm_reconstruida():
    pop = rc(POP)
    assert any(r["contains_raw_features"] == "true" for r in pop)
    norm = rc(NORM)
    assert any(r["population_reconstructed"] == "true" for r in norm)


def test_normalizacao_canary_only_proibida():
    for r in rc(NORM):
        assert r["canary_only_normalization_used"] == "false"
    for r in rc(LEAKAGE):
        assert r["uses_canary_only_normalization"] == "false"


def test_hidrografia_oficial_e_proxy_diferenciados():
    dr = rc(DRAINAGE)
    for r in dr:
        if r["official_drainage_available"] == "false":
            assert r["drainage_proxy_used"] == "true"
        if r["drainage_proxy_used"] == "true":
            assert r["drainage_source_authority"] == "osm_proxy"
    for r in rc(LEAKAGE):
        assert r["uses_osm_as_official_hydrography"] == "false"


def test_hand_proxy_nao_vira_hand_full():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["HAND_full_rows_count"] == 0
    for r in rc(HAND):
        assert r["not_full_hydrologic_HAND"] == "true"
    for r in rc(LEAKAGE):
        assert r["uses_hand_proxy_as_hand_full"] == "false"


def test_landcover_oficial_e_proxy_diferenciados():
    for r in rc(LC):
        if r["landcover_available"] == "false":
            assert r["proxy_only"] == "true"


def test_feature_parity_e_readiness_existem():
    assert len(rc(PARITY)) >= 11
    assert len(rc(READY)) >= 11


def test_score_numerico_exige_populacao_reconstruida():
    for r in rc(REPLAY):
        if r["score_v6_review_only"] != "not_available":
            assert r["normalization_population_reconstructed"] == "true"
        # neste marco score final e bloqueado (topografia)
        assert r["not_official_score"] == "true"
        assert r["does_not_replace_score_v6"] == "true"


def test_full_replay_bloqueado_partial_permitido():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["score_v6_full_replay_computable"] is False
    assert s["score_v6_partial_component_replay_computable"] is True
    assert s["feature_family_available_count"] >= 6


def test_ocorrencia_nao_feature_controle_nao_negativo():
    for r in rc(LEAKAGE):
        assert r["uses_occurrence_as_feature"] == "false"
        assert r["uses_control_as_negative_truth"] == "false"
        assert r["passes_no_leakage"] == "true"
    for r in rc(CONTRAST):
        assert r["absence_of_occurrence_is_true_negative"] == "false"


def test_score_v7_e_gt_treino_ausentes():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["score_v7_created"] is False
    assert s["ground_truth_created"] is False
    assert s["training_labels_created"] is False
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()


def test_raster_pesado_nao_commitado():
    if LIGHT.exists():
        for p in LIGHT.glob("**/*"):
            if p.is_file():
                assert p.suffix.lower() not in FORBIDDEN, p.name
                assert p.stat().st_size <= 5_000_000, p.name


def test_extracao_bloqueia_sem_env():
    r = run_script("susc_17c35_feature_parity_score_replay.py", "run-hydrography-acquisition")
    assert "BLOCKED" in (r.stdout + r.stderr)
