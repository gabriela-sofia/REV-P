"""Testes do SUSC-17C37 - validacao de equivalencia metodologica topography/hydrology."""

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
LIGHT = OUT / "susc_17c37_light_artifacts"
FORBIDDEN = (".tif", ".tiff", ".safe", ".zip", ".nc", ".jp2", ".gz")
CRITICAL = ["hand_mean", "twi_mean", "tpi_250m_mean", "flow_acc_log_mean"]

POP = OUT / "susc_17c37_official_patch_population_inventory.csv"
SAMPLE = OUT / "susc_17c37_official_patch_equivalence_sample.csv"
REC = OUT / "susc_17c37_official_patch_recomputed_topography_features.csv"
COMP = OUT / "susc_17c37_topography_feature_equivalence_comparison.csv"
METRICS = OUT / "susc_17c37_method_equivalence_metrics.csv"
DECISION = OUT / "susc_17c37_method_equivalence_decision.json"
READY = OUT / "susc_17c37_score_v6_replay_readiness_after_equivalence.csv"
INTEG = OUT / "susc_17c37_score_integrity_audit.csv"
LEAKAGE = OUT / "susc_17c37_no_leakage_audit.csv"
SUMMARY = OUT / "susc_17c37_readiness_summary.json"

EXPECTED = [
    OUT / "SUSC_17C37_TOPOGRAPHY_METHOD_EQUIVALENCE_REPORT.md", POP, SAMPLE, REC, COMP,
    METRICS, DECISION, READY, INTEG, LEAKAGE, SUMMARY, OUT / "susc_17c37_promotion_blockers.csv",
]


def rc(path):
    with path.open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def run_script(name, *args):
    env = os.environ.copy()
    for v in ["SUSC_17C37_ALLOW_NETWORK", "SUSC_17C37_ALLOW_PUBLIC_DOWNLOAD",
              "SUSC_17C37_ALLOW_LIGHTWEIGHT_RASTER", "SUSC_17C37_ALLOW_DEM", "SUSC_17C37_ALLOW_HYDROLOGIC_PROCESSING"]:
        env.pop(v, None)
    return subprocess.run([sys.executable, str(SCRIPTS / name), *args],
                          cwd=ROOT, env=env, text=True, capture_output=True, timeout=600, check=False)


def test_build_offline_byte_identico_e_valida():
    r = run_script("build_susc_17c37_topography_method_equivalence.py")
    assert r.returncode == 0, r.stderr + r.stdout
    first = {p: p.read_bytes() for p in EXPECTED}
    r = run_script("build_susc_17c37_topography_method_equivalence.py")
    assert r.returncode == 0, r.stderr + r.stdout
    assert {p: p.read_bytes() for p in EXPECTED} == first
    r = run_script("validate_susc_17c37_topography_method_equivalence.py")
    assert r.returncode == 0, r.stderr + r.stdout


def test_populacao_amostra_recompute_comparacao_minimas():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["official_patch_population_rows_count"] >= 100
    assert s["official_patch_sample_rows_count"] >= 20
    assert s["official_patch_recomputed_rows_count"] >= 20
    assert s["topography_feature_comparison_rows_count"] >= 80


def test_metricas_para_4_features():
    feats = {r["feature_name"] for r in rc(METRICS)}
    assert set(CRITICAL).issubset(feats)
    for r in rc(METRICS):
        assert r["equivalence_status"] in ("equivalent", "calibratable", "incompatible", "insufficient_data")


def test_decisao_equivalencia_existe():
    d = json.loads(DECISION.read_text(encoding="utf-8"))
    assert d["method_equivalence_decision_created"] is True


def test_full_equivalence_exige_4_features_equivalent():
    d = json.loads(DECISION.read_text(encoding="utf-8"))
    metrics = {r["feature_name"]: r["equivalence_status"] for r in rc(METRICS)}
    if d["method_equivalence_accepted"]:
        assert all(metrics[f] == "equivalent" for f in CRITICAL)


def test_score_final_nao_libera_sem_full_equivalence():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    d = json.loads(DECISION.read_text(encoding="utf-8"))
    if not d["method_equivalence_accepted"]:
        assert s["can_compute_score_v6_final_replay"] is False
    for r in rc(INTEG):
        if r["final_score_computed"] == "true":
            assert d["method_equivalence_accepted"] is True


def test_canarios_nao_calibram_equivalencia():
    for r in rc(LEAKAGE):
        assert r["uses_canary_to_calibrate_equivalence"] == "false"
        assert r["equivalence_evaluated_on_official_patches"] == "true"
        assert r["accepts_equivalence_just_because_values_generated"] == "false"
    for r in rc(SAMPLE):
        assert r["patch_id"].startswith("recife")


def test_score_v6_e_matriz_oficial_intactos():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["score_v6_original_file_changed"] is False
    assert s["official_matrix_changed"] is False
    assert s["score_v7_created"] is False
    for target in ["datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv",
                   "datasets/suscetibilidade/susc_features_by_patch_v1.csv"]:
        r = subprocess.run(["git", "diff", "--name-only", "--", target], cwd=ROOT, text=True, capture_output=True, check=False)
        assert r.stdout.strip() == ""
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()


def test_gt_treino_nao_criados():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["ground_truth_created"] is False
    assert s["training_labels_created"] is False
    assert s["eligible_for_17b_now"] is False


def test_no_leakage_e_controle_nao_negativo():
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


def test_recompute_bloqueia_sem_env():
    r = run_script("susc_17c37_topography_method_equivalence.py", "run-recompute-official-patches")
    assert "BLOCKED" in (r.stdout + r.stderr)
