"""Testes do SUSC-17C38 - flow accumulation basin-aware e revalidacao de equivalencia."""

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
LIGHT = OUT / "susc_17c38_light_artifacts"
FORBIDDEN = (".tif", ".tiff", ".safe", ".zip", ".nc", ".jp2", ".gz")
CRITICAL = ["hand_mean", "twi_mean", "tpi_250m_mean", "flow_acc_log_mean"]

DOMAIN = OUT / "susc_17c38_basin_aware_domain_definition.csv"
HGRID = OUT / "susc_17c38_basin_hydrologic_grid_metadata.csv"
FEATURES = OUT / "susc_17c38_official_patch_basin_aware_topography_features.csv"
COMP = OUT / "susc_17c38_basin_aware_topography_feature_comparison.csv"
METRICS = OUT / "susc_17c38_basin_aware_equivalence_metrics.csv"
DECISION = OUT / "susc_17c38_method_equivalence_decision_update.json"
READY = OUT / "susc_17c38_score_v6_replay_readiness_after_basin_flow.csv"
INTEG = OUT / "susc_17c38_score_integrity_audit.csv"
LEAKAGE = OUT / "susc_17c38_no_leakage_audit.csv"
SUMMARY = OUT / "susc_17c38_readiness_summary.json"

EXPECTED = [
    OUT / "SUSC_17C38_BASIN_AWARE_FLOW_EQUIVALENCE_REPORT.md", DOMAIN,
    OUT / "susc_17c38_dem_acquisition_attempts.csv", OUT / "susc_17c38_dem_artifact_manifest.csv",
    OUT / "susc_17c38_basin_dem_grid_metadata.csv", HGRID,
    OUT / "susc_17c38_basin_flow_accumulation_diagnostics.csv", FEATURES, COMP, METRICS, DECISION,
    READY, INTEG, LEAKAGE, SUMMARY, OUT / "susc_17c38_promotion_blockers.csv",
]


def rc(path):
    with path.open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def run_script(name, *args):
    env = os.environ.copy()
    for v in ["SUSC_17C38_ALLOW_NETWORK", "SUSC_17C38_ALLOW_PUBLIC_DOWNLOAD",
              "SUSC_17C38_ALLOW_LIGHTWEIGHT_RASTER", "SUSC_17C38_ALLOW_DEM",
              "SUSC_17C38_ALLOW_HYDROLOGIC_PROCESSING", "SUSC_17C38_ALLOW_BASIN_AWARE_FLOW"]:
        env.pop(v, None)
    return subprocess.run([sys.executable, str(SCRIPTS / name), *args],
                          cwd=ROOT, env=env, text=True, capture_output=True, timeout=600, check=False)


def test_build_offline_byte_identico_e_valida():
    r = run_script("build_susc_17c38_basin_aware_flow_equivalence.py")
    assert r.returncode == 0, r.stderr + r.stdout
    first = {p: p.read_bytes() for p in EXPECTED}
    r = run_script("build_susc_17c38_basin_aware_flow_equivalence.py")
    assert r.returncode == 0, r.stderr + r.stdout
    assert {p: p.read_bytes() for p in EXPECTED} == first
    r = run_script("validate_susc_17c38_basin_aware_flow_equivalence.py")
    assert r.returncode == 0, r.stderr + r.stdout


def test_populacao_e_recompute_100():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["official_patch_population_rows_count"] >= 100
    assert s["official_patch_recomputed_rows_count"] >= 100


def test_dominio_basin_aware_existe():
    assert len(rc(DOMAIN)) >= 1
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["basin_aware_domain_created"] is True


def test_flow_acc_nao_por_janela_de_patch():
    for r in rc(HGRID):
        assert r["flow_acc_per_patch_window"] == "false"
        assert "single_domain" in r["flow_accumulation_method"]
    for r in rc(LEAKAGE):
        assert r["flow_acc_per_patch_window"] == "false"
        assert r["flow_acc_single_basin_domain"] == "true"


def test_comparacao_400_e_metricas_4():
    assert len(rc(COMP)) >= 400
    feats = {r["feature_name"] for r in rc(METRICS)}
    assert set(CRITICAL).issubset(feats)


def test_flow_acc_retestado_e_status_registrado():
    d = json.loads(DECISION.read_text(encoding="utf-8"))
    assert d["flow_acc_equivalence_retested"] is True
    m = {r["feature_name"]: r for r in rc(METRICS)}
    assert m["flow_acc_log_mean"]["equivalence_status_17c37"] != ""
    assert m["flow_acc_log_mean"]["equivalence_status_17c38"] != ""


def test_score_final_nao_libera_sem_full_equivalence():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    d = json.loads(DECISION.read_text(encoding="utf-8"))
    if not d["method_equivalence_accepted"]:
        assert s["can_compute_score_v6_final_replay"] is False
    m = {r["feature_name"]: r["equivalence_status"] for r in rc(METRICS)}
    if d["method_equivalence_accepted"]:
        assert all(m[f] == "equivalent" for f in CRITICAL)


def test_canarios_nao_decidem_equivalencia():
    for r in rc(LEAKAGE):
        assert r["uses_canary_to_decide_equivalence"] == "false"
        assert r["equivalence_evaluated_on_official_patches"] == "true"
    for r in rc(FEATURES):
        assert r["patch_id"].startswith("recife")


def test_score_v6_e_matriz_intactos_sem_v7():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["score_v6_original_file_changed"] is False
    assert s["official_matrix_changed"] is False
    assert s["score_v7_created"] is False
    for target in ["datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv",
                   "datasets/suscetibilidade/susc_features_by_patch_v1.csv"]:
        r = subprocess.run(["git", "diff", "--name-only", "--", target], cwd=ROOT, text=True, capture_output=True, check=False)
        assert r.stdout.strip() == ""
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()


def test_gt_treino_nao_criados_17b_bloqueado():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["ground_truth_created"] is False
    assert s["training_labels_created"] is False
    assert s["eligible_for_17b_now"] is False


def test_no_leakage_controle_nao_negativo():
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


def test_processamento_bloqueia_sem_env():
    r = run_script("susc_17c38_basin_aware_flow_equivalence.py", "build-basin-hydrologic-grid")
    assert "BLOCKED" in (r.stdout + r.stderr)
