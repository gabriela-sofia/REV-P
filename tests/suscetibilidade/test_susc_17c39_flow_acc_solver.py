"""Testes do SUSC-17C39 - flow accumulation solver (variantes + calibracao transferivel)."""

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
LIGHT = OUT / "susc_17c39_light_artifacts"
FORBIDDEN = (".tif", ".tiff", ".safe", ".zip", ".nc", ".jp2", ".gz")

TRACE = OUT / "susc_17c39_flow_acc_method_trace_inventory.csv"
REG = OUT / "susc_17c39_flow_acc_variant_registry.csv"
RESULTS = OUT / "susc_17c39_official_patch_flow_acc_variants.csv"
VMETRICS = OUT / "susc_17c39_flow_acc_variant_metrics.csv"
CREG = OUT / "susc_17c39_flow_acc_calibration_model_registry.csv"
CV = OUT / "susc_17c39_flow_acc_calibration_crossval.csv"
CMETRICS = OUT / "susc_17c39_flow_acc_calibration_metrics.csv"
SOLUTION = OUT / "susc_17c39_flow_acc_solution_selection.csv"
CANARY = OUT / "susc_17c39_canary_flow_acc_solution.csv"
READY = OUT / "susc_17c39_score_v6_replay_readiness_after_flow_solution.csv"
INTEG = OUT / "susc_17c39_score_integrity_audit.csv"
LEAKAGE = OUT / "susc_17c39_no_leakage_audit.csv"
SUMMARY = OUT / "susc_17c39_readiness_summary.json"

EXPECTED = [
    OUT / "SUSC_17C39_FLOW_ACC_SOLVER_REPORT.md", TRACE,
    OUT / "susc_17c39_official_flow_acc_diagnostic.csv", REG, RESULTS, VMETRICS, CREG, CV,
    CMETRICS, SOLUTION, CANARY, READY, INTEG, LEAKAGE, SUMMARY,
    OUT / "susc_17c39_promotion_blockers.csv",
]


def rc(path):
    with path.open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def run_script(name, *args):
    env = os.environ.copy()
    for v in ["SUSC_17C39_ALLOW_NETWORK", "SUSC_17C39_ALLOW_PUBLIC_DOWNLOAD",
              "SUSC_17C39_ALLOW_LIGHTWEIGHT_RASTER", "SUSC_17C39_ALLOW_DEM",
              "SUSC_17C39_ALLOW_HYDROLOGIC_PROCESSING", "SUSC_17C39_ALLOW_FLOW_VARIANTS",
              "SUSC_17C39_ALLOW_CALIBRATION"]:
        env.pop(v, None)
    return subprocess.run([sys.executable, str(SCRIPTS / name), *args],
                          cwd=ROOT, env=env, text=True, capture_output=True, timeout=600, check=False)


def test_build_offline_byte_identico_e_valida():
    r = run_script("build_susc_17c39_flow_acc_solver.py")
    assert r.returncode == 0, r.stderr + r.stdout
    first = {p: p.read_bytes() for p in EXPECTED}
    r = run_script("build_susc_17c39_flow_acc_solver.py")
    assert r.returncode == 0, r.stderr + r.stdout
    assert {p: p.read_bytes() for p in EXPECTED} == first
    r = run_script("validate_susc_17c39_flow_acc_solver.py")
    assert r.returncode == 0, r.stderr + r.stdout


def test_populacao_trace_variantes_resultados():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["official_patch_population_rows_count"] >= 100
    assert s["method_trace_rows_count"] > 0
    assert s["flow_acc_variant_count"] >= 12
    assert s["official_patch_variant_recomputed_rows_count"] >= 1200
    assert s["flow_acc_variant_metric_rows_count"] >= 12


def test_calibradores_e_crossval():
    assert len(rc(CREG)) >= 4
    assert len(rc(CV)) >= 4
    assert len(rc(CMETRICS)) >= 4


def test_solucao_selecionada_nao_blocked():
    sel = rc(SOLUTION)[0]
    assert sel["selected_solution_type"] in ("method_recovered", "variant_equivalent",
                                             "calibrated_surrogate_validated", "replacement_contract_review_only")
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["best_solution_selected"] is True
    assert s["minimum_success_achieved"] is True
    assert s["flow_acc_solution_status"] != "blocked"


def test_calibrador_nao_usa_canarios():
    for r in rc(CREG):
        assert r["uses_canaries"] == "false"
    for r in rc(CANARY):
        assert r["uses_canary_for_training"] == "false"
        assert r["not_original_method"] == "true"
    for r in rc(LEAKAGE):
        assert r["uses_canary_for_training"] == "false"
        assert r["calibration_on_official_only"] == "true"


def test_surrogate_validado_antes_de_aplicar():
    sel = rc(SOLUTION)[0]
    if sel["selected_solution_type"] == "calibrated_surrogate_validated":
        assert sel["solution_passes_threshold"] == "true"
    for r in rc(LEAKAGE):
        assert r["surrogate_validated_before_apply"] == "true"


def test_canary_solution_11_e_score_final_nao_computado():
    assert len(rc(CANARY)) >= 11
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["can_compute_score_v6_final_replay"] is False
    for r in rc(INTEG):
        assert r["final_score_computed"] == "false"


def test_readiness_atualizada():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["score_v6_replay_readiness_updated"] is True
    assert len(rc(READY)) >= 11


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


def test_gt_treino_nao_criados_no_leakage():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["ground_truth_created"] is False
    assert s["training_labels_created"] is False
    for r in rc(LEAKAGE):
        assert r["uses_occurrence_as_feature"] == "false"
        assert r["uses_control_as_negative_truth"] == "false"
        assert r["masks_calibrated_as_original"] == "false"
        assert r["passes_no_leakage"] == "true"


def test_raster_pesado_nao_commitado():
    if LIGHT.exists():
        for p in LIGHT.glob("**/*"):
            if p.is_file():
                assert p.suffix.lower() not in FORBIDDEN, p.name
                assert p.stat().st_size <= 5_000_000, p.name


def test_recompute_bloqueia_sem_env():
    r = run_script("susc_17c39_flow_acc_solver.py", "run-flow-variant-recompute")
    assert "BLOCKED" in (r.stdout + r.stderr)
