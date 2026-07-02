"""Testes do SUSC-17C40 - indice transparente review-only com contrato de substituicao do flow accumulation."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"

CONTRACT = OUT / "susc_17c40_flow_acc_replacement_contract_operationalized.json"
POLICY = OUT / "susc_17c40_transparent_index_policy.json"
COMP = OUT / "susc_17c40_transparent_component_matrix.csv"
NORM = OUT / "susc_17c40_component_normalization_audit.csv"
INDEX = OUT / "susc_17c40_transparent_review_only_index.csv"
CONTRAST = OUT / "susc_17c40_canary_vs_control_operational_contrast.csv"
ROB = OUT / "susc_17c40_index_robustness_analysis.csv"
ALIGN = OUT / "susc_17c40_observational_alignment_metrics.csv"
INTERP = OUT / "susc_17c40_operational_interpretation.json"
INTEG = OUT / "susc_17c40_score_integrity_audit.csv"
LEAKAGE = OUT / "susc_17c40_no_leakage_audit.csv"
SUMMARY = OUT / "susc_17c40_readiness_summary.json"

EXPECTED = [
    OUT / "SUSC_17C40_TRANSPARENT_REVIEW_INDEX_REPORT.md", CONTRACT, POLICY, COMP, NORM, INDEX,
    CONTRAST, ROB, ALIGN, INTERP, INTEG, LEAKAGE, SUMMARY, OUT / "susc_17c40_promotion_blockers.csv",
]


def rc(path):
    with path.open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def run_script(name, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / name), *args],
                          cwd=ROOT, text=True, capture_output=True, timeout=600, check=False)


def test_build_offline_byte_identico_e_valida():
    r = run_script("build_susc_17c40_transparent_review_index.py")
    assert r.returncode == 0, r.stderr + r.stdout
    first = {p: p.read_bytes() for p in EXPECTED}
    r = run_script("build_susc_17c40_transparent_review_index.py")
    assert r.returncode == 0, r.stderr + r.stdout
    assert {p: p.read_bytes() for p in EXPECTED} == first
    r = run_script("validate_susc_17c40_transparent_review_index.py")
    assert r.returncode == 0, r.stderr + r.stdout


def test_canarios_controles_e_matriz():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["canary_patch_rows_consumed_count"] >= 11
    assert s["original_control_patch_rows_consumed_count"] >= 5
    assert len(rc(COMP)) >= 16


def test_contrato_flow_acc_consumido():
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert c["flow_acc_replacement_contract_review_only"] is True
    assert c["flow_acc_official_equivalent"] is False
    assert c["score_v6_replay_final_allowed"] is False


def test_indice_nao_e_score_v6_nem_v7():
    pol = json.loads(POLICY.read_text(encoding="utf-8"))
    assert pol["not_score_v6"] is True and pol["not_score_v7"] is True and pol["does_not_replace_score_v6"] is True
    for r in rc(INDEX):
        assert r["not_score_v6"] == "true"
        assert r["not_score_v7"] == "true"
        assert r["does_not_replace_score_v6"] == "true"
    for r in rc(INTEG):
        assert r["index_is_score_v6"] == "false"
        assert r["index_is_score_v7"] == "false"
        assert r["index_is_replay_v6"] == "false"


def test_tres_variantes_e_16_linhas():
    variants = {r["index_variant"] for r in rc(INDEX)}
    assert len(variants) >= 3
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["transparent_index_rows_count"] >= 16


def test_contraste_robustez_interpretacao():
    assert len(rc(CONTRAST)) >= 11
    assert len(rc(ROB)) >= 3
    interp = json.loads(INTERP.read_text(encoding="utf-8"))
    assert interp["operational_interpretation_created"] is True


def test_flow_acc_nao_equivalente_ocorrencia_nao_causal():
    for r in rc(LEAKAGE):
        assert r["uses_flow_acc_as_equivalent"] == "false"
        assert r["uses_occurrence_as_causal_feature"] == "false"
        assert r["occurrence_only_observational_anchor"] == "true"
        assert r["hides_missing_feature"] == "false"
        assert r["uses_canary_to_calibrate_scale"] == "false"
        assert r["passes_no_leakage"] == "true"


def test_controle_nao_negativo_verdadeiro():
    for r in rc(COMP):
        assert r["absence_of_occurrence_is_true_negative"] == "false"
    for r in rc(CONTRAST):
        assert r["absence_of_occurrence_is_true_negative"] == "false"


def test_score_v6_matriz_intactos_sem_v7():
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
    interp = json.loads(INTERP.read_text(encoding="utf-8"))
    assert interp["can_unlock_17b"] is False
    assert interp["can_support_training"] is False
    assert interp["can_support_ground_truth"] is False


def test_nao_termina_blocked():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["minimum_success_achieved"] is True
    assert s["result_class"] != "blocked"
    assert s["transparent_index_operational"] is True
