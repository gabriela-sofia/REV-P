"""Testes do SUSC-18B - multimodal ML readiness, dataset builder e experiment harness."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"

CONTRACT = OUT / "susc_18b_ml_dataset_contract.json"
DATASET = OUT / "susc_18b_ml_ready_dataset.csv"
GROUPS = OUT / "susc_18b_feature_group_registry.csv"
MASK = OUT / "susc_18b_missingness_mask.csv"
NORM_PIPE = OUT / "susc_18b_ml_normalization_pipeline.json"
NORM_AUDIT = OUT / "susc_18b_ml_normalization_audit.csv"
TENSOR = OUT / "susc_18b_ml_feature_tensor.csv"
SPLITS = OUT / "susc_18b_split_registry.csv"
UNSUP = OUT / "susc_18b_unsupervised_experiment_registry.csv"
EMB = OUT / "susc_18b_tabular_embedding_projection.csv"
CLUSTERS = OUT / "susc_18b_cluster_assignments.csv"
OUTLIERS = OUT / "susc_18b_outlier_scores.csv"
SIM = OUT / "susc_18b_patch_similarity_index.csv"
CAND = OUT / "susc_18b_candidate_discovery_queue.csv"
GATE = OUT / "susc_18b_trainability_gate.csv"
HARNESS = OUT / "susc_18b_supervised_harness_status.csv"
LABEL_CONTRACT = OUT / "susc_18b_future_label_contract.json"
EXP_REG = OUT / "susc_18b_experiment_registry.csv"
CARDS = OUT / "susc_18b_model_cards.csv"
LEAKAGE = OUT / "susc_18b_no_leakage_audit.csv"
SUMMARY = OUT / "susc_18b_readiness_summary.json"
BLOCKERS = OUT / "susc_18b_promotion_blockers.csv"
REPORT = OUT / "SUSC_18B_MULTIMODAL_ML_READINESS_REPORT.md"

SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

EXPECTED = [
    CONTRACT, DATASET, GROUPS, MASK, NORM_PIPE, NORM_AUDIT, TENSOR, SPLITS, UNSUP, EMB, CLUSTERS,
    OUTLIERS, SIM, CAND, GATE, HARNESS, LABEL_CONTRACT, EXP_REG, CARDS,
    OUT / "susc_18b_score_integrity_audit.csv", LEAKAGE, SUMMARY, BLOCKERS, REPORT,
]


def rc(path):
    with path.open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def rj(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_script(name, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / name), *args],
                          cwd=ROOT, text=True, capture_output=True, timeout=600, check=False)


def test_18a_commitado_antes_do_18b():
    r = subprocess.run(["git", "cat-file", "-e", "HEAD:outputs_public/suscetibilidade/susc_18a_multimodal_patch_feature_store.csv"],
                       cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.returncode == 0, "18A precisa estar commitado antes do 18B"


def test_build_offline_byte_identico_e_valida():
    r = run_script("build_susc_18b_multimodal_ml_readiness.py")
    assert r.returncode == 0, r.stderr + r.stdout
    first = {p: p.read_bytes() for p in EXPECTED}
    r = run_script("build_susc_18b_multimodal_ml_readiness.py")
    assert r.returncode == 0, r.stderr + r.stdout
    assert {p: p.read_bytes() for p in EXPECTED} == first
    r = run_script("validate_susc_18b_multimodal_ml_readiness.py")
    assert r.returncode == 0, r.stderr + r.stdout


def test_feature_store_consumida_e_dataset():
    s = rj(SUMMARY)
    assert s["feature_store_rows_consumed_count"] >= 316
    assert s["ml_dataset_rows_count"] >= 316
    assert len(rc(DATASET)) >= 316


def test_feature_groups_e_numeric_features():
    assert len(rc(GROUPS)) >= 6
    assert rj(SUMMARY)["numeric_feature_count"] >= 25


def test_missingness_mask_normalization_tensor_splits():
    assert len(rc(MASK)) > 0 and rj(SUMMARY)["missingness_mask_created"] is True
    assert NORM_PIPE.exists() and rj(SUMMARY)["normalization_pipeline_created"] is True
    assert len(rc(NORM_AUDIT)) >= 25
    assert len(rc(TENSOR)) >= 316
    assert len(rc(SPLITS)) > 0


def test_unsupervised_experiments_e_projecoes():
    assert len(rc(UNSUP)) >= 5
    assert len(rc(EMB)) >= 316
    assert len(rc(CLUSTERS)) >= 316
    assert len(rc(OUTLIERS)) >= 316


def test_similarity_e_candidatos():
    assert len(rc(SIM)) >= 1580
    assert len(rc(CAND)) >= 50


def test_trainability_gate_e_harness():
    gate = {r["training_mode"]: r["is_allowed_now"] for r in rc(GATE)}
    assert gate["supervised_training"] == "false"
    assert gate["weak_supervision"] == "false"
    assert gate["positive_unlabeled_learning"] == "false"
    assert gate["self_supervised_learning"] == "true"
    assert gate["unsupervised_clustering"] == "true"
    assert gate["candidate_screening"] == "true"
    harness = rc(HARNESS)
    assert len(harness) >= 4


def test_supervised_training_nao_executado():
    s = rj(SUMMARY)
    assert s["supervised_training_attempted"] is False
    assert s["supervised_training_executed"] is False
    assert s["supervised_training_allowed"] is False
    assert s["supervised_training_blocked"] is True
    for r in rc(HARNESS):
        assert r["attempted_training"] == "false"
        assert r["training_executed"] == "false"


def test_future_label_contract_experiment_registry_model_cards():
    lc = rj(LABEL_CONTRACT)
    assert lc["minimum_criteria_for_future_training"]["accepted_ground_reference_count"] == 50
    assert lc["current_status"]["training_enabled_now"] is False
    assert len(rc(EXP_REG)) >= 5
    assert len(rc(CARDS)) >= 5


def test_canario_nao_positivo_controle_nao_negativo():
    s = rj(SUMMARY)
    assert s["canary_used_as_positive_label"] is False
    assert s["control_used_as_negative_label"] is False
    for r in rc(LEAKAGE):
        assert r["uses_canary_as_positive_label"] == "false"
        assert r["uses_control_as_negative_label"] == "false"


def test_score_v6_nao_target_ocorrencia_nao_feature():
    s = rj(SUMMARY)
    assert s["score_v6_used_as_training_target"] is False
    assert s["occurrence_used_as_causal_feature"] is False
    for r in rc(DATASET):
        assert r["score_v6_is_target"] == "false"
        assert r["occurrence_anchor_is_target"] == "false"
    for r in rc(LEAKAGE):
        assert r["uses_score_v6_as_target"] == "false"
        assert r["uses_occurrence_as_feature"] == "false"


def test_flow_acc_nao_equivalente():
    assert rj(SUMMARY)["flow_acc_treated_as_equivalent"] is False
    for r in rc(DATASET):
        assert r["flow_acc_equivalent"] == "false"
    for r in rc(LEAKAGE):
        assert r["uses_flow_acc_as_equivalent"] == "false"


def test_score_v6_intacto_v7_inexistente():
    r = subprocess.run(["git", "diff", "--name-only", "--", "datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv"],
                       cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.stdout.strip() == ""
    assert rj(SUMMARY)["score_v6_original_file_changed"] is False
    assert not SCORE_V7.exists()
    assert rj(SUMMARY)["score_v7_created"] is False


def test_ground_truth_treino_nao_criados():
    s = rj(SUMMARY)
    assert s["ground_truth_created"] is False
    assert s["training_labels_created"] is False
    for r in rc(DATASET):
        assert r["ground_truth"] == "false"
        assert r["training_label"] == "false"


def test_17b_bloqueado():
    assert rj(SUMMARY)["eligible_for_17b_now"] is False
    types = {r["blocker_type"] for r in rc(BLOCKERS)}
    assert "17b_blocked" in types
    assert "supervised_training_blocked_no_ground_reference_labels" in types


def test_no_leakage_passa():
    leak = rc(LEAKAGE)
    assert len(leak) > 0
    for r in leak:
        assert r["passes_no_leakage"] == "true"
        assert r["uses_post_event_feature_as_pre_event"] == "false"


def test_minimum_success_achieved():
    assert rj(SUMMARY)["minimum_success_achieved"] is True
