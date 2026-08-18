"""Testes do SUSC-18C - self-supervised pretraining e ground reference factory."""

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

CONTRACT = OUT / "susc_18c_pretraining_and_reference_contract.json"
SS_TENSOR = OUT / "susc_18c_self_supervised_training_tensor.csv"
MFM = OUT / "susc_18c_masked_feature_modeling_metrics.csv"
DENOISE_EMB = OUT / "susc_18c_denoising_embedding.csv"
DENOISE_METRICS = OUT / "susc_18c_denoising_model_metrics.csv"
CONTRASTIVE = OUT / "susc_18c_contrastive_similarity_index.csv"
REP_QUALITY = OUT / "susc_18c_representation_quality_audit.csv"
EMB_CAND = OUT / "susc_18c_embedding_candidate_discovery_queue.csv"
GR_TARGETS = OUT / "susc_18c_ground_reference_target_pack.csv"
GR_ATTEMPTS = OUT / "susc_18c_ground_reference_acquisition_attempts.csv"
GR_CANDS = OUT / "susc_18c_ground_reference_candidate_registry.csv"
PATCH_LINKS = OUT / "susc_18c_patch_reference_link_candidates.csv"
QA_QUEUE = OUT / "susc_18c_ground_reference_qa_queue.csv"
LABEL_EXEC = OUT / "susc_18c_label_contract_execution.csv"
GATE_RECHECK = OUT / "susc_18c_supervised_gate_recheck.csv"
EXP_REG = OUT / "susc_18c_experiment_registry.csv"
CARDS = OUT / "susc_18c_model_cards.csv"
LEAKAGE = OUT / "susc_18c_no_leakage_audit.csv"
SUMMARY = OUT / "susc_18c_readiness_summary.json"
BLOCKERS = OUT / "susc_18c_promotion_blockers.csv"
REPORT = OUT / "SUSC_18C_SELF_SUPERVISED_AND_REFERENCE_FACTORY_REPORT.md"

SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

EXPECTED = [
    CONTRACT, SS_TENSOR, MFM, DENOISE_EMB, DENOISE_METRICS, CONTRASTIVE, REP_QUALITY, EMB_CAND,
    GR_TARGETS, GR_ATTEMPTS, GR_CANDS, PATCH_LINKS, QA_QUEUE, LABEL_EXEC, GATE_RECHECK, EXP_REG,
    CARDS, OUT / "susc_18c_score_integrity_audit.csv", LEAKAGE, SUMMARY, BLOCKERS, REPORT,
]


def rc(path):
    with path.open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def rj(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_script(name, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / name), *args],
                          cwd=ROOT, text=True, capture_output=True, timeout=600, check=False)


def test_18a_18b_commitados():
    for p in ("susc_18a_multimodal_patch_feature_store.csv", "susc_18b_ml_feature_tensor.csv"):
        r = subprocess.run(["git", "cat-file", "-e", f"HEAD:outputs_public/suscetibilidade/{p}"],
                           cwd=ROOT, text=True, capture_output=True, check=False)
        assert r.returncode == 0, f"{p} precisa estar commitado"


def test_build_offline_byte_identico_e_valida():
    r = run_script("build_susc_18c_self_supervised_reference_factory.py")
    assert r.returncode == 0, r.stderr + r.stdout
    first = {p: p.read_bytes() for p in EXPECTED}
    r = run_script("build_susc_18c_self_supervised_reference_factory.py")
    assert r.returncode == 0, r.stderr + r.stdout
    assert {p: p.read_bytes() for p in EXPECTED} == first
    r = run_script("validate_susc_18c_self_supervised_reference_factory.py")
    assert r.returncode == 0, r.stderr + r.stdout


def test_dataset_e_tensor_consumidos():
    s = rj(SUMMARY)
    assert s["feature_tensor_rows_consumed_count"] >= 316
    assert len(rc(SS_TENSOR)) >= 316


def test_self_supervised_experimentos():
    s = rj(SUMMARY)
    assert s["self_supervised_experiment_count"] >= 5
    assert len(rc(MFM)) >= 5
    assert len(rc(DENOISE_EMB)) >= 316
    assert len(rc(REP_QUALITY)) >= 5


def test_contrastive_similarity():
    assert len(rc(CONTRASTIVE)) >= 3160


def test_embedding_candidatos_e_targets():
    assert len(rc(EMB_CAND)) >= 75
    assert len(rc(GR_TARGETS)) >= 75


def test_acquisition_attempts_e_candidates():
    assert len(rc(GR_ATTEMPTS)) >= 50
    assert len(rc(GR_CANDS)) >= 1
    assert len(rc(PATCH_LINKS)) >= 1
    assert len(rc(QA_QUEUE)) >= 1


def test_ground_reference_candidato_nao_e_ground_truth():
    for r in rc(GR_CANDS):
        assert r["eligible_for_training"] == "false"
        assert r["eligible_for_17b"] == "false"
        assert r["qa_status"] == "pending"


def test_label_contract_e_gate_recheck():
    assert LABEL_EXEC.exists() and rj(SUMMARY)["label_contract_execution_created"] is True
    assert GATE_RECHECK.exists() and rj(SUMMARY)["supervised_gate_rechecked"] is True
    lce = rc(LABEL_EXEC)[0]
    assert lce["label_contract_training_allowed"] == "false"


def test_supervised_training_nao_executado():
    s = rj(SUMMARY)
    assert s["supervised_training_executed"] is False
    for r in rc(GATE_RECHECK):
        assert r["supervised_training_executed"] == "false"


def test_canario_nao_positivo_controle_nao_negativo():
    s = rj(SUMMARY)
    assert s["canary_used_as_positive_label"] is False
    assert s["control_used_as_negative_label"] is False
    for r in rc(LEAKAGE):
        assert r["uses_canary_as_positive_label"] == "false"
        assert r["uses_control_as_negative_label"] == "false"


def test_score_v6_nao_target_ocorrencia_nao_causal():
    s = rj(SUMMARY)
    assert s["score_v6_used_as_training_target"] is False
    assert s["occurrence_used_as_causal_feature"] is False
    for r in rc(LEAKAGE):
        assert r["uses_score_v6_as_target"] == "false"
        assert r["uses_occurrence_as_feature"] == "false"


def test_flow_acc_nao_equivalente():
    assert rj(SUMMARY)["flow_acc_treated_as_equivalent"] is False
    for r in rc(LEAKAGE):
        assert r["uses_flow_acc_as_equivalent"] == "false"


def test_post_evento_nao_pre_evento():
    for r in rc(LEAKAGE):
        assert r["uses_post_event_feature_as_pre_event"] == "false"


def test_score_v6_intacto_v7_inexistente():
    r = subprocess.run(["git", "diff", "--name-only", "--", "datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv"],
                       cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.stdout.strip() == ""
    assert rj(SUMMARY)["score_v6_original_file_changed"] is False
    assert not SCORE_V7.exists()
    assert rj(SUMMARY)["score_v7_created"] is False


def test_ground_truth_e_labels_nao_criados():
    s = rj(SUMMARY)
    assert s["ground_truth_created"] is False
    assert s["training_labels_created"] is False


def test_17b_bloqueado():
    s = rj(SUMMARY)
    assert s["eligible_for_17b_now"] is False
    types = {r["blocker_type"] for r in rc(BLOCKERS)}
    assert "17b_blocked" in types
    assert "supervised_training_blocked_no_accepted_ground_reference" in types


def test_no_leakage_passa():
    leak = rc(LEAKAGE)
    assert len(leak) > 0
    for r in leak:
        assert r["passes_no_leakage"] == "true"
        assert r["ground_reference_candidate_treated_as_ground_truth"] == "false"


def test_minimum_success_achieved():
    assert rj(SUMMARY)["minimum_success_achieved"] is True
