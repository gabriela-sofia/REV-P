"""Testes do SUSC-17C30 - Gate Promotion Engine, Source Fingerprint e Replication Pipeline."""

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
OFFICIAL_LEVELS = {"official", "official_institutional", "official_institutional_public_agency"}

REPORT = OUT / "SUSC_17C30_GATE_PROMOTION_SOURCE_FINGERPRINT_REPLICATION_ENGINE_REPORT.md"
FINGERPRINT = OUT / "susc_17c30_source_quality_fingerprint.csv"
REFERENCE_PROFILE = OUT / "susc_17c30_reference_source_profile_agencia_brasil_ebc.json"
SCORING_POLICY = OUT / "susc_17c30_candidate_source_scoring_policy.json"
CLASSIFICATION = OUT / "susc_17c30_source_quality_classification.csv"
DOC_GATE = OUT / "susc_17c30_documentary_gate_evaluation.csv"
G4_SG = OUT / "susc_17c30_g4_subgate_evaluation.csv"
G5_SG = OUT / "susc_17c30_g5_subgate_evaluation.csv"
PROMOTION = OUT / "susc_17c30_gate_promotion_decisions.csv"
LEDGER = OUT / "susc_17c30_gate_transition_ledger.csv"
SUBGATE_MATRIX = OUT / "susc_17c30_subgate_matrix.csv"
MAPPING = OUT / "susc_17c30_source_to_event_candidate_mapping.csv"
ACQ = OUT / "susc_17c30_candidate_acquisition_queue.csv"
MMFD = OUT / "susc_17c30_multimodal_followup_decision.csv"
DOC_EVIDENCE = OUT / "susc_17c30_documentary_evidence_registry.csv"
EVENT_RECORD = OUT / "susc_17c30_event_record_candidate_registry.csv"
REJECTIONS = OUT / "susc_17c30_source_rejection_registry.csv"
BLUEPRINT = OUT / "susc_17c30_replication_pipeline_blueprint.json"
LEAKAGE = OUT / "susc_17c30_no_leakage_audit.csv"
GATE_MATRIX = OUT / "susc_17c30_gate_evaluation_matrix.csv"
SUMMARY_PATH = OUT / "susc_17c30_readiness_summary.json"
BLOCKERS = OUT / "susc_17c30_promotion_blockers.csv"

EXPECTED = [
    REPORT, FINGERPRINT, REFERENCE_PROFILE, SCORING_POLICY, CLASSIFICATION,
    DOC_GATE, G4_SG, G5_SG, PROMOTION, LEDGER, SUBGATE_MATRIX, MAPPING,
    ACQ, MMFD, DOC_EVIDENCE, EVENT_RECORD, REJECTIONS, BLUEPRINT,
    LEAKAGE, GATE_MATRIX, SUMMARY_PATH, BLOCKERS,
]


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str, *args: str):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=ROOT, text=True, capture_output=True, timeout=600, check=False,
    )


def test_build_offline_byte_identico_e_valida():
    result = run_script("build_susc_17c30_gate_promotion_replication.py")
    assert result.returncode == 0, result.stderr + result.stdout
    first = {p: p.read_bytes() for p in EXPECTED}
    result = run_script("build_susc_17c30_gate_promotion_replication.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {p: p.read_bytes() for p in EXPECTED} == first
    result = run_script("validate_susc_17c30_gate_promotion_replication.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_reference_source_profile_agencia_brasil_ebc():
    profile = json.loads(REFERENCE_PROFILE.read_text(encoding="utf-8"))
    assert profile["reference_source_name"] == "Agencia Brasil / EBC"
    assert profile["source_type"] == "official_institutional_public_agency"
    assert profile["authority_tier"] == "institutional_public"
    assert "documentary_evidence" in profile["allowed_uses"]
    assert "ground_truth" in profile["blocked_uses"]
    assert "training_label" in profile["blocked_uses"]
    assert "17B_unlock" in profile["blocked_uses"]
    assert profile["review_only"] is True
    assert profile["ground_truth"] is False


def test_scoring_policy_10_sinais_e_6_classes():
    policy = json.loads(SCORING_POLICY.read_text(encoding="utf-8"))
    assert len(policy["signals"]) == 10
    class_codes = {c["class_code"] for c in policy["classes"]}
    assert class_codes == {"A", "B", "C", "D", "E", "F"}


def test_fingerprint_pelo_menos_10_fontes_classificadas():
    fp = read_csv_rows(FINGERPRINT)
    assert len(fp) >= 10
    classes = {row["source_quality_class"] for row in fp}
    assert classes.issubset({
        "A_institutional_event_candidate",
        "B_official_context_candidate",
        "C_technical_sensor_candidate",
        "D_documentary_support_only",
        "E_weak_news_context",
        "F_rejected",
    })


def test_documentary_gates_g1_g2_g3_g6_g7_abrem():
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    for gate in ("G1_true_count", "G2_true_count", "G3_true_count", "G6_true_count", "G7_true_count"):
        assert summary[gate] > 0, gate
    assert summary["documentary_gates_opened_count"] > 0


def test_g4_subgates_abrem_sem_g4_full():
    g4 = read_csv_rows(G4_SG)
    assert any(row["G4a_location_text_present"] == "true" for row in g4)
    assert any(row["G4b_geocodable_locality_present"] == "true" for row in g4)
    assert all(row["G4_full_spatial_patch_link"] == "false" for row in g4)


def test_g5_subgates_abrem_sem_g5_full():
    g5 = read_csv_rows(G5_SG)
    assert any(row["G5a_hydrological_signal_present"] == "true" or row["G5b_mass_movement_signal_present"] == "true" for row in g5)
    assert any(row["G5c_mixed_phenomenon_detected"] == "true" for row in g5)
    assert all(row["G5_full_phenomenon_separation"] == "false" for row in g5)


def test_bairro_cidade_nao_vira_g4_full():
    g4 = read_csv_rows(G4_SG)
    for row in g4:
        if row["G4_full_spatial_patch_link"] == "true":
            assert row["G4c_patch_level_geometry_present"] == "true"
    # bairro-level uncertainty nunca resulta em G4_full
    for row in g4:
        if row["uncertainty_m"].startswith(("3000", "10000")):
            assert row["G4_full_spatial_patch_link"] == "false"


def test_fenomeno_misto_nao_vira_g5_full():
    g5 = read_csv_rows(G5_SG)
    for row in g5:
        if row["G5c_mixed_phenomenon_detected"] == "true":
            assert row["G5_full_phenomenon_separation"] == "false"


def test_fonte_adquirida_exige_sha256():
    fp = read_csv_rows(FINGERPRINT)
    for row in fp:
        if row["artifact_acquired"] == "true":
            assert row["sha256"] and len(row["sha256"]) >= 40


def test_classe_A_exige_autoridade_aquisicao_parse_event_specific():
    fp = read_csv_rows(FINGERPRINT)
    for row in fp:
        if row["source_quality_class"] == "A_institutional_event_candidate":
            assert int(row["source_authority_score"]) >= 2
            assert int(row["artifact_acquisition_score"]) >= 2
            assert int(row["parseability_score"]) >= 2
            assert int(row["event_specificity_score"]) >= 2
            assert int(row["total_source_quality_score"]) >= 23


def test_nenhuma_fonte_vira_ground_truth_ou_treino():
    fp = read_csv_rows(FINGERPRINT)
    for row in fp:
        assert row["eligible_for_ground_truth"] == "false"
        assert row["eligible_for_training"] == "false"
        assert row["eligible_for_score_v7"] == "false"
    mapping = read_csv_rows(MAPPING)
    for row in mapping:
        assert row["can_be_ground_truth"] == "false"
        assert row["can_be_training_label"] == "false"
        assert row["can_unlock_17b"] == "false"


def test_nenhuma_fonte_libera_score_v7_ou_17b():
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["sources_eligible_for_score_v7_count"] == 0
    assert summary["sources_eligible_for_ground_truth_count"] == 0
    assert summary["sources_eligible_for_training_count"] == 0
    assert summary["eligible_for_17b_now"] is False


def test_noticia_comercial_nao_vira_ground_reference():
    fp = read_csv_rows(FINGERPRINT)
    for row in fp:
        if row["source_quality_class"] == "E_weak_news_context":
            assert row["eligible_for_ground_reference_candidate"] == "false"
    # sem noticia comercial neste marco (todas as fontes sao official_institutional_public_agency),
    # mas o guardrail estrutural precisa estar presente para futuras aquisicoes
    for row in fp:
        assert row["eligible_for_ground_reference_candidate"] == "false"


def test_followup_multimodal_diferencia_documento_de_tile():
    mmfd = read_csv_rows(MMFD)
    for row in mmfd:
        # embedding nunca recebe noticia como input
        assert row["followup_embedding"] == "false"
        assert row["blocked_from_ground_truth"] == "true"
        assert row["blocked_from_training"] == "true"
        assert row["blocked_from_score_v7"] == "true"


def test_chirps_sentinel_nao_viram_evento():
    leakage = read_csv_rows(LEAKAGE)
    for row in leakage:
        assert row["uses_chirps_as_event_reference"] == "false"
        assert row["uses_sensor_as_event_observation"] == "false"
        assert row["uses_embedding_as_label"] == "false"
        assert row["uses_news_as_ground_reference"] == "false"
        assert row["uses_documentary_source_as_ground_truth"] == "false"
        assert row["passes_no_leakage"] == "true"


def test_score_v6_intacto_e_score_v7_inexistente():
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
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()


def test_fila_aquisicao_e_mappings_e_decisoes_multimodais_minimas():
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["candidate_acquisition_queue_rows_count"] >= 30
    assert summary["source_to_event_candidate_mapping_rows_count"] >= 10
    assert summary["multimodal_followup_decision_rows_count"] >= 10


def test_promotion_blockers_expressivos():
    blockers = read_csv_rows(BLOCKERS)
    types = {row["blocker_type"] for row in blockers}
    for required in (
        "G4_full_blocked_no_patch_level_geometry",
        "G4_full_blocked_no_patch_buffer_link",
        "G5_full_blocked_mixed_phenomenon",
        "G5_full_blocked_no_hydrological_separation",
        "bairro_only_not_patch_link",
        "news_context_not_ground_reference",
        "embedding_not_label",
        "chirps_not_event_reference",
        "sentinel_not_event_reference",
        "score_v7_blocked",
        "17b_blocked",
    ):
        assert required in types, required
