"""Testes do SUSC-17C14 - agente de coleta oficial e intake condicionado."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

EXPECTED = [
    OUT / "SUSC_17C14_AGENTE_COLETA_OFICIAL_INTAKE_CONDICIONADO_REPORT.md",
    OUT / "susc_17c14_collection_request_queue.csv",
    OUT / "susc_17c14_collection_attempt_log.csv",
    OUT / "susc_17c14_official_response_intake_registry.csv",
    OUT / "susc_17c14_artifact_manifest.csv",
    OUT / "susc_17c14_gate_evaluation_matrix.csv",
    OUT / "susc_17c14_spatial_validation_status.csv",
    OUT / "susc_17c14_temporal_validation_status.csv",
    OUT / "susc_17c14_phenomenon_separation_status.csv",
    OUT / "susc_17c14_accepted_ground_reference_candidates.csv",
    OUT / "susc_17c14_rejected_or_pending_artifacts.csv",
    OUT / "susc_17c14_anti_leakage_audit.csv",
    OUT / "susc_17c14_readiness_summary.json",
    OUT / "susc_17c14_promotion_blockers.csv",
    SCHEMAS / "susc_17c14_gate_evaluation_schema_v1.json",
    SCHEMAS / "susc_17c14_official_response_intake_schema_v1.json",
    SCHEMAS / "susc_17c14_ground_reference_candidate_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c14_official_collection_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c14_collection_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str, *args: str):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )


def test_build_gera_todos_artefatos_e_valida():
    result = run_script("build_susc_17c14_official_collection_agent.py")
    assert result.returncode == 0, result.stderr + result.stdout
    for path in EXPECTED:
        assert path.exists(), path
    result = run_script("validate_susc_17c14_official_collection_agent.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_validos():
    s17c14 = _load_common()
    gate_schema = json.loads((SCHEMAS / "susc_17c14_gate_evaluation_schema_v1.json").read_text(encoding="utf-8"))
    intake_schema = json.loads((SCHEMAS / "susc_17c14_official_response_intake_schema_v1.json").read_text(encoding="utf-8"))
    candidate_schema = json.loads((SCHEMAS / "susc_17c14_ground_reference_candidate_schema_v1.json").read_text(encoding="utf-8"))
    for row in read_csv(OUT / "susc_17c14_gate_evaluation_matrix.csv"):
        assert s17c14._schema_violations(row, gate_schema) == []
    for row in read_csv(OUT / "susc_17c14_official_response_intake_registry.csv"):
        assert s17c14._schema_violations(row, intake_schema) == []
    for row in read_csv(OUT / "susc_17c14_accepted_ground_reference_candidates.csv"):
        assert s17c14._schema_violations(row, candidate_schema) == []


def test_build_duas_vezes_byte_identico_e_ids_deterministicos():
    first = tuple(path.read_bytes() for path in EXPECTED)
    result = run_script("build_susc_17c14_official_collection_agent.py")
    assert result.returncode == 0, result.stderr + result.stdout
    second = tuple(path.read_bytes() for path in EXPECTED)
    assert first == second
    for path, key in [
        (OUT / "susc_17c14_collection_request_queue.csv", "collection_queue_id"),
        (OUT / "susc_17c14_collection_attempt_log.csv", "attempt_id"),
        (OUT / "susc_17c14_gate_evaluation_matrix.csv", "gate_evaluation_id"),
        (OUT / "susc_17c14_rejected_or_pending_artifacts.csv", "rejected_or_pending_id"),
    ]:
        ids = [row[key] for row in read_csv(path)]
        assert ids == sorted(ids)
        assert len(ids) == len(set(ids))


def test_gates_bloqueiam_ausencia_de_artefato_hash_temporal_espacial_fenomeno():
    gates = read_csv(OUT / "susc_17c14_gate_evaluation_matrix.csv")
    for gate_id in ("G1", "G3", "G4", "G5", "G6"):
        rows = [row for row in gates if row["gate_id"] == gate_id]
        assert rows
        assert all(row["gate_pass"] == "false" for row in rows)
    assert all(row["gate_pass"] == "false" for row in gates if row["gate_id"] == "G2" and row["source_is_official"] == "false")


def test_gate_g7_e_anti_leakage_nao_criam_feature_pos_evento():
    gates = [row for row in read_csv(OUT / "susc_17c14_gate_evaluation_matrix.csv") if row["gate_id"] == "G7"]
    assert gates
    assert all(row["gate_pass"] == "true" for row in gates)
    for row in read_csv(OUT / "susc_17c14_anti_leakage_audit.csv"):
        assert row["passes"] == "true"
        assert row["violation_found"] == "false"


def test_pdf_sem_geometria_nao_vira_ground_reference_e_spatial_bloqueado():
    spatial = read_csv(OUT / "susc_17c14_spatial_validation_status.csv")
    assert all(row["spatial_validation_status"] == "geometry_missing_blocked" for row in spatial)
    assert read_csv(OUT / "susc_17c14_accepted_ground_reference_candidates.csv") == []


def test_resposta_valida_ainda_nao_vira_ground_truth_no_build_inicial():
    intake = read_csv(OUT / "susc_17c14_official_response_intake_registry.csv")
    assert all(row["response_received"] == "false" for row in intake)
    assert all(row["ground_truth"] == "false" for row in intake)
    assert all(row["trainable"] == "false" for row in intake)
    assert all(row["review_only"] == "true" for row in intake)


def test_accepted_candidates_exigem_todos_os_gates_e_rejected_contem_falhas():
    accepted = read_csv(OUT / "susc_17c14_accepted_ground_reference_candidates.csv")
    assert accepted == []
    rejected = read_csv(OUT / "susc_17c14_rejected_or_pending_artifacts.csv")
    assert len(rejected) == 9
    assert all("G1" in row["failed_gates"] and "G6" in row["failed_gates"] for row in rejected)


def test_cli_modos_seguros():
    queue = run_script("susc_17c14_official_collection_agent.py", "queue")
    assert queue.returncode == 0
    assert "collection_ready_dry_run_only" in queue.stdout
    collect = run_script("susc_17c14_official_collection_agent.py", "collect")
    assert collect.returncode == 0
    assert "dry-run" in collect.stdout
    submit = run_script("susc_17c14_official_collection_agent.py", "submit", "--request-id", "P0_DRM_RJ_PKG_FR_PET_001", "--dry-run=false")
    assert submit.returncode == 2
    assert "blocked_guardrail" in submit.stdout
    audit = run_script("susc_17c14_official_collection_agent.py", "audit")
    assert audit.returncode == 0


def test_summary_guardrails_score_v6_score_v7_17b():
    summary = json.loads((OUT / "susc_17c14_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["formal_requests_count"] == 9
    assert summary["collection_attempts_count"] == 9
    assert summary["artifacts_ingested_count"] == 0
    assert summary["artifacts_all_gates_passed_count"] == 0
    assert summary["accepted_ground_reference_candidates_count"] == 0
    assert summary["rejected_or_pending_artifacts_count"] == 9
    assert summary["ground_truth_created"] is False
    assert summary["training_label_created"] is False
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False


def test_datasets_oficiais_intactos_e_sem_score_v7():
    assert not (ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    for path in (
        "datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv",
        "datasets/suscetibilidade/susc_features_by_patch_v1.csv",
        "datasets/suscetibilidade/susc_patches_official_v1.csv",
        "datasets/suscetibilidade/susc_patch_links_official_v1.csv",
    ):
        changed = subprocess.run(["git", "diff", "--name-only", "--", path], cwd=ROOT, text=True, capture_output=True, check=False)
        assert changed.stdout.strip() == ""
