"""Testes do SUSC-17C12 - orquestrador de submissao assistida."""

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
    OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C12.md",
    OUT / "SUSC_17C12_ORQUESTRADOR_SUBMISSAO_ASSISTIDA_E_INTAKE_OPERACIONAL_REPORT.md",
    OUT / "susc_17c12_operational_request_queue.csv",
    OUT / "susc_17c12_submission_package_registry.csv",
    OUT / "susc_17c12_agent_action_plan.csv",
    OUT / "susc_17c12_prepared_messages.md",
    OUT / "susc_17c12_submission_status_registry.csv",
    OUT / "susc_17c12_submission_attempt_log.csv",
    OUT / "susc_17c12_response_intake_registry.csv",
    OUT / "susc_17c12_response_manifest_template.csv",
    OUT / "susc_17c12_operational_risk_policy.json",
    OUT / "susc_17c12_readiness_summary.json",
    OUT / "susc_17c12_promotion_blockers.csv",
    SCHEMAS / "susc_17c12_operational_request_queue_schema_v1.json",
    SCHEMAS / "susc_17c12_submission_status_schema_v1.json",
    SCHEMAS / "susc_17c12_response_intake_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c12_submission_orchestrator_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c12_common", path)
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
    result = run_script("build_susc_17c12_submission_orchestrator.py")
    assert result.returncode == 0, result.stderr + result.stdout
    for path in EXPECTED:
        assert path.exists(), path
    result = run_script("validate_susc_17c12_submission_orchestrator.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_validos():
    s17c12 = _load_common()
    queue_schema = json.loads((SCHEMAS / "susc_17c12_operational_request_queue_schema_v1.json").read_text(encoding="utf-8"))
    status_schema = json.loads((SCHEMAS / "susc_17c12_submission_status_schema_v1.json").read_text(encoding="utf-8"))
    intake_schema = json.loads((SCHEMAS / "susc_17c12_response_intake_schema_v1.json").read_text(encoding="utf-8"))
    for row in read_csv(OUT / "susc_17c12_operational_request_queue.csv"):
        assert s17c12._schema_violations(row, queue_schema) == []
    for row in read_csv(OUT / "susc_17c12_submission_status_registry.csv"):
        assert s17c12._schema_violations(row, status_schema) == []
    for row in read_csv(OUT / "susc_17c12_response_intake_registry.csv"):
        assert s17c12._schema_violations(row, intake_schema) == []


def test_build_duas_vezes_byte_identico_e_ids_deterministicos():
    first = tuple(path.read_bytes() for path in EXPECTED)
    result = run_script("build_susc_17c12_submission_orchestrator.py")
    assert result.returncode == 0, result.stderr + result.stdout
    second = tuple(path.read_bytes() for path in EXPECTED)
    assert first == second
    queue_ids = [row["queue_id"] for row in read_csv(OUT / "susc_17c12_operational_request_queue.csv")]
    assert queue_ids == sorted(queue_ids)
    assert len(queue_ids) == len(set(queue_ids))


def test_fila_operacional_cobre_9_e_separa_status_de_canais():
    queue = read_csv(OUT / "susc_17c12_operational_request_queue.csv")
    summary = json.loads((OUT / "susc_17c12_readiness_summary.json").read_text(encoding="utf-8"))
    assert len(queue) == 9
    assert summary["operational_queue_rows_count"] == 9
    assert summary["submission_packages_count"] == 9
    assert summary["prepared_messages_count"] == 9
    assert summary["requests_ready_to_prepare_count"] == 5
    assert summary["requests_ready_for_manual_submission_count"] == 0
    assert summary["candidate_channel_requests_count"] == 2
    assert summary["blocked_no_channel_requests_count"] == 2
    assert len([row for row in queue if row["operational_status"] == "ready_to_prepare"]) == 5
    assert len([row for row in queue if row["operational_status"] == "needs_manual_channel_verification"]) == 2
    assert len([row for row in queue if row["operational_status"] == "blocked_no_official_channel"]) == 2


def test_sem_envio_protocolo_resposta_contato_feature_patch_ou_score_v7():
    summary = json.loads((OUT / "susc_17c12_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["requests_marked_sent_count"] == 0
    assert summary["protocols_opened_count"] == 0
    assert summary["responses_received_count"] == 0
    assert summary["contacts_invented_count"] == 0
    assert summary["features_extracted_this_sprint"] == 0
    assert summary["official_patch_created"] is False
    assert summary["official_patch_link_created"] is False
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
    assert not (ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv").exists()


def test_governanca_e_intake_inicial_sem_hash_path_inventado():
    for path in (
        OUT / "susc_17c12_operational_request_queue.csv",
        OUT / "susc_17c12_submission_package_registry.csv",
        OUT / "susc_17c12_submission_status_registry.csv",
        OUT / "susc_17c12_response_intake_registry.csv",
    ):
        for row in read_csv(path):
            assert row["review_only"] == "true"
            assert row["trainable"] == "false"
            assert row["ground_truth"] == "false"
    for row in read_csv(OUT / "susc_17c12_submission_status_registry.csv"):
        assert row["submitted_manually"] == "false"
        assert row["protocol_number"] == "not_informed"
        assert row["response_received"] == "false"
    for row in read_csv(OUT / "susc_17c12_response_intake_registry.csv"):
        assert row["raw_response_storage_path"] == "not_available"
        assert row["raw_response_sha256"] == "not_available"
        assert row["eligible_for_future_ingestion"] == "false"


def test_submit_assisted_bloqueado_por_padrao_e_opt_in_exigido():
    policy = json.loads((OUT / "susc_17c12_operational_risk_policy.json").read_text(encoding="utf-8"))
    assert policy["default_dry_run"] is True
    assert policy["allow_external_submission"] is False
    assert policy["allow_browser_action"] is False
    assert policy["allow_email_send"] is False
    assert policy["allow_form_submit"] is False
    assert policy["required_env_for_external_action"] == "SUSC_17C12_ALLOW_EXTERNAL_ACTION=1"
    actions = read_csv(OUT / "susc_17c12_agent_action_plan.csv")
    submit = [row for row in actions if row["action_mode"] == "submit-assisted"]
    assert len(submit) == 9
    assert all(row["allowed_by_default"] == "false" for row in submit)
    assert all(row["requires_env_opt_in"] == "true" for row in submit)
    result = run_script(
        "susc_17c12_submission_orchestrator.py",
        "submit-assisted",
        "--request-id",
        "P0_DRM_RJ_PKG_FR_PET_001",
    )
    assert result.returncode == 0
    assert "dry_run_only" in result.stdout
    result = run_script(
        "susc_17c12_submission_orchestrator.py",
        "submit-assisted",
        "--request-id",
        "P0_DRM_RJ_PKG_FR_PET_001",
        "--dry-run=false",
    )
    assert result.returncode == 2
    assert "blocked_missing_opt_in" in result.stdout


def test_mensagens_em_portugues_e_modos_cli_sem_efeito_externo():
    text = (OUT / "susc_17c12_prepared_messages.md").read_text(encoding="utf-8")
    assert text.count("## S17C12_MSG_") == 9
    assert "Prezadas(os)" in text
    assert "Nenhuma mensagem foi enviada" in text
    assert "nao deve ser interpretada como validacao operacional" in text
    plan = run_script("susc_17c12_submission_orchestrator.py", "plan")
    assert plan.returncode == 0
    assert "Fila operacional SUSC-17C12" in plan.stdout
    status = run_script("susc_17c12_submission_orchestrator.py", "status")
    assert status.returncode == 0
    assert "Status operacional SUSC-17C12" in status.stdout
    audit = run_script("susc_17c12_submission_orchestrator.py", "audit")
    assert audit.returncode == 0
    assert "17C12 ->" in audit.stdout


def test_datasets_oficiais_e_score_v6_intactos():
    for path in (
        "datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv",
        "datasets/suscetibilidade/susc_features_by_patch_v1.csv",
        "datasets/suscetibilidade/susc_patches_official_v1.csv",
        "datasets/suscetibilidade/susc_patch_links_official_v1.csv",
    ):
        changed = subprocess.run(["git", "diff", "--name-only", "--", path], cwd=ROOT, text=True, capture_output=True, check=False)
        assert changed.stdout.strip() == ""
