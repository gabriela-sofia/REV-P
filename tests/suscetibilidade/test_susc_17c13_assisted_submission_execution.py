"""Testes do SUSC-17C13 - execucao assistida de submissoes manuais."""

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
    OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C13.md",
    OUT / "SUSC_17C13_EXECUCAO_ASSISTIDA_SUBMISSOES_MANUAIS_REPORT.md",
    OUT / "susc_17c13_orchestrator_run_log.csv",
    OUT / "susc_17c13_prepared_submission_packages.csv",
    OUT / "susc_17c13_channel_open_instructions.csv",
    OUT / "susc_17c13_manual_execution_board.csv",
    OUT / "susc_17c13_submission_status_snapshot.csv",
    OUT / "susc_17c13_submission_evidence_requirements.csv",
    OUT / "susc_17c13_pending_channel_verification_queue.csv",
    OUT / "susc_17c13_response_intake_watchlist.csv",
    OUT / "susc_17c13_operational_safety_audit.csv",
    OUT / "susc_17c13_readiness_summary.json",
    OUT / "susc_17c13_promotion_blockers.csv",
    SCHEMAS / "susc_17c13_orchestrator_run_schema_v1.json",
    SCHEMAS / "susc_17c13_execution_board_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c13_assisted_submission_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c13_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )


def test_build_gera_todos_artefatos_e_valida():
    result = run_script("build_susc_17c13_assisted_submission_execution.py")
    assert result.returncode == 0, result.stderr + result.stdout
    for path in EXPECTED:
        assert path.exists(), path
    result = run_script("validate_susc_17c13_assisted_submission_execution.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_validos():
    s17c13 = _load_common()
    run_schema = json.loads((SCHEMAS / "susc_17c13_orchestrator_run_schema_v1.json").read_text(encoding="utf-8"))
    board_schema = json.loads((SCHEMAS / "susc_17c13_execution_board_schema_v1.json").read_text(encoding="utf-8"))
    for row in read_csv(OUT / "susc_17c13_orchestrator_run_log.csv"):
        assert s17c13._schema_violations(row, run_schema) == []
    for row in read_csv(OUT / "susc_17c13_manual_execution_board.csv"):
        assert s17c13._schema_violations(row, board_schema) == []


def test_build_duas_vezes_byte_identico_e_ids_deterministicos():
    first = tuple(path.read_bytes() for path in EXPECTED)
    result = run_script("build_susc_17c13_assisted_submission_execution.py")
    assert result.returncode == 0, result.stderr + result.stdout
    second = tuple(path.read_bytes() for path in EXPECTED)
    assert first == second
    for path, key in [
        (OUT / "susc_17c13_orchestrator_run_log.csv", "run_log_id"),
        (OUT / "susc_17c13_prepared_submission_packages.csv", "prepared_package_id"),
        (OUT / "susc_17c13_manual_execution_board.csv", "execution_task_id"),
    ]:
        ids = [row[key] for row in read_csv(path)]
        assert ids == sorted(ids)
        assert len(ids) == len(set(ids))


def test_contagens_operacionais_e_pacotes_confirmados():
    summary = json.loads((OUT / "susc_17c13_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["formal_requests_count"] == 9
    assert summary["orchestrator_runs_logged_count"] == 15
    assert summary["prepared_packages_count"] == 9
    assert summary["copy_paste_ready_packages_count"] == 5
    assert summary["channel_open_instructions_count"] == 7
    assert summary["manual_execution_tasks_count"] == 90
    assert summary["ready_for_manual_submission_after_review_count"] == 5
    assert summary["needs_manual_channel_verification_count"] == 2
    assert summary["blocked_no_channel_count"] == 2
    packages = read_csv(OUT / "susc_17c13_prepared_submission_packages.csv")
    assert len([row for row in packages if row["copy_paste_ready"] == "true"]) == 5
    assert all(row["copy_paste_ready"] == "false" for row in packages if row["channel_status"] != "confirmed_official_source")


def test_sem_envio_protocolo_resposta_contato_efeito_externo_feature_patch_ou_score_v7():
    summary = json.loads((OUT / "susc_17c13_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["requests_marked_sent_count"] == 0
    assert summary["protocols_opened_count"] == 0
    assert summary["responses_received_count"] == 0
    assert summary["external_effects_count"] == 0
    assert summary["contacts_invented_count"] == 0
    assert summary["features_extracted_this_sprint"] == 0
    assert summary["official_patch_created"] is False
    assert summary["official_patch_link_created"] is False
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
    assert not (ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv").exists()


def test_run_log_sem_efeito_externo_e_submit_assisted_nao_executado():
    rows = read_csv(OUT / "susc_17c13_orchestrator_run_log.csv")
    modes = {row["orchestrator_mode"] for row in rows}
    assert modes == {"plan", "status", "audit", "prepare", "open-channel"}
    assert "submit-assisted" not in modes
    assert all(row["executed"] == "true" for row in rows)
    assert all(row["external_effect"] == "false" for row in rows)
    assert all(row["request_marked_sent"] == "false" for row in rows)
    assert all(row["protocol_opened"] == "false" for row in rows)
    assert all(row["response_received"] == "false" for row in rows)


def test_canais_candidatos_exigem_verificacao_e_ausentes_bloqueados():
    pending = read_csv(OUT / "susc_17c13_pending_channel_verification_queue.csv")
    assert len(pending) == 4
    assert len([row for row in pending if row["channel_status"] == "candidate_needs_manual_verification"]) == 2
    assert len([row for row in pending if row["channel_status"] == "not_found"]) == 2
    snapshot = read_csv(OUT / "susc_17c13_submission_status_snapshot.csv")
    assert len([row for row in snapshot if row["needs_manual_channel_verification"] == "true"]) == 2
    assert len([row for row in snapshot if row["blocked_no_channel"] == "true"]) == 2


def test_action_board_nao_concluido_e_watchlist_sem_resposta():
    for row in read_csv(OUT / "susc_17c13_manual_execution_board.csv"):
        assert row["done"] == "false"
        assert row["task_status"] == "pending"
        assert row["requires_human_action"] == "true"
    for row in read_csv(OUT / "susc_17c13_response_intake_watchlist.csv"):
        assert row["submission_required_first"] == "true"
        assert row["hash_required"] == "true"
        assert row["blocks_17b"] == "true"


def test_governanca_score_v6_e_datasets_oficiais_intactos():
    for path in [
        OUT / "susc_17c13_orchestrator_run_log.csv",
        OUT / "susc_17c13_prepared_submission_packages.csv",
        OUT / "susc_17c13_submission_status_snapshot.csv",
    ]:
        for row in read_csv(path):
            assert row["review_only"] == "true"
            assert row["trainable"] == "false"
            assert row["ground_truth"] == "false"
    for path in (
        "datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv",
        "datasets/suscetibilidade/susc_features_by_patch_v1.csv",
        "datasets/suscetibilidade/susc_patches_official_v1.csv",
        "datasets/suscetibilidade/susc_patch_links_official_v1.csv",
    ):
        changed = subprocess.run(["git", "diff", "--name-only", "--", path], cwd=ROOT, text=True, capture_output=True, check=False)
        assert changed.stdout.strip() == ""


def test_bloqueadores_e_relatorio_em_portugues():
    assert read_csv(OUT / "susc_17c13_promotion_blockers.csv")
    text = (OUT / "SUSC_17C13_EXECUCAO_ASSISTIDA_SUBMISSOES_MANUAIS_REPORT.md").read_text(encoding="utf-8")
    assert "Execucao assistida" in text
    assert "Nenhum envio foi simulado" in text
    assert "score v7 continua inexistente" in text
