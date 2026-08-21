"""Testes do SUSC-17C14 - verificacao manual de canais pendentes."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

EXPECTED = [
    OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C14.md",
    OUT / "SUSC_17C14_VERIFICACAO_MANUAL_CANAIS_PENDENTES_REPORT.md",
    OUT / "susc_17c14_pending_channel_reaudit.csv",
    OUT / "susc_17c14_official_channel_resolution.csv",
    OUT / "susc_17c14_channel_evidence_update.csv",
    OUT / "susc_17c14_submission_readiness_delta.csv",
    OUT / "susc_17c14_updated_channel_open_instructions.csv",
    OUT / "susc_17c14_manual_verification_board.csv",
    OUT / "susc_17c14_unresolved_channel_blockers.csv",
    OUT / "susc_17c14_operational_safety_audit.csv",
    OUT / "susc_17c14_readiness_summary.json",
    OUT / "susc_17c14_promotion_blockers.csv",
    SCHEMAS / "susc_17c14_channel_resolution_schema_v1.json",
    SCHEMAS / "susc_17c14_readiness_delta_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c14_channel_verification_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c14_common", path)
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
    result = run_script("build_susc_17c14_pending_channel_verification.py")
    assert result.returncode == 0, result.stderr + result.stdout
    for path in EXPECTED:
        assert path.exists(), path
    result = run_script("validate_susc_17c14_pending_channel_verification.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_validos():
    s17c14 = _load_common()
    resolution_schema = json.loads((SCHEMAS / "susc_17c14_channel_resolution_schema_v1.json").read_text(encoding="utf-8"))
    delta_schema = json.loads((SCHEMAS / "susc_17c14_readiness_delta_schema_v1.json").read_text(encoding="utf-8"))
    for row in read_csv(OUT / "susc_17c14_official_channel_resolution.csv"):
        assert s17c14._schema_violations(row, resolution_schema) == []
    for row in read_csv(OUT / "susc_17c14_submission_readiness_delta.csv"):
        assert s17c14._schema_violations(row, delta_schema) == []


def test_build_duas_vezes_byte_identico_e_ids_deterministicos():
    first = tuple(path.read_bytes() for path in EXPECTED)
    result = run_script("build_susc_17c14_pending_channel_verification.py")
    assert result.returncode == 0, result.stderr + result.stdout
    second = tuple(path.read_bytes() for path in EXPECTED)
    assert first == second
    for path, key in [
        (OUT / "susc_17c14_pending_channel_reaudit.csv", "reaudit_id"),
        (OUT / "susc_17c14_official_channel_resolution.csv", "resolution_id"),
        (OUT / "susc_17c14_submission_readiness_delta.csv", "readiness_delta_id"),
        (OUT / "susc_17c14_manual_verification_board.csv", "manual_verification_task_id"),
    ]:
        ids = [row[key] for row in read_csv(path)]
        assert ids == sorted(ids)
        assert len(ids) == len(set(ids))


def test_contagens_reauditoria_e_prontidao():
    summary = json.loads((OUT / "susc_17c14_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["pending_requests_reaudited_count"] == 4
    assert summary["official_channels_confirmed_this_sprint_count"] == 1
    assert summary["requests_became_ready_for_manual_submission_count"] == 1
    assert summary["requests_still_pending_manual_verification_count"] == 2
    assert summary["requests_still_blocked_no_channel_count"] == 1
    resolution = read_csv(OUT / "susc_17c14_official_channel_resolution.csv")
    assert len([row for row in resolution if row["resolved_status"] == "canal_oficial_confirmado"]) == 1
    assert len([row for row in resolution if row["resolved_status"] == "canal_candidato_ainda_pendente"]) == 1
    assert len([row for row in resolution if row["resolved_status"] == "sem_canal_oficial_encontrado"]) == 1
    assert len([row for row in resolution if row["resolved_status"] == "usar_canal_alternativo_institucional"]) == 1


def test_canal_confirmado_exige_fonte_oficial_e_prontidao_exige_confirmacao():
    for row in read_csv(OUT / "susc_17c14_official_channel_resolution.csv"):
        if row["resolved_status"] == "canal_oficial_confirmado":
            assert row["source_is_official"] == "true"
            assert row["ready_for_manual_submission_after_review"] == "true"
            assert row["still_blocked"] == "false"
        else:
            assert row["ready_for_manual_submission_after_review"] == "false"
            assert row["still_blocked"] == "true"
    for row in read_csv(OUT / "susc_17c14_channel_evidence_update.csv"):
        if row["evidence_is_official"] == "false":
            assert row["evidence_supports_resolution"] == "false"


def test_sem_envio_protocolo_resposta_contato_url_feature_patch_ou_score_v7():
    summary = json.loads((OUT / "susc_17c14_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["requests_marked_sent_count"] == 0
    assert summary["protocols_opened_count"] == 0
    assert summary["responses_received_count"] == 0
    assert summary["external_effects_count"] == 0
    assert summary["contacts_invented_count"] == 0
    assert summary["urls_invented_count"] == 0
    assert summary["features_extracted_this_sprint"] == 0
    assert summary["official_patch_created"] is False
    assert summary["official_patch_link_created"] is False
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
    assert not (ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv").exists()


def test_delta_nao_inventa_estado_e_instrucoes_nao_submetem():
    for row in read_csv(OUT / "susc_17c14_submission_readiness_delta.csv"):
        assert row["request_marked_sent"] == "false"
        assert row["protocol_opened"] == "false"
        assert row["response_received"] == "false"
    for row in read_csv(OUT / "susc_17c14_updated_channel_open_instructions.csv"):
        assert row["requires_manual_action"] == "true"
        assert row["can_be_opened_by_agent"] == "false"
        assert row["can_be_submitted_by_agent"] == "false"


def test_action_board_nao_concluido_e_safety_sem_violacao():
    board = read_csv(OUT / "susc_17c14_manual_verification_board.csv")
    assert len(board) == 32
    for row in board:
        assert row["done"] == "false"
        assert row["task_status"] == "pending"
        assert row["requires_human_action"] == "true"
    safety = read_csv(OUT / "susc_17c14_operational_safety_audit.csv")
    assert safety
    assert all(row["passes"] == "true" and row["violation_found"] == "false" for row in safety)


def test_governanca_score_v6_e_datasets_oficiais_intactos():
    for path in [
        OUT / "susc_17c14_pending_channel_reaudit.csv",
        OUT / "susc_17c14_official_channel_resolution.csv",
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
    assert read_csv(OUT / "susc_17c14_promotion_blockers.csv")
    assert read_csv(OUT / "susc_17c14_unresolved_channel_blockers.csv")
    text = (OUT / "SUSC_17C14_VERIFICACAO_MANUAL_CANAIS_PENDENTES_REPORT.md").read_text(encoding="utf-8")
    assert "Verificacao manual" in text
    assert "Nenhum envio foi feito" in text
    assert "score v7 continua bloqueado" in text
