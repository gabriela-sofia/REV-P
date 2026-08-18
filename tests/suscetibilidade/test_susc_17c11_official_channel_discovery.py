"""Testes do SUSC-17C11 - descoberta de canais oficiais."""

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
    OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C11.md",
    OUT / "SUSC_17C11_DESCOBERTA_CANAIS_OFICIAIS_REPORT.md",
    OUT / "susc_17c11_official_channel_registry.csv",
    OUT / "susc_17c11_channel_source_evidence.csv",
    OUT / "susc_17c11_request_channel_match_matrix.csv",
    OUT / "susc_17c11_manual_channel_verification_checklist.csv",
    OUT / "susc_17c11_submission_readiness_update.csv",
    OUT / "susc_17c11_contact_discovery_search_plan.csv",
    OUT / "susc_17c11_channel_risk_policy.json",
    OUT / "susc_17c11_readiness_summary.json",
    OUT / "susc_17c11_promotion_blockers.csv",
    SCHEMAS / "susc_17c11_channel_registry_schema_v1.json",
    SCHEMAS / "susc_17c11_request_channel_match_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c11_official_channel_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c11_common", path)
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
    result = run_script("build_susc_17c11_official_channel_discovery.py")
    assert result.returncode == 0, result.stderr + result.stdout
    for path in EXPECTED:
        assert path.exists(), path
    result = run_script("validate_susc_17c11_official_channel_discovery.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_validos():
    s17c11 = _load_common()
    channel_schema = json.loads((SCHEMAS / "susc_17c11_channel_registry_schema_v1.json").read_text(encoding="utf-8"))
    match_schema = json.loads((SCHEMAS / "susc_17c11_request_channel_match_schema_v1.json").read_text(encoding="utf-8"))
    for row in read_csv(OUT / "susc_17c11_official_channel_registry.csv"):
        assert s17c11._schema_violations(row, channel_schema) == []
    for row in read_csv(OUT / "susc_17c11_request_channel_match_matrix.csv"):
        assert s17c11._schema_violations(row, match_schema) == []


def test_build_duas_vezes_byte_identico_e_ids_deterministicos():
    first = tuple(path.read_bytes() for path in EXPECTED)
    result = run_script("build_susc_17c11_official_channel_discovery.py")
    assert result.returncode == 0, result.stderr + result.stdout
    second = tuple(path.read_bytes() for path in EXPECTED)
    assert first == second
    ids = [row["channel_id"] for row in read_csv(OUT / "susc_17c11_official_channel_registry.csv")]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_sem_envio_protocolo_contato_inventado_ou_fonte_nao_oficial_confirmada():
    summary = json.loads((OUT / "susc_17c11_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["requests_marked_sent_count"] == 0
    assert summary["responses_received_count"] == 0
    assert summary["external_protocols_opened_count"] == 0
    assert summary["external_forms_filled_count"] == 0
    assert summary["contacts_invented_count"] == 0
    assert summary["nonofficial_sources_confirmed_count"] == 0
    for row in read_csv(OUT / "susc_17c11_official_channel_registry.csv"):
        assert row["source_is_nonofficial"] == "false"
        assert row["contact_was_invented"] == "false"
        assert row["can_submit_automatically"] == "false"
        assert row["do_not_submit_automatically"] == "true"
    for row in read_csv(OUT / "susc_17c11_request_channel_match_matrix.csv"):
        assert row["should_submit_automatically"] == "false"
        assert row["request_marked_sent"] == "false"
        assert row["response_marked_received"] == "false"


def test_contagens_e_bloqueios_review_only():
    summary = json.loads((OUT / "susc_17c11_readiness_summary.json").read_text(encoding="utf-8"))
    registry = read_csv(OUT / "susc_17c11_official_channel_registry.csv")
    assert summary["formal_requests_count"] == 9
    assert summary["official_channel_registry_rows_count"] == len(registry)
    assert summary["confirmed_official_channels_count"] == 5
    assert summary["candidate_channels_needing_manual_verification_count"] == 2
    assert summary["not_found_channels_count"] == 2
    assert summary["requests_ready_after_manual_verification_count"] == 5
    assert summary["requests_still_needing_contact_channel_count"] == 4
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
    assert summary["review_only"] is True
    assert read_csv(OUT / "susc_17c11_promotion_blockers.csv")


def test_sem_score_v6_score_v7_features_patch_treino_ou_ground_truth():
    assert not (ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    summary = json.loads((OUT / "susc_17c11_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["features_extracted_this_sprint"] == 0
    assert summary["official_patch_created"] is False
    assert summary["official_patch_link_created"] is False
    assert summary["training_or_model_created"] is False
    assert summary["label_or_ground_truth_created"] is False
    for path in (
        "datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv",
        "datasets/suscetibilidade/susc_features_by_patch_v1.csv",
        "datasets/suscetibilidade/susc_patches_official_v1.csv",
        "datasets/suscetibilidade/susc_patch_links_official_v1.csv",
    ):
        changed = subprocess.run(["git", "diff", "--name-only", "--", path], cwd=ROOT, text=True, capture_output=True, check=False)
        assert changed.stdout.strip() == ""


def test_checklist_nao_concluido_e_politica_bloqueia_execucao_externa():
    policy = json.loads((OUT / "susc_17c11_channel_risk_policy.json").read_text(encoding="utf-8"))
    assert policy["automatic_submission_allowed"] is False
    assert policy["external_protocol_opening_allowed"] is False
    assert policy["external_form_filling_allowed"] is False
    assert policy["external_download_allowed"] is False
    assert policy["training_or_label_allowed"] is False
    for row in read_csv(OUT / "susc_17c11_manual_channel_verification_checklist.csv"):
        assert row["done"] == "false"
        assert row["external_action_performed"] == "false"
