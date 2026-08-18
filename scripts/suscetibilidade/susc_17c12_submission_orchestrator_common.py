"""SUSC-17C12 orquestrador de submissao assistida e intake operacional.

O pacote operacionaliza canais oficiais/candidatos do SUSC-17C11 sem executar
submissao externa, sem abrir protocolo, sem preencher formulario, sem baixar
dado externo e sem promover resposta futura a ground truth.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
LOCAL_STATE = ROOT / "local_runs" / "suscetibilidade" / "17c12_submission_orchestrator"

FEATURES = DAT / "susc_features_by_patch_v1.csv"
SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

AUDIT = OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C12.md"

C11_INPUTS = [
    OUT / "susc_17c11_official_channel_registry.csv",
    OUT / "susc_17c11_channel_source_evidence.csv",
    OUT / "susc_17c11_request_channel_match_matrix.csv",
    OUT / "susc_17c11_manual_channel_verification_checklist.csv",
    OUT / "susc_17c11_submission_readiness_update.csv",
    OUT / "susc_17c11_contact_discovery_search_plan.csv",
    OUT / "susc_17c11_channel_risk_policy.json",
    OUT / "susc_17c11_readiness_summary.json",
    OUT / "susc_17c11_promotion_blockers.csv",
]
C10_INPUTS = [
    OUT / "susc_17c10_formal_request_registry.csv",
    OUT / "susc_17c10_provider_package_index.csv",
    OUT / "susc_17c10_required_fields_by_request.csv",
    OUT / "susc_17c10_request_to_blocker_traceability.csv",
    OUT / "susc_17c10_request_impact_matrix.csv",
    OUT / "susc_17c10_manual_submission_checklist.csv",
    OUT / "susc_17c10_request_message_templates.md",
    OUT / "susc_17c10_attachment_manifest.csv",
    OUT / "susc_17c10_response_intake_schema_plan.csv",
    OUT / "susc_17c10_readiness_summary.json",
    OUT / "susc_17c10_promotion_blockers.csv",
]
C9_INPUTS = [
    OUT / "susc_17c9_feature_extraction_unblock_matrix.csv",
    OUT / "susc_17c9_promotion_blockers.csv",
]

REPORT = OUT / "SUSC_17C12_ORQUESTRADOR_SUBMISSAO_ASSISTIDA_E_INTAKE_OPERACIONAL_REPORT.md"
QUEUE = OUT / "susc_17c12_operational_request_queue.csv"
PACKAGE_REGISTRY = OUT / "susc_17c12_submission_package_registry.csv"
ACTION_PLAN = OUT / "susc_17c12_agent_action_plan.csv"
PREPARED_MESSAGES = OUT / "susc_17c12_prepared_messages.md"
STATUS_REGISTRY = OUT / "susc_17c12_submission_status_registry.csv"
ATTEMPT_LOG = OUT / "susc_17c12_submission_attempt_log.csv"
INTAKE_REGISTRY = OUT / "susc_17c12_response_intake_registry.csv"
MANIFEST_TEMPLATE = OUT / "susc_17c12_response_manifest_template.csv"
RISK_POLICY = OUT / "susc_17c12_operational_risk_policy.json"
SUMMARY = OUT / "susc_17c12_readiness_summary.json"
BLOCKERS = OUT / "susc_17c12_promotion_blockers.csv"

QUEUE_SCHEMA = SCHEMAS / "susc_17c12_operational_request_queue_schema_v1.json"
STATUS_SCHEMA = SCHEMAS / "susc_17c12_submission_status_schema_v1.json"
INTAKE_SCHEMA = SCHEMAS / "susc_17c12_response_intake_schema_v1.json"

REQUIRED_INPUTS = [FEATURES, SCORE_V6, AUDIT, *C11_INPUTS, *C10_INPUTS, *C9_INPUTS]
REQUIRED_OUTPUTS = [
    AUDIT,
    REPORT,
    QUEUE,
    PACKAGE_REGISTRY,
    ACTION_PLAN,
    PREPARED_MESSAGES,
    STATUS_REGISTRY,
    ATTEMPT_LOG,
    INTAKE_REGISTRY,
    MANIFEST_TEMPLATE,
    RISK_POLICY,
    SUMMARY,
    BLOCKERS,
    QUEUE_SCHEMA,
    STATUS_SCHEMA,
    INTAKE_SCHEMA,
]

QUEUE_FIELDS = [
    "queue_id",
    "formal_request_id",
    "provider_package_id",
    "provider_name",
    "priority",
    "request_title",
    "channel_id",
    "channel_type",
    "channel_status",
    "operational_status",
    "can_prepare_now",
    "can_open_channel_now",
    "can_submit_assisted_now",
    "requires_manual_channel_verification",
    "requires_human_submission",
    "requires_external_action_opt_in",
    "blocked_reason",
    "review_only",
    "trainable",
    "ground_truth",
]

STATUS_FIELDS = [
    "status_id",
    "formal_request_id",
    "provider_name",
    "current_status",
    "prepared_at",
    "submitted_manually",
    "submitted_at",
    "submitted_channel_id",
    "protocol_number",
    "response_received",
    "response_intake_id",
    "last_agent_mode",
    "status_source",
    "review_only",
    "trainable",
    "ground_truth",
]

INTAKE_FIELDS = [
    "response_intake_id",
    "formal_request_id",
    "provider_name",
    "response_received",
    "response_received_at",
    "raw_response_storage_path",
    "raw_response_sha256",
    "response_format",
    "contains_geometry",
    "contains_coordinates",
    "contains_event_date",
    "contains_crs",
    "contains_license_or_use_terms",
    "contains_sensitive_data",
    "manual_review_required",
    "eligible_for_future_ingestion",
    "blocking_reason",
    "review_only",
    "trainable",
    "ground_truth",
]

ATTEMPT_FIELDS = [
    "attempt_id",
    "formal_request_id",
    "attempt_mode",
    "attempted_at",
    "dry_run",
    "external_action_requested",
    "external_action_allowed",
    "result_status",
    "result_message",
    "protocol_number",
    "review_only",
]

MODES = ["plan", "prepare", "open-channel", "submit-assisted", "record-submission", "intake-response", "status", "audit"]
GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}
BUILD_TIMESTAMP = "not_executed_in_initial_build"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _run_git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _require_inputs() -> None:
    missing = [path for path in REQUIRED_INPUTS if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(path) for path in missing))
    for path in C11_INPUTS + C10_INPUTS + C9_INPUTS:
        if path.suffix == ".json":
            read_json(path)
        elif path.suffix == ".csv":
            read_csv(path)
        else:
            path.read_text(encoding="utf-8")


def _requests() -> list[dict]:
    return read_csv(OUT / "susc_17c10_formal_request_registry.csv")


def _channels() -> list[dict]:
    return read_csv(OUT / "susc_17c11_official_channel_registry.csv")


def _provider_packages() -> list[dict]:
    return read_csv(OUT / "susc_17c10_provider_package_index.csv")


def _attachments() -> list[dict]:
    return read_csv(OUT / "susc_17c10_attachment_manifest.csv")


def _required_fields() -> list[dict]:
    return read_csv(OUT / "susc_17c10_required_fields_by_request.csv")


def _by_id(rows: list[dict], key: str) -> dict[str, dict]:
    return {row[key]: row for row in rows}


def _provider_package_for_request(request_id: str) -> str:
    for row in _provider_packages():
        ids = row["requests_in_package"].split(";")
        if request_id in ids:
            return row["provider_package_id"]
    return "not_available"


def _attachment_ids(request_id: str) -> str:
    ids = [row["attachment_id"] for row in _attachments() if row["formal_request_id"] == request_id]
    return ";".join(ids) if ids else "none"


def _field_names(request_id: str) -> str:
    names = [row["field_name"] for row in _required_fields() if row["formal_request_id"] == request_id]
    return ";".join(names) if names else "sem_campos_minimos_registrados"


def _message_section_id(request_id: str) -> str:
    return f"S17C12_MSG_{request_id}"


def _channel_for_request(request_id: str) -> dict:
    return _by_id(_channels(), "formal_request_id")[request_id]


def _operational_status(channel_status: str) -> tuple[str, bool, bool, bool, str]:
    if channel_status == "confirmed_official_source":
        return ("ready_to_prepare", True, True, False, "not_blocked_prepare_only")
    if channel_status == "candidate_needs_manual_verification":
        return ("needs_manual_channel_verification", False, True, False, "channel_candidate_requires_manual_verification")
    return ("blocked_no_official_channel", False, False, False, "no_official_channel_available")


def build_operational_queue() -> list[dict]:
    rows = []
    for idx, req in enumerate(sorted(_requests(), key=lambda row: row["formal_request_id"]), start=1):
        channel = _channel_for_request(req["formal_request_id"])
        status, can_prepare, can_open, can_submit, reason = _operational_status(channel["channel_status"])
        rows.append({
            "queue_id": f"S17C12_QUEUE_{idx:04d}",
            "formal_request_id": req["formal_request_id"],
            "provider_package_id": _provider_package_for_request(req["formal_request_id"]),
            "provider_name": req["provider_name"],
            "priority": req["priority"],
            "request_title": req["request_title"],
            "channel_id": channel["channel_id"],
            "channel_type": channel["channel_type"],
            "channel_status": channel["channel_status"],
            "operational_status": status,
            "can_prepare_now": _bool_text(can_prepare),
            "can_open_channel_now": _bool_text(can_open),
            "can_submit_assisted_now": _bool_text(can_submit),
            "requires_manual_channel_verification": _bool_text(channel["channel_status"] != "confirmed_official_source"),
            "requires_human_submission": "true",
            "requires_external_action_opt_in": "true",
            "blocked_reason": reason,
            **GOV,
        })
    return rows


def build_submission_package_registry() -> list[dict]:
    rows = []
    for idx, req in enumerate(sorted(_requests(), key=lambda row: row["formal_request_id"]), start=1):
        channel = _channel_for_request(req["formal_request_id"])
        if channel["channel_status"] == "confirmed_official_source":
            package_status = "prepared_for_manual_review"
        elif channel["channel_status"] == "candidate_needs_manual_verification":
            package_status = "blocked_channel_candidate"
        else:
            package_status = "blocked_missing_channel"
        rows.append({
            "submission_package_id": f"S17C12_PKG_{idx:04d}",
            "formal_request_id": req["formal_request_id"],
            "provider_name": req["provider_name"],
            "priority": req["priority"],
            "message_section_id": _message_section_id(req["formal_request_id"]),
            "attachment_ids": _attachment_ids(req["formal_request_id"]),
            "channel_id": channel["channel_id"],
            "package_status": package_status,
            "message_ready": "true",
            "attachments_ready": "false",
            "manual_review_required": "true",
            "external_submission_allowed_by_default": "false",
            "request_marked_sent": "false",
            "response_marked_received": "false",
            **GOV,
        })
    return rows


def build_agent_action_plan() -> list[dict]:
    descriptions = {
        "plan": ("Listar fila priorizada sem efeito externo.", "nenhum_arquivo"),
        "prepare": ("Gerar pacote local copiavel para revisao humana.", "local_runs/suscetibilidade/17c12_submission_orchestrator/packages"),
        "open-channel": ("Imprimir URL e instrucao do canal para acao humana.", "stdout"),
        "submit-assisted": ("Tentar fluxo assistido somente com opt-in completo; portais permanecem bloqueados para acao manual.", "local_runs/suscetibilidade/17c12_submission_orchestrator/attempts"),
        "record-submission": ("Registrar envio manual ja feito fora do agente, com evidencia literal informada pela pessoa.", "local_runs/suscetibilidade/17c12_submission_orchestrator/state/submission_records.csv"),
        "intake-response": ("Registrar resposta recebida manualmente, calcular hash e gerar manifesto local.", "local_runs/suscetibilidade/17c12_submission_orchestrator/intake"),
        "status": ("Mostrar estado operacional atual sem efeito externo.", "stdout"),
        "audit": ("Validar guardrails contra envio, protocolo, resposta e score inventados.", "stdout"),
    }
    rows = []
    for req in sorted(_requests(), key=lambda row: row["formal_request_id"]):
        for mode in MODES:
            has_external = mode in {"open-channel", "submit-assisted"}
            requires_env = mode == "submit-assisted"
            rows.append({
                "agent_action_id": f"S17C12_ACTION_{len(rows) + 1:05d}",
                "formal_request_id": req["formal_request_id"],
                "action_mode": mode,
                "action_description": descriptions[mode][0],
                "command_example": f"python scripts\\suscetibilidade\\susc_17c12_submission_orchestrator.py {mode} --request-id {req['formal_request_id']}" if mode not in {"plan", "status", "audit"} else f"python scripts\\suscetibilidade\\susc_17c12_submission_orchestrator.py {mode}",
                "requires_user_confirmation": _bool_text(mode in {"submit-assisted", "record-submission", "intake-response"}),
                "requires_env_opt_in": _bool_text(requires_env),
                "has_external_effect": _bool_text(has_external),
                "allowed_by_default": _bool_text(mode != "submit-assisted"),
                "expected_local_output": descriptions[mode][1],
                "blocked_reason": "external_action_blocked_by_default" if mode == "submit-assisted" else "not_blocked_local_or_instruction_only",
                "review_only": "true",
            })
    return rows


def build_prepared_messages() -> str:
    parts = [
        "# SUSC-17C12 - mensagens preparadas para submissao assistida",
        "",
        "Este arquivo contem texto copiavel para revisao humana. Nenhuma mensagem foi enviada e nenhum protocolo foi aberto.",
    ]
    for req in sorted(_requests(), key=lambda row: row["formal_request_id"]):
        channel = _channel_for_request(req["formal_request_id"])
        subject = f"Solicitacao formal REV-P - {req['request_title']}"
        parts.extend([
            "",
            f"## {_message_section_id(req['formal_request_id'])}",
            "",
            f"- ID da solicitacao: `{req['formal_request_id']}`",
            f"- Fornecedor: {req['provider_name']}",
            f"- Prioridade: {req['priority']}",
            f"- Canal: {channel['channel_name']}",
            f"- Status do canal: `{channel['channel_status']}`",
            f"- Assunto sugerido: {subject}",
            "",
            "### Mensagem copiavel",
            "",
            "Prezadas(os),",
            "",
            f"Solicito, para fins de pesquisa academica e auditoria metodologica review-only do projeto REV-P, informacoes referentes a: {req['artifact_requested']}.",
            "",
            f"Contexto da solicitacao: {req['why_needed']}.",
            "",
            f"Campos minimos solicitados: {_field_names(req['formal_request_id'])}.",
            "",
            "Caso existam restricoes de acesso, formato, licenca, sensibilidade ou procedimento formal, solicito orientacao sobre o canal correto e os requisitos aplicaveis.",
            "",
            "Esta solicitacao nao deve ser interpretada como validacao operacional, ground truth, aprovacao institucional ou autorizacao para treino de modelo.",
            "",
            "Atenciosamente,",
            "[preencher manualmente]",
            "",
            "### Controle operacional",
            "",
            f"- Anexos recomendados: {_attachment_ids(req['formal_request_id'])}",
            "- Instrucao de revisao humana: revisar canal, texto, anexos e sensibilidade antes de qualquer acao externa.",
            "- Instrucao de nao submissao automatica: nao enviar por agente; usar apenas como texto copiavel.",
            f"- Comando local sugerido para prepare: `python scripts\\suscetibilidade\\susc_17c12_submission_orchestrator.py prepare --request-id {req['formal_request_id']}`",
            f"- Comando local sugerido para open-channel: `python scripts\\suscetibilidade\\susc_17c12_submission_orchestrator.py open-channel --request-id {req['formal_request_id']}`",
            f"- Comando local sugerido para record-submission: `python scripts\\suscetibilidade\\susc_17c12_submission_orchestrator.py record-submission --request-id {req['formal_request_id']} --submitted-at AAAA-MM-DD --channel-id {channel['channel_id']} --evidence-note \"valor literal informado pela pessoa\"`",
            "- Instrucao de intake futuro: registrar somente depois de resposta real recebida e arquivo local disponivel.",
        ])
    return "\n".join(parts)


def build_submission_status_registry() -> list[dict]:
    rows = []
    for idx, req in enumerate(sorted(_requests(), key=lambda row: row["formal_request_id"]), start=1):
        channel = _channel_for_request(req["formal_request_id"])
        current_status = "not_prepared" if channel["channel_status"] == "confirmed_official_source" else "blocked"
        rows.append({
            "status_id": f"S17C12_STATUS_{idx:04d}",
            "formal_request_id": req["formal_request_id"],
            "provider_name": req["provider_name"],
            "current_status": current_status,
            "prepared_at": "not_prepared",
            "submitted_manually": "false",
            "submitted_at": "not_submitted",
            "submitted_channel_id": "not_submitted",
            "protocol_number": "not_informed",
            "response_received": "false",
            "response_intake_id": "not_registered",
            "last_agent_mode": "initial_build",
            "status_source": "build_initial_no_mode_execution",
            **GOV,
        })
    return rows


def build_attempt_log() -> list[dict]:
    return []


def build_response_intake_registry() -> list[dict]:
    rows = []
    for idx, req in enumerate(sorted(_requests(), key=lambda row: row["formal_request_id"]), start=1):
        rows.append({
            "response_intake_id": f"S17C12_INTAKE_{idx:04d}",
            "formal_request_id": req["formal_request_id"],
            "provider_name": req["provider_name"],
            "response_received": "false",
            "response_received_at": "not_received",
            "raw_response_storage_path": "not_available",
            "raw_response_sha256": "not_available",
            "response_format": "not_available",
            "contains_geometry": "unknown_until_manual_review",
            "contains_coordinates": "unknown_until_manual_review",
            "contains_event_date": "unknown_until_manual_review",
            "contains_crs": "unknown_until_manual_review",
            "contains_license_or_use_terms": "unknown_until_manual_review",
            "contains_sensitive_data": "unknown_until_manual_review",
            "manual_review_required": "true",
            "eligible_for_future_ingestion": "false",
            "blocking_reason": "response_not_received",
            **GOV,
        })
    return rows


def build_manifest_template() -> list[dict]:
    rows = []
    for idx, req in enumerate(sorted(_requests(), key=lambda row: row["formal_request_id"]), start=1):
        rows.append({
            "manifest_id": f"S17C12_MANIFEST_{idx:04d}",
            "response_intake_id": f"S17C12_INTAKE_{idx:04d}",
            "formal_request_id": req["formal_request_id"],
            "provider_name": req["provider_name"],
            "artifact_name": "not_available_until_real_response",
            "artifact_local_path": "not_available",
            "sha256": "not_available",
            "size_bytes": "not_available",
            "format": "not_available",
            "is_raw_heavy": "unknown_until_file_review",
            "store_in_outputs_public": "false",
            "store_in_local_runs": "true",
            "requires_manual_qa": "true",
            "review_only": "true",
        })
    return rows


def build_risk_policy() -> dict:
    return {
        "milestone": "SUSC-17C12",
        "default_dry_run": True,
        "allow_external_submission": False,
        "allow_browser_action": False,
        "allow_email_send": False,
        "allow_form_submit": False,
        "required_env_for_external_action": "SUSC_17C12_ALLOW_EXTERNAL_ACTION=1",
        "required_flags_for_submit_assisted": ["--request-id", "--confirm-human-reviewed", "--dry-run=false"],
        "automatic_submission_blocked_by_default": True,
        "block_without_confirm_human_reviewed": True,
        "block_without_env_opt_in": True,
        "block_unconfirmed_channel": True,
        "block_candidate_channel": True,
        "block_manual_portal_automation": True,
        "block_protocol_invention": True,
        "block_response_invention": True,
        "block_hash_invention": True,
        "block_path_invention": True,
        "block_ingestion_without_manifest": True,
        "block_raw_heavy_in_outputs_public": True,
        "block_ground_truth_promotion": True,
        "block_17b_without_real_features_policy_and_qa": True,
        "feature_extraction_allowed": False,
        "score_v6_change_allowed": False,
        "score_v7_creation_allowed": False,
        "training_or_label_allowed": False,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
    }


def build_summary() -> dict:
    queue = build_operational_queue()
    channels = _channels()
    packages = build_submission_package_registry()
    messages_count = len(_requests())
    return {
        "formal_requests_count": len(_requests()),
        "operational_queue_rows_count": len(queue),
        "submission_packages_count": len(packages),
        "prepared_messages_count": messages_count,
        "confirmed_channel_requests_count": len([c for c in channels if c["channel_status"] == "confirmed_official_source"]),
        "candidate_channel_requests_count": len([c for c in channels if c["channel_status"] == "candidate_needs_manual_verification"]),
        "blocked_no_channel_requests_count": len([c for c in channels if c["channel_status"] == "not_found"]),
        "agent_modes_count": len(MODES),
        "requests_ready_to_prepare_count": len([r for r in queue if r["operational_status"] == "ready_to_prepare"]),
        "requests_ready_for_manual_submission_count": len([r for r in queue if r["operational_status"] == "ready_for_manual_submission"]),
        "requests_requiring_manual_channel_verification_count": len([r for r in queue if r["requires_manual_channel_verification"] == "true"]),
        "requests_marked_sent_count": 0,
        "protocols_opened_count": 0,
        "responses_received_count": 0,
        "contacts_invented_count": 0,
        "features_extracted_this_sprint": 0,
        "score_v6_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
        "official_patch_created": False,
        "official_patch_link_created": False,
        "raw_raster_committed": False,
        "eligible_for_17b_now": False,
        "eligible_for_score_v7": False,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "recommended_next_milestone": "SUSC-17C13 Execucao Assistida de Submissoes Manuais para canais confirmados; verificar canais candidatos em paralelo",
    }


def build_blockers() -> list[dict]:
    blocker_types = [
        "external_submission_requires_human_opt_in",
        "manual_submission_not_performed",
        "responses_not_received",
        "response_intake_not_executed",
        "candidate_specific_artifacts_missing",
        "candidate_patch_policy_missing",
        "pipeline_adapters_missing",
        "official_patch_id_dependency",
        "sentinel2_tile_missing",
        "embedding_tile_missing",
        "sar_runtime_missing",
        "qa_not_accepted",
        "17b_blocked_until_real_features_and_policy",
    ]
    return [
        {
            "blocker_id": f"S17C12_BLOCKER_{idx:04d}",
            "blocker_type": blocker,
            "description": "Bloqueio preservado pelo orquestrador 17C12; requer acao humana, resposta real, politica ou QA futura antes de promocao.",
            "blocks_external_submission": _bool_text(blocker in {"external_submission_requires_human_opt_in", "manual_submission_not_performed"}),
            "blocks_17b": "true",
            "blocks_score_v7": "true",
            "review_only": "true",
        }
        for idx, blocker in enumerate(blocker_types, start=1)
    ]


def _schema(required: list[str], props: dict, title: str) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "required": required,
        "properties": props,
    }


def build_queue_schema() -> dict:
    return _schema(QUEUE_FIELDS, {
        "queue_id": {"type": "string", "pattern": "^S17C12_QUEUE_"},
        "operational_status": {"enum": ["ready_to_prepare", "prepared", "ready_for_manual_submission", "needs_manual_channel_verification", "blocked_no_official_channel", "blocked_ambiguous_channel", "blocked_external_action_not_allowed", "submitted_manually", "response_received_pending_intake", "intake_registered_pending_validation"]},
        "can_submit_assisted_now": {"const": "false"},
        "requires_external_action_opt_in": {"const": "true"},
        "review_only": {"const": "true"},
        "trainable": {"const": "false"},
        "ground_truth": {"const": "false"},
    }, "SUSC-17C12 operational request queue schema v1")


def build_status_schema() -> dict:
    return _schema(STATUS_FIELDS, {
        "status_id": {"type": "string", "pattern": "^S17C12_STATUS_"},
        "current_status": {"enum": ["not_prepared", "prepared_for_manual_review", "ready_for_manual_submission", "submitted_manually", "response_received_pending_intake", "intake_registered_pending_validation", "blocked"]},
        "submitted_manually": {"const": "false"},
        "protocol_number": {"const": "not_informed"},
        "response_received": {"const": "false"},
        "review_only": {"const": "true"},
        "trainable": {"const": "false"},
        "ground_truth": {"const": "false"},
    }, "SUSC-17C12 submission status schema v1")


def build_intake_schema() -> dict:
    return _schema(INTAKE_FIELDS, {
        "response_intake_id": {"type": "string", "pattern": "^S17C12_INTAKE_"},
        "response_received": {"const": "false"},
        "raw_response_storage_path": {"const": "not_available"},
        "raw_response_sha256": {"const": "not_available"},
        "eligible_for_future_ingestion": {"const": "false"},
        "review_only": {"const": "true"},
        "trainable": {"const": "false"},
        "ground_truth": {"const": "false"},
    }, "SUSC-17C12 response intake schema v1")


def build_report() -> str:
    summary = build_summary()
    return "\n".join([
        "# SUSC-17C12 - Orquestrador de submissao assistida e intake operacional",
        "",
        "## O que o 17C11 resolveu",
        "O SUSC-17C11 associou as 9 solicitacoes formais do 17C10 a canais oficiais, candidatos ou ausentes: 5 canais confirmados, 2 candidatos e 2 sem canal externo oficial.",
        "",
        "## O que o 17C12 acrescenta",
        "O SUSC-17C12 transforma esse inventario em fila operacional, pacotes copiaveis, mensagens finais, modos CLI seguros, registro local de acao humana e intake futuro de respostas reais. O marco nao e apenas dossie estatico: ele cria comandos auditaveis para o agente preparar, orientar e registrar, sem executar acao externa automaticamente.",
        "",
        "## Como a fila funciona",
        f"A fila possui {summary['operational_queue_rows_count']} linhas. Pedidos com canal confirmado ficam `ready_to_prepare`; canais candidatos ficam `needs_manual_channel_verification`; canais ausentes ficam `blocked_no_official_channel`.",
        "",
        "## Modos do orquestrador",
        "`plan`, `prepare`, `open-channel`, `submit-assisted`, `record-submission`, `intake-response`, `status` e `audit` foram implementados. `prepare`, `status` e `audit` nao tem efeito externo. `submit-assisted` e bloqueado por padrao e exige opt-in ambiental e revisao humana.",
        "",
        "## Preparacao e canal",
        "Use `prepare --request-id <ID>` para gerar pacote local em `local_runs`. Use `open-channel --request-id <ID>` para imprimir URL e checklist humano. O comando nao preenche formulario, nao autentica e nao registra envio.",
        "",
        "## Registro manual e intake",
        "`record-submission` registra envio manual ja feito fora do agente, com data, canal e evidencia literal. `intake-response` exige arquivo local real, calcula hash e grava manifesto local; nada bruto pesado e copiado para `outputs_public`.",
        "",
        "## Bloqueios",
        "Envio automatico, protocolo inventado, resposta inventada, hash inventado, bruto pesado em `outputs_public`, score v7, 17B, treino, modelo, label e ground truth permanecem bloqueados.",
        "",
        "## Prontidao",
        f"- Prontos para preparacao: {summary['requests_ready_to_prepare_count']}.",
        f"- Prontos para submissao manual no build inicial: {summary['requests_ready_for_manual_submission_count']}.",
        f"- Dependem de verificacao manual de canal: {summary['candidate_channel_requests_count']}.",
        f"- Sem canal externo oficial: {summary['blocked_no_channel_requests_count']}.",
        "",
        "## Proximo marco recomendado",
        summary["recommended_next_milestone"],
    ])


def build_all() -> None:
    _require_inputs()
    write_csv(QUEUE, build_operational_queue(), QUEUE_FIELDS)
    write_csv(PACKAGE_REGISTRY, build_submission_package_registry())
    write_csv(ACTION_PLAN, build_agent_action_plan())
    write_markdown(PREPARED_MESSAGES, build_prepared_messages())
    write_csv(STATUS_REGISTRY, build_submission_status_registry(), STATUS_FIELDS)
    write_csv(ATTEMPT_LOG, build_attempt_log(), ATTEMPT_FIELDS)
    write_csv(INTAKE_REGISTRY, build_response_intake_registry(), INTAKE_FIELDS)
    write_csv(MANIFEST_TEMPLATE, build_manifest_template())
    write_json(RISK_POLICY, build_risk_policy())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_json(QUEUE_SCHEMA, build_queue_schema())
    write_json(STATUS_SCHEMA, build_status_schema())
    write_json(INTAKE_SCHEMA, build_intake_schema())
    write_markdown(REPORT, build_report())


def _schema_violations(row: dict, schema: dict) -> list[str]:
    violations = []
    for field in schema.get("required", []):
        if field not in row or row[field] == "":
            violations.append(f"missing:{field}")
    for field, rules in schema.get("properties", {}).items():
        if field not in row:
            continue
        value = row[field]
        if "const" in rules and value != rules["const"]:
            violations.append(f"{field}:const:{rules['const']}")
        if "enum" in rules and value not in rules["enum"]:
            violations.append(f"{field}:enum")
        if "pattern" in rules and not value.startswith(rules["pattern"].replace("^", "")):
            violations.append(f"{field}:pattern")
    return violations


def validate() -> int:
    missing = [path for path in REQUIRED_OUTPUTS if not path.exists()]
    if missing:
        print("MISSING: " + "; ".join(rel(path) for path in missing), file=sys.stderr)
        return 1
    queue = read_csv(QUEUE)
    packages = read_csv(PACKAGE_REGISTRY)
    actions = read_csv(ACTION_PLAN)
    statuses = read_csv(STATUS_REGISTRY)
    attempts = read_csv(ATTEMPT_LOG)
    intake = read_csv(INTAKE_REGISTRY)
    manifest = read_csv(MANIFEST_TEMPLATE)
    blockers = read_csv(BLOCKERS)
    summary = read_json(SUMMARY)
    policy = read_json(RISK_POLICY)
    errors = []

    for row in queue:
        errors.extend(_schema_violations(row, read_json(QUEUE_SCHEMA)))
    for row in statuses:
        errors.extend(_schema_violations(row, read_json(STATUS_SCHEMA)))
    for row in intake:
        errors.extend(_schema_violations(row, read_json(INTAKE_SCHEMA)))
    if len(queue) != 9 or len(packages) != 9 or len(statuses) != 9 or len(intake) != 9 or len(manifest) != 9:
        errors.append("expected_nine_request_rows")
    if len(actions) != 9 * len(MODES):
        errors.append("expected_action_plan_for_all_modes")
    for rows, key in [(queue, "queue_id"), (packages, "submission_package_id"), (actions, "agent_action_id"), (statuses, "status_id"), (intake, "response_intake_id"), (manifest, "manifest_id")]:
        values = [row[key] for row in rows]
        if values != sorted(values) or len(values) != len(set(values)):
            errors.append(f"ids_not_unique_or_sorted:{key}")
    if attempts:
        if any(row["protocol_number"] not in {"", "not_informed"} for row in attempts):
            errors.append("initial_attempt_log_contains_protocol")
    if any(row["review_only"] != "true" or row.get("trainable", "false") != "false" or row.get("ground_truth", "false") != "false" for row in queue + packages + statuses + intake):
        errors.append("governance_flags_invalid")
    if any(row["can_submit_assisted_now"] != "false" for row in queue):
        errors.append("queue_allows_submit_assisted")
    if any(row["submitted_manually"] != "false" or row["protocol_number"] != "not_informed" or row["response_received"] != "false" for row in statuses):
        errors.append("initial_status_invents_submission_protocol_or_response")
    if any(row["raw_response_storage_path"] != "not_available" or row["raw_response_sha256"] != "not_available" for row in intake):
        errors.append("initial_intake_invents_path_or_hash")
    if any(row["store_in_outputs_public"] != "false" for row in manifest):
        errors.append("manifest_allows_raw_storage_in_outputs_public")
    submit_actions = [row for row in actions if row["action_mode"] == "submit-assisted"]
    if not submit_actions or any(row["allowed_by_default"] != "false" or row["requires_env_opt_in"] != "true" for row in submit_actions):
        errors.append("submit_assisted_not_blocked_by_default")
    text = PREPARED_MESSAGES.read_text(encoding="utf-8")
    if text.count("## S17C12_MSG_") != 9 or "Prezadas(os)" not in text or "Nenhuma mensagem foi enviada" not in text:
        errors.append("prepared_messages_missing_portuguese_sections")
    expected_summary = build_summary()
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            errors.append(f"summary_mismatch:{key}")
    for path in [FEATURES, SCORE_V6, DAT / "susc_patches_official_v1.csv", DAT / "susc_patch_links_official_v1.csv"]:
        if path.exists() and _run_git(["diff", "--name-only", "--", rel(path)]):
            errors.append(f"official_dataset_changed:{rel(path)}")
    if SCORE_V7.exists() or summary["score_v7_created"]:
        errors.append("score_v7_exists")
    if not blockers:
        errors.append("empty_blockers")
    if policy["allow_external_submission"] or policy["allow_email_send"] or policy["allow_form_submit"]:
        errors.append("risk_policy_allows_external_submission")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "17C12 -> "
        f"queue={summary['operational_queue_rows_count']} "
        f"packages={summary['submission_packages_count']} "
        f"messages={summary['prepared_messages_count']} "
        f"ready_to_prepare={summary['requests_ready_to_prepare_count']} "
        f"sent={summary['requests_marked_sent_count']} "
        f"responses={summary['responses_received_count']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
    return 0


def _write_local_json(path: Path, payload: dict) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _write_local_text(path: Path, text: str) -> Path:
    ensure_dir(path.parent)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return path


def plan_text() -> str:
    lines = ["Fila operacional SUSC-17C12"]
    for row in build_operational_queue():
        lines.append(f"{row['priority']} {row['formal_request_id']} {row['operational_status']} {row['channel_status']}")
    return "\n".join(lines)


def status_text() -> str:
    lines = ["Status operacional SUSC-17C12"]
    for row in build_operational_queue():
        if row["operational_status"] == "ready_to_prepare":
            status = "ready_to_prepare"
        elif row["operational_status"] == "needs_manual_channel_verification":
            status = "needs_manual_channel_verification"
        else:
            status = "blocked"
        lines.append(f"{row['formal_request_id']}: {status}")
    return "\n".join(lines)


def prepare_request(request_id: str) -> Path:
    req = _by_id(_requests(), "formal_request_id").get(request_id)
    if not req:
        raise ValueError(f"request_id desconhecido: {request_id}")
    channel = _channel_for_request(request_id)
    target = LOCAL_STATE / "packages" / request_id
    message = f"# Pacote local {request_id}\n\nCanal: {channel['channel_name']}\nURL: {channel['channel_entrypoint_url']}\n\nRevise manualmente antes de qualquer acao externa.\n"
    _write_local_text(target / "mensagem_copiavel.md", message)
    _write_local_text(target / "anexos_recomendados.txt", _attachment_ids(request_id))
    _write_local_json(target / "manifesto_pacote.json", {
        "formal_request_id": request_id,
        "channel_id": channel["channel_id"],
        "channel_status": channel["channel_status"],
        "external_submission_performed": False,
        "review_only": True,
    })
    return target


def open_channel_instruction(request_id: str) -> str:
    channel = _channel_for_request(request_id)
    if channel["channel_entrypoint_url"] == "not_available":
        return f"{request_id}: sem canal externo oficial confirmado. Nao abrir canal."
    return f"{request_id}: abrir manualmente {channel['channel_entrypoint_url']} e nao preencher nem enviar formulario pelo agente."


def submit_assisted(request_id: str, dry_run: bool, confirm_human_reviewed: bool) -> tuple[str, str]:
    channel = _channel_for_request(request_id)
    if dry_run:
        return ("dry_run_only", "Dry-run: nenhuma acao externa executada.")
    if not confirm_human_reviewed:
        return ("blocked_missing_opt_in", "Bloqueado: faltou --confirm-human-reviewed.")
    if os.environ.get("SUSC_17C12_ALLOW_EXTERNAL_ACTION") != "1":
        return ("blocked_missing_opt_in", "Bloqueado: faltou SUSC_17C12_ALLOW_EXTERNAL_ACTION=1.")
    if channel["channel_status"] != "confirmed_official_source":
        return ("blocked_channel_not_confirmed", "Bloqueado: canal nao confirmado.")
    return ("blocked_manual_portal_required", "Bloqueado por seguranca: executar portal, login, e-SIC, ouvidoria ou formulario manualmente.")


def record_manual_submission(request_id: str, submitted_at: str, channel_id: str, evidence_note: str, protocol_number: str = "not_informed") -> Path:
    if not evidence_note.strip():
        raise ValueError("evidence_note obrigatorio para registro manual")
    path = LOCAL_STATE / "state" / "submission_records.csv"
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["formal_request_id", "submitted_at", "channel_id", "evidence_note", "protocol_number", "review_only"], lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({
            "formal_request_id": request_id,
            "submitted_at": submitted_at,
            "channel_id": channel_id,
            "evidence_note": evidence_note,
            "protocol_number": protocol_number,
            "review_only": "true",
        })
    return path


def intake_response(request_id: str, response_file: Path, received_at: str) -> Path:
    if not response_file.exists() or not response_file.is_file():
        raise FileNotFoundError(response_file)
    digest = hashlib.sha256(response_file.read_bytes()).hexdigest()
    target = LOCAL_STATE / "intake" / request_id / "response_manifest.json"
    _write_local_json(target, {
        "formal_request_id": request_id,
        "response_received_at": received_at,
        "raw_response_storage_path": str(response_file.resolve()),
        "raw_response_sha256": digest,
        "size_bytes": response_file.stat().st_size,
        "eligible_for_future_ingestion": False,
        "manual_review_required": True,
        "review_only": True,
    })
    return target
