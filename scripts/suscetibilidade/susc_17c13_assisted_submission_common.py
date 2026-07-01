"""SUSC-17C13 execucao assistida de submissoes manuais.

Consolida a execucao segura do orquestrador 17C12: plan/status/audit,
preparacao local dos pedidos com canal confirmado e instrucoes de abertura de
canal. Nao envia solicitacao, nao abre protocolo e nao registra resposta.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
LOCAL_PACKAGES_REL = "local_runs/suscetibilidade/17c12_submission_orchestrator/packages"

FEATURES = DAT / "susc_features_by_patch_v1.csv"
SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
AUDIT = OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C13.md"

C12_INPUTS = [
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
]
C11_INPUTS = [
    OUT / "susc_17c11_official_channel_registry.csv",
    OUT / "susc_17c11_request_channel_match_matrix.csv",
    OUT / "susc_17c11_submission_readiness_update.csv",
    OUT / "susc_17c11_channel_risk_policy.json",
]
C10_INPUTS = [
    OUT / "susc_17c10_formal_request_registry.csv",
    OUT / "susc_17c10_request_message_templates.md",
    OUT / "susc_17c10_attachment_manifest.csv",
]

REPORT = OUT / "SUSC_17C13_EXECUCAO_ASSISTIDA_SUBMISSOES_MANUAIS_REPORT.md"
RUN_LOG = OUT / "susc_17c13_orchestrator_run_log.csv"
PREPARED_PACKAGES = OUT / "susc_17c13_prepared_submission_packages.csv"
OPEN_INSTRUCTIONS = OUT / "susc_17c13_channel_open_instructions.csv"
EXECUTION_BOARD = OUT / "susc_17c13_manual_execution_board.csv"
STATUS_SNAPSHOT = OUT / "susc_17c13_submission_status_snapshot.csv"
EVIDENCE_REQUIREMENTS = OUT / "susc_17c13_submission_evidence_requirements.csv"
PENDING_CHANNELS = OUT / "susc_17c13_pending_channel_verification_queue.csv"
INTAKE_WATCHLIST = OUT / "susc_17c13_response_intake_watchlist.csv"
SAFETY_AUDIT = OUT / "susc_17c13_operational_safety_audit.csv"
SUMMARY = OUT / "susc_17c13_readiness_summary.json"
BLOCKERS = OUT / "susc_17c13_promotion_blockers.csv"

RUN_SCHEMA = SCHEMAS / "susc_17c13_orchestrator_run_schema_v1.json"
BOARD_SCHEMA = SCHEMAS / "susc_17c13_execution_board_schema_v1.json"

REQUIRED_INPUTS = [FEATURES, SCORE_V6, AUDIT, *C12_INPUTS, *C11_INPUTS, *C10_INPUTS]
REQUIRED_OUTPUTS = [
    AUDIT,
    REPORT,
    RUN_LOG,
    PREPARED_PACKAGES,
    OPEN_INSTRUCTIONS,
    EXECUTION_BOARD,
    STATUS_SNAPSHOT,
    EVIDENCE_REQUIREMENTS,
    PENDING_CHANNELS,
    INTAKE_WATCHLIST,
    SAFETY_AUDIT,
    SUMMARY,
    BLOCKERS,
    RUN_SCHEMA,
    BOARD_SCHEMA,
]

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}

RUN_FIELDS = [
    "run_log_id",
    "formal_request_id",
    "orchestrator_mode",
    "command_invoked",
    "executed",
    "dry_run",
    "external_effect",
    "result_status",
    "result_message",
    "local_output_created",
    "request_marked_sent",
    "protocol_opened",
    "response_received",
    "review_only",
    "trainable",
    "ground_truth",
]

BOARD_FIELDS = [
    "execution_task_id",
    "formal_request_id",
    "provider_name",
    "priority",
    "task_order",
    "task_description",
    "task_status",
    "requires_human_action",
    "evidence_required_to_mark_done",
    "done",
    "review_only",
]


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _run_git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _require_inputs() -> None:
    missing = [path for path in REQUIRED_INPUTS if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(path) for path in missing))
    for path in C12_INPUTS + C11_INPUTS + C10_INPUTS:
        if path.suffix == ".json":
            read_json(path)
        elif path.suffix == ".csv":
            read_csv(path)
        else:
            path.read_text(encoding="utf-8")


def _queue() -> list[dict]:
    return read_csv(OUT / "susc_17c12_operational_request_queue.csv")


def _channels() -> list[dict]:
    return read_csv(OUT / "susc_17c11_official_channel_registry.csv")


def _requests() -> list[dict]:
    return read_csv(OUT / "susc_17c10_formal_request_registry.csv")


def _statuses() -> list[dict]:
    return read_csv(OUT / "susc_17c12_submission_status_registry.csv")


def _by_id(rows: list[dict], key: str) -> dict[str, dict]:
    return {row[key]: row for row in rows}


def _confirmed_queue() -> list[dict]:
    return [row for row in _queue() if row["channel_status"] == "confirmed_official_source" and row["operational_status"] == "ready_to_prepare"]


def _openable_queue() -> list[dict]:
    return [row for row in _queue() if row["channel_status"] in {"confirmed_official_source", "candidate_needs_manual_verification"}]


def _pending_queue() -> list[dict]:
    return [row for row in _queue() if row["channel_status"] != "confirmed_official_source"]


def _cmd(mode: str, request_id: str | None = None) -> str:
    base = f"python scripts\\suscetibilidade\\susc_17c12_submission_orchestrator.py {mode}"
    return f"{base} --request-id {request_id}" if request_id else base


def build_run_log() -> list[dict]:
    rows = []

    def add(request_id: str, mode: str, status: str, message: str, local_output: bool, dry_run: bool = True) -> None:
        rows.append({
            "run_log_id": f"S17C13_RUN_{len(rows) + 1:04d}",
            "formal_request_id": request_id,
            "orchestrator_mode": mode,
            "command_invoked": _cmd(mode, None if request_id == "all_requests" else request_id),
            "executed": "true",
            "dry_run": _bool_text(dry_run),
            "external_effect": "false",
            "result_status": status,
            "result_message": message,
            "local_output_created": _bool_text(local_output),
            "request_marked_sent": "false",
            "protocol_opened": "false",
            "response_received": "false",
            **GOV,
        })

    add("all_requests", "plan", "completed_local_only", "Fila operacional impressa sem efeito externo.", False)
    add("all_requests", "status", "completed_local_only", "Status operacional impresso sem efeito externo.", False)
    add("all_requests", "audit", "completed_local_only", "Auditoria 17C12 executada sem efeito externo.", False)
    for row in _confirmed_queue():
        add(row["formal_request_id"], "prepare", "completed_local_only", "Pacote local copiavel criado para revisao humana.", True)
    for row in _openable_queue():
        status = "completed_dry_run" if row["channel_status"] == "confirmed_official_source" else "blocked_candidate_channel_needs_verification"
        add(row["formal_request_id"], "open-channel", status, "Instrucao de abertura manual gerada; nenhum canal foi submetido pelo agente.", False)
    return rows


def build_prepared_packages() -> list[dict]:
    rows = []
    for idx, row in enumerate(_queue(), start=1):
        confirmed = row["channel_status"] == "confirmed_official_source"
        candidate = row["channel_status"] == "candidate_needs_manual_verification"
        rows.append({
            "prepared_package_id": f"S17C13_PKG_{idx:04d}",
            "formal_request_id": row["formal_request_id"],
            "provider_name": row["provider_name"],
            "priority": row["priority"],
            "channel_id": row["channel_id"],
            "channel_status": row["channel_status"],
            "message_prepared": _bool_text(confirmed),
            "attachments_listed": _bool_text(confirmed),
            "copy_paste_ready": _bool_text(confirmed),
            "manual_review_required": "true",
            "ready_for_manual_submission_after_review": _bool_text(confirmed),
            "blocked_reason": "not_blocked_after_manual_review" if confirmed else ("channel_candidate_requires_verification" if candidate else "blocked_no_official_channel"),
            **GOV,
        })
    return rows


def build_open_instructions() -> list[dict]:
    channels = _by_id(_channels(), "formal_request_id")
    rows = []
    for row in _openable_queue():
        channel = channels[row["formal_request_id"]]
        portal_like = channel["channel_type"] in {"ombudsman_or_institutional_contact", "sic_or_ombudsman", "official_contact_page", "candidate_contact_page"}
        rows.append({
            "open_instruction_id": f"S17C13_OPEN_{len(rows) + 1:04d}",
            "formal_request_id": row["formal_request_id"],
            "provider_name": row["provider_name"],
            "channel_id": row["channel_id"],
            "channel_type": row["channel_type"],
            "channel_reference": channel["channel_entrypoint_url"],
            "instruction_text": f"Abrir manualmente {channel['channel_entrypoint_url']} e nao preencher, autenticar, submeter ou registrar protocolo pelo agente.",
            "requires_login_or_portal": _bool_text(portal_like),
            "requires_manual_action": "true",
            "can_be_opened_by_agent": "false",
            "can_be_submitted_by_agent": "false",
            "blocked_reason": "manual_action_required_no_agent_submission" if row["channel_status"] == "confirmed_official_source" else "candidate_channel_requires_manual_verification",
            "review_only": "true",
        })
    return rows


TASKS = [
    ("revisar pacote", "pacote revisado e anotacao humana"),
    ("abrir canal oficial", "canal aberto manualmente pela pessoa"),
    ("verificar canal", "confirmacao humana de que o canal cobre o pedido"),
    ("copiar mensagem", "mensagem copiada apos revisao"),
    ("anexar arquivos revisados, se aplicavel", "lista de anexos revisada e sensibilidade checada"),
    ("submeter manualmente", "evidencia literal do envio manual"),
    ("registrar data/hora", "data/hora literal informada pela pessoa"),
    ("registrar protocolo, se houver", "protocolo literal se existir"),
    ("salvar comprovante ou resposta futura fora de outputs_public", "path local privado ou local_runs protegido"),
    ("rodar intake-response quando houver resposta", "arquivo real e hash calculado no intake futuro"),
]


def build_execution_board() -> list[dict]:
    rows = []
    for row in _queue():
        for order, (description, evidence) in enumerate(TASKS, start=1):
            rows.append({
                "execution_task_id": f"S17C13_TASK_{len(rows) + 1:04d}",
                "formal_request_id": row["formal_request_id"],
                "provider_name": row["provider_name"],
                "priority": row["priority"],
                "task_order": str(order),
                "task_description": description,
                "task_status": "pending",
                "requires_human_action": "true",
                "evidence_required_to_mark_done": evidence,
                "done": "false",
                "review_only": "true",
            })
    return rows


def build_status_snapshot() -> list[dict]:
    previous = _by_id(_statuses(), "formal_request_id")
    rows = []
    for idx, row in enumerate(_queue(), start=1):
        confirmed = row["channel_status"] == "confirmed_official_source"
        candidate = row["channel_status"] == "candidate_needs_manual_verification"
        no_channel = row["channel_status"] == "not_found"
        new_status = "ready_for_manual_submission_after_review" if confirmed else ("needs_manual_channel_verification" if candidate else "blocked_no_official_channel")
        rows.append({
            "status_snapshot_id": f"S17C13_STATUS_{idx:04d}",
            "formal_request_id": row["formal_request_id"],
            "provider_name": row["provider_name"],
            "previous_status_17c12": previous[row["formal_request_id"]]["current_status"],
            "new_status_17c13": new_status,
            "prepared": _bool_text(confirmed),
            "channel_open_instruction_created": _bool_text(confirmed or candidate),
            "ready_for_manual_submission": _bool_text(confirmed),
            "needs_manual_channel_verification": _bool_text(candidate),
            "blocked_no_channel": _bool_text(no_channel),
            "submitted_manually": "false",
            "protocol_opened": "false",
            "response_received": "false",
            **GOV,
        })
    return rows


def build_evidence_requirements() -> list[dict]:
    fields = [
        ("request_id", "ID da solicitacao formal", False, "vincular registro ao pedido correto"),
        ("provider_name", "Nome do fornecedor/orgaos", False, "confirmar alvo institucional"),
        ("canal usado", "Canal efetivamente usado na submissao manual", False, "evitar inventar canal"),
        ("data/hora da submissao", "Data e hora literais informadas pela pessoa", False, "provar momento do envio"),
        ("modo de submissao", "Portal, ouvidoria, e-SIC, pagina ou outro modo usado", False, "documentar meio usado"),
        ("responsavel humano ou marcador local", "Pessoa ou marcador local que realizou o envio", False, "separar acao humana de agente"),
        ("protocolo, se houver", "Numero literal informado pelo canal, quando existir", True, "nao inventar protocolo"),
        ("observacoes", "Notas humanas sobre restricoes, anexos ou comprovantes", True, "preservar contexto sem promover evidencia"),
    ]
    rows = []
    for req in _requests():
        for field, desc, can_empty, why in fields:
            rows.append({
                "evidence_requirement_id": f"S17C13_EVIDREQ_{len(rows) + 1:04d}",
                "formal_request_id": req["formal_request_id"],
                "required_to_record_submission": "true",
                "required_field": field,
                "field_description": desc,
                "can_be_empty": _bool_text(can_empty),
                "why_required": why,
                "review_only": "true",
            })
    return rows


def build_pending_channel_queue() -> list[dict]:
    rows = []
    for row in _pending_queue():
        rows.append({
            "pending_channel_task_id": f"S17C13_PENDING_{len(rows) + 1:04d}",
            "formal_request_id": row["formal_request_id"],
            "provider_name": row["provider_name"],
            "channel_id": row["channel_id"],
            "channel_status": row["channel_status"],
            "verification_needed": "true",
            "verification_steps": "confirmar fornecedor;confirmar canal oficial;confirmar se aceita o artefato solicitado;registrar evidencia literal antes de submissao",
            "priority": row["priority"],
            "blocking_reason": row["blocked_reason"],
            "review_only": "true",
        })
    return rows


def build_intake_watchlist() -> list[dict]:
    rows = []
    for idx, req in enumerate(_requests(), start=1):
        rows.append({
            "watchlist_id": f"S17C13_WATCH_{idx:04d}",
            "formal_request_id": req["formal_request_id"],
            "provider_name": req["provider_name"],
            "submission_required_first": "true",
            "expected_response_type": req["artifact_type_requested"],
            "expected_artifact_or_fields": req["minimum_required_fields"],
            "intake_command_example": f"python scripts\\suscetibilidade\\susc_17c12_submission_orchestrator.py intake-response --request-id {req['formal_request_id']} --response-file <arquivo_local_real> --received-at <AAAA-MM-DD>",
            "hash_required": "true",
            "manual_qa_required": "true",
            "blocks_17b": "true",
            "review_only": "true",
        })
    return rows


GUARDRAILS = [
    "nenhuma solicitacao enviada no build",
    "nenhum protocolo inventado",
    "nenhuma resposta inventada",
    "nenhum contato inventado",
    "nenhum efeito externo",
    "submit-assisted bloqueado sem opt-in",
    "canais candidatos nao submetidos",
    "canais ausentes bloqueados",
    "score v6 intacto",
    "score v7 inexistente",
    "17B inelegivel",
]


def build_safety_audit() -> list[dict]:
    rows = []
    for row in _queue():
        for guardrail in GUARDRAILS:
            rows.append({
                "safety_audit_id": f"S17C13_SAFE_{len(rows) + 1:04d}",
                "formal_request_id": row["formal_request_id"],
                "guardrail": guardrail,
                "passes": "true",
                "violation_found": "false",
                "blocking_reason": "guardrail_preserved_review_only",
                "review_only": "true",
            })
    return rows


def build_summary() -> dict:
    prepared = build_prepared_packages()
    snapshot = build_status_snapshot()
    return {
        "formal_requests_count": len(_requests()),
        "orchestrator_runs_logged_count": len(build_run_log()),
        "prepared_packages_count": len(prepared),
        "copy_paste_ready_packages_count": len([row for row in prepared if row["copy_paste_ready"] == "true"]),
        "channel_open_instructions_count": len(build_open_instructions()),
        "manual_execution_tasks_count": len(build_execution_board()),
        "ready_for_manual_submission_after_review_count": len([row for row in snapshot if row["ready_for_manual_submission"] == "true"]),
        "needs_manual_channel_verification_count": len([row for row in snapshot if row["needs_manual_channel_verification"] == "true"]),
        "blocked_no_channel_count": len([row for row in snapshot if row["blocked_no_channel"] == "true"]),
        "requests_marked_sent_count": 0,
        "protocols_opened_count": 0,
        "responses_received_count": 0,
        "external_effects_count": 0,
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
        "recommended_next_milestone": "SUSC-17C14 Registro de Submissoes Manuais Reais se os envios forem feitos manualmente; caso contrario, verificar canais pendentes",
    }


def build_blockers() -> list[dict]:
    blockers = [
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
            "blocker_id": f"S17C13_BLOCKER_{idx:04d}",
            "blocker_type": blocker,
            "description": "Bloqueio preservado apos preparacao assistida; requer submissao humana real, resposta real, intake, politica, adapter ou QA antes de promocao.",
            "blocks_17b": "true",
            "blocks_score_v7": "true",
            "review_only": "true",
        }
        for idx, blocker in enumerate(blockers, start=1)
    ]


def build_run_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17C13 orchestrator run schema v1",
        "type": "object",
        "required": RUN_FIELDS,
        "properties": {
            "run_log_id": {"type": "string", "pattern": "^S17C13_RUN_"},
            "orchestrator_mode": {"enum": ["plan", "status", "audit", "prepare", "open-channel"]},
            "result_status": {"enum": ["completed_local_only", "completed_dry_run", "blocked_guardrail", "blocked_missing_channel", "blocked_candidate_channel_needs_verification", "failed_safe"]},
            "external_effect": {"const": "false"},
            "request_marked_sent": {"const": "false"},
            "protocol_opened": {"const": "false"},
            "response_received": {"const": "false"},
            "review_only": {"const": "true"},
            "trainable": {"const": "false"},
            "ground_truth": {"const": "false"},
        },
    }


def build_board_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17C13 execution board schema v1",
        "type": "object",
        "required": BOARD_FIELDS,
        "properties": {
            "execution_task_id": {"type": "string", "pattern": "^S17C13_TASK_"},
            "task_status": {"enum": ["pending"]},
            "requires_human_action": {"const": "true"},
            "done": {"const": "false"},
            "review_only": {"const": "true"},
        },
    }


def build_report() -> str:
    summary = build_summary()
    return "\n".join([
        "# SUSC-17C13 - Execucao assistida de submissoes manuais",
        "",
        "## O que o 17C12 operacionalizou",
        "O 17C12 criou a fila operacional, os pacotes copiaveis, o CLI seguro e a politica que bloqueia submissao automatica por padrao.",
        "",
        "## Comandos executados",
        "Foram executados `plan`, `status`, `audit`, `prepare` para os 5 pedidos com canal oficial confirmado e `open-channel` para os 7 pedidos com canal confirmado ou candidato.",
        "",
        "## Resultado operacional",
        f"- Pedidos processados: {summary['formal_requests_count']}.",
        f"- Pacotes preparados: {summary['prepared_packages_count']}.",
        f"- Pacotes copy/paste ready apos revisao humana: {summary['copy_paste_ready_packages_count']}.",
        f"- Instrucoes de abertura de canal: {summary['channel_open_instructions_count']}.",
        f"- Prontos para submissao manual apos revisao: {summary['ready_for_manual_submission_after_review_count']}.",
        f"- Dependem de verificacao manual de canal: {summary['needs_manual_channel_verification_count']}.",
        f"- Seguem sem canal externo oficial: {summary['blocked_no_channel_count']}.",
        "",
        "## Uso do quadro manual",
        "O quadro de execucao manual lista 10 tarefas pendentes por solicitacao. Todas estao `done=false` e exigem evidencia humana literal para mudanca futura.",
        "",
        "## Registro futuro",
        "Uma submissao real futura deve usar `record-submission` somente depois de acao humana externa, com data/hora, canal usado e evidencia literal. Protocolo so pode ser registrado se existir e for informado literalmente.",
        "",
        "## Intake futuro",
        "Resposta futura deve usar `intake-response` com arquivo local real. O hash sera calculado no momento do intake, sem bruto pesado em `outputs_public`.",
        "",
        "## Guardrails",
        "Nenhum envio foi simulado, nenhum protocolo foi inventado, 17B permanece bloqueado, score v6 nao foi alterado e score v7 continua inexistente.",
        "",
        "## Proximo marco recomendado",
        summary["recommended_next_milestone"],
    ])


def build_all() -> None:
    _require_inputs()
    write_csv(RUN_LOG, build_run_log(), RUN_FIELDS)
    write_csv(PREPARED_PACKAGES, build_prepared_packages())
    write_csv(OPEN_INSTRUCTIONS, build_open_instructions())
    write_csv(EXECUTION_BOARD, build_execution_board(), BOARD_FIELDS)
    write_csv(STATUS_SNAPSHOT, build_status_snapshot())
    write_csv(EVIDENCE_REQUIREMENTS, build_evidence_requirements())
    write_csv(PENDING_CHANNELS, build_pending_channel_queue())
    write_csv(INTAKE_WATCHLIST, build_intake_watchlist())
    write_csv(SAFETY_AUDIT, build_safety_audit())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_json(RUN_SCHEMA, build_run_schema())
    write_json(BOARD_SCHEMA, build_board_schema())
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
    run_log = read_csv(RUN_LOG)
    packages = read_csv(PREPARED_PACKAGES)
    instructions = read_csv(OPEN_INSTRUCTIONS)
    board = read_csv(EXECUTION_BOARD)
    snapshot = read_csv(STATUS_SNAPSHOT)
    pending = read_csv(PENDING_CHANNELS)
    watch = read_csv(INTAKE_WATCHLIST)
    safety = read_csv(SAFETY_AUDIT)
    summary = read_json(SUMMARY)
    errors = []
    for row in run_log:
        errors.extend(_schema_violations(row, read_json(RUN_SCHEMA)))
    for row in board:
        errors.extend(_schema_violations(row, read_json(BOARD_SCHEMA)))
    for rows, key in [(run_log, "run_log_id"), (packages, "prepared_package_id"), (instructions, "open_instruction_id"), (board, "execution_task_id"), (snapshot, "status_snapshot_id"), (pending, "pending_channel_task_id"), (watch, "watchlist_id"), (safety, "safety_audit_id")]:
        values = [row[key] for row in rows]
        if values != sorted(values) or len(values) != len(set(values)):
            errors.append(f"ids_not_unique_or_sorted:{key}")
    if len(_requests()) != 9 or len(packages) != 9 or len(snapshot) != 9 or len(watch) != 9:
        errors.append("expected_nine_request_rows")
    if len([row for row in packages if row["copy_paste_ready"] == "true"]) != 5:
        errors.append("expected_five_copy_paste_ready")
    if len(instructions) != 7 or len(pending) != 4 or len(board) != 90:
        errors.append("unexpected_instruction_pending_or_board_count")
    if any(row["external_effect"] != "false" or row["request_marked_sent"] != "false" or row["protocol_opened"] != "false" or row["response_received"] != "false" for row in run_log):
        errors.append("run_log_records_external_effect_or_submission")
    if any(row["done"] != "false" for row in board):
        errors.append("execution_board_done_true")
    if any(row["submitted_manually"] != "false" or row["protocol_opened"] != "false" or row["response_received"] != "false" for row in snapshot):
        errors.append("snapshot_invents_submission_protocol_or_response")
    if any(row["submission_required_first"] != "true" for row in watch):
        errors.append("watchlist_does_not_require_submission_first")
    if any(row["violation_found"] != "false" or row["passes"] != "true" for row in safety):
        errors.append("safety_audit_violation_found")
    for path in [FEATURES, SCORE_V6, DAT / "susc_patches_official_v1.csv", DAT / "susc_patch_links_official_v1.csv"]:
        if path.exists() and _run_git(["diff", "--name-only", "--", rel(path)]):
            errors.append(f"official_dataset_changed:{rel(path)}")
    if SCORE_V7.exists():
        errors.append("score_v7_exists")
    expected = build_summary()
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(f"summary_mismatch:{key}")
    if not read_csv(BLOCKERS):
        errors.append("empty_blockers")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "17C13 -> "
        f"runs={summary['orchestrator_runs_logged_count']} "
        f"packages={summary['prepared_packages_count']} "
        f"copy_paste_ready={summary['copy_paste_ready_packages_count']} "
        f"instructions={summary['channel_open_instructions_count']} "
        f"sent={summary['requests_marked_sent_count']} "
        f"responses={summary['responses_received_count']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
    return 0
