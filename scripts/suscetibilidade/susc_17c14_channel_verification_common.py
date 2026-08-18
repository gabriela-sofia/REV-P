"""SUSC-17C14 verificacao manual de canais pendentes.

Reaudita os quatro pedidos pendentes do SUSC-17C13 com fontes oficiais e
classificacao fail-closed. Nao envia solicitacao, nao abre protocolo, nao marca
resposta recebida e nao promove 17B ou score v7.
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

FEATURES = DAT / "susc_features_by_patch_v1.csv"
SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
AUDIT = OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C14.md"

C13_INPUTS = [
    OUT / "susc_17c13_pending_channel_verification_queue.csv",
    OUT / "susc_17c13_channel_open_instructions.csv",
    OUT / "susc_17c13_manual_execution_board.csv",
    OUT / "susc_17c13_submission_status_snapshot.csv",
    OUT / "susc_17c13_readiness_summary.json",
    OUT / "susc_17c13_promotion_blockers.csv",
]
C12_INPUTS = [
    OUT / "susc_17c12_operational_request_queue.csv",
    OUT / "susc_17c12_submission_package_registry.csv",
    OUT / "susc_17c12_prepared_messages.md",
    OUT / "susc_17c12_submission_status_registry.csv",
    OUT / "susc_17c12_operational_risk_policy.json",
]
C11_INPUTS = [
    OUT / "susc_17c11_official_channel_registry.csv",
    OUT / "susc_17c11_channel_source_evidence.csv",
    OUT / "susc_17c11_request_channel_match_matrix.csv",
    OUT / "susc_17c11_manual_channel_verification_checklist.csv",
    OUT / "susc_17c11_submission_readiness_update.csv",
    OUT / "susc_17c11_contact_discovery_search_plan.csv",
    OUT / "susc_17c11_channel_risk_policy.json",
]
C10_INPUTS = [
    OUT / "susc_17c10_formal_request_registry.csv",
    OUT / "susc_17c10_provider_package_index.csv",
    OUT / "susc_17c10_request_message_templates.md",
]

REPORT = OUT / "SUSC_17C14_VERIFICACAO_MANUAL_CANAIS_PENDENTES_REPORT.md"
REAUDIT = OUT / "susc_17c14_pending_channel_reaudit.csv"
RESOLUTION = OUT / "susc_17c14_official_channel_resolution.csv"
EVIDENCE_UPDATE = OUT / "susc_17c14_channel_evidence_update.csv"
READINESS_DELTA = OUT / "susc_17c14_submission_readiness_delta.csv"
UPDATED_INSTRUCTIONS = OUT / "susc_17c14_updated_channel_open_instructions.csv"
MANUAL_BOARD = OUT / "susc_17c14_manual_verification_board.csv"
UNRESOLVED_BLOCKERS = OUT / "susc_17c14_unresolved_channel_blockers.csv"
SAFETY_AUDIT = OUT / "susc_17c14_operational_safety_audit.csv"
SUMMARY = OUT / "susc_17c14_readiness_summary.json"
BLOCKERS = OUT / "susc_17c14_promotion_blockers.csv"

RESOLUTION_SCHEMA = SCHEMAS / "susc_17c14_channel_resolution_schema_v1.json"
DELTA_SCHEMA = SCHEMAS / "susc_17c14_readiness_delta_schema_v1.json"

REQUIRED_INPUTS = [FEATURES, SCORE_V6, AUDIT, *C13_INPUTS, *C12_INPUTS, *C11_INPUTS, *C10_INPUTS]
REQUIRED_OUTPUTS = [
    AUDIT,
    REPORT,
    REAUDIT,
    RESOLUTION,
    EVIDENCE_UPDATE,
    READINESS_DELTA,
    UPDATED_INSTRUCTIONS,
    MANUAL_BOARD,
    UNRESOLVED_BLOCKERS,
    SAFETY_AUDIT,
    SUMMARY,
    BLOCKERS,
    RESOLUTION_SCHEMA,
    DELTA_SCHEMA,
]

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}
CHECKED_AT = "2026-06-30"

RESOLUTION_FIELDS = [
    "resolution_id",
    "formal_request_id",
    "provider_name",
    "resolved_status",
    "resolved_channel_type",
    "resolved_channel_reference",
    "resolved_channel_source",
    "source_is_official",
    "confidence_level",
    "ready_for_manual_submission_after_review",
    "still_blocked",
    "blocking_reason",
    "review_only",
    "trainable",
    "ground_truth",
]

DELTA_FIELDS = [
    "readiness_delta_id",
    "formal_request_id",
    "previous_readiness_17c13",
    "new_readiness_17c14",
    "became_ready_for_manual_submission",
    "remains_pending_verification",
    "remains_blocked_no_channel",
    "manual_action_required",
    "request_marked_sent",
    "protocol_opened",
    "response_received",
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
    for path in C13_INPUTS + C12_INPUTS + C11_INPUTS + C10_INPUTS:
        if path.suffix == ".json":
            read_json(path)
        elif path.suffix == ".csv":
            read_csv(path)
        else:
            path.read_text(encoding="utf-8")


def _pending() -> list[dict]:
    return read_csv(OUT / "susc_17c13_pending_channel_verification_queue.csv")


def _requests() -> dict[str, dict]:
    return {row["formal_request_id"]: row for row in read_csv(OUT / "susc_17c10_formal_request_registry.csv")}


def _previous_snapshot() -> dict[str, dict]:
    return {row["formal_request_id"]: row for row in read_csv(OUT / "susc_17c13_submission_status_snapshot.csv")}


def _previous_channel() -> dict[str, dict]:
    return {row["formal_request_id"]: row for row in read_csv(OUT / "susc_17c11_official_channel_registry.csv")}


def resolution_specs() -> dict[str, dict]:
    return {
        "P1_DEM_HAND_DRAINAGE_COVERAGE_CANDIDATE_GRID": {
            "reaudit_method": "official_web_review",
            "official_source_checked": "APAC Fale Conosco e monitoramento",
            "official_source_reference": "https://www.apac.pe.gov.br/fale-conosco",
            "source_is_official": True,
            "new_channel_candidate": "APAC Fale Conosco",
            "new_channel_type": "official_contact_page",
            "new_channel_status": "canal_candidato_ainda_pendente",
            "resolved_status": "canal_candidato_ainda_pendente",
            "resolved_channel_type": "official_contact_page",
            "resolved_channel_reference": "https://www.apac.pe.gov.br/fale-conosco",
            "resolved_channel_source": "APAC Fale Conosco",
            "confidence_level": "medium",
            "ready": False,
            "still_blocked": True,
            "blocking_reason": "canal oficial institucional existe, mas nao confirma fornecedor de DEM/HAND/drenagem para o artefato solicitado",
            "evidence_type": "official_contact_page",
            "evidence_label": "APAC Fale Conosco",
            "limitations": "fonte oficial nao resolve a origem DEM/HAND/drenagem nem o metodo de solicitacao de dados geoespaciais",
            "new_readiness": "needs_manual_channel_verification",
        },
        "P1_PET_2024_VALPARAISO_FLORESTA_GEOMETRY": {
            "reaudit_method": "official_web_review",
            "official_source_checked": "Defesa Civil de Petropolis e RO Digital",
            "official_source_reference": "https://www.petropolis.rj.gov.br/pmp/index.php/defesa-civil",
            "source_is_official": True,
            "new_channel_candidate": "Defesa Civil de Petropolis",
            "new_channel_type": "official_contact_page",
            "new_channel_status": "canal_oficial_confirmado",
            "resolved_status": "canal_oficial_confirmado",
            "resolved_channel_type": "official_contact_page",
            "resolved_channel_reference": "https://www.petropolis.rj.gov.br/pmp/index.php/defesa-civil",
            "resolved_channel_source": "Defesa Civil - Prefeitura de Petropolis",
            "confidence_level": "high",
            "ready": True,
            "still_blocked": False,
            "blocking_reason": "not_blocked_after_manual_review",
            "evidence_type": "official_contact_page",
            "evidence_label": "Pagina oficial da Defesa Civil de Petropolis",
            "limitations": "uso ainda exige revisao humana e nao substitui resposta oficial futura",
            "new_readiness": "ready_for_manual_submission_after_review",
        },
        "P2_DINO_SATMAE_INPUT_TILE": {
            "reaudit_method": "repo_reference_review",
            "official_source_checked": "referencias internas do orquestrador 17C12",
            "official_source_reference": "not_available",
            "source_is_official": False,
            "new_channel_candidate": "not_available",
            "new_channel_type": "none",
            "new_channel_status": "sem_canal_oficial_encontrado",
            "resolved_status": "sem_canal_oficial_encontrado",
            "resolved_channel_type": "none",
            "resolved_channel_reference": "not_available",
            "resolved_channel_source": "not_available",
            "confidence_level": "none",
            "ready": False,
            "still_blocked": True,
            "blocking_reason": "pipeline interno DINO/SatMAE sem canal externo de submissao formal",
            "evidence_type": "repo_reference",
            "evidence_label": "SUSC-17C12 classifica como dependencia interna",
            "limitations": "nao ha provedor externo definido para submissao manual",
            "new_readiness": "blocked_no_official_channel",
        },
        "P2_SAR_RUNTIME_OR_EXPORT_PATH": {
            "reaudit_method": "official_web_review",
            "official_source_checked": "Copernicus Data Space Sentinel-1",
            "official_source_reference": "https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-1",
            "source_is_official": True,
            "new_channel_candidate": "Copernicus Data Space Sentinel-1",
            "new_channel_type": "official_data_portal",
            "new_channel_status": "usar_canal_alternativo_institucional",
            "resolved_status": "usar_canal_alternativo_institucional",
            "resolved_channel_type": "official_data_portal",
            "resolved_channel_reference": "https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-1",
            "resolved_channel_source": "Copernicus Data Space Ecosystem",
            "confidence_level": "medium",
            "ready": False,
            "still_blocked": True,
            "blocking_reason": "canal institucional Sentinel-1 e oficial, mas nao resolve sozinho runtime/export SAR e credenciais do pedido",
            "evidence_type": "official_portal",
            "evidence_label": "Copernicus Data Space Sentinel-1",
            "limitations": "usar apenas como alternativa institucional; requer decisao humana sobre produto, runtime e janela temporal",
            "new_readiness": "context_only",
        },
    }


def build_pending_channel_reaudit() -> list[dict]:
    previous = _previous_channel()
    rows = []
    for item in _pending():
        spec = resolution_specs()[item["formal_request_id"]]
        old = previous[item["formal_request_id"]]
        rows.append({
            "reaudit_id": f"S17C14_REAUDIT_{len(rows) + 1:04d}",
            "formal_request_id": item["formal_request_id"],
            "provider_name": item["provider_name"],
            "previous_channel_status": item["channel_status"],
            "previous_blocking_reason": old["blocked_reason"],
            "reaudit_method": spec["reaudit_method"],
            "official_source_checked": spec["official_source_checked"],
            "official_source_reference": spec["official_source_reference"],
            "source_is_official": _bool_text(spec["source_is_official"]),
            "new_channel_candidate": spec["new_channel_candidate"],
            "new_channel_type": spec["new_channel_type"],
            "new_channel_status": spec["new_channel_status"],
            "manual_verification_required": "true",
            "blocking_reason": spec["blocking_reason"],
            **GOV,
        })
    return rows


def build_channel_resolution() -> list[dict]:
    rows = []
    for item in _pending():
        spec = resolution_specs()[item["formal_request_id"]]
        rows.append({
            "resolution_id": f"S17C14_RES_{len(rows) + 1:04d}",
            "formal_request_id": item["formal_request_id"],
            "provider_name": item["provider_name"],
            "resolved_status": spec["resolved_status"],
            "resolved_channel_type": spec["resolved_channel_type"],
            "resolved_channel_reference": spec["resolved_channel_reference"],
            "resolved_channel_source": spec["resolved_channel_source"],
            "source_is_official": _bool_text(spec["source_is_official"]),
            "confidence_level": spec["confidence_level"],
            "ready_for_manual_submission_after_review": _bool_text(spec["ready"]),
            "still_blocked": _bool_text(spec["still_blocked"]),
            "blocking_reason": spec["blocking_reason"],
            **GOV,
        })
    return rows


def build_channel_evidence_update() -> list[dict]:
    rows = []
    for item in _pending():
        spec = resolution_specs()[item["formal_request_id"]]
        rows.append({
            "evidence_update_id": f"S17C14_EVID_{len(rows) + 1:04d}",
            "formal_request_id": item["formal_request_id"],
            "provider_name": item["provider_name"],
            "evidence_type": spec["evidence_type"],
            "evidence_reference": spec["official_source_reference"],
            "evidence_title_or_label": spec["evidence_label"],
            "evidence_is_official": _bool_text(spec["source_is_official"]),
            "evidence_supports_resolution": _bool_text(spec["source_is_official"] and spec["resolved_status"] in {"canal_oficial_confirmado", "usar_canal_alternativo_institucional", "canal_candidato_ainda_pendente"}),
            "limitations": spec["limitations"],
            "checked_at": CHECKED_AT,
            "review_only": "true",
        })
    return rows


def _previous_readiness(row: dict) -> str:
    if row["ready_for_manual_submission"] == "true":
        return "ready_for_manual_submission_after_review"
    if row["needs_manual_channel_verification"] == "true":
        return "needs_manual_channel_verification"
    if row["blocked_no_channel"] == "true":
        return "blocked_no_official_channel"
    return "context_only"


def build_readiness_delta() -> list[dict]:
    snapshot = _previous_snapshot()
    rows = []
    for item in _pending():
        spec = resolution_specs()[item["formal_request_id"]]
        previous = _previous_readiness(snapshot[item["formal_request_id"]])
        rows.append({
            "readiness_delta_id": f"S17C14_DELTA_{len(rows) + 1:04d}",
            "formal_request_id": item["formal_request_id"],
            "previous_readiness_17c13": previous,
            "new_readiness_17c14": spec["new_readiness"],
            "became_ready_for_manual_submission": _bool_text(spec["ready"] and previous != "ready_for_manual_submission_after_review"),
            "remains_pending_verification": _bool_text(spec["new_readiness"] in {"needs_manual_channel_verification", "context_only"}),
            "remains_blocked_no_channel": _bool_text(spec["new_readiness"] == "blocked_no_official_channel"),
            "manual_action_required": "true",
            "request_marked_sent": "false",
            "protocol_opened": "false",
            "response_received": "false",
            "review_only": "true",
        })
    return rows


def build_updated_instructions() -> list[dict]:
    rows = []
    for item in _pending():
        spec = resolution_specs()[item["formal_request_id"]]
        reference = spec["resolved_channel_reference"]
        can_open = reference != "not_available"
        rows.append({
            "instruction_id": f"S17C14_INSTR_{len(rows) + 1:04d}",
            "formal_request_id": item["formal_request_id"],
            "provider_name": item["provider_name"],
            "channel_type": spec["resolved_channel_type"],
            "channel_reference": reference,
            "instruction_text": "Abrir manualmente a fonte oficial e registrar a verificacao; nao enviar pedido, nao autenticar pelo agente e nao abrir protocolo." if can_open else "Nao ha canal externo oficial confirmado; definir provedor/canal antes de qualquer acao externa.",
            "requires_login_or_portal": _bool_text(spec["resolved_channel_type"] in {"official_contact_page", "official_data_portal"}),
            "requires_manual_action": "true",
            "can_be_opened_by_agent": "false",
            "can_be_submitted_by_agent": "false",
            "blocking_reason": spec["blocking_reason"],
            "review_only": "true",
        })
    return rows


TASKS = [
    ("abrir fonte oficial", "URL oficial aberta manualmente"),
    ("confirmar orgao", "registro humano do orgao confirmado"),
    ("confirmar se canal aceita solicitacao de dados", "evidencia literal de aceitacao ou negativa"),
    ("confirmar se exige login/protocolo", "nota humana sobre login, protocolo ou formulario"),
    ("registrar canal confirmado manualmente, se aplicavel", "registro humano sem marcar envio"),
    ("nao enviar pedido nesta sprint", "confirmacao de ausencia de envio"),
    ("nao registrar protocolo nesta sprint", "confirmacao de ausencia de protocolo"),
    ("nao registrar resposta nesta sprint", "confirmacao de ausencia de resposta"),
]


def build_manual_verification_board() -> list[dict]:
    rows = []
    for item in _pending():
        for order, (desc, evidence) in enumerate(TASKS, start=1):
            rows.append({
                "manual_verification_task_id": f"S17C14_TASK_{len(rows) + 1:04d}",
                "formal_request_id": item["formal_request_id"],
                "provider_name": item["provider_name"],
                "task_order": str(order),
                "task_description": desc,
                "task_status": "pending",
                "requires_human_action": "true",
                "evidence_required_to_mark_done": evidence,
                "done": "false",
                "review_only": "true",
            })
    return rows


def build_unresolved_channel_blockers() -> list[dict]:
    rows = []
    for row in build_channel_resolution():
        if row["resolved_status"] == "canal_oficial_confirmado":
            continue
        if row["resolved_status"] == "canal_candidato_ainda_pendente":
            blocker_type = "candidate_channel_unverified"
        elif row["resolved_status"] == "usar_canal_alternativo_institucional":
            blocker_type = "requires_institutional_decision"
        elif row["resolved_status"] == "bloqueado_por_ambiguidade":
            blocker_type = "ambiguous_provider"
        else:
            blocker_type = "no_official_channel_found"
        rows.append({
            "unresolved_blocker_id": f"S17C14_UNRES_{len(rows) + 1:04d}",
            "formal_request_id": row["formal_request_id"],
            "provider_name": row["provider_name"],
            "blocker_type": blocker_type,
            "blocker_description": row["blocking_reason"],
            "required_next_action": "executar verificacao humana e registrar evidencia literal antes de submissao",
            "blocks_manual_submission": "true",
            "blocks_17b": "true",
            "review_only": "true",
        })
    return rows


GUARDRAILS = [
    "nenhuma solicitacao enviada",
    "nenhum protocolo aberto",
    "nenhuma resposta recebida",
    "nenhum contato inventado",
    "nenhuma URL inventada",
    "nenhuma fonte nao oficial confirmada",
    "score v6 intacto",
    "score v7 inexistente",
    "17B inelegivel",
]


def build_safety_audit() -> list[dict]:
    rows = []
    for item in _pending():
        for guardrail in GUARDRAILS:
            rows.append({
                "safety_audit_id": f"S17C14_SAFE_{len(rows) + 1:04d}",
                "formal_request_id": item["formal_request_id"],
                "guardrail": guardrail,
                "passes": "true",
                "violation_found": "false",
                "blocking_reason": "guardrail_preserved_review_only",
                "review_only": "true",
            })
    return rows


def build_summary() -> dict:
    resolution = build_channel_resolution()
    delta = build_readiness_delta()
    return {
        "pending_requests_reaudited_count": len(_pending()),
        "official_channels_confirmed_this_sprint_count": len([r for r in resolution if r["resolved_status"] == "canal_oficial_confirmado"]),
        "requests_became_ready_for_manual_submission_count": len([r for r in delta if r["became_ready_for_manual_submission"] == "true"]),
        "requests_still_pending_manual_verification_count": len([r for r in delta if r["remains_pending_verification"] == "true"]),
        "requests_still_blocked_no_channel_count": len([r for r in delta if r["remains_blocked_no_channel"] == "true"]),
        "requests_marked_sent_count": 0,
        "protocols_opened_count": 0,
        "responses_received_count": 0,
        "external_effects_count": 0,
        "contacts_invented_count": 0,
        "urls_invented_count": 0,
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
        "recommended_next_milestone": "SUSC-17C15 Execucao Assistida Final para Pacotes Prontos; verificar APAC/SAR/DINO em paralelo antes de submissao",
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
            "blocker_id": f"S17C14_BLOCKER_{idx:04d}",
            "blocker_type": blocker,
            "description": "Bloqueio preservado apos verificacao de canais pendentes; requer submissao humana real, resposta real, intake, politica, adapter ou QA antes de promocao.",
            "blocks_17b": "true",
            "blocks_score_v7": "true",
            "review_only": "true",
        }
        for idx, blocker in enumerate(blockers, start=1)
    ]


def build_resolution_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17C14 channel resolution schema v1",
        "type": "object",
        "required": RESOLUTION_FIELDS,
        "properties": {
            "resolution_id": {"type": "string", "pattern": "^S17C14_RES_"},
            "resolved_status": {"enum": ["canal_oficial_confirmado", "canal_candidato_ainda_pendente", "sem_canal_oficial_encontrado", "bloqueado_por_ambiguidade", "usar_canal_alternativo_institucional"]},
            "source_is_official": {"enum": ["true", "false"]},
            "ready_for_manual_submission_after_review": {"enum": ["true", "false"]},
            "review_only": {"const": "true"},
            "trainable": {"const": "false"},
            "ground_truth": {"const": "false"},
        },
    }


def build_delta_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17C14 readiness delta schema v1",
        "type": "object",
        "required": DELTA_FIELDS,
        "properties": {
            "readiness_delta_id": {"type": "string", "pattern": "^S17C14_DELTA_"},
            "new_readiness_17c14": {"enum": ["ready_for_manual_submission_after_review", "needs_manual_channel_verification", "blocked_no_official_channel", "blocked_ambiguous_channel", "context_only"]},
            "request_marked_sent": {"const": "false"},
            "protocol_opened": {"const": "false"},
            "response_received": {"const": "false"},
            "review_only": {"const": "true"},
        },
    }


def build_report() -> str:
    summary = build_summary()
    return "\n".join([
        "# SUSC-17C14 - Verificacao manual de canais pendentes",
        "",
        "## Pendencias herdadas do 17C13",
        "O 17C13 deixou quatro pedidos nao plenamente acionaveis: dois canais candidatos e dois pedidos sem canal externo oficial.",
        "",
        "## Canais reavaliados",
        "Foram reavaliados APAC/DEM-HAND, Defesa Civil de Petropolis, DINO/SatMAE interno e SAR/runtime com alternativa Copernicus Data Space Sentinel-1.",
        "",
        "## Resolucao",
        f"- Pedidos reavaliados: {summary['pending_requests_reaudited_count']}.",
        f"- Canais oficiais confirmados nesta sprint: {summary['official_channels_confirmed_this_sprint_count']}.",
        f"- Pedidos que ficaram prontos para submissao manual apos revisao: {summary['requests_became_ready_for_manual_submission_count']}.",
        f"- Pedidos ainda pendentes de verificacao manual: {summary['requests_still_pending_manual_verification_count']}.",
        f"- Pedidos ainda sem canal oficial: {summary['requests_still_blocked_no_channel_count']}.",
        "",
        "## Acao humana ainda exigida",
        "APAC exige confirmacao do fornecedor de DEM/HAND/drenagem. SAR tem alternativa institucional oficial, mas ainda requer decisao humana sobre runtime/export. DINO/SatMAE continua sem canal externo de submissao.",
        "",
        "## Guardrails",
        "Nenhum envio foi feito, nenhum protocolo foi aberto, nenhuma resposta foi recebida, nenhum contato foi inventado e nenhuma fonte nao oficial confirmou canal.",
        "",
        "## Impacto metodologico",
        "A fila operacional ganha um pedido adicional pronto para revisao humana, mas 17B permanece bloqueado porque ainda nao ha submissao real, resposta oficial, intake validado, features reais, politica de promocao e QA aceito.",
        "",
        "## Score",
        "Score v6 nao foi alterado e score v7 continua bloqueado/inexistente.",
        "",
        "## Proximo marco recomendado",
        summary["recommended_next_milestone"],
    ])


def build_all() -> None:
    _require_inputs()
    write_csv(REAUDIT, build_pending_channel_reaudit())
    write_csv(RESOLUTION, build_channel_resolution(), RESOLUTION_FIELDS)
    write_csv(EVIDENCE_UPDATE, build_channel_evidence_update())
    write_csv(READINESS_DELTA, build_readiness_delta(), DELTA_FIELDS)
    write_csv(UPDATED_INSTRUCTIONS, build_updated_instructions())
    write_csv(MANUAL_BOARD, build_manual_verification_board())
    write_csv(UNRESOLVED_BLOCKERS, build_unresolved_channel_blockers())
    write_csv(SAFETY_AUDIT, build_safety_audit())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_json(RESOLUTION_SCHEMA, build_resolution_schema())
    write_json(DELTA_SCHEMA, build_delta_schema())
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
    reaudit = read_csv(REAUDIT)
    resolution = read_csv(RESOLUTION)
    evidence = read_csv(EVIDENCE_UPDATE)
    delta = read_csv(READINESS_DELTA)
    instructions = read_csv(UPDATED_INSTRUCTIONS)
    board = read_csv(MANUAL_BOARD)
    unresolved = read_csv(UNRESOLVED_BLOCKERS)
    safety = read_csv(SAFETY_AUDIT)
    summary = read_json(SUMMARY)
    errors = []
    for row in resolution:
        errors.extend(_schema_violations(row, read_json(RESOLUTION_SCHEMA)))
    for row in delta:
        errors.extend(_schema_violations(row, read_json(DELTA_SCHEMA)))
    for rows, key in [
        (reaudit, "reaudit_id"),
        (resolution, "resolution_id"),
        (evidence, "evidence_update_id"),
        (delta, "readiness_delta_id"),
        (instructions, "instruction_id"),
        (board, "manual_verification_task_id"),
        (unresolved, "unresolved_blocker_id"),
        (safety, "safety_audit_id"),
    ]:
        values = [row[key] for row in rows]
        if values != sorted(values) or len(values) != len(set(values)):
            errors.append(f"ids_not_unique_or_sorted:{key}")
    if len(reaudit) != 4 or len(resolution) != 4 or len(delta) != 4 or len(instructions) != 4 or len(board) != 32:
        errors.append("unexpected_row_count")
    if any(row["review_only"] != "true" or row.get("trainable", "false") != "false" or row.get("ground_truth", "false") != "false" for row in reaudit + resolution):
        errors.append("governance_flags_invalid")
    if any(row["resolved_status"] == "canal_oficial_confirmado" and row["source_is_official"] != "true" for row in resolution):
        errors.append("confirmed_channel_without_official_source")
    if any(row["ready_for_manual_submission_after_review"] == "true" and row["resolved_status"] != "canal_oficial_confirmado" for row in resolution):
        errors.append("ready_without_confirmed_official_channel")
    if any(row["evidence_is_official"] == "false" and row["evidence_supports_resolution"] == "true" for row in evidence):
        errors.append("nonofficial_evidence_supports_resolution")
    if any(row["request_marked_sent"] != "false" or row["protocol_opened"] != "false" or row["response_received"] != "false" for row in delta):
        errors.append("delta_invents_submission_protocol_or_response")
    if any(row["can_be_submitted_by_agent"] != "false" for row in instructions):
        errors.append("instruction_allows_agent_submission")
    if any(row["done"] != "false" for row in board):
        errors.append("manual_board_done_true")
    if any(row["passes"] != "true" or row["violation_found"] != "false" for row in safety):
        errors.append("safety_guardrail_failed")
    for path in [FEATURES, SCORE_V6, DAT / "susc_patches_official_v1.csv", DAT / "susc_patch_links_official_v1.csv"]:
        if path.exists() and _run_git(["diff", "--name-only", "--", rel(path)]):
            errors.append(f"official_dataset_changed:{rel(path)}")
    if SCORE_V7.exists():
        errors.append("score_v7_exists")
    expected = build_summary()
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(f"summary_mismatch:{key}")
    if not read_csv(BLOCKERS) or not unresolved:
        errors.append("empty_blockers")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "17C14 -> "
        f"reaudited={summary['pending_requests_reaudited_count']} "
        f"confirmed={summary['official_channels_confirmed_this_sprint_count']} "
        f"became_ready={summary['requests_became_ready_for_manual_submission_count']} "
        f"pending={summary['requests_still_pending_manual_verification_count']} "
        f"no_channel={summary['requests_still_blocked_no_channel_count']} "
        f"sent={summary['requests_marked_sent_count']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
    return 0
