"""SUSC-17C11 descoberta de canais oficiais para solicitacoes formais.

Pacote review-only: registra canais oficiais ou candidatos para submissao
manual futura, sem enviar mensagens, abrir protocolos, preencher formularios,
baixar dados externos ou promover qualquer evidencia a treino/label.
"""

from __future__ import annotations

import csv
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

AUDIT = OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C11.md"

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
    OUT / "susc_17c9_source_artifact_inventory.csv",
    OUT / "susc_17c9_external_artifact_request_plan.csv",
    OUT / "susc_17c9_feature_extraction_unblock_matrix.csv",
    OUT / "susc_17c9_readiness_summary.json",
]
C4_INPUTS = [
    OUT / "susc_17c4_formal_request_queue.csv",
    OUT / "susc_17c4_summary.json",
]
C3_INPUTS = [
    OUT / "susc_17c3_official_source_acquisition_targets.csv",
    OUT / "susc_17c3_missing_official_artifacts_manifest.csv",
    OUT / "susc_17c3_summary.json",
]

REPORT = OUT / "SUSC_17C11_DESCOBERTA_CANAIS_OFICIAIS_REPORT.md"
CHANNEL_REGISTRY = OUT / "susc_17c11_official_channel_registry.csv"
SOURCE_EVIDENCE = OUT / "susc_17c11_channel_source_evidence.csv"
MATCH_MATRIX = OUT / "susc_17c11_request_channel_match_matrix.csv"
CHECKLIST = OUT / "susc_17c11_manual_channel_verification_checklist.csv"
READINESS_UPDATE = OUT / "susc_17c11_submission_readiness_update.csv"
SEARCH_PLAN = OUT / "susc_17c11_contact_discovery_search_plan.csv"
RISK_POLICY = OUT / "susc_17c11_channel_risk_policy.json"
SUMMARY = OUT / "susc_17c11_readiness_summary.json"
BLOCKERS = OUT / "susc_17c11_promotion_blockers.csv"

CHANNEL_SCHEMA = SCHEMAS / "susc_17c11_channel_registry_schema_v1.json"
MATCH_SCHEMA = SCHEMAS / "susc_17c11_request_channel_match_schema_v1.json"

REQUIRED_INPUTS = [FEATURES, SCORE_V6, AUDIT, *C10_INPUTS, *C9_INPUTS, *C4_INPUTS, *C3_INPUTS]
REQUIRED_OUTPUTS = [
    AUDIT,
    REPORT,
    CHANNEL_REGISTRY,
    SOURCE_EVIDENCE,
    MATCH_MATRIX,
    CHECKLIST,
    READINESS_UPDATE,
    SEARCH_PLAN,
    RISK_POLICY,
    SUMMARY,
    BLOCKERS,
    CHANNEL_SCHEMA,
    MATCH_SCHEMA,
]

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}
CHECKED_AT = "2026-06-30"

CHANNEL_FIELDS = [
    "channel_id",
    "formal_request_id",
    "provider_name_from_17c10",
    "provider_name_normalized",
    "channel_name",
    "channel_type",
    "channel_status",
    "channel_entrypoint_url",
    "official_domain",
    "official_source_title",
    "official_source_url",
    "official_source_supports_channel",
    "official_source_checked_at",
    "evidence_summary",
    "source_is_official",
    "source_is_nonofficial",
    "source_access_requires_account",
    "manual_verification_required",
    "can_submit_automatically",
    "do_not_submit_automatically",
    "contact_value_recorded",
    "contact_value_type",
    "contact_was_invented",
    "confidence",
    "blocked_reason",
    "review_only",
    "trainable",
    "ground_truth",
]

MATCH_FIELDS = [
    "match_id",
    "formal_request_id",
    "channel_id",
    "provider_name_from_request",
    "matched_channel_status",
    "request_priority",
    "submission_readiness",
    "can_submit_now",
    "should_submit_automatically",
    "manual_verification_required",
    "no_external_submission_performed",
    "request_marked_sent",
    "response_marked_received",
    "blocker_after_17c11",
    "next_manual_step",
    "review_only",
    "trainable",
    "ground_truth",
]


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _run_git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _require_inputs() -> None:
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))
    for path in C10_INPUTS + C9_INPUTS + C4_INPUTS + C3_INPUTS:
        if path.suffix == ".json":
            read_json(path)
        elif path.suffix == ".csv":
            read_csv(path)
        else:
            path.read_text(encoding="utf-8")


def _requests() -> list[dict]:
    return read_csv(OUT / "susc_17c10_formal_request_registry.csv")


def _request_by_id() -> dict[str, dict]:
    return {row["formal_request_id"]: row for row in _requests()}


def _spec(
    request_id: str,
    provider_normalized: str,
    channel_name: str,
    channel_type: str,
    channel_status: str,
    entrypoint: str,
    domain: str,
    source_title: str,
    source_url: str,
    evidence: str,
    confidence: str,
    blocked_reason: str,
    requires_account: bool = False,
) -> dict:
    req = _request_by_id()[request_id]
    return {
        "formal_request_id": request_id,
        "provider_name_from_17c10": req["provider_name"],
        "provider_name_normalized": provider_normalized,
        "channel_name": channel_name,
        "channel_type": channel_type,
        "channel_status": channel_status,
        "channel_entrypoint_url": entrypoint,
        "official_domain": domain,
        "official_source_title": source_title,
        "official_source_url": source_url,
        "official_source_supports_channel": _bool_text(channel_status in {"confirmed_official_source", "candidate_needs_manual_verification"}),
        "official_source_checked_at": CHECKED_AT,
        "evidence_summary": evidence,
        "source_is_official": _bool_text(source_url != "not_available"),
        "source_is_nonofficial": "false",
        "source_access_requires_account": _bool_text(requires_account),
        "manual_verification_required": "true",
        "can_submit_automatically": "false",
        "do_not_submit_automatically": "true",
        "contact_value_recorded": "official_url_only" if source_url != "not_available" else "not_available",
        "contact_value_type": "official_url" if source_url != "not_available" else "none",
        "contact_was_invented": "false",
        "confidence": confidence,
        "blocked_reason": blocked_reason,
        **GOV,
    }


def channel_specs() -> list[dict]:
    return [
        _spec(
            "P0_COMPDEC_RECIFE_PKG_FR_REC_002",
            "Secretaria Executiva de Defesa Civil do Recife",
            "Pagina oficial da Secretaria Executiva de Defesa Civil do Recife",
            "official_contact_page",
            "confirmed_official_source",
            "https://www2.recife.pe.gov.br/pagina/secretaria-executiva-de-defesa-civil",
            "recife.pe.gov.br",
            "Secretaria-Executiva de Defesa Civil - Prefeitura do Recife",
            "https://www2.recife.pe.gov.br/pagina/secretaria-executiva-de-defesa-civil",
            "Fonte oficial municipal identifica a Defesa Civil do Recife e canais de atendimento; usar apenas apos verificacao humana.",
            "high",
            "not_blocked_channel_found_manual_only",
        ),
        _spec(
            "P0_DRM_RJ_PKG_FR_PET_001",
            "Departamento de Recursos Minerais do Estado do Rio de Janeiro",
            "DRM-RJ e OuveRJ",
            "ombudsman_or_institutional_contact",
            "confirmed_official_source",
            "https://www.rj.gov.br/ouverj/",
            "rj.gov.br",
            "Portal DRM-RJ e OuveRJ",
            "https://www.rj.gov.br/drm/",
            "Portal oficial do DRM-RJ identifica o orgao, contatos institucionais e link para OuveRJ; submissao permanece manual.",
            "high",
            "not_blocked_channel_found_manual_only",
        ),
        _spec(
            "P1_CHIRPS_PRE_EVENT_WINDOW_REC_2022_05_24_30",
            "UCSB Climate Hazards Center / Google Earth Engine",
            "CHIRPS official data page e catalogo Earth Engine",
            "official_data_portal",
            "confirmed_official_source",
            "https://www.chc.ucsb.edu/data/chirps",
            "chc.ucsb.edu;developers.google.com",
            "CHIRPS e Earth Engine Data Catalog",
            "https://www.chc.ucsb.edu/data/chirps",
            "CHIRPS e descrito por fonte oficial do Climate Hazards Center; catalogo GEE confirma dataset tecnico.",
            "high",
            "not_blocked_channel_found_manual_only_no_download",
        ),
        _spec(
            "P1_DEM_HAND_DRAINAGE_COVERAGE_CANDIDATE_GRID",
            "APAC / fonte geoespacial a confirmar",
            "APAC Monitoramento e Fale Conosco",
            "candidate_contact_page",
            "candidate_needs_manual_verification",
            "https://www.apac.pe.gov.br/fale-conosco",
            "apac.pe.gov.br",
            "APAC Monitoramento e Fale Conosco",
            "https://www.apac.pe.gov.br/monitoramento",
            "APAC possui paginas oficiais de monitoramento e contato, mas o fornecedor de DEM/HAND/drenagem segue a confirmar.",
            "medium",
            "provider_ambiguous_dem_hand_drainage_requires_manual_source_confirmation",
        ),
        _spec(
            "P1_PET_2024_VALPARAISO_FLORESTA_GEOMETRY",
            "Secretaria de Protecao e Defesa Civil de Petropolis / DRM-RJ a confirmar",
            "Defesa Civil de Petropolis",
            "candidate_contact_page",
            "candidate_needs_manual_verification",
            "https://www.petropolis.rj.gov.br/pmp/index.php/defesa-civil",
            "petropolis.rj.gov.br",
            "Defesa Civil - Prefeitura de Petropolis",
            "https://www.petropolis.rj.gov.br/pmp/index.php/defesa-civil",
            "Pagina oficial municipal confirma canal da Defesa Civil; a relacao com DRM-RJ/NADE para este pedido precisa verificacao humana.",
            "medium",
            "provider_split_between_petropolis_and_drm_rj_requires_manual_confirmation",
        ),
        _spec(
            "P1_SENTINEL2_TILE_CANDIDATE_GRID",
            "Copernicus Data Space Ecosystem / Google Earth Engine",
            "Copernicus Data Space Sentinel-2 e STAC",
            "official_data_portal",
            "confirmed_official_source",
            "https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2",
            "dataspace.copernicus.eu",
            "Sentinel-2 - Copernicus Data Space Ecosystem",
            "https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2",
            "CDSE identifica acesso a dados Sentinel-2 e documentacao STAC; nenhum download ou query foi executado.",
            "high",
            "not_blocked_channel_found_manual_only_no_download",
            requires_account=False,
        ),
        _spec(
            "P1_SGB_CPRM_VECTOR_OR_COORDINATES",
            "Servico Geologico do Brasil - SGB/CPRM",
            "SIC e Ouvidoria SGB",
            "sic_or_ombudsman",
            "confirmed_official_source",
            "https://www.sgb.gov.br/servi%C3%A7o-de-informa%C3%A7%C3%B5es-ao-cidad%C3%A3o-sic",
            "sgb.gov.br",
            "Servico de Informacoes ao Cidadao - SIC - SGB",
            "https://www.sgb.gov.br/servi%C3%A7o-de-informa%C3%A7%C3%B5es-ao-cidad%C3%A3o-sic",
            "Pagina oficial do SGB informa canal SIC e referencia Fala.BR para pedidos de informacao.",
            "high",
            "not_blocked_channel_found_manual_only",
        ),
        _spec(
            "P2_DINO_SATMAE_INPUT_TILE",
            "Pipeline interno DINO/SatMAE",
            "Sem canal externo oficial confirmado",
            "none",
            "not_found",
            "not_available",
            "not_available",
            "not_available",
            "not_available",
            "Dependencia interna de runtime/tile real; nao ha fonte externa oficial para submissao nesta sprint.",
            "none",
            "internal_pipeline_no_external_submission_channel",
        ),
        _spec(
            "P2_SAR_RUNTIME_OR_EXPORT_PATH",
            "Runtime tecnico SAR a confirmar",
            "Sem canal externo oficial confirmado",
            "none",
            "not_found",
            "not_available",
            "not_available",
            "not_available",
            "not_available",
            "Runtime tecnico SAR e caminho de exportacao seguem sem provedor oficial definido.",
            "none",
            "ambiguous_runtime_provider_no_official_channel_found",
        ),
    ]


def build_channel_registry() -> list[dict]:
    rows = []
    for idx, spec in enumerate(sorted(channel_specs(), key=lambda item: item["formal_request_id"]), start=1):
        row = {"channel_id": f"S17C11_CH_{idx:04d}", **spec}
        rows.append(row)
    return rows


def build_source_evidence() -> list[dict]:
    evidence = []
    extra_urls = {
        "P0_DRM_RJ_PKG_FR_PET_001": [
            ("https://www.rj.gov.br/ouverj/", "Portal OuveRJ", "Portal oficial estadual de manifestacoes."),
        ],
        "P1_CHIRPS_PRE_EVENT_WINDOW_REC_2022_05_24_30": [
            ("https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY", "Earth Engine Data Catalog CHIRPS", "Catalogo oficial GEE descreve dataset CHIRPS v2."),
        ],
        "P1_SENTINEL2_TILE_CANDIDATE_GRID": [
            ("https://documentation.dataspace.copernicus.eu/APIs.html", "CDSE APIs", "Documentacao oficial cita STAC do CDSE."),
            ("https://documentation.dataspace.copernicus.eu/Support.html", "CDSE Support", "Documentacao oficial descreve suporte com conta CDSE."),
        ],
        "P1_DEM_HAND_DRAINAGE_COVERAGE_CANDIDATE_GRID": [
            ("https://www.apac.pe.gov.br/fale-conosco", "APAC Fale Conosco", "Pagina oficial de contato institucional da APAC."),
        ],
    }
    for channel in build_channel_registry():
        urls = [(channel["official_source_url"], channel["official_source_title"], channel["evidence_summary"])]
        urls.extend(extra_urls.get(channel["formal_request_id"], []))
        for url, title, summary in urls:
            official = url != "not_available"
            evidence.append({
                "evidence_id": f"S17C11_EVID_{len(evidence) + 1:04d}",
                "channel_id": channel["channel_id"],
                "formal_request_id": channel["formal_request_id"],
                "official_source_url": url,
                "official_source_domain": channel["official_domain"] if official else "not_available",
                "official_source_type": "institutional_page" if official else "none",
                "evidence_kind": "official_channel_reference" if official else "negative_channel_discovery_result",
                "evidence_summary": summary,
                "supports_channel_confirmation": _bool_text(channel["channel_status"] == "confirmed_official_source"),
                "source_is_official": _bool_text(official),
                "source_is_nonofficial": "false",
                "checked_at": CHECKED_AT,
                "used_for_confirmed_channel": _bool_text(channel["channel_status"] == "confirmed_official_source"),
                "review_only": "true",
            })
    return evidence


def _readiness_for_status(status: str) -> tuple[str, str, str]:
    if status == "confirmed_official_source":
        return (
            "ready_after_manual_verification",
            "manual_review_required_before_any_external_submission",
            "Revisar pagina oficial e registrar decisao humana antes de qualquer envio fora do repositorio.",
        )
    if status == "candidate_needs_manual_verification":
        return (
            "needs_provider_confirmation",
            "official_source_found_but_provider_mapping_unconfirmed",
            "Confirmar manualmente se o canal cobre exatamente o artefato solicitado.",
        )
    return (
        "blocked_no_channel_found",
        "no_official_external_channel_confirmed",
        "Definir provedor ou canal oficial antes de preparar submissao manual.",
    )


def build_match_matrix() -> list[dict]:
    reqs = _request_by_id()
    rows = []
    for channel in build_channel_registry():
        readiness, blocker, step = _readiness_for_status(channel["channel_status"])
        rid = channel["formal_request_id"]
        rows.append({
            "match_id": f"S17C11_MATCH_{len(rows) + 1:04d}",
            "formal_request_id": rid,
            "channel_id": channel["channel_id"],
            "provider_name_from_request": reqs[rid]["provider_name"],
            "matched_channel_status": channel["channel_status"],
            "request_priority": reqs[rid]["priority"],
            "submission_readiness": readiness,
            "can_submit_now": "false",
            "should_submit_automatically": "false",
            "manual_verification_required": "true",
            "no_external_submission_performed": "true",
            "request_marked_sent": "false",
            "response_marked_received": "false",
            "blocker_after_17c11": blocker,
            "next_manual_step": step,
            **GOV,
        })
    return rows


def build_manual_checklist() -> list[dict]:
    steps = [
        ("abrir_pagina_oficial", "Abrir manualmente a pagina oficial registrada."),
        ("confirmar_orgao", "Confirmar que o orgao cobre o pedido e o artefato."),
        ("confirmar_tipo_manifestacao", "Confirmar se o canal aceita pedido de informacao, solicitacao tecnica ou ouvidoria."),
        ("revisar_template_17c10", "Revisar o template 17C10 e anexos antes de qualquer uso externo."),
        ("registrar_decisao", "Registrar a decisao humana em intake futuro, sem marcar envio nesta sprint."),
    ]
    rows = []
    for match in build_match_matrix():
        for order, (code, desc) in enumerate(steps, start=1):
            rows.append({
                "checklist_id": f"S17C11_CHECK_{len(rows) + 1:04d}",
                "formal_request_id": match["formal_request_id"],
                "channel_id": match["channel_id"],
                "step_order": str(order),
                "step_code": code,
                "step_description": desc,
                "requires_human_action": "true",
                "done": "false",
                "external_action_performed": "false",
                "notes": "SUSC-17C11 registra canal; nao executa submissao externa.",
                "review_only": "true",
            })
    return rows


def build_readiness_update() -> list[dict]:
    reqs = _request_by_id()
    rows = []
    by_request = {row["formal_request_id"]: row for row in build_match_matrix()}
    for channel in build_channel_registry():
        rid = channel["formal_request_id"]
        match = by_request[rid]
        rows.append({
            "readiness_update_id": f"S17C11_READY_{len(rows) + 1:04d}",
            "formal_request_id": rid,
            "previous_17c10_request_status": reqs[rid]["request_status"],
            "channel_status_after_17c11": channel["channel_status"],
            "submission_readiness_after_17c11": match["submission_readiness"],
            "submission_channel_known_after_17c11": _bool_text(channel["channel_status"] == "confirmed_official_source"),
            "manual_verification_required": "true",
            "request_marked_sent": "false",
            "response_marked_received": "false",
            "external_protocol_opened": "false",
            "external_form_filled": "false",
            "external_download_performed": "false",
            "impact_on_17b": "none_17b_remains_blocked",
            "impact_on_score_v6": "none",
            "impact_on_score_v7": "none_score_v7_not_created",
            **GOV,
        })
    return rows


def build_search_plan() -> list[dict]:
    rows = []
    for channel in build_channel_registry():
        rows.append({
            "search_plan_id": f"S17C11_SEARCH_{len(rows) + 1:04d}",
            "formal_request_id": channel["formal_request_id"],
            "provider_name_normalized": channel["provider_name_normalized"],
            "official_domain_allowlist": channel["official_domain"],
            "search_scope": "official_provider_pages_only",
            "nonofficial_sources_allowed_for_confirmation": "false",
            "web_runtime_available": "true",
            "status_after_search": channel["channel_status"],
            "candidate_url_or_blocker": channel["channel_entrypoint_url"],
            "manual_follow_up": channel["blocked_reason"],
            "no_external_submission_performed": "true",
            "review_only": "true",
        })
    return rows


def build_risk_policy() -> dict:
    return {
        "milestone": "SUSC-17C11",
        "policy_status": "review_only_fail_closed",
        "official_sources_only": True,
        "nonofficial_sources_can_confirm_channel": False,
        "automatic_submission_allowed": False,
        "external_protocol_opening_allowed": False,
        "external_form_filling_allowed": False,
        "external_download_allowed": False,
        "contact_invention_allowed": False,
        "mark_request_sent_allowed": False,
        "mark_response_received_allowed": False,
        "feature_extraction_allowed": False,
        "official_patch_creation_allowed": False,
        "official_patch_link_creation_allowed": False,
        "score_v6_change_allowed": False,
        "score_v7_creation_allowed": False,
        "training_or_label_allowed": False,
        "blocked_statuses": ["candidate_needs_manual_verification", "not_found", "blocked_no_web_or_runtime", "blocked_ambiguous_provider"],
        "checked_at": CHECKED_AT,
    }


def build_summary() -> dict:
    channels = build_channel_registry()
    matches = build_match_matrix()
    confirmed = [c for c in channels if c["channel_status"] == "confirmed_official_source"]
    candidates = [c for c in channels if c["channel_status"] == "candidate_needs_manual_verification"]
    not_found = [c for c in channels if c["channel_status"] == "not_found"]
    return {
        "milestone": "SUSC-17C11",
        "formal_requests_count": len(_requests()),
        "official_channel_registry_rows_count": len(channels),
        "channels_discovered_count": len(confirmed) + len(candidates),
        "confirmed_official_channels_count": len(confirmed),
        "candidate_channels_needing_manual_verification_count": len(candidates),
        "not_found_channels_count": len(not_found),
        "blocked_no_web_or_runtime_count": len([c for c in channels if c["channel_status"] == "blocked_no_web_or_runtime"]),
        "requests_ready_after_manual_verification_count": len([m for m in matches if m["submission_readiness"] == "ready_after_manual_verification"]),
        "requests_still_needing_contact_channel_count": len([m for m in matches if m["submission_readiness"] != "ready_after_manual_verification"]),
        "requests_marked_sent_count": 0,
        "responses_received_count": 0,
        "external_protocols_opened_count": 0,
        "external_forms_filled_count": 0,
        "contacts_invented_count": 0,
        "nonofficial_sources_confirmed_count": 0,
        "external_downloads_performed_count": 0,
        "features_extracted_this_sprint": 0,
        "official_patch_created": False,
        "official_patch_link_created": False,
        "score_v6_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
        "eligible_for_17b_now": False,
        "eligible_for_score_v7": False,
        "training_or_model_created": False,
        "label_or_ground_truth_created": False,
        "raw_heavy_data_committed": False,
        "review_only": True,
    }


def build_blockers() -> list[dict]:
    rows = [
        ("S17C11_BLOCKER_001", "all_requests", "manual_verification_required_before_submission", "Nenhum canal pode ser usado automaticamente; toda submissao exige revisao humana fora do 17C11."),
        ("S17C11_BLOCKER_002", "SUSC-17B", "not_eligible_for_17b", "Descoberta de canal nao equivale a resposta oficial recebida nem evidencia observacional validada."),
        ("S17C11_BLOCKER_003", "score_v7", "score_v7_not_allowed", "Sem treino, labels, ground truth ou score v7 nesta sprint."),
    ]
    for channel in build_channel_registry():
        if channel["channel_status"] != "confirmed_official_source":
            rows.append((
                f"S17C11_BLOCKER_{len(rows) + 1:03d}",
                channel["formal_request_id"],
                channel["blocked_reason"],
                "Canal permanece pendente ou inexistente; requer decisao humana futura.",
            ))
    return [
        {
            "blocker_id": bid,
            "scope": scope,
            "blocker_type": btype,
            "description": desc,
            "blocks_submission_now": "true",
            "blocks_17b": "true",
            "blocks_score_v7": "true",
            "review_only": "true",
        }
        for bid, scope, btype, desc in rows
    ]


def build_channel_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17C11 channel registry schema v1",
        "type": "object",
        "required": CHANNEL_FIELDS,
        "properties": {
            "channel_id": {"type": "string", "pattern": "^S17C11_CH_"},
            "formal_request_id": {"type": "string"},
            "channel_type": {"enum": ["official_contact_page", "ombudsman_or_institutional_contact", "official_data_portal", "candidate_contact_page", "sic_or_ombudsman", "none"]},
            "channel_status": {"enum": ["confirmed_official_source", "candidate_needs_manual_verification", "not_found", "blocked_no_web_or_runtime", "blocked_ambiguous_provider"]},
            "source_is_official": {"enum": ["true", "false"]},
            "source_is_nonofficial": {"const": "false"},
            "manual_verification_required": {"const": "true"},
            "can_submit_automatically": {"const": "false"},
            "do_not_submit_automatically": {"const": "true"},
            "contact_was_invented": {"const": "false"},
            "review_only": {"const": "true"},
            "trainable": {"const": "false"},
            "ground_truth": {"const": "false"},
        },
    }


def build_match_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17C11 request channel match schema v1",
        "type": "object",
        "required": MATCH_FIELDS,
        "properties": {
            "match_id": {"type": "string", "pattern": "^S17C11_MATCH_"},
            "submission_readiness": {"enum": ["ready_after_manual_verification", "needs_provider_confirmation", "blocked_no_channel_found"]},
            "can_submit_now": {"const": "false"},
            "should_submit_automatically": {"const": "false"},
            "manual_verification_required": {"const": "true"},
            "no_external_submission_performed": {"const": "true"},
            "request_marked_sent": {"const": "false"},
            "response_marked_received": {"const": "false"},
            "review_only": {"const": "true"},
            "trainable": {"const": "false"},
            "ground_truth": {"const": "false"},
        },
    }


def build_report() -> str:
    summary = build_summary()
    return "\n".join([
        "# SUSC-17C11 - Descoberta de canais oficiais",
        "",
        "## Escopo",
        "",
        "Este pacote registra canais oficiais ou candidatos para as 9 solicitacoes formais do SUSC-17C10. O resultado e review-only: nao houve envio, protocolo, preenchimento de formulario externo, download, feature extraction, treino, label, ground truth, patch oficial, patch-link oficial, score v6 alterado ou score v7 criado.",
        "",
        "## Resultado",
        "",
        f"- Solicitacoes avaliadas: {summary['formal_requests_count']}.",
        f"- Linhas no registro de canais: {summary['official_channel_registry_rows_count']}.",
        f"- Canais descobertos em fonte oficial ou candidata: {summary['channels_discovered_count']}.",
        f"- Canais confirmados por fonte oficial: {summary['confirmed_official_channels_count']}.",
        f"- Canais candidatos que exigem verificacao manual: {summary['candidate_channels_needing_manual_verification_count']}.",
        f"- Solicitações sem canal externo oficial encontrado: {summary['not_found_channels_count']}.",
        f"- Solicitações prontas apenas apos verificacao manual: {summary['requests_ready_after_manual_verification_count']}.",
        f"- Solicitações que ainda precisam de confirmacao de canal/provedor: {summary['requests_still_needing_contact_channel_count']}.",
        "",
        "## Politica de uso",
        "",
        "- Fontes nao oficiais nao confirmam canal.",
        "- URL oficial registrada nao autoriza submissao automatica.",
        "- Todo uso externo exige decisao humana futura e registro em intake posterior.",
        "- Contatos nao foram inventados; somente URLs institucionais oficiais foram registradas.",
        "",
        "## Bloqueios preservados",
        "",
        "- 17B permanece bloqueado.",
        "- Score v7 permanece inexistente.",
        "- Score v6 permanece sem alteracao.",
        "- Nenhum dataset oficial de patches foi alterado.",
    ])


def build_all() -> None:
    _require_inputs()
    write_csv(CHANNEL_REGISTRY, build_channel_registry(), CHANNEL_FIELDS)
    write_csv(SOURCE_EVIDENCE, build_source_evidence())
    write_csv(MATCH_MATRIX, build_match_matrix(), MATCH_FIELDS)
    write_csv(CHECKLIST, build_manual_checklist())
    write_csv(READINESS_UPDATE, build_readiness_update())
    write_csv(SEARCH_PLAN, build_search_plan())
    write_json(RISK_POLICY, build_risk_policy())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_json(CHANNEL_SCHEMA, build_channel_schema())
    write_json(MATCH_SCHEMA, build_match_schema())
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
    missing = [p for p in REQUIRED_OUTPUTS if not p.exists()]
    if missing:
        print("MISSING: " + "; ".join(rel(p) for p in missing), file=sys.stderr)
        return 1

    channel_schema = read_json(CHANNEL_SCHEMA)
    match_schema = read_json(MATCH_SCHEMA)
    channels = read_csv(CHANNEL_REGISTRY)
    evidence = read_csv(SOURCE_EVIDENCE)
    matches = read_csv(MATCH_MATRIX)
    checklist = read_csv(CHECKLIST)
    readiness = read_csv(READINESS_UPDATE)
    summary = read_json(SUMMARY)
    policy = read_json(RISK_POLICY)

    errors = []
    if len(channels) != len(_requests()) or len(matches) != len(_requests()) or len(readiness) != len(_requests()):
        errors.append("expected_one_channel_match_readiness_row_per_formal_request")
    if [r["formal_request_id"] for r in channels] != sorted(r["formal_request_id"] for r in channels):
        errors.append("channel_registry_not_sorted")
    if len({r["channel_id"] for r in channels}) != len(channels):
        errors.append("duplicate_channel_id")
    for row in channels:
        errors.extend(_schema_violations(row, channel_schema))
        if row["channel_status"] == "confirmed_official_source" and row["source_is_official"] != "true":
            errors.append(f"{row['channel_id']}:confirmed_without_official_source")
        if row["source_is_nonofficial"] != "false":
            errors.append(f"{row['channel_id']}:nonofficial_source_used")
    for row in matches:
        errors.extend(_schema_violations(row, match_schema))
    if any(row["done"] != "false" or row["external_action_performed"] != "false" for row in checklist):
        errors.append("checklist_marks_external_action_done")
    if any(row["source_is_nonofficial"] != "false" for row in evidence):
        errors.append("nonofficial_evidence_registered")
    if any(row["request_marked_sent"] != "false" or row["response_marked_received"] != "false" for row in readiness):
        errors.append("readiness_marks_sent_or_received")
    if not read_csv(BLOCKERS):
        errors.append("empty_blockers")
    expected_summary = build_summary()
    for key in [
        "formal_requests_count",
        "channels_discovered_count",
        "confirmed_official_channels_count",
        "candidate_channels_needing_manual_verification_count",
        "not_found_channels_count",
        "requests_marked_sent_count",
        "responses_received_count",
        "contacts_invented_count",
        "nonofficial_sources_confirmed_count",
        "features_extracted_this_sprint",
    ]:
        if summary.get(key) != expected_summary[key]:
            errors.append(f"summary_mismatch:{key}")
    if summary["score_v6_changed"] or summary["score_v7_created"]:
        errors.append("score_guardrail_failed")
    if policy["automatic_submission_allowed"] or policy["training_or_label_allowed"]:
        errors.append("risk_policy_allows_forbidden_action")
    for path in [
        FEATURES,
        SCORE_V6,
        DAT / "susc_patches_official_v1.csv",
        DAT / "susc_patch_links_official_v1.csv",
    ]:
        if path.exists() and _run_git(["diff", "--name-only", "--", rel(path)]):
            errors.append(f"official_dataset_changed:{rel(path)}")
    if SCORE_V7.exists():
        errors.append("score_v7_exists")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "17C11 -> "
        f"requests={summary['formal_requests_count']} "
        f"confirmed_channels={summary['confirmed_official_channels_count']} "
        f"candidate_channels={summary['candidate_channels_needing_manual_verification_count']} "
        f"not_found={summary['not_found_channels_count']} "
        f"sent={summary['requests_marked_sent_count']} "
        f"responses={summary['responses_received_count']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
    return 0
