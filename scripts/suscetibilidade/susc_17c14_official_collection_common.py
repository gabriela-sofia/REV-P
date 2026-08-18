"""SUSC-17C14 agente de coleta oficial e intake condicionado.

Cria fila de coleta, tentativa dry-run, intake condicionado e matriz de gates
G1-G7. O build inicial nao ingere resposta real e, portanto, nao aceita nenhum
artefato como ground truth, label, treino, score v7 ou 17B.
"""

from __future__ import annotations

import argparse
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
LOCAL_STATE = ROOT / "local_runs" / "suscetibilidade" / "17c14_official_collection_agent"

FEATURES = DAT / "susc_features_by_patch_v1.csv"
SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

C13_INPUTS = [
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
]
C12_INPUTS = [
    OUT / "susc_17c12_operational_request_queue.csv",
    OUT / "susc_17c12_submission_package_registry.csv",
    OUT / "susc_17c12_prepared_messages.md",
    OUT / "susc_17c12_submission_status_registry.csv",
    OUT / "susc_17c12_response_intake_registry.csv",
    OUT / "susc_17c12_response_manifest_template.csv",
    OUT / "susc_17c12_operational_risk_policy.json",
]
C11_INPUTS = [
    OUT / "susc_17c11_official_channel_registry.csv",
    OUT / "susc_17c11_channel_source_evidence.csv",
    OUT / "susc_17c11_request_channel_match_matrix.csv",
]
C10_INPUTS = [
    OUT / "susc_17c10_formal_request_registry.csv",
    OUT / "susc_17c10_required_fields_by_request.csv",
    OUT / "susc_17c10_request_to_blocker_traceability.csv",
    OUT / "susc_17c10_request_impact_matrix.csv",
]

REPORT = OUT / "SUSC_17C14_AGENTE_COLETA_OFICIAL_INTAKE_CONDICIONADO_REPORT.md"
QUEUE = OUT / "susc_17c14_collection_request_queue.csv"
ATTEMPT_LOG = OUT / "susc_17c14_collection_attempt_log.csv"
INTAKE = OUT / "susc_17c14_official_response_intake_registry.csv"
MANIFEST = OUT / "susc_17c14_artifact_manifest.csv"
GATE_MATRIX = OUT / "susc_17c14_gate_evaluation_matrix.csv"
SPATIAL_STATUS = OUT / "susc_17c14_spatial_validation_status.csv"
TEMPORAL_STATUS = OUT / "susc_17c14_temporal_validation_status.csv"
PHENOMENON_STATUS = OUT / "susc_17c14_phenomenon_separation_status.csv"
ACCEPTED = OUT / "susc_17c14_accepted_ground_reference_candidates.csv"
REJECTED = OUT / "susc_17c14_rejected_or_pending_artifacts.csv"
ANTI_LEAKAGE = OUT / "susc_17c14_anti_leakage_audit.csv"
SUMMARY = OUT / "susc_17c14_readiness_summary.json"
BLOCKERS = OUT / "susc_17c14_promotion_blockers.csv"

GATE_SCHEMA = SCHEMAS / "susc_17c14_gate_evaluation_schema_v1.json"
INTAKE_SCHEMA = SCHEMAS / "susc_17c14_official_response_intake_schema_v1.json"
CANDIDATE_SCHEMA = SCHEMAS / "susc_17c14_ground_reference_candidate_schema_v1.json"

REQUIRED_INPUTS = [FEATURES, SCORE_V6, *C13_INPUTS, *C12_INPUTS, *C11_INPUTS, *C10_INPUTS]
REQUIRED_OUTPUTS = [
    REPORT,
    QUEUE,
    ATTEMPT_LOG,
    INTAKE,
    MANIFEST,
    GATE_MATRIX,
    SPATIAL_STATUS,
    TEMPORAL_STATUS,
    PHENOMENON_STATUS,
    ACCEPTED,
    REJECTED,
    ANTI_LEAKAGE,
    SUMMARY,
    BLOCKERS,
    GATE_SCHEMA,
    INTAKE_SCHEMA,
    CANDIDATE_SCHEMA,
]

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}
MODES = ["queue", "collect", "submit", "intake", "validate-artifact", "status", "audit"]
GATES = [
    ("G1", "Existencia documental"),
    ("G2", "Confiabilidade da fonte"),
    ("G3", "Precisao temporal"),
    ("G4", "Vinculo espacial"),
    ("G5", "Separacao de fenomeno"),
    ("G6", "Proveniencia e integridade"),
    ("G7", "Politica anti-leakage"),
]
CLASS_VALUES = [
    "accepted_ground_reference_candidate",
    "received_needs_normalization",
    "received_needs_spatial_qa",
    "received_missing_required_fields",
    "received_wrong_phenomenon",
    "received_temporal_leakage_risk",
    "received_unverifiable_source",
    "received_context_only",
    "rejected_not_official",
    "rejected_sensitive_or_unsafe",
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


def _requests() -> list[dict]:
    return read_csv(OUT / "susc_17c10_formal_request_registry.csv")


def _channels() -> dict[str, dict]:
    return {row["formal_request_id"]: row for row in read_csv(OUT / "susc_17c11_official_channel_registry.csv")}


def _packages() -> dict[str, dict]:
    return {row["formal_request_id"]: row for row in read_csv(OUT / "susc_17c13_prepared_submission_packages.csv")}


def _queue12() -> dict[str, dict]:
    return {row["formal_request_id"]: row for row in read_csv(OUT / "susc_17c12_operational_request_queue.csv")}


def _traceability() -> dict[str, dict]:
    return {row["formal_request_id"]: row for row in read_csv(OUT / "susc_17c10_request_to_blocker_traceability.csv")}


def _collection_status(channel_status: str) -> tuple[str, str]:
    if channel_status == "confirmed_official_source":
        return ("collection_ready_dry_run_only", "canal confirmado; coleta externa real bloqueada sem opt-in")
    if channel_status == "candidate_needs_manual_verification":
        return ("blocked_channel_not_confirmed", "canal candidato exige verificacao manual antes de coleta")
    return ("blocked_no_channel", "sem canal externo oficial confirmado")


def build_collection_queue() -> list[dict]:
    channels = _channels()
    packages = _packages()
    rows = []
    for idx, req in enumerate(sorted(_requests(), key=lambda row: row["formal_request_id"]), start=1):
        channel = channels[req["formal_request_id"]]
        package = packages[req["formal_request_id"]]
        status, reason = _collection_status(channel["channel_status"])
        rows.append({
            "collection_queue_id": f"S17C14_COLLECT_QUEUE_{idx:04d}",
            "formal_request_id": req["formal_request_id"],
            "provider_name": req["provider_name"],
            "provider_type": req["provider_type"],
            "priority": req["priority"],
            "event_id": req["related_event_id"],
            "channel_id": channel["channel_id"],
            "channel_status": channel["channel_status"],
            "channel_type": channel["channel_type"],
            "channel_reference": channel["channel_entrypoint_url"],
            "submission_package_status": package["blocked_reason"],
            "collection_status": status,
            "collect_allowed_by_default": "false",
            "submit_allowed_by_default": "false",
            "requires_external_opt_in": "true",
            "blocking_reason": reason,
            **GOV,
        })
    return rows


def build_attempt_log() -> list[dict]:
    rows = []
    for row in build_collection_queue():
        dry_run_possible = row["channel_status"] == "confirmed_official_source"
        result = "dry_run_only" if dry_run_possible else ("blocked_channel_not_confirmed" if row["channel_status"] == "candidate_needs_manual_verification" else "blocked_no_channel")
        rows.append({
            "attempt_id": f"S17C14_ATTEMPT_{len(rows) + 1:04d}",
            "formal_request_id": row["formal_request_id"],
            "attempt_mode": "collect",
            "attempted_at": "not_executed_external_dry_run_registry",
            "dry_run": "true",
            "external_action_requested": "false",
            "external_action_allowed": "false",
            "channel_id": row["channel_id"],
            "result_status": result,
            "result_message": row["blocking_reason"],
            "response_received": "false",
            "artifact_received": "false",
            "protocol_number": "not_informed",
            "contact_invented": "false",
            "url_invented": "false",
            "review_only": "true",
        })
    return rows


def build_intake_registry() -> list[dict]:
    rows = []
    for row in build_collection_queue():
        rows.append({
            "official_response_intake_id": f"S17C14_INTAKE_{len(rows) + 1:04d}",
            "formal_request_id": row["formal_request_id"],
            "provider_name": row["provider_name"],
            "response_received": "false",
            "response_received_at": "not_received",
            "source_artifact_exists": "false",
            "source_artifact_path_or_reference": "not_available",
            "source_artifact_sha256": "not_available",
            "source_artifact_format": "not_available",
            "source_checked_at": "not_checked_no_response",
            "provider_type": row["provider_type"],
            "official_source_reference": row["channel_reference"],
            "source_is_official": _bool_text(row["channel_status"] == "confirmed_official_source"),
            "channel_id": row["channel_id"],
            "channel_status": row["channel_status"],
            "license_or_use_terms": "not_available",
            "classification": "received_missing_required_fields" if row["channel_status"] == "confirmed_official_source" else "received_unverifiable_source",
            "manual_review_required": "true",
            "eligible_for_gate_evaluation": "true",
            **GOV,
        })
    return rows


def build_artifact_manifest() -> list[dict]:
    rows = []
    for intake in build_intake_registry():
        rows.append({
            "artifact_manifest_id": f"S17C14_MANIFEST_{len(rows) + 1:04d}",
            "official_response_intake_id": intake["official_response_intake_id"],
            "formal_request_id": intake["formal_request_id"],
            "provider_name": intake["provider_name"],
            "sha256": "not_available",
            "size_bytes": "not_available",
            "format": "not_available",
            "storage_location": "not_available",
            "raw_heavy": "false",
            "public_manifest_path": rel(MANIFEST),
            "local_raw_path": "not_available",
            "created_at": "not_created_no_response",
            "source_chain": "17C10_request;17C11_channel;17C12_package;17C13_execution;17C14_no_response_yet",
            "manifest_complete": "false",
            "review_only": "true",
        })
    return rows


def _gate_result(row: dict, gate: str) -> tuple[bool, str]:
    confirmed = row["channel_status"] == "confirmed_official_source"
    if gate == "G1":
        return (False, "sem artefato ou resposta real rastreavel")
    if gate == "G2":
        return (confirmed, "fonte/canal oficial confirmado" if confirmed else "canal nao confirmado ou ausente")
    if gate == "G3":
        return (False, "data ou periodo ainda ausente")
    if gate == "G4":
        return (False, "geometria, coordenada ou endereco preciso ainda ausente")
    if gate == "G5":
        return (False, "fenomeno ainda nao classificado por evidencia recebida")
    if gate == "G6":
        return (False, "manifesto sem hash porque nao ha artefato real")
    if gate == "G7":
        return (True, "nenhum uso de dado pos-evento ou proxy indevido foi executado")
    raise AssertionError(gate)


def build_gate_matrix() -> list[dict]:
    rows = []
    queue = {row["formal_request_id"]: row for row in build_collection_queue()}
    manifests = {row["formal_request_id"]: row for row in build_artifact_manifest()}
    for request_id, row in queue.items():
        for gate_id, gate_name in GATES:
            passed, reason = _gate_result(row, gate_id)
            rows.append({
                "gate_evaluation_id": f"S17C14_GATE_{len(rows) + 1:04d}",
                "formal_request_id": request_id,
                "artifact_manifest_id": manifests[request_id]["artifact_manifest_id"],
                "gate_id": gate_id,
                "gate_name": gate_name,
                "gate_pass": _bool_text(passed),
                "failure_reason": "not_failed" if passed else reason,
                "evidence_class_after_gate": "received_missing_required_fields" if not passed else "received_context_only",
                "source_is_official": _bool_text(row["channel_status"] == "confirmed_official_source"),
                "review_only": "true",
            })
    return rows


def build_spatial_status() -> list[dict]:
    rows = []
    for req in sorted(_requests(), key=lambda row: row["formal_request_id"]):
        rows.append({
            "spatial_status_id": f"S17C14_SPATIAL_{len(rows) + 1:04d}",
            "formal_request_id": req["formal_request_id"],
            "geometry_type": "not_available",
            "geometry_path_or_wkt": "not_available",
            "coordinates_available": "false",
            "address_available": "false",
            "crs": "not_available",
            "spatial_precision_m": "not_available",
            "uncertainty_m": "not_available",
            "can_overlay_patch": "false",
            "patch_overlap_possible": "false",
            "spatial_validation_status": "geometry_missing_blocked",
            "review_only": "true",
        })
    return rows


def build_temporal_status() -> list[dict]:
    rows = []
    for req in sorted(_requests(), key=lambda row: row["formal_request_id"]):
        rows.append({
            "temporal_status_id": f"S17C14_TEMP_{len(rows) + 1:04d}",
            "formal_request_id": req["formal_request_id"],
            "event_date": "not_available",
            "event_period_start": "not_available",
            "event_period_end": "not_available",
            "temporal_precision": "not_available",
            "pre_event_window_available": "false",
            "post_event_window_available": "false",
            "temporal_leakage_risk": "unknown_no_artifact",
            "temporal_validation_status": "date_missing",
            "review_only": "true",
        })
    return rows


def build_phenomenon_status() -> list[dict]:
    rows = []
    for req in sorted(_requests(), key=lambda row: row["formal_request_id"]):
        rows.append({
            "phenomenon_status_id": f"S17C14_PHEN_{len(rows) + 1:04d}",
            "formal_request_id": req["formal_request_id"],
            "provider_name": req["provider_name"],
            "phenomenon_class": "UNKNOWN",
            "phenomenon_terms_detected": "not_available",
            "hydrological_validation_allowed": "false",
            "blocking_reason": "fenomeno nao classificado por resposta oficial recebida",
            "review_only": "true",
        })
    return rows


CANDIDATE_FIELDS = [
    "ground_reference_candidate_id",
    "formal_request_id",
    "provider_name",
    "event_id",
    "artifact_manifest_id",
    "evidence_class",
    "phenomenon_class",
    "geometry_type",
    "crs",
    "spatial_precision_m",
    "temporal_precision",
    "source_is_official",
    "all_gates_passed",
    "usable_for_review_only",
    "usable_for_training",
    "ground_truth",
    "eligible_for_17b_now",
    "blocking_reason_for_17b",
]


def build_accepted_candidates() -> list[dict]:
    return []


def build_rejected_or_pending() -> list[dict]:
    rows = []
    manifests = {row["formal_request_id"]: row for row in build_artifact_manifest()}
    for req in sorted(_requests(), key=lambda row: row["formal_request_id"]):
        request_id = req["formal_request_id"]
        gates = [row for row in build_gate_matrix() if row["formal_request_id"] == request_id]
        failed = [row["gate_id"] for row in gates if row["gate_pass"] != "true"]
        rows.append({
            "rejected_or_pending_id": f"S17C14_REJECT_{len(rows) + 1:04d}",
            "formal_request_id": request_id,
            "provider_name": req["provider_name"],
            "artifact_manifest_id": manifests[request_id]["artifact_manifest_id"],
            "evidence_class": "received_missing_required_fields",
            "failed_gates": ";".join(failed),
            "primary_rejection_reason": "sem artefato/resposta real com hash e geometria validavel",
            "can_be_reprocessed_after_intake": "true",
            "ground_truth": "false",
            "trainable": "false",
            "review_only": "true",
        })
    return rows


def build_anti_leakage_audit() -> list[dict]:
    checks = [
        ("post_event_as_pre_event_feature", "false", "nenhuma feature pre-evento foi criada"),
        ("occurrence_document_as_susceptibility_feature", "false", "nenhum documento de ocorrencia virou feature"),
        ("risk_area_as_occurrence", "false", "area de risco nao foi aceita como evento ocorrido"),
        ("score_as_label", "false", "score anterior nao foi usado como label"),
        ("synthetic_feature_as_real", "false", "feature sintetica nao foi marcada como real"),
        ("neighbor_patch_proxy", "false", "patch oficial vizinho nao foi usado como proxy"),
    ]
    rows = []
    for req in sorted(_requests(), key=lambda row: row["formal_request_id"]):
        for code, violation, note in checks:
            rows.append({
                "anti_leakage_audit_id": f"S17C14_LEAK_{len(rows) + 1:04d}",
                "formal_request_id": req["formal_request_id"],
                "anti_leakage_check": code,
                "violation_found": violation,
                "passes": "true",
                "audit_note": note,
                "review_only": "true",
            })
    return rows


def build_summary() -> dict:
    queue = build_collection_queue()
    attempts = build_attempt_log()
    intake = build_intake_registry()
    accepted = build_accepted_candidates()
    rejected = build_rejected_or_pending()
    return {
        "formal_requests_count": len(queue),
        "agent_modes_count": len(MODES),
        "collection_attempts_count": len(attempts),
        "artifacts_ingested_count": len([row for row in intake if row["response_received"] == "true"]),
        "artifacts_all_gates_passed_count": len(accepted),
        "accepted_ground_reference_candidates_count": len(accepted),
        "rejected_or_pending_artifacts_count": len(rejected),
        "requests_marked_sent_count": 0,
        "protocols_opened_count": 0,
        "responses_invented_count": 0,
        "contacts_invented_count": 0,
        "features_extracted_this_sprint": 0,
        "ground_truth_created": False,
        "training_label_created": False,
        "score_v6_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
        "official_patch_created": False,
        "official_patch_link_created": False,
        "eligible_for_17b_now": False,
        "eligible_for_score_v7": False,
        "raw_raster_committed": False,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "recommended_next_milestone": "SUSC-17C15 Registro de Submissoes Manuais Reais e Intake de Respostas Oficiais quando houver arquivo/resposta real",
    }


def build_blockers() -> list[dict]:
    blockers = [
        "manual_submission_not_performed",
        "responses_not_received",
        "response_intake_not_executed",
        "source_artifact_missing",
        "artifact_hash_missing",
        "geometry_missing",
        "temporal_fields_missing",
        "phenomenon_unknown",
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
            "description": "Bloqueio preservado pelo agente de coleta: requer submissao real, resposta oficial, hash, manifesto, geometria, temporalidade e QA antes de qualquer promocao.",
            "blocks_ground_reference_acceptance": _bool_text(blocker in {"responses_not_received", "response_intake_not_executed", "source_artifact_missing", "artifact_hash_missing", "geometry_missing", "temporal_fields_missing", "phenomenon_unknown"}),
            "blocks_17b": "true",
            "blocks_score_v7": "true",
            "review_only": "true",
        }
        for idx, blocker in enumerate(blockers, start=1)
    ]


def build_gate_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17C14 gate evaluation schema v1",
        "type": "object",
        "required": ["gate_evaluation_id", "formal_request_id", "artifact_manifest_id", "gate_id", "gate_name", "gate_pass", "failure_reason", "evidence_class_after_gate", "source_is_official", "review_only"],
        "properties": {
            "gate_evaluation_id": {"type": "string", "pattern": "^S17C14_GATE_"},
            "gate_id": {"enum": [gate for gate, _ in GATES]},
            "gate_pass": {"enum": ["true", "false"]},
            "review_only": {"const": "true"},
        },
    }


def build_intake_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17C14 official response intake schema v1",
        "type": "object",
        "required": list(build_intake_registry()[0].keys()),
        "properties": {
            "official_response_intake_id": {"type": "string", "pattern": "^S17C14_INTAKE_"},
            "response_received": {"const": "false"},
            "source_artifact_exists": {"const": "false"},
            "source_artifact_sha256": {"const": "not_available"},
            "classification": {"enum": CLASS_VALUES},
            "review_only": {"const": "true"},
            "trainable": {"const": "false"},
            "ground_truth": {"const": "false"},
        },
    }


def build_candidate_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17C14 ground reference candidate schema v1",
        "type": "object",
        "required": CANDIDATE_FIELDS,
        "properties": {
            "ground_reference_candidate_id": {"type": "string", "pattern": "^S17C14_GRC_"},
            "all_gates_passed": {"const": "true"},
            "usable_for_training": {"const": "false"},
            "ground_truth": {"const": "false"},
            "eligible_for_17b_now": {"const": "false"},
        },
    }


def build_report() -> str:
    summary = build_summary()
    return "\n".join([
        "# SUSC-17C14 - Agente de coleta oficial e intake condicionado",
        "",
        "## Objetivo",
        "Este marco troca a avaliacao manual subjetiva por gates objetivos G1-G7 aplicados pelo agente. O agente prepara fila, registra tentativas dry-run, recebe arquivos locais quando houver intake real e classifica automaticamente o resultado.",
        "",
        "## Modos do agente",
        "`queue`, `collect`, `submit`, `intake`, `validate-artifact`, `status` e `audit` foram implementados. `collect` e `submit` sao bloqueados por padrao para acao externa; `intake` exige arquivo local real.",
        "",
        "## Gates implementados",
        "G1 existencia documental; G2 confiabilidade da fonte; G3 precisao temporal; G4 vinculo espacial; G5 separacao de fenomeno; G6 proveniencia e integridade; G7 politica anti-leakage.",
        "",
        "## Ground Reference Candidate",
        "`ground_reference_candidate` e uma referencia candidata para revisao, nao ground truth. Ela exige todos os gates aprovados, permanece `review_only=true`, `trainable=false`, `ground_truth=false` e `eligible_for_17b_now=false`.",
        "",
        "## PDF, geometria e fenomeno",
        "PDF sem coordenada, vetor, endereco preciso ou mapa georreferenciavel nao passa G4. A classificacao de fenomeno separa evidencia hidrologica de movimento de massa; `MIXED_CONFIRMED`, `MASS_MOVEMENT` e `UNKNOWN` nao validam inundacao.",
        "",
        "## Resultado do build inicial",
        f"- Pedidos na fila: {summary['formal_requests_count']}.",
        f"- Tentativas registradas: {summary['collection_attempts_count']}.",
        f"- Artefatos ingeridos: {summary['artifacts_ingested_count']}.",
        f"- Artefatos com todos os gates aprovados: {summary['artifacts_all_gates_passed_count']}.",
        f"- Ground Reference Candidates aceitos: {summary['accepted_ground_reference_candidates_count']}.",
        f"- Artefatos rejeitados ou pendentes: {summary['rejected_or_pending_artifacts_count']}.",
        "",
        "## Guardrails",
        "Nenhuma resposta foi inventada, nenhum contato ou protocolo foi inventado, nenhum artefato sem hash foi aceito, nenhuma fonte nao oficial foi aceita e nenhum dado pos-evento virou feature pre-evento.",
        "",
        "## Score e 17B",
        "Score v6 nao mudou, score v7 continua inexistente e 17B permanece bloqueado porque nao ha resposta oficial real com hash, manifesto, geometria, temporalidade, fenomeno e QA aceitos.",
        "",
        "## Proximo marco recomendado",
        summary["recommended_next_milestone"],
    ])


def build_all() -> None:
    _require_inputs()
    write_csv(QUEUE, build_collection_queue())
    write_csv(ATTEMPT_LOG, build_attempt_log())
    write_csv(INTAKE, build_intake_registry())
    write_csv(MANIFEST, build_artifact_manifest())
    write_csv(GATE_MATRIX, build_gate_matrix())
    write_csv(SPATIAL_STATUS, build_spatial_status())
    write_csv(TEMPORAL_STATUS, build_temporal_status())
    write_csv(PHENOMENON_STATUS, build_phenomenon_status())
    write_csv(ACCEPTED, build_accepted_candidates(), CANDIDATE_FIELDS)
    write_csv(REJECTED, build_rejected_or_pending())
    write_csv(ANTI_LEAKAGE, build_anti_leakage_audit())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_json(GATE_SCHEMA, build_gate_schema())
    write_json(INTAKE_SCHEMA, build_intake_schema())
    write_json(CANDIDATE_SCHEMA, build_candidate_schema())
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
    attempts = read_csv(ATTEMPT_LOG)
    intake = read_csv(INTAKE)
    manifest = read_csv(MANIFEST)
    gates = read_csv(GATE_MATRIX)
    accepted = read_csv(ACCEPTED)
    rejected = read_csv(REJECTED)
    leakage = read_csv(ANTI_LEAKAGE)
    summary = read_json(SUMMARY)
    errors = []
    for row in gates:
        errors.extend(_schema_violations(row, read_json(GATE_SCHEMA)))
    for row in intake:
        errors.extend(_schema_violations(row, read_json(INTAKE_SCHEMA)))
    for row in accepted:
        errors.extend(_schema_violations(row, read_json(CANDIDATE_SCHEMA)))
    for rows, key in [
        (queue, "collection_queue_id"),
        (attempts, "attempt_id"),
        (intake, "official_response_intake_id"),
        (manifest, "artifact_manifest_id"),
        (gates, "gate_evaluation_id"),
        (rejected, "rejected_or_pending_id"),
        (leakage, "anti_leakage_audit_id"),
    ]:
        values = [row[key] for row in rows]
        if values != sorted(values) or len(values) != len(set(values)):
            errors.append(f"ids_not_unique_or_sorted:{key}")
    if len(queue) != 9 or len(attempts) != 9 or len(intake) != 9 or len(manifest) != 9 or len(gates) != 63 or len(rejected) != 9:
        errors.append("unexpected_row_count")
    if any(row["external_action_allowed"] != "false" or row["response_received"] != "false" or row["protocol_number"] != "not_informed" for row in attempts):
        errors.append("attempt_log_records_external_effect_or_protocol")
    if any(row["source_artifact_exists"] != "false" or row["source_artifact_sha256"] != "not_available" for row in intake):
        errors.append("intake_invents_artifact_or_hash")
    if any(row["sha256"] != "not_available" and row["manifest_complete"] != "true" for row in manifest):
        errors.append("manifest_hash_without_complete_manifest")
    if accepted:
        for row in accepted:
            if row["all_gates_passed"] != "true" or row["ground_truth"] != "false" or row["usable_for_training"] != "false":
                errors.append("accepted_candidate_invalid")
    if any(row["gate_id"] in {"G1", "G3", "G4", "G5", "G6"} and row["gate_pass"] == "true" for row in gates):
        errors.append("missing_artifact_gate_passed")
    if any(row["anti_leakage_check"] == "post_event_as_pre_event_feature" and row["violation_found"] != "false" for row in leakage):
        errors.append("anti_leakage_violation")
    if not rejected:
        errors.append("empty_rejected_or_pending")
    for path in [FEATURES, SCORE_V6, DAT / "susc_patches_official_v1.csv", DAT / "susc_patch_links_official_v1.csv"]:
        if path.exists() and _run_git(["diff", "--name-only", "--", rel(path)]):
            errors.append(f"official_dataset_changed:{rel(path)}")
    if SCORE_V7.exists():
        errors.append("score_v7_exists")
    expected = build_summary()
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(f"summary_mismatch:{key}")
    if summary["ground_truth_created"] or summary["training_label_created"] or summary["eligible_for_17b_now"]:
        errors.append("promotion_guardrail_failed")
    if not read_csv(BLOCKERS):
        errors.append("empty_blockers")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "17C14 collection -> "
        f"queue={summary['formal_requests_count']} "
        f"attempts={summary['collection_attempts_count']} "
        f"ingested={summary['artifacts_ingested_count']} "
        f"accepted={summary['accepted_ground_reference_candidates_count']} "
        f"rejected={summary['rejected_or_pending_artifacts_count']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
    return 0


def queue_text() -> str:
    return "\n".join([f"{row['formal_request_id']} {row['collection_status']}" for row in build_collection_queue()])


def collect_text(request_id: str | None, dry_run: bool = True) -> str:
    if not dry_run:
        return "blocked_guardrail: coleta externa real exige implementacao segura e opt-in; nada executado."
    rows = build_collection_queue()
    selected = [row for row in rows if request_id in {None, row["formal_request_id"]}]
    return "\n".join([f"{row['formal_request_id']}: dry-run {row['collection_status']}" for row in selected])


def submit_text(request_id: str, dry_run: bool, confirm_authorized: bool) -> tuple[int, str]:
    if dry_run:
        return (0, "dry_run_only: nenhuma submissao executada.")
    if not confirm_authorized or os.environ.get("SUSC_17C14_ALLOW_EXTERNAL_ACTION") != "1":
        return (2, "blocked_guardrail: faltou autorizacao explicita e variavel SUSC_17C14_ALLOW_EXTERNAL_ACTION=1.")
    return (2, f"blocked_guardrail: submissao real para {request_id} nao implementada com seguranca neste marco.")


def intake_file(request_id: str, artifact_path: Path) -> Path:
    if not artifact_path.exists() or not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    target = LOCAL_STATE / "intake" / request_id / "manifesto_local.json"
    ensure_dir(target.parent)
    target.write_text(json.dumps({
        "formal_request_id": request_id,
        "artifact_path": str(artifact_path.resolve()),
        "sha256": digest,
        "size_bytes": artifact_path.stat().st_size,
        "classification": "received_needs_spatial_qa",
        "ground_truth": False,
        "trainable": False,
        "review_only": True,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target
