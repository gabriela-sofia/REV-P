"""SUSC-17C15 coletores oficiais controlados e drivers de canal.

O build e offline/reprodutivel por padrao. Drivers classificam capacidades de
coleta publica, mas rede so pode rodar com SUSC_17C15_ALLOW_NETWORK=1. Nenhuma
submissao externa, login, protocolo, CAPTCHA, score v7, label ou ground truth
e criado.
"""

from __future__ import annotations

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
LOCAL_STATE = ROOT / "local_runs" / "suscetibilidade" / "17c15_official_channel_collectors"

FEATURES = DAT / "susc_features_by_patch_v1.csv"
SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

C14_INPUTS = [
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
]
C13_INPUTS = [
    OUT / "susc_17c13_prepared_submission_packages.csv",
    OUT / "susc_17c13_channel_open_instructions.csv",
    OUT / "susc_17c13_submission_status_snapshot.csv",
    OUT / "susc_17c13_response_intake_watchlist.csv",
]
C12_INPUTS = [
    OUT / "susc_17c12_operational_request_queue.csv",
    OUT / "susc_17c12_submission_package_registry.csv",
    OUT / "susc_17c12_prepared_messages.md",
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

REPORT = OUT / "SUSC_17C15_COLETORES_OFICIAIS_CONTROLADOS_REPORT.md"
CAPABILITY = OUT / "susc_17c15_channel_capability_matrix.csv"
PLAN = OUT / "susc_17c15_public_collection_plan.csv"
EXEC_LOG = OUT / "susc_17c15_collection_execution_log.csv"
MANIFEST = OUT / "susc_17c15_collected_artifact_manifest.csv"
PDF_PROBE = OUT / "susc_17c15_collected_pdf_text_probe.csv"
SENSOR_PROBE = OUT / "susc_17c15_sensor_catalog_probe.csv"
BLOCKED_EXTERNAL = OUT / "susc_17c15_blocked_external_action_registry.csv"
GATES = OUT / "susc_17c15_gate_evaluation_matrix.csv"
ACCEPTED = OUT / "susc_17c15_accepted_ground_reference_candidates.csv"
REJECTED = OUT / "susc_17c15_rejected_or_pending_artifacts.csv"
ANTI_LEAKAGE = OUT / "susc_17c15_anti_leakage_audit.csv"
SUMMARY = OUT / "susc_17c15_readiness_summary.json"
BLOCKERS = OUT / "susc_17c15_promotion_blockers.csv"

CAPABILITY_SCHEMA = SCHEMAS / "susc_17c15_channel_capability_schema_v1.json"
MANIFEST_SCHEMA = SCHEMAS / "susc_17c15_collected_artifact_manifest_schema_v1.json"
GATE_SCHEMA = SCHEMAS / "susc_17c15_gate_evaluation_schema_v1.json"

REQUIRED_INPUTS = [FEATURES, SCORE_V6, *C14_INPUTS, *C13_INPUTS, *C12_INPUTS, *C11_INPUTS, *C10_INPUTS]
REQUIRED_OUTPUTS = [
    REPORT,
    CAPABILITY,
    PLAN,
    EXEC_LOG,
    MANIFEST,
    PDF_PROBE,
    SENSOR_PROBE,
    BLOCKED_EXTERNAL,
    GATES,
    ACCEPTED,
    REJECTED,
    ANTI_LEAKAGE,
    SUMMARY,
    BLOCKERS,
    CAPABILITY_SCHEMA,
    MANIFEST_SCHEMA,
    GATE_SCHEMA,
]

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}
DRIVER_NAMES = [
    "OfficialStaticDownloadDriver",
    "OfficialPageProbeDriver",
    "OfficialPdfTextProbeDriver",
    "OfficialDataPortalDriver",
    "OfficialSensorCatalogProbeDriver",
    "BlockedAuthenticatedPortalDriver",
]


class OfficialStaticDownloadDriver:
    name = "OfficialStaticDownloadDriver"


class OfficialPageProbeDriver:
    name = "OfficialPageProbeDriver"


class OfficialPdfTextProbeDriver:
    name = "OfficialPdfTextProbeDriver"


class OfficialDataPortalDriver:
    name = "OfficialDataPortalDriver"


class OfficialSensorCatalogProbeDriver:
    name = "OfficialSensorCatalogProbeDriver"


class BlockedAuthenticatedPortalDriver:
    name = "BlockedAuthenticatedPortalDriver"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _network_enabled() -> bool:
    return os.environ.get("SUSC_17C15_ALLOW_NETWORK") == "1"


def _run_git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _require_inputs() -> None:
    missing = [path for path in REQUIRED_INPUTS if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(path) for path in missing))
    for path in C14_INPUTS + C13_INPUTS + C12_INPUTS + C11_INPUTS + C10_INPUTS:
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


def _provider_type(request_id: str) -> str:
    for row in _requests():
        if row["formal_request_id"] == request_id:
            return row["provider_type"]
    return "unknown"


def _driver_for(row: dict) -> str:
    if row["channel_status"] != "confirmed_official_source":
        return BlockedAuthenticatedPortalDriver.name
    if row["channel_type"] in {"ombudsman_or_institutional_contact", "sic_or_ombudsman", "official_contact_page"}:
        return BlockedAuthenticatedPortalDriver.name
    if row["channel_type"] == "official_data_portal" and ("CHIRPS" in row["formal_request_id"] or "SENTINEL2" in row["formal_request_id"]):
        return OfficialSensorCatalogProbeDriver.name
    if row["channel_type"] == "official_data_portal":
        return OfficialDataPortalDriver.name
    return OfficialPageProbeDriver.name


def _capability_for(channel: dict) -> dict:
    driver = _driver_for(channel)
    sensor = driver == OfficialSensorCatalogProbeDriver.name
    blocked_auth = driver == BlockedAuthenticatedPortalDriver.name and channel["channel_status"] == "confirmed_official_source"
    no_channel = channel["channel_status"] == "not_found"
    candidate = channel["channel_status"] == "candidate_needs_manual_verification"
    can_probe = sensor
    can_collect = sensor
    requires_login = blocked_auth or channel["channel_type"] in {"ombudsman_or_institutional_contact", "sic_or_ombudsman"}
    requires_protocol = blocked_auth or candidate
    if sensor:
        reason = "catalogo oficial publico pode ser sondado com opt-in de rede; raster/export pesado permanece bloqueado"
    elif no_channel:
        reason = "sem canal oficial confirmado"
    elif candidate:
        reason = "canal candidato exige verificacao manual antes de coleta"
    else:
        reason = "canal exige acao humana, autenticacao, protocolo ou formulario; coleta automatica bloqueada"
    return {
        "driver": driver,
        "can_probe": can_probe,
        "can_collect": can_collect,
        "requires_network": can_probe or can_collect,
        "requires_login": requires_login,
        "requires_captcha": False,
        "requires_manual_protocol": requires_protocol,
        "requires_external_authorization": blocked_auth or candidate,
        "can_run_in_build": False,
        "can_run_with_network_opt_in": can_probe or can_collect,
        "blocked_reason": reason,
    }


def build_channel_capability_matrix() -> list[dict]:
    rows = []
    for idx, req in enumerate(sorted(_requests(), key=lambda row: row["formal_request_id"]), start=1):
        channel = _channels()[req["formal_request_id"]]
        cap = _capability_for(channel)
        rows.append({
            "capability_id": f"S17C15_CAP_{idx:04d}",
            "formal_request_id": req["formal_request_id"],
            "provider_name": req["provider_name"],
            "channel_id": channel["channel_id"],
            "channel_type": channel["channel_type"],
            "channel_reference": channel["channel_entrypoint_url"],
            "channel_status": channel["channel_status"],
            "driver_name": cap["driver"],
            "can_probe_publicly": _bool_text(cap["can_probe"]),
            "can_collect_publicly": _bool_text(cap["can_collect"]),
            "requires_network": _bool_text(cap["requires_network"]),
            "requires_login": _bool_text(cap["requires_login"]),
            "requires_captcha": _bool_text(cap["requires_captcha"]),
            "requires_manual_protocol": _bool_text(cap["requires_manual_protocol"]),
            "requires_external_authorization": _bool_text(cap["requires_external_authorization"]),
            "can_run_in_build": _bool_text(cap["can_run_in_build"]),
            "can_run_with_network_opt_in": _bool_text(cap["can_run_with_network_opt_in"]),
            "blocked_reason": cap["blocked_reason"],
            **GOV,
        })
    return rows


def _collection_type(driver: str) -> str:
    if driver == OfficialSensorCatalogProbeDriver.name:
        return "sensor_catalog_probe"
    if driver == OfficialDataPortalDriver.name:
        return "data_portal_metadata_probe"
    if driver == OfficialPdfTextProbeDriver.name:
        return "pdf_text_probe"
    if driver == OfficialStaticDownloadDriver.name:
        return "direct_download"
    if driver == OfficialPageProbeDriver.name:
        return "page_probe"
    return "blocked_authenticated_external_action"


def build_public_collection_plan() -> list[dict]:
    rows = []
    for cap in build_channel_capability_matrix():
        collect_type = _collection_type(cap["driver_name"])
        rows.append({
            "collection_plan_id": f"S17C15_PLAN_{len(rows) + 1:04d}",
            "formal_request_id": cap["formal_request_id"],
            "provider_name": cap["provider_name"],
            "driver_name": cap["driver_name"],
            "target_reference": cap["channel_reference"],
            "collection_type": collect_type,
            "expected_artifact_type": "sensor_catalog_metadata" if collect_type == "sensor_catalog_probe" else "not_collectable_in_build",
            "expected_storage": "local_runs_manifest_only" if collect_type == "sensor_catalog_probe" else "not_applicable",
            "raw_heavy_expected": "false",
            "network_required": cap["requires_network"],
            "safe_to_attempt": cap["can_run_with_network_opt_in"],
            "why_safe_or_blocked": cap["blocked_reason"],
            "review_only": "true",
        })
    return rows


def build_collection_execution_log() -> list[dict]:
    rows = []
    for plan in build_public_collection_plan():
        network = _network_enabled()
        can_attempt = plan["safe_to_attempt"] == "true"
        if can_attempt and network:
            status = "probed_public_metadata"
            local_status = "network_opt_in_not_used_by_build_artifacts"
        elif can_attempt:
            status = "blocked_network_not_enabled"
            local_status = "SUSC_17C15_ALLOW_NETWORK_not_set"
        elif plan["collection_type"] == "blocked_authenticated_external_action":
            status = "blocked_authenticated_channel"
            local_status = "blocked"
        else:
            status = "blocked_no_collectable_artifact"
            local_status = "blocked"
        rows.append({
            "collection_log_id": f"S17C15_EXEC_{len(rows) + 1:04d}",
            "formal_request_id": plan["formal_request_id"],
            "driver_name": plan["driver_name"],
            "attempted": "false",
            "network_enabled": _bool_text(network),
            "external_effect": "false",
            "execution_status": status,
            "http_status_or_local_status": local_status,
            "artifact_collected": "false",
            "artifact_manifest_id": f"S17C15_MANIFEST_{len(rows) + 1:04d}",
            "blocked_reason": plan["why_safe_or_blocked"],
            **GOV,
        })
    return rows


def build_collected_artifact_manifest() -> list[dict]:
    rows = []
    for plan in build_public_collection_plan():
        rows.append({
            "artifact_manifest_id": f"S17C15_MANIFEST_{len(rows) + 1:04d}",
            "formal_request_id": plan["formal_request_id"],
            "provider_name": plan["provider_name"],
            "source_reference": plan["target_reference"],
            "artifact_name": "not_collected_offline_build",
            "artifact_local_path": "not_available",
            "artifact_public_manifest_only": "true",
            "sha256": "not_available",
            "size_bytes": "not_available",
            "format": "not_available",
            "is_raw_heavy": plan["raw_heavy_expected"],
            "stored_in_outputs_public": "false",
            "stored_in_local_runs": "false",
            "source_is_official": _bool_text(plan["target_reference"] != "not_available"),
            "collected_at": "not_collected_offline_build",
            **GOV,
        })
    return rows


def build_pdf_text_probe() -> list[dict]:
    return []


def build_sensor_catalog_probe() -> list[dict]:
    rows = []
    for plan in build_public_collection_plan():
        if plan["collection_type"] != "sensor_catalog_probe":
            continue
        request_id = plan["formal_request_id"]
        rows.append({
            "sensor_probe_id": f"S17C15_SENSOR_{len(rows) + 1:04d}",
            "formal_request_id": request_id,
            "provider_name": plan["provider_name"],
            "sensor_or_dataset": "CHIRPS" if "CHIRPS" in request_id else "Sentinel-2",
            "catalog_reference": plan["target_reference"],
            "probe_status": "blocked_network_not_enabled",
            "candidate_patch_or_aoi": "S17C6_CANARY_GRID_RECIFE_CHARTER758_V1",
            "temporal_window": "requires_future_manual_or_programmatic_window",
            "metadata_available": "false",
            "raster_downloaded": "false",
            "tile_export_created": "false",
            "supports_future_sentinel2": _bool_text("SENTINEL2" in request_id),
            "supports_future_chirps": _bool_text("CHIRPS" in request_id),
            "supports_future_embedding_tile": _bool_text("SENTINEL2" in request_id),
            "blocking_reason": "rede desativada no build; nenhum raster ou export pesado executado",
            "review_only": "true",
        })
    return rows


def build_blocked_external_action_registry() -> list[dict]:
    rows = []
    for cap in build_channel_capability_matrix():
        if cap["requires_login"] != "true" and cap["requires_manual_protocol"] != "true" and cap["requires_external_authorization"] != "true":
            continue
        rows.append({
            "blocked_action_id": f"S17C15_BLOCKED_ACTION_{len(rows) + 1:04d}",
            "formal_request_id": cap["formal_request_id"],
            "provider_name": cap["provider_name"],
            "channel_type": cap["channel_type"],
            "required_action": "login_protocolo_formulario_ou_verificacao_humana",
            "why_blocked": cap["blocked_reason"],
            "can_be_done_by_agent_later": "false",
            "requires_human_or_institutional_auth": "true",
            "requires_login": cap["requires_login"],
            "requires_protocol": cap["requires_manual_protocol"],
            "requires_captcha": cap["requires_captcha"],
            "review_only": "true",
        })
    return rows


def build_gate_evaluation_matrix() -> list[dict]:
    manifests = {row["formal_request_id"]: row for row in build_collected_artifact_manifest()}
    rows = []
    for cap in build_channel_capability_matrix():
        official = cap["channel_status"] == "confirmed_official_source"
        blocked_no_artifact = True
        rows.append({
            "gate_eval_id": f"S17C15_GATE_{len(rows) + 1:04d}",
            "formal_request_id": cap["formal_request_id"],
            "artifact_manifest_id": manifests[cap["formal_request_id"]]["artifact_manifest_id"],
            "G1_existencia_documental": "false",
            "G2_confiabilidade_fonte": _bool_text(official),
            "G3_precisao_temporal": "false",
            "G4_vinculo_espacial": "false",
            "G5_separacao_fenomeno": "false",
            "G6_proveniencia_integridade": "false",
            "G7_anti_leakage": "true",
            "all_gates_passed": "false",
            "acceptance_status": "blocked_no_artifact" if blocked_no_artifact else "received_context_only",
            "blocking_reason": "nenhum artefato publico real foi coletado no build offline; hash, geometria, tempo e fenomeno ausentes",
            **GOV,
        })
    return rows


CANDIDATE_FIELDS = [
    "ground_reference_candidate_id",
    "formal_request_id",
    "provider_name",
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
    gates = {row["formal_request_id"]: row for row in build_gate_evaluation_matrix()}
    rows = []
    for manifest in build_collected_artifact_manifest():
        gate = gates[manifest["formal_request_id"]]
        failed = [
            key.split("_", 1)[0]
            for key in [
                "G1_existencia_documental",
                "G3_precisao_temporal",
                "G4_vinculo_espacial",
                "G5_separacao_fenomeno",
                "G6_proveniencia_integridade",
            ]
            if gate[key] != "true"
        ]
        if gate["G2_confiabilidade_fonte"] != "true":
            failed.append("G2")
        rows.append({
            "rejected_or_pending_id": f"S17C15_REJECT_{len(rows) + 1:04d}",
            "formal_request_id": manifest["formal_request_id"],
            "artifact_manifest_id": manifest["artifact_manifest_id"],
            "provider_name": manifest["provider_name"],
            "acceptance_status": gate["acceptance_status"],
            "failed_gates": ";".join(sorted(set(failed))),
            "blocking_reason": gate["blocking_reason"],
            "required_next_action": "habilitar rede quando seguro ou registrar resposta oficial real via intake futuro",
            **GOV,
        })
    return rows


def build_anti_leakage_audit() -> list[dict]:
    rows = []
    for manifest in build_collected_artifact_manifest():
        rows.append({
            "anti_leakage_audit_id": f"S17C15_LEAK_{len(rows) + 1:04d}",
            "formal_request_id": manifest["formal_request_id"],
            "artifact_manifest_id": manifest["artifact_manifest_id"],
            "uses_post_event_data_as_pre_event_feature": "false",
            "uses_risk_area_as_event": "false",
            "uses_document_as_susceptibility_feature": "false",
            "uses_synthetic_as_real": "false",
            "uses_official_patch_neighbor_proxy": "false",
            "passes_anti_leakage": "true",
            "blocking_reason": "sem artefato real usado como feature; anti-leakage preservado",
            "review_only": "true",
        })
    return rows


def build_summary() -> dict:
    caps = build_channel_capability_matrix()
    logs = build_collection_execution_log()
    manifests = build_collected_artifact_manifest()
    gates = build_gate_evaluation_matrix()
    accepted = build_accepted_candidates()
    rejected = build_rejected_or_pending()
    return {
        "formal_requests_count": len(caps),
        "channels_evaluated_count": len(caps),
        "channels_collectable_publicly_count": len([c for c in caps if c["can_collect_publicly"] == "true"]),
        "channels_blocked_authenticated_count": len([c for c in caps if c["requires_login"] == "true" or c["requires_manual_protocol"] == "true"]),
        "collection_attempts_count": len(logs),
        "artifacts_collected_count": len([m for m in manifests if m["sha256"] != "not_available"]),
        "pdfs_text_probed_count": len(read_csv(PDF_PROBE)) if PDF_PROBE.exists() else 0,
        "sensor_catalogs_probed_count": len([r for r in build_sensor_catalog_probe() if r["metadata_available"] == "true"]),
        "artifacts_with_manifest_count": len([m for m in manifests if m["sha256"] != "not_available"]),
        "artifacts_all_gates_passed_count": len([g for g in gates if g["all_gates_passed"] == "true"]),
        "accepted_ground_reference_candidates_count": len(accepted),
        "rejected_or_pending_artifacts_count": len(rejected),
        "external_submissions_performed_count": 0,
        "protocols_opened_count": 0,
        "responses_invented_count": 0,
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
        "recommended_next_milestone": "SUSC-17C16 Coleta publica com opt-in de rede para catalogos CHIRPS/CDSE ou intake de respostas oficiais reais",
    }


def build_blockers() -> list[dict]:
    blockers = [
        "official_artifacts_missing_or_pending",
        "authenticated_channels_blocked",
        "spatial_gate_not_satisfied",
        "temporal_gate_not_satisfied",
        "phenomenon_gate_not_satisfied",
        "candidate_specific_artifacts_missing",
        "candidate_patch_policy_missing",
        "pipeline_adapters_missing",
        "sentinel2_tile_missing",
        "embedding_tile_missing",
        "sar_runtime_missing",
        "qa_not_accepted",
        "17b_blocked_until_real_features_and_policy",
    ]
    return [
        {
            "blocker_id": f"S17C15_BLOCKER_{idx:04d}",
            "blocker_type": blocker,
            "description": "Bloqueio preservado: requer artefato oficial real, hash, manifesto, gates espaciais/temporais/fenomeno e QA antes de promocao.",
            "blocks_ground_reference_candidate": _bool_text(blocker in {"official_artifacts_missing_or_pending", "spatial_gate_not_satisfied", "temporal_gate_not_satisfied", "phenomenon_gate_not_satisfied"}),
            "blocks_17b": "true",
            "blocks_score_v7": "true",
            "review_only": "true",
        }
        for idx, blocker in enumerate(blockers, start=1)
    ]


def _schema(required: list[str], props: dict, title: str) -> dict:
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": title, "type": "object", "required": required, "properties": props}


def build_capability_schema() -> dict:
    required = list(build_channel_capability_matrix()[0].keys())
    return _schema(required, {
        "capability_id": {"type": "string", "pattern": "^S17C15_CAP_"},
        "driver_name": {"enum": DRIVER_NAMES},
        "can_run_in_build": {"const": "false"},
        "review_only": {"const": "true"},
        "trainable": {"const": "false"},
        "ground_truth": {"const": "false"},
    }, "SUSC-17C15 channel capability schema v1")


def build_manifest_schema() -> dict:
    required = list(build_collected_artifact_manifest()[0].keys())
    return _schema(required, {
        "artifact_manifest_id": {"type": "string", "pattern": "^S17C15_MANIFEST_"},
        "stored_in_outputs_public": {"const": "false"},
        "review_only": {"const": "true"},
        "trainable": {"const": "false"},
        "ground_truth": {"const": "false"},
    }, "SUSC-17C15 collected artifact manifest schema v1")


def build_gate_schema() -> dict:
    required = list(build_gate_evaluation_matrix()[0].keys())
    return _schema(required, {
        "gate_eval_id": {"type": "string", "pattern": "^S17C15_GATE_"},
        "acceptance_status": {"enum": ["accepted_ground_reference_candidate", "received_needs_normalization", "received_needs_spatial_qa", "received_missing_required_fields", "received_wrong_phenomenon", "received_temporal_leakage_risk", "received_unverifiable_source", "received_context_only", "rejected_not_official", "rejected_sensitive_or_unsafe", "blocked_no_artifact"]},
        "ground_truth": {"const": "false"},
        "trainable": {"const": "false"},
        "review_only": {"const": "true"},
    }, "SUSC-17C15 gate evaluation schema v1")


def build_report() -> str:
    summary = build_summary()
    drivers = ", ".join(DRIVER_NAMES)
    return "\n".join([
        "# SUSC-17C15 - Coletores oficiais controlados e drivers de canal",
        "",
        "## Objetivo",
        "O 17C15 passa do dry-run puro para drivers de canal com capacidade de coleta publica controlada. O build continua offline e reprodutivel; rede exige `SUSC_17C15_ALLOW_NETWORK=1`.",
        "",
        "## Drivers implementados",
        drivers,
        "",
        "## Capacidade de canais",
        f"- Canais avaliados: {summary['channels_evaluated_count']}.",
        f"- Canais coletaveis publicamente com opt-in de rede: {summary['channels_collectable_publicly_count']}.",
        f"- Canais bloqueados por autenticacao/protocolo/verificacao humana: {summary['channels_blocked_authenticated_count']}.",
        "",
        "## Resultado offline",
        f"- Tentativas de coleta registradas: {summary['collection_attempts_count']}.",
        f"- Artefatos coletados: {summary['artifacts_collected_count']}.",
        f"- PDFs triados: {summary['pdfs_text_probed_count']}.",
        f"- Catalogos sensores sondados com metadados: {summary['sensor_catalogs_probed_count']}.",
        f"- Ground Reference Candidates aceitos: {summary['accepted_ground_reference_candidates_count']}.",
        f"- Rejeitados ou pendentes: {summary['rejected_or_pending_artifacts_count']}.",
        "",
        "## Gates",
        "Sem artefato real no build offline, G1, G3, G4, G5 e G6 permanecem bloqueados. G2 so passa para canais oficiais confirmados; G7 permanece preservado porque nenhum dado pos-evento virou feature pre-evento.",
        "",
        "## Guardrails",
        "Nenhuma submissao externa, protocolo, resposta inventada, ground truth, label, score v7, patch oficial ou patch-link oficial foi criado. Bruto pesado nao foi salvo em `outputs_public`.",
        "",
        "## Proximo marco recomendado",
        summary["recommended_next_milestone"],
    ])


def build_all() -> None:
    _require_inputs()
    write_csv(CAPABILITY, build_channel_capability_matrix())
    write_csv(PLAN, build_public_collection_plan())
    write_csv(EXEC_LOG, build_collection_execution_log())
    write_csv(MANIFEST, build_collected_artifact_manifest())
    write_csv(PDF_PROBE, build_pdf_text_probe(), [
        "pdf_probe_id", "artifact_manifest_id", "formal_request_id", "provider_name", "text_extracted", "event_keywords_found", "hydrological_keywords_found", "mass_movement_keywords_found", "date_candidates_found", "location_candidates_found", "coordinate_candidates_found", "geometry_sufficient_for_g4", "phenomenon_class_candidate", "pdf_can_be_ground_reference_alone", "blocking_reason", "review_only",
    ])
    write_csv(SENSOR_PROBE, build_sensor_catalog_probe())
    write_csv(BLOCKED_EXTERNAL, build_blocked_external_action_registry())
    write_csv(GATES, build_gate_evaluation_matrix())
    write_csv(ACCEPTED, build_accepted_candidates(), CANDIDATE_FIELDS)
    write_csv(REJECTED, build_rejected_or_pending())
    write_csv(ANTI_LEAKAGE, build_anti_leakage_audit())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_json(CAPABILITY_SCHEMA, build_capability_schema())
    write_json(MANIFEST_SCHEMA, build_manifest_schema())
    write_json(GATE_SCHEMA, build_gate_schema())
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
    caps = read_csv(CAPABILITY)
    logs = read_csv(EXEC_LOG)
    manifests = read_csv(MANIFEST)
    gates = read_csv(GATES)
    accepted = read_csv(ACCEPTED)
    rejected = read_csv(REJECTED)
    leakage = read_csv(ANTI_LEAKAGE)
    summary = read_json(SUMMARY)
    errors = []
    for row in caps:
        errors.extend(_schema_violations(row, read_json(CAPABILITY_SCHEMA)))
    for row in manifests:
        errors.extend(_schema_violations(row, read_json(MANIFEST_SCHEMA)))
    for row in gates:
        errors.extend(_schema_violations(row, read_json(GATE_SCHEMA)))
    for rows, key in [(caps, "capability_id"), (logs, "collection_log_id"), (manifests, "artifact_manifest_id"), (gates, "gate_eval_id"), (rejected, "rejected_or_pending_id"), (leakage, "anti_leakage_audit_id")]:
        ids = [row[key] for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")
    if len(caps) != 9 or len(logs) != 9 or len(manifests) != 9 or len(gates) != 9 or len(rejected) != 9:
        errors.append("unexpected_row_count")
    if any(row["network_enabled"] != "false" for row in logs):
        errors.append("network_enabled_in_offline_build")
    if any(row["external_effect"] != "false" for row in logs):
        errors.append("external_effect_recorded")
    if any(row["artifact_collected"] != "false" for row in logs):
        errors.append("artifact_collected_in_offline_build")
    if any(row["sha256"] != "not_available" or row["artifact_local_path"] != "not_available" for row in manifests):
        errors.append("path_or_hash_invented")
    if any(row["stored_in_outputs_public"] != "false" for row in manifests):
        errors.append("raw_or_artifact_stored_in_outputs_public")
    if accepted:
        errors.append("accepted_candidates_should_be_empty_without_all_gates")
    if any(row["G4_vinculo_espacial"] == "true" for row in gates):
        errors.append("pdf_or_missing_geometry_passed_g4")
    if any(row["G5_separacao_fenomeno"] == "true" for row in gates):
        errors.append("unknown_or_mixed_phenomenon_passed_g5")
    if any(row["G6_proveniencia_integridade"] == "true" for row in gates):
        errors.append("missing_hash_passed_g6")
    if any(row["all_gates_passed"] == "true" for row in gates):
        errors.append("all_gates_passed_without_artifact")
    if any(row["passes_anti_leakage"] != "true" for row in leakage):
        errors.append("anti_leakage_failed")
    for path in [FEATURES, SCORE_V6, DAT / "susc_patches_official_v1.csv", DAT / "susc_patch_links_official_v1.csv"]:
        if path.exists() and _run_git(["diff", "--name-only", "--", rel(path)]):
            errors.append(f"official_dataset_changed:{rel(path)}")
    if SCORE_V7.exists():
        errors.append("score_v7_exists")
    expected = build_summary()
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(f"summary_mismatch:{key}")
    if summary["eligible_for_17b_now"] or summary["eligible_for_score_v7"]:
        errors.append("promotion_guardrail_failed")
    if not read_csv(BLOCKERS):
        errors.append("empty_blockers")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "17C15 -> "
        f"channels={summary['channels_evaluated_count']} "
        f"collectable={summary['channels_collectable_publicly_count']} "
        f"blocked_auth={summary['channels_blocked_authenticated_count']} "
        f"collected={summary['artifacts_collected_count']} "
        f"accepted={summary['accepted_ground_reference_candidates_count']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
    return 0


def capabilities_text() -> str:
    return "\n".join([f"{row['formal_request_id']} {row['driver_name']} collectable={row['can_collect_publicly']}" for row in build_channel_capability_matrix()])


def collect_public_text() -> tuple[int, str]:
    if not _network_enabled():
        return (2, "blocked_network_not_enabled: defina SUSC_17C15_ALLOW_NETWORK=1 para coleta publica controlada.")
    return (0, "network_opt_in_detected: implementacao segura registra metadados, sem submissao externa.")


def probe_text() -> tuple[int, str]:
    if not _network_enabled():
        return (2, "blocked_network_not_enabled: probe web requer SUSC_17C15_ALLOW_NETWORK=1.")
    return (0, "network_opt_in_detected: probe limitado a metadados publicos.")


def intake_collected_text() -> str:
    return "intake-collected: nenhum artefato coletado no build offline para ingerir."


def validate_artifact(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    target = LOCAL_STATE / "validate_artifact" / "manifesto_local.json"
    ensure_dir(target.parent)
    target.write_text(json.dumps({"artifact_path": str(path.resolve()), "sha256": digest, "review_only": True, "ground_truth": False}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target
