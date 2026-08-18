"""SUSC-17C16 coleta publica com opt-in de rede para catalogos.

O build publico e offline e deterministico. Os modos de rede ficam atras de
SUSC_17C16_ALLOW_NETWORK=1 e gravam bruto leve apenas em local_runs. Nenhuma
submissao externa, protocolo, raster pesado, tile, embedding, label, ground
truth, score v7, patch oficial ou patch-link oficial e criado.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
LOCAL_STATE = ROOT / "local_runs" / "suscetibilidade" / "17c16_public_network_collection"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_PATCHES = DAT / "susc_patches_official_v1.csv"
OFFICIAL_PATCH_LINKS = DAT / "susc_patch_links_official_v1.csv"

C15_INPUTS = [
    OUT / "susc_17c15_channel_capability_matrix.csv",
    OUT / "susc_17c15_public_collection_plan.csv",
    OUT / "susc_17c15_collection_execution_log.csv",
    OUT / "susc_17c15_collected_artifact_manifest.csv",
    OUT / "susc_17c15_collected_pdf_text_probe.csv",
    OUT / "susc_17c15_sensor_catalog_probe.csv",
    OUT / "susc_17c15_blocked_external_action_registry.csv",
    OUT / "susc_17c15_gate_evaluation_matrix.csv",
    OUT / "susc_17c15_accepted_ground_reference_candidates.csv",
    OUT / "susc_17c15_rejected_or_pending_artifacts.csv",
    OUT / "susc_17c15_anti_leakage_audit.csv",
    OUT / "susc_17c15_readiness_summary.json",
    OUT / "susc_17c15_promotion_blockers.csv",
]
C14_INPUTS = [
    OUT / "susc_17c14_collection_request_queue.csv",
    OUT / "susc_17c14_official_response_intake_registry.csv",
    OUT / "susc_17c14_gate_evaluation_matrix.csv",
    OUT / "susc_17c14_readiness_summary.json",
]
C13_C12_C11_INPUTS = [
    OUT / "susc_17c13_prepared_submission_packages.csv",
    OUT / "susc_17c12_operational_request_queue.csv",
    OUT / "susc_17c11_official_channel_registry.csv",
]

REPORT = OUT / "SUSC_17C16_COLETA_PUBLICA_OPT_IN_REDE_CATALOGOS_REPORT.md"
CAPABILITY = OUT / "susc_17c16_network_capability_update.csv"
PROBE_LOG = OUT / "susc_17c16_public_probe_log.csv"
COLLECTION_LOG = OUT / "susc_17c16_lightweight_collection_log.csv"
MANIFEST = OUT / "susc_17c16_collected_artifact_manifest.csv"
CHIRPS_PROBE = OUT / "susc_17c16_chirps_catalog_probe.csv"
SENTINEL2_PROBE = OUT / "susc_17c16_sentinel2_catalog_probe.csv"
HEAVY_RASTER = OUT / "susc_17c16_blocked_heavy_raster_registry.csv"
GATES = OUT / "susc_17c16_gate_evaluation_matrix.csv"
ACCEPTED = OUT / "susc_17c16_accepted_ground_reference_candidates.csv"
REJECTED = OUT / "susc_17c16_rejected_or_pending_artifacts.csv"
ANTI_LEAKAGE = OUT / "susc_17c16_anti_leakage_audit.csv"
SUMMARY = OUT / "susc_17c16_readiness_summary.json"
BLOCKERS = OUT / "susc_17c16_promotion_blockers.csv"

NETWORK_SCHEMA = SCHEMAS / "susc_17c16_network_probe_schema_v1.json"
CATALOG_SCHEMA = SCHEMAS / "susc_17c16_catalog_probe_schema_v1.json"
MANIFEST_SCHEMA = SCHEMAS / "susc_17c16_collected_artifact_manifest_schema_v1.json"

REQUIRED_INPUTS = [SCORE_V6, *C15_INPUTS, *C14_INPUTS, *C13_C12_C11_INPUTS]
REQUIRED_OUTPUTS = [
    REPORT,
    CAPABILITY,
    PROBE_LOG,
    COLLECTION_LOG,
    MANIFEST,
    CHIRPS_PROBE,
    SENTINEL2_PROBE,
    HEAVY_RASTER,
    GATES,
    ACCEPTED,
    REJECTED,
    ANTI_LEAKAGE,
    SUMMARY,
    BLOCKERS,
    NETWORK_SCHEMA,
    CATALOG_SCHEMA,
    MANIFEST_SCHEMA,
]

DRIVER_NAMES = [
    "ChirpsCatalogProbeDriver",
    "CopernicusSentinel2CatalogProbeDriver",
    "OfficialLightweightArtifactDriver",
    "BlockedHeavyRasterDriver",
    "BlockedAuthenticatedActionDriver",
]
GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}
MAX_LIGHTWEIGHT_BYTES = 1024 * 1024


class ChirpsCatalogProbeDriver:
    name = "ChirpsCatalogProbeDriver"


class CopernicusSentinel2CatalogProbeDriver:
    name = "CopernicusSentinel2CatalogProbeDriver"


class OfficialLightweightArtifactDriver:
    name = "OfficialLightweightArtifactDriver"


class BlockedHeavyRasterDriver:
    name = "BlockedHeavyRasterDriver"


class BlockedAuthenticatedActionDriver:
    name = "BlockedAuthenticatedActionDriver"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _network_enabled() -> bool:
    return os.environ.get("SUSC_17C16_ALLOW_NETWORK") == "1"


def _run_git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _require_inputs() -> None:
    missing = [path for path in REQUIRED_INPUTS if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(path) for path in missing))
    for path in REQUIRED_INPUTS:
        if path.suffix == ".json":
            read_json(path)
        elif path.suffix == ".csv":
            read_csv(path)


def _c15_capabilities() -> list[dict]:
    return read_csv(OUT / "susc_17c15_channel_capability_matrix.csv")


def _driver_for(row: dict) -> str:
    request_id = row["formal_request_id"]
    if row["can_collect_publicly"] == "true" and "CHIRPS" in request_id:
        return ChirpsCatalogProbeDriver.name
    if row["can_collect_publicly"] == "true" and "SENTINEL2" in request_id:
        return CopernicusSentinel2CatalogProbeDriver.name
    if row["requires_login"] == "true" or row["requires_manual_protocol"] == "true":
        return BlockedAuthenticatedActionDriver.name
    if row["channel_status"] != "confirmed_official_source":
        return BlockedAuthenticatedActionDriver.name
    return OfficialLightweightArtifactDriver.name


def _safe_to_probe(row: dict) -> bool:
    return row["can_collect_publicly"] == "true" and row["requires_login"] == "false" and row["requires_manual_protocol"] == "false"


def _blocking_reason(row: dict) -> str:
    if _safe_to_probe(row):
        return "rede desativada no build publico; probe real exige SUSC_17C16_ALLOW_NETWORK=1"
    if row["requires_login"] == "true" or row["requires_manual_protocol"] == "true":
        return "canal exige login, protocolo, autorizacao institucional ou verificacao humana"
    if row["channel_status"] != "confirmed_official_source":
        return "fonte oficial publica nao confirmada para coleta automatica"
    return "sem artefato leve coletavel definido para este canal"


def build_network_capability_update() -> list[dict]:
    rows = []
    for idx, cap in enumerate(_c15_capabilities(), start=1):
        driver = _driver_for(cap)
        safe = _safe_to_probe(cap)
        rows.append({
            "network_capability_id": f"S17C16_NETCAP_{idx:04d}",
            "formal_request_id": cap["formal_request_id"],
            "provider_name": cap["provider_name"],
            "channel_id": cap["channel_id"],
            "driver_name": driver,
            "network_required": _bool_text(safe),
            "network_enabled": "false",
            "can_probe_with_network": _bool_text(safe),
            "can_collect_lightweight_with_network": "false",
            "can_collect_heavy_raster": "false",
            "requires_login": cap["requires_login"],
            "requires_protocol": cap["requires_manual_protocol"],
            "requires_authentication": cap["requires_external_authorization"],
            "safe_to_run_now": "false",
            "blocking_reason": _blocking_reason(cap),
            **GOV,
        })
    return rows


def build_public_probe_log() -> list[dict]:
    rows = []
    for idx, cap in enumerate(_c15_capabilities(), start=1):
        safe = _safe_to_probe(cap)
        blocked_auth = cap["requires_login"] == "true" or cap["requires_manual_protocol"] == "true"
        if safe:
            result_class = "blocked_network_not_enabled"
            reason = "rede desativada no build publico"
        elif blocked_auth:
            result_class = "blocked_authenticated"
            reason = "acao autenticada, protocolo ou formulario humano bloqueado"
        elif cap["channel_status"] != "confirmed_official_source":
            result_class = "blocked_not_official"
            reason = "fonte oficial publica nao confirmada"
        else:
            result_class = "failed_safe"
            reason = "sem probe publico leve definido"
        rows.append({
            "probe_log_id": f"S17C16_PROBE_{idx:04d}",
            "formal_request_id": cap["formal_request_id"],
            "provider_name": cap["provider_name"],
            "channel_reference": cap["channel_reference"],
            "driver_name": _driver_for(cap),
            "network_enabled": "false",
            "attempted": "false",
            "http_status": "not_attempted",
            "content_type": "not_available",
            "content_length": "not_available",
            "final_reference": "not_available",
            "source_is_official": _bool_text(cap["channel_status"] == "confirmed_official_source"),
            "probe_success": "false",
            "probe_result_class": result_class,
            "blocking_reason": reason,
            **GOV,
        })
    return rows


def build_lightweight_collection_log() -> list[dict]:
    rows = []
    for idx, cap in enumerate(_c15_capabilities(), start=1):
        safe = _safe_to_probe(cap)
        if safe:
            status = "blocked_network_not_enabled"
            reason = "rede desativada no build publico"
        elif cap["requires_login"] == "true" or cap["requires_manual_protocol"] == "true":
            status = "blocked_authenticated"
            reason = "login, protocolo, autorizacao institucional ou formulario humano bloqueado"
        else:
            status = "blocked_no_collectable_artifact"
            reason = "sem artefato leve oficial coletavel para este canal"
        rows.append({
            "collection_log_id": f"S17C16_COLLECT_{idx:04d}",
            "formal_request_id": cap["formal_request_id"],
            "provider_name": cap["provider_name"],
            "driver_name": _driver_for(cap),
            "network_enabled": "false",
            "attempted": "false",
            "collection_status": status,
            "artifact_collected": "false",
            "artifact_manifest_id": "not_available",
            "stored_in_outputs_public": "false",
            "stored_in_local_runs": "false",
            "raw_heavy": "false",
            "blocking_reason": reason,
            **GOV,
        })
    return rows


MANIFEST_FIELDS = [
    "artifact_manifest_id",
    "formal_request_id",
    "provider_name",
    "source_reference",
    "artifact_name",
    "artifact_local_path",
    "sha256",
    "size_bytes",
    "format",
    "content_type",
    "is_raw_heavy",
    "stored_in_outputs_public",
    "stored_in_local_runs",
    "source_is_official",
    "collected_at",
    "review_only",
    "trainable",
    "ground_truth",
]


def build_collected_artifact_manifest() -> list[dict]:
    return []


def _sensor_capabilities(kind: str) -> list[dict]:
    token = "CHIRPS" if kind == "chirps" else "SENTINEL2"
    return [row for row in _c15_capabilities() if token in row["formal_request_id"]]


def build_chirps_catalog_probe() -> list[dict]:
    rows = []
    for idx, cap in enumerate(_sensor_capabilities("chirps"), start=1):
        rows.append({
            "chirps_probe_id": f"S17C16_CHIRPS_{idx:04d}",
            "formal_request_id": cap["formal_request_id"],
            "provider_name": cap["provider_name"],
            "catalog_reference": cap["channel_reference"],
            "network_enabled": "false",
            "probe_status": "blocked_network_not_enabled",
            "dataset_identified": "false",
            "metadata_available": "false",
            "temporal_coverage_available": "false",
            "spatial_resolution_available": "false",
            "download_or_export_performed": "false",
            "raster_downloaded": "false",
            "supports_future_rainfall_trigger": "true",
            "blocking_reason": "rede desativada no build; nenhum raster CHIRPS baixado e nenhuma feature calculada",
            "review_only": "true",
        })
    return rows


def build_sentinel2_catalog_probe() -> list[dict]:
    rows = []
    for idx, cap in enumerate(_sensor_capabilities("sentinel2"), start=1):
        rows.append({
            "sentinel2_probe_id": f"S17C16_SENTINEL2_{idx:04d}",
            "formal_request_id": cap["formal_request_id"],
            "provider_name": cap["provider_name"],
            "catalog_reference": cap["channel_reference"],
            "network_enabled": "false",
            "probe_status": "blocked_network_not_enabled",
            "dataset_identified": "false",
            "metadata_available": "false",
            "mission_confirmed": "false",
            "product_type_or_collection": "not_available",
            "download_or_export_performed": "false",
            "raster_downloaded": "false",
            "tile_created": "false",
            "supports_future_sentinel2_tile": "true",
            "supports_future_embedding_tile": "true",
            "blocking_reason": "rede desativada no build; nenhum produto Sentinel-2, tile ou embedding foi criado",
            "review_only": "true",
        })
    return rows


def build_blocked_heavy_raster_registry() -> list[dict]:
    rows = []
    for cap in _sensor_capabilities("chirps"):
        rows.append({
            "blocked_heavy_raster_id": f"S17C16_HEAVY_{len(rows) + 1:04d}",
            "formal_request_id": cap["formal_request_id"],
            "provider_name": cap["provider_name"],
            "dataset_or_source": "CHIRPS",
            "requested_or_possible_action": "download_raster_or_export_precipitation_grid",
            "why_blocked": "raster bruto pesado e feature CHIRPS estao fora do escopo do 17C16",
            "can_be_done_in_future_with_policy": "true",
            "requires_storage_policy": "true",
            "requires_runtime_credentials": "false",
            "requires_patch_aoi": "true",
            "review_only": "true",
        })
    for cap in _sensor_capabilities("sentinel2"):
        rows.append({
            "blocked_heavy_raster_id": f"S17C16_HEAVY_{len(rows) + 1:04d}",
            "formal_request_id": cap["formal_request_id"],
            "provider_name": cap["provider_name"],
            "dataset_or_source": "Sentinel-2/CDSE",
            "requested_or_possible_action": "download_product_create_tile_or_embedding",
            "why_blocked": "produto Sentinel-2, tile e embedding DINO/SatMAE estao fora do escopo do 17C16",
            "can_be_done_in_future_with_policy": "true",
            "requires_storage_policy": "true",
            "requires_runtime_credentials": "true",
            "requires_patch_aoi": "true",
            "review_only": "true",
        })
    return rows


def build_gate_evaluation_matrix() -> list[dict]:
    rows = []
    for idx, cap in enumerate(_c15_capabilities(), start=1):
        official = cap["channel_status"] == "confirmed_official_source"
        rows.append({
            "gate_eval_id": f"S17C16_GATE_{idx:04d}",
            "formal_request_id": cap["formal_request_id"],
            "artifact_manifest_id": "not_available",
            "G1_existencia_documental": "false",
            "G2_confiabilidade_fonte": _bool_text(official),
            "G3_precisao_temporal": "false",
            "G4_vinculo_espacial": "false",
            "G5_separacao_fenomeno": "false",
            "G6_proveniencia_integridade": "false",
            "G7_anti_leakage": "true",
            "all_gates_passed": "false",
            "acceptance_status": "blocked_network_not_enabled" if _safe_to_probe(cap) else "blocked_no_event_artifact",
            "blocking_reason": "sem artefato oficial real coletado no build offline; catalogo sensorial nao e evento observado nem geometria de evento",
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
    rows = []
    for idx, gate in enumerate(build_gate_evaluation_matrix(), start=1):
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
            "rejected_or_pending_id": f"S17C16_REJECT_{idx:04d}",
            "formal_request_id": gate["formal_request_id"],
            "artifact_manifest_id": gate["artifact_manifest_id"],
            "provider_name": next(row["provider_name"] for row in _c15_capabilities() if row["formal_request_id"] == gate["formal_request_id"]),
            "acceptance_status": gate["acceptance_status"],
            "failed_gates": ";".join(sorted(set(failed))),
            "blocking_reason": gate["blocking_reason"],
            "required_next_action": "executar coleta publica com opt-in ou ingerir artefato oficial real com data, geometria, fenomeno e hash",
            **GOV,
        })
    return rows


def build_anti_leakage_audit() -> list[dict]:
    rows = []
    for idx, gate in enumerate(build_gate_evaluation_matrix(), start=1):
        rows.append({
            "anti_leakage_audit_id": f"S17C16_LEAK_{idx:04d}",
            "formal_request_id": gate["formal_request_id"],
            "artifact_manifest_id": gate["artifact_manifest_id"],
            "uses_post_event_data_as_pre_event_feature": "false",
            "uses_risk_area_as_event": "false",
            "uses_document_as_susceptibility_feature": "false",
            "uses_synthetic_as_real": "false",
            "uses_official_patch_neighbor_proxy": "false",
            "catalog_metadata_misused_as_event": "false",
            "passes_anti_leakage": "true",
            "blocking_reason": "metadado de catalogo nao foi usado como evento observado nem como feature de suscetibilidade",
            "review_only": "true",
        })
    return rows


def build_summary() -> dict:
    caps = build_network_capability_update()
    probes = build_public_probe_log()
    collections = build_lightweight_collection_log()
    manifests = build_collected_artifact_manifest()
    gates = build_gate_evaluation_matrix()
    accepted = build_accepted_candidates()
    rejected = build_rejected_or_pending()
    chirps = build_chirps_catalog_probe()
    sentinel = build_sentinel2_catalog_probe()
    return {
        "formal_requests_count": len(caps),
        "channels_evaluated_count": len(caps),
        "network_enabled": False,
        "public_probes_attempted_count": len([row for row in probes if row["attempted"] == "true"]),
        "lightweight_collection_attempts_count": len([row for row in collections if row["attempted"] == "true"]),
        "lightweight_artifacts_collected_count": len([row for row in collections if row["artifact_collected"] == "true"]),
        "artifacts_with_manifest_count": len(manifests),
        "chirps_catalogs_probed_count": len([row for row in chirps if row["metadata_available"] == "true"]),
        "sentinel2_catalogs_probed_count": len([row for row in sentinel if row["metadata_available"] == "true"]),
        "heavy_raster_downloads_count": 0,
        "raster_tiles_created_count": 0,
        "dino_satmae_embeddings_created_count": 0,
        "artifacts_all_gates_passed_count": len([row for row in gates if row["all_gates_passed"] == "true"]),
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
        "recommended_next_milestone": "SUSC-17C17 Politica de execucao para raster leve, AOI de patch e intake de metadados coletados com rede",
    }


def build_blockers() -> list[dict]:
    blockers = [
        "ground_reference_event_artifacts_still_missing",
        "catalog_metadata_not_event_reference",
        "spatial_gate_not_satisfied",
        "phenomenon_gate_not_satisfied",
        "candidate_specific_event_geometry_missing",
        "heavy_raster_download_blocked_by_policy",
        "sentinel2_tile_not_created",
        "chirps_feature_not_extracted",
        "embedding_tile_missing",
        "sar_runtime_missing",
        "qa_not_accepted",
        "17b_blocked_until_real_event_artifacts_and_policy",
    ]
    return [
        {
            "blocker_id": f"S17C16_BLOCKER_{idx:04d}",
            "blocker_type": blocker,
            "description": "Bloqueio preservado: catalogo sensorial e metadado publico nao bastam como evento observado; faltam artefato de evento, geometria, fenomeno, hash e politica para raster/tile.",
            "blocks_ground_reference_candidate": _bool_text(blocker in {"ground_reference_event_artifacts_still_missing", "catalog_metadata_not_event_reference", "spatial_gate_not_satisfied", "phenomenon_gate_not_satisfied"}),
            "blocks_17b": "true",
            "blocks_score_v7": "true",
            "review_only": "true",
        }
        for idx, blocker in enumerate(blockers, start=1)
    ]


def _schema(required: list[str], props: dict, title: str) -> dict:
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": title, "type": "object", "required": required, "properties": props}


def build_network_schema() -> dict:
    required = list(build_public_probe_log()[0].keys())
    return _schema(required, {
        "probe_log_id": {"type": "string", "pattern": "^S17C16_PROBE_"},
        "probe_result_class": {"enum": ["official_public_metadata_available", "official_public_artifact_available", "official_page_only_context", "blocked_network_not_enabled", "blocked_authenticated", "blocked_heavy_raster", "blocked_not_official", "failed_safe"]},
        "review_only": {"const": "true"},
        "trainable": {"const": "false"},
        "ground_truth": {"const": "false"},
    }, "SUSC-17C16 network probe schema v1")


def build_catalog_schema() -> dict:
    required = [
        "formal_request_id",
        "provider_name",
        "catalog_reference",
        "network_enabled",
        "probe_status",
        "dataset_identified",
        "metadata_available",
        "download_or_export_performed",
        "raster_downloaded",
        "review_only",
    ]
    return _schema(required, {
        "download_or_export_performed": {"const": "false"},
        "raster_downloaded": {"const": "false"},
        "review_only": {"const": "true"},
    }, "SUSC-17C16 catalog probe schema v1")


def build_manifest_schema() -> dict:
    return _schema(MANIFEST_FIELDS, {
        "artifact_manifest_id": {"type": "string", "pattern": "^S17C16_MANIFEST_"},
        "is_raw_heavy": {"const": "false"},
        "stored_in_outputs_public": {"const": "false"},
        "review_only": {"const": "true"},
        "trainable": {"const": "false"},
        "ground_truth": {"const": "false"},
    }, "SUSC-17C16 collected artifact manifest schema v1")


def build_report() -> str:
    summary = build_summary()
    return "\n".join([
        "# SUSC-17C16 - Coleta publica com opt-in de rede para catalogos CHIRPS/CDSE",
        "",
        "## Objetivo",
        "O 17C16 especializa a coleta publica controlada para catalogos CHIRPS e Sentinel-2/CDSE. O pacote versionado permanece offline e deterministico; rede real exige `SUSC_17C16_ALLOW_NETWORK=1` nos modos CLI.",
        "",
        "## Drivers",
        ", ".join(DRIVER_NAMES),
        "",
        "## Resultado do build publico",
        f"- Rede habilitada no build: {summary['network_enabled']}.",
        f"- Canais avaliados: {summary['channels_evaluated_count']}.",
        f"- Probes publicos tentados: {summary['public_probes_attempted_count']}.",
        f"- Coletas leves tentadas: {summary['lightweight_collection_attempts_count']}.",
        f"- Artefatos leves coletados: {summary['lightweight_artifacts_collected_count']}.",
        f"- Catalogos CHIRPS sondados com metadados: {summary['chirps_catalogs_probed_count']}.",
        f"- Catalogos Sentinel-2/CDSE sondados com metadados: {summary['sentinel2_catalogs_probed_count']}.",
        "",
        "## Gates",
        "Metadado de catalogo nao vira evento observado. Sem artefato real de evento, G4 e G5 permanecem bloqueados e nenhum Ground Reference Candidate e aceito.",
        "",
        "## Guardrails",
        "Nao houve submissao externa, protocolo, resposta inventada, contato inventado, raster pesado, tile, embedding, score v7, patch oficial, patch-link oficial, ground truth ou label de treino.",
        "",
        "## Proximo marco recomendado",
        summary["recommended_next_milestone"],
    ])


def build_all() -> None:
    _require_inputs()
    write_csv(CAPABILITY, build_network_capability_update())
    write_csv(PROBE_LOG, build_public_probe_log())
    write_csv(COLLECTION_LOG, build_lightweight_collection_log())
    write_csv(MANIFEST, build_collected_artifact_manifest(), MANIFEST_FIELDS)
    write_csv(CHIRPS_PROBE, build_chirps_catalog_probe())
    write_csv(SENTINEL2_PROBE, build_sentinel2_catalog_probe())
    write_csv(HEAVY_RASTER, build_blocked_heavy_raster_registry())
    write_csv(GATES, build_gate_evaluation_matrix())
    write_csv(ACCEPTED, build_accepted_candidates(), CANDIDATE_FIELDS)
    write_csv(REJECTED, build_rejected_or_pending())
    write_csv(ANTI_LEAKAGE, build_anti_leakage_audit())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_json(NETWORK_SCHEMA, build_network_schema())
    write_json(CATALOG_SCHEMA, build_catalog_schema())
    write_json(MANIFEST_SCHEMA, build_manifest_schema())
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


def _validate_byte_identical() -> list[str]:
    before = {path: path.read_bytes() for path in REQUIRED_OUTPUTS if path.exists()}
    build_all()
    errors = []
    for path, content in before.items():
        if path.read_bytes() != content:
            errors.append(f"offline_build_not_byte_identical:{rel(path)}")
    return errors


def validate() -> int:
    missing = [path for path in REQUIRED_OUTPUTS if not path.exists()]
    if missing:
        print("MISSING: " + "; ".join(rel(path) for path in missing), file=sys.stderr)
        return 1
    errors = _validate_byte_identical()
    caps = read_csv(CAPABILITY)
    probes = read_csv(PROBE_LOG)
    collections = read_csv(COLLECTION_LOG)
    manifests = read_csv(MANIFEST)
    chirps = read_csv(CHIRPS_PROBE)
    sentinel = read_csv(SENTINEL2_PROBE)
    heavy = read_csv(HEAVY_RASTER)
    gates = read_csv(GATES)
    accepted = read_csv(ACCEPTED)
    rejected = read_csv(REJECTED)
    leakage = read_csv(ANTI_LEAKAGE)
    summary = read_json(SUMMARY)
    net_schema = read_json(NETWORK_SCHEMA)
    catalog_schema = read_json(CATALOG_SCHEMA)
    manifest_schema = read_json(MANIFEST_SCHEMA)
    for row in probes:
        errors.extend(_schema_violations(row, net_schema))
    for row in chirps + sentinel:
        errors.extend(_schema_violations(row, catalog_schema))
    for row in manifests:
        errors.extend(_schema_violations(row, manifest_schema))
    for rows, key in [(caps, "network_capability_id"), (probes, "probe_log_id"), (collections, "collection_log_id"), (chirps, "chirps_probe_id"), (sentinel, "sentinel2_probe_id"), (heavy, "blocked_heavy_raster_id"), (gates, "gate_eval_id"), (rejected, "rejected_or_pending_id"), (leakage, "anti_leakage_audit_id")]:
        ids = [row[key] for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")
    if len(caps) != 9 or len(probes) != 9 or len(collections) != 9 or len(gates) != 9 or len(rejected) != 9:
        errors.append("unexpected_row_count")
    if len(chirps) != 1 or len(sentinel) != 1 or len(heavy) != 2:
        errors.append("unexpected_catalog_or_heavy_count")
    if any(row["network_enabled"] != "false" for row in probes + collections + chirps + sentinel):
        errors.append("network_enabled_in_public_build")
    if any(row["attempted"] != "false" for row in probes + collections):
        errors.append("network_attempted_without_opt_in")
    if manifests:
        errors.append("manifest_should_be_empty_without_real_artifact")
    if any(row["download_or_export_performed"] != "false" or row["raster_downloaded"] != "false" for row in chirps + sentinel):
        errors.append("raster_or_export_performed")
    if any(row.get("tile_created", "false") != "false" for row in sentinel):
        errors.append("tile_created")
    if accepted:
        errors.append("accepted_candidates_should_be_empty")
    if any(row["G4_vinculo_espacial"] == "true" or row["G5_separacao_fenomeno"] == "true" for row in gates):
        errors.append("catalog_metadata_promoted_to_event")
    if any(row["all_gates_passed"] == "true" for row in gates):
        errors.append("all_gates_passed_without_event_artifact")
    if any(row["catalog_metadata_misused_as_event"] != "false" or row["passes_anti_leakage"] != "true" for row in leakage):
        errors.append("anti_leakage_failed")
    for path in [SCORE_V6, OFFICIAL_PATCHES, OFFICIAL_PATCH_LINKS]:
        if path.exists() and _run_git(["diff", "--name-only", "--", rel(path)]):
            errors.append(f"official_dataset_changed:{rel(path)}")
    if SCORE_V7.exists():
        errors.append("score_v7_exists")
    expected = build_summary()
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(f"summary_mismatch:{key}")
    if summary["eligible_for_17b_now"] or summary["eligible_for_score_v7"] or summary["trainable"] or summary["ground_truth"]:
        errors.append("promotion_guardrail_failed")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "17C16 -> "
        f"channels={summary['channels_evaluated_count']} "
        f"network={summary['network_enabled']} "
        f"probes={summary['public_probes_attempted_count']} "
        f"collected={summary['lightweight_artifacts_collected_count']} "
        f"accepted={summary['accepted_ground_reference_candidates_count']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
    return 0


def capabilities_text() -> str:
    lines = []
    for row in build_network_capability_update():
        if row["can_probe_with_network"] == "true":
            lines.append(f"{row['formal_request_id']} {row['driver_name']} network_required={row['network_required']}")
    return "\n".join(lines)


def _network_guard(mode: str) -> tuple[int, str] | None:
    if _network_enabled():
        return None
    return (2, f"blocked_network_not_enabled: {mode} exige SUSC_17C16_ALLOW_NETWORK=1.")


def _http_probe(reference: str) -> dict:
    req = Request(reference, method="HEAD", headers={"User-Agent": "REV-P-SUSC-17C16-review-only"})
    try:
        with urlopen(req, timeout=20) as response:
            return {
                "http_status": str(response.status),
                "content_type": response.headers.get("content-type", "not_available"),
                "content_length": response.headers.get("content-length", "not_available"),
                "final_reference": response.geturl(),
                "ok": True,
            }
    except HTTPError as exc:
        return {"http_status": str(exc.code), "content_type": exc.headers.get("content-type", "not_available"), "content_length": exc.headers.get("content-length", "not_available"), "final_reference": reference, "ok": False}
    except URLError as exc:
        return {"http_status": "failed_safe", "content_type": "not_available", "content_length": "not_available", "final_reference": reference, "ok": False, "error": str(exc.reason)}


def _safe_filename(request_id: str, suffix: str) -> str:
    return f"{request_id.lower()}_{suffix}".replace("/", "_").replace("\\", "_")


def probe_public_text() -> tuple[int, str]:
    guard = _network_guard("probe-public")
    if guard:
        return guard
    results = []
    for row in build_network_capability_update():
        if row["can_probe_with_network"] != "true":
            continue
        cap = next(cap for cap in _c15_capabilities() if cap["formal_request_id"] == row["formal_request_id"])
        probe = _http_probe(cap["channel_reference"])
        results.append({"formal_request_id": row["formal_request_id"], **probe})
    ensure_dir(LOCAL_STATE)
    (LOCAL_STATE / "probe_public_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return (0, f"probe-public: {len(results)} probes publicos leves registrados em {rel(LOCAL_STATE / 'probe_public_results.json')}.")


def collect_lightweight_text() -> tuple[int, str]:
    guard = _network_guard("collect-lightweight")
    if guard:
        return guard
    collected = []
    ensure_dir(LOCAL_STATE / "raw_lightweight")
    for row in build_network_capability_update():
        if row["can_probe_with_network"] != "true":
            continue
        cap = next(cap for cap in _c15_capabilities() if cap["formal_request_id"] == row["formal_request_id"])
        req = Request(cap["channel_reference"], headers={"User-Agent": "REV-P-SUSC-17C16-review-only"})
        try:
            with urlopen(req, timeout=20) as response:
                content_type = response.headers.get("content-type", "not_available")
                data = response.read(MAX_LIGHTWEIGHT_BYTES + 1)
                if len(data) > MAX_LIGHTWEIGHT_BYTES:
                    collected.append({"formal_request_id": row["formal_request_id"], "status": "blocked_heavy_raster_or_large_artifact"})
                    continue
                path = LOCAL_STATE / "raw_lightweight" / _safe_filename(row["formal_request_id"], "artifact.bin")
                path.write_bytes(data)
                collected.append({
                    "formal_request_id": row["formal_request_id"],
                    "status": "collected_lightweight_artifact",
                    "artifact_local_path": str(path),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                    "content_type": content_type,
                })
        except (HTTPError, URLError) as exc:
            collected.append({"formal_request_id": row["formal_request_id"], "status": "failed_safe", "error": str(exc)})
    manifest = LOCAL_STATE / "lightweight_collection_manifest.json"
    manifest.write_text(json.dumps(collected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return (0, f"collect-lightweight: {len([r for r in collected if r['status'] == 'collected_lightweight_artifact'])} artefatos leves locais registrados em {rel(manifest)}.")


def probe_sensor_catalog_text() -> tuple[int, str]:
    guard = _network_guard("probe-sensor-catalog")
    if guard:
        return guard
    code, text = probe_public_text()
    if code != 0:
        return (code, text)
    return (0, "probe-sensor-catalog: metadados de catalogo sondados em modo publico leve; nenhum raster, tile ou export executado.")


def intake_collected_text() -> str:
    return "intake-collected: nenhum artefato leve versionado foi ingerido; gates permanecem bloqueados para evento observado."


def status_text() -> str:
    summary = build_summary()
    return (
        f"status 17C16: network_enabled={summary['network_enabled']} "
        f"probes={summary['public_probes_attempted_count']} "
        f"artifacts={summary['lightweight_artifacts_collected_count']} "
        f"accepted={summary['accepted_ground_reference_candidates_count']}"
    )
