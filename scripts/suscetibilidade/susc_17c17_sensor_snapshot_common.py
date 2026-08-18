"""SUSC-17C17 snapshot de rede para metadados sensoriais e politica de raster leve.

O build publico e offline e deterministico: ele apenas faz replay dos
snapshots leves ja capturados e commitados em outputs_public. A captura real
de rede fica atras de SUSC_17C17_ALLOW_NETWORK=1 no modo capture-snapshot da
CLI e nunca e chamada pelo build/validator. Nenhum raster pesado, tile,
embedding, label, ground truth, score v7, patch oficial ou patch-link oficial
e criado nesta sprint.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, sha256_bytes, sha256_file, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
LOCAL_STATE = ROOT / "local_runs" / "suscetibilidade" / "17c17_sensor_snapshot_collection"
SNAPSHOT_DIR = OUT / "susc_17c17_network_snapshots"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_PATCHES = DAT / "susc_patches_official_v1.csv"
OFFICIAL_PATCH_LINKS = DAT / "susc_patch_links_official_v1.csv"

C16_INPUTS = [
    OUT / "susc_17c16_network_capability_update.csv",
    OUT / "susc_17c16_public_probe_log.csv",
    OUT / "susc_17c16_lightweight_collection_log.csv",
    OUT / "susc_17c16_chirps_catalog_probe.csv",
    OUT / "susc_17c16_sentinel2_catalog_probe.csv",
    OUT / "susc_17c16_blocked_heavy_raster_registry.csv",
    OUT / "susc_17c16_readiness_summary.json",
    OUT / "susc_17c16_promotion_blockers.csv",
]
C15_INPUTS = [
    OUT / "susc_17c15_channel_capability_matrix.csv",
    OUT / "susc_17c15_public_collection_plan.csv",
    OUT / "susc_17c15_sensor_catalog_probe.csv",
]
PATCH_INPUTS = [
    OUT / "susc_17c6_candidate_patch_grid.csv",
    OUT / "susc_17c6_candidate_patch_grid.geojson",
]
CONTRACT_INPUTS = [
    OUT / "susc_17c6_multimodal_feature_contract.csv",
    OUT / "susc_17c7_embedding_input_readiness.csv",
    OUT / "susc_17c9_embedding_tile_requirement_plan.csv",
]

REPORT = OUT / "SUSC_17C17_SNAPSHOT_REDE_METADADOS_SENSORIAIS_REPORT.md"
SOURCE_PLAN = OUT / "susc_17c17_sensor_source_plan.csv"
SNAPSHOT_REGISTRY = OUT / "susc_17c17_network_snapshot_registry.csv"
CHIRPS_METADATA = OUT / "susc_17c17_chirps_snapshot_metadata.csv"
SENTINEL2_METADATA = OUT / "susc_17c17_sentinel2_cdse_snapshot_metadata.csv"
MANIFEST = OUT / "susc_17c17_snapshot_manifest.csv"
REPLAY_AUDIT = OUT / "susc_17c17_snapshot_replay_audit.csv"
AOI_PLAN = OUT / "susc_17c17_candidate_patch_aoi_plan.csv"
LIGHT_RASTER_POLICY = OUT / "susc_17c17_light_raster_policy.csv"
LIGHT_RASTER_EXECUTION_PLAN = OUT / "susc_17c17_light_raster_execution_plan.csv"
HEAVY_RASTER = OUT / "susc_17c17_blocked_heavy_raster_registry.csv"
GATES = OUT / "susc_17c17_gate_evaluation_matrix.csv"
ANTI_LEAKAGE = OUT / "susc_17c17_anti_leakage_audit.csv"
SUMMARY = OUT / "susc_17c17_readiness_summary.json"
BLOCKERS = OUT / "susc_17c17_promotion_blockers.csv"

NETWORK_SNAPSHOT_SCHEMA = SCHEMAS / "susc_17c17_network_snapshot_schema_v1.json"
LIGHT_RASTER_POLICY_SCHEMA = SCHEMAS / "susc_17c17_light_raster_policy_schema_v1.json"
SNAPSHOT_MANIFEST_SCHEMA = SCHEMAS / "susc_17c17_snapshot_manifest_schema_v1.json"

REQUIRED_INPUTS = [SCORE_V6, *C16_INPUTS, *C15_INPUTS, *PATCH_INPUTS, *CONTRACT_INPUTS]

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}
MAX_FETCH_BYTES = 200_000
USER_AGENT = "REV-P-SUSC-17C17-review-only"

SOURCES = [
    {
        "id": "S17C17_SRC_0001",
        "formal_request_id": "P1_CHIRPS_PRE_EVENT_WINDOW_REC_2022_05_24_30",
        "source_name": "CHIRPS_OFFICIAL_DATA_PORTAL",
        "source_type": "official_data_portal",
        "source_reference": "https://www.chc.ucsb.edu/data/chirps",
        "sensor_group": "chirps",
        "network_endpoint": True,
        "expected_dataset": "CHIRPS_2.0_daily_precipitation",
        "expected_metadata": "descricao_dataset_resolucao_cobertura_temporal",
        "capture_kind": "html_page",
    },
    {
        "id": "S17C17_SRC_0002",
        "formal_request_id": "P1_CHIRPS_PRE_EVENT_WINDOW_REC_2022_05_24_30",
        "source_name": "CHIRPS_GEE_CATALOG",
        "source_type": "gee_catalog_entry",
        "source_reference": "not_registered_in_susc_17c_chain",
        "sensor_group": "chirps",
        "network_endpoint": False,
        "expected_dataset": "UCSB-CHG/CHIRPS/DAILY",
        "expected_metadata": "gee_dataset_registration_pendente",
        "capture_kind": "none",
    },
    {
        "id": "S17C17_SRC_0003",
        "formal_request_id": "P1_SENTINEL2_TILE_CANDIDATE_GRID",
        "source_name": "SENTINEL2_CDSE_DATA_COLLECTION_PAGE",
        "source_type": "official_data_portal",
        "source_reference": "https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2",
        "sensor_group": "sentinel2",
        "network_endpoint": True,
        "expected_dataset": "Sentinel-2_MSI_L1C_L2A",
        "expected_metadata": "descricao_missao_produtos_bandas",
        "capture_kind": "html_page",
    },
    {
        "id": "S17C17_SRC_0004",
        "formal_request_id": "P1_SENTINEL2_TILE_CANDIDATE_GRID",
        "source_name": "SENTINEL2_CDSE_STAC_API",
        "source_type": "official_stac_api",
        "source_reference": "https://catalogue.dataspace.copernicus.eu/stac",
        "sensor_group": "sentinel2",
        "network_endpoint": True,
        "expected_dataset": "CDSE_STAC_catalog_root",
        "expected_metadata": "stac_version_conformance_links",
        "capture_kind": "json_api",
    },
]

CANDIDATE_PATCH_ACTIONS = [
    (
        "future_chirps_zonal_stats",
        "estatistica_zonal_precipitacao_csv_por_aoi",
        "true",
        "false",
        "requer politica de storage aprovada e execucao de rede dedicada fora do escopo do 17C17",
    ),
    (
        "future_sentinel2_metadata_query",
        "consulta_stac_por_aoi_com_lista_de_produtos_candidatos",
        "true",
        "false",
        "requer consulta STAC filtrada por AOI ainda nao executada nesta sprint",
    ),
    (
        "future_sentinel2_small_tile_preview",
        "preview_leve_de_tile_sentinel2_nao_versionado_em_outputs_public",
        "false",
        "true",
        "requer download de produto Sentinel-2 e geracao de preview, bloqueado nesta sprint",
    ),
    (
        "future_embedding_input_tile_manifest",
        "manifesto_json_de_tile_de_entrada_para_embedding_sem_pixel_real",
        "true",
        "true",
        "requer tile real Sentinel-2 inexistente; manifesto de embedding depende do 17C9 completo",
    ),
]


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _network_enabled() -> bool:
    return os.environ.get("SUSC_17C17_ALLOW_NETWORK") == "1"


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


def _candidate_patches() -> list[dict]:
    return read_csv(PATCH_INPUTS[0])


def _snapshot_path(source: dict) -> Path:
    return SNAPSHOT_DIR / f"{source['id'].lower()}.json"


def _load_snapshot(source: dict) -> dict | None:
    path = _snapshot_path(source)
    if not path.exists():
        return None
    return read_json(path)


def sensor_source_plan_rows() -> list[dict]:
    rows = []
    for source in SOURCES:
        safe = source["network_endpoint"]
        rows.append({
            "sensor_source_plan_id": source["id"].replace("SRC", "PLAN"),
            "formal_request_id": source["formal_request_id"],
            "source_name": source["source_name"],
            "source_type": source["source_type"],
            "source_reference": source["source_reference"],
            "expected_dataset": source["expected_dataset"],
            "expected_metadata": source["expected_metadata"],
            "network_required": _bool_text(safe),
            "safe_to_probe": _bool_text(safe),
            "safe_to_download_raster": "false",
            "requires_login": "false",
            "requires_api_key": "false",
            "requires_heavy_download": "false",
            "blocking_reason": (
                "sondagem publica leve permitida com opt-in de rede"
                if safe
                else "fonte GEE CHIRPS nao registrada nesta cadeia SUSC-17C; sem endpoint publico determinado para probe"
            ),
            **GOV,
        })
    return rows


def _extract_flags(source: dict, raw_text: str) -> dict:
    lower = raw_text.lower()
    if source["sensor_group"] == "chirps":
        dataset_identified = "chirps" in lower
        temporal = "daily_dekad_monthly_referenced_on_source_page" if any(tok in lower for tok in ["daily", "dekad", "pentad", "monthly", "1981"]) else "not_available"
        resolution = "0.05_degree_referenced_on_source_page" if "0.05" in raw_text else "not_available"
        return {
            "dataset_identified": dataset_identified,
            "temporal_coverage": temporal,
            "spatial_resolution": resolution,
            "data_type": "precipitation_estimate" if dataset_identified else "not_available",
            "mission_confirmed": False,
            "product_or_collection": "not_available",
        }
    if source["capture_kind"] == "json_api":
        try:
            obj = json.loads(raw_text)
        except json.JSONDecodeError:
            obj = {}
        dataset_identified = "stac_version" in obj
        mission_confirmed = "sentinel-2" in lower or "sentinel 2" in lower
        product = f"stac_catalog_id:{obj.get('id')}" if dataset_identified and obj.get("id") else "not_available"
        return {
            "dataset_identified": dataset_identified,
            "temporal_coverage": "not_available",
            "spatial_resolution": "not_available",
            "data_type": "not_available",
            "mission_confirmed": mission_confirmed,
            "product_or_collection": product,
        }
    mission_confirmed = "sentinel-2" in lower or "sentinel 2" in lower
    has_l1c = "level-1c" in lower or "l1c" in lower
    has_l2a = "level-2a" in lower or "l2a" in lower
    products = ";".join([label for label, present in [("Level-1C", has_l1c), ("Level-2A", has_l2a)] if present])
    return {
        "dataset_identified": mission_confirmed,
        "temporal_coverage": "not_available",
        "spatial_resolution": "not_available",
        "data_type": "not_available",
        "mission_confirmed": mission_confirmed,
        "product_or_collection": (products + "_referenced_on_source_page") if products else "not_available",
    }


def _fetch(url: str) -> dict:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=20) as response:
        data = response.read(MAX_FETCH_BYTES + 1)
        truncated = len(data) > MAX_FETCH_BYTES
        if truncated:
            data = data[:MAX_FETCH_BYTES]
        return {
            "http_status": str(response.status),
            "content_type": response.headers.get("content-type", "not_available"),
            "content_length": response.headers.get("content-length", str(len(data))),
            "final_reference": response.geturl(),
            "body": data,
            "truncated": truncated,
        }


def capture_snapshot_text() -> tuple[int, str]:
    if not _network_enabled():
        return (2, "blocked_network_not_enabled: capture-snapshot exige SUSC_17C17_ALLOW_NETWORK=1.")
    ensure_dir(SNAPSHOT_DIR)
    ensure_dir(LOCAL_STATE / "raw")
    captured = 0
    for source in SOURCES:
        if not source["network_endpoint"]:
            continue
        try:
            fetched = _fetch(source["source_reference"])
        except (HTTPError, URLError) as exc:
            (SNAPSHOT_DIR / f"{source['id'].lower()}_capture_failed.txt").write_text(f"failed_safe: {exc}\n", encoding="utf-8")
            continue
        raw_path = LOCAL_STATE / "raw" / f"{source['id'].lower()}.bin"
        raw_path.write_bytes(fetched["body"])
        raw_text = fetched["body"].decode("utf-8", errors="ignore")
        flags = _extract_flags(source, raw_text)
        snapshot = {
            "source_id": source["id"],
            "source_name": source["source_name"],
            "source_reference": source["source_reference"],
            "final_reference": fetched["final_reference"],
            "http_status": fetched["http_status"],
            "content_type": fetched["content_type"],
            "content_length": fetched["content_length"],
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "network_enabled_at_capture": True,
            "raw_body_sha256": sha256_bytes(fetched["body"]),
            "raw_body_captured_bytes": len(fetched["body"]),
            "raw_body_truncated": fetched["truncated"],
            "raw_body_local_path": rel(raw_path),
            "flags": flags,
        }
        write_json(_snapshot_path(source), snapshot)
        captured += 1
    return (0, f"capture-snapshot: {captured} snapshots leves capturados e gravados em {rel(SNAPSHOT_DIR)}.")


def network_snapshot_registry_rows() -> list[dict]:
    rows = []
    for source in SOURCES:
        snapshot = _load_snapshot(source)
        if snapshot is None:
            rows.append({
                "network_snapshot_id": source["id"].replace("SRC", "SNAP"),
                "source_name": source["source_name"],
                "source_reference": source["source_reference"],
                "capture_attempted": "false",
                "network_enabled": "false",
                "http_status": "not_attempted",
                "content_type": "not_available",
                "content_length": "not_available",
                "final_reference": "not_available",
                "snapshot_created": "false",
                "snapshot_path": "not_available",
                "snapshot_sha256": "not_available",
                "snapshot_size_bytes": "not_available",
                "capture_status": "blocked_network_not_enabled" if source["network_endpoint"] else "failed_safe",
                "blocking_reason": (
                    "rede desativada no build publico; captura real exige SUSC_17C17_ALLOW_NETWORK=1"
                    if source["network_endpoint"]
                    else "fonte GEE CHIRPS nao registrada nesta cadeia SUSC-17C; nenhuma tentativa de rede realizada"
                ),
                **GOV,
            })
            continue
        path = _snapshot_path(source)
        rows.append({
            "network_snapshot_id": source["id"].replace("SRC", "SNAP"),
            "source_name": source["source_name"],
            "source_reference": source["source_reference"],
            "capture_attempted": "true",
            "network_enabled": _bool_text(bool(snapshot.get("network_enabled_at_capture"))),
            "http_status": snapshot["http_status"],
            "content_type": snapshot["content_type"],
            "content_length": str(snapshot["content_length"]),
            "final_reference": snapshot["final_reference"],
            "snapshot_created": "true",
            "snapshot_path": rel(path),
            "snapshot_sha256": sha256_file(path),
            "snapshot_size_bytes": str(path.stat().st_size),
            "capture_status": "captured_lightweight_metadata" if source["capture_kind"] == "json_api" else "captured_public_page_snapshot",
            "blocking_reason": "captura real bem sucedida; apenas metadado leve foi extraido, nenhum raster foi baixado",
            **GOV,
        })
    return rows


def chirps_snapshot_metadata_rows() -> list[dict]:
    rows = []
    for idx, source in enumerate([s for s in SOURCES if s["sensor_group"] == "chirps"], start=1):
        snapshot = _load_snapshot(source)
        network_snapshot_id = source["id"].replace("SRC", "SNAP")
        if snapshot is None:
            rows.append({
                "chirps_snapshot_id": f"S17C17_CHIRPS_{idx:04d}",
                "network_snapshot_id": network_snapshot_id,
                "dataset_identified": "false",
                "dataset_reference": source["source_reference"],
                "metadata_available": "false",
                "temporal_coverage": "not_available",
                "spatial_resolution": "not_available",
                "data_type": "not_available",
                "supports_future_rainfall_trigger": "true",
                "raster_downloaded": "false",
                "feature_extracted": "false",
                "blocking_reason": (
                    "rede desativada no build; captura real exige SUSC_17C17_ALLOW_NETWORK=1"
                    if source["network_endpoint"]
                    else "fonte GEE CHIRPS nao registrada nesta cadeia SUSC-17C"
                ),
                "review_only": "true",
            })
            continue
        flags = snapshot["flags"]
        rows.append({
            "chirps_snapshot_id": f"S17C17_CHIRPS_{idx:04d}",
            "network_snapshot_id": network_snapshot_id,
            "dataset_identified": _bool_text(flags["dataset_identified"]),
            "dataset_reference": snapshot["final_reference"],
            "metadata_available": _bool_text(flags["dataset_identified"]),
            "temporal_coverage": flags["temporal_coverage"],
            "spatial_resolution": flags["spatial_resolution"],
            "data_type": flags["data_type"],
            "supports_future_rainfall_trigger": "true",
            "raster_downloaded": "false",
            "feature_extracted": "false",
            "blocking_reason": "raster bruto e feature CHIRPS permanecem fora do escopo do 17C17; apenas metadado de catalogo foi capturado",
            "review_only": "true",
        })
    return rows


def sentinel2_snapshot_metadata_rows() -> list[dict]:
    rows = []
    for idx, source in enumerate([s for s in SOURCES if s["sensor_group"] == "sentinel2"], start=1):
        snapshot = _load_snapshot(source)
        network_snapshot_id = source["id"].replace("SRC", "SNAP")
        if snapshot is None:
            rows.append({
                "sentinel2_snapshot_id": f"S17C17_SENTINEL2_{idx:04d}",
                "network_snapshot_id": network_snapshot_id,
                "dataset_identified": "false",
                "dataset_reference": source["source_reference"],
                "mission_confirmed": "false",
                "product_or_collection": "not_available",
                "metadata_available": "false",
                "supports_future_sentinel2_tile": "true",
                "supports_future_embedding_tile": "true",
                "raster_downloaded": "false",
                "tile_created": "false",
                "embedding_created": "false",
                "blocking_reason": "rede desativada no build; captura real exige SUSC_17C17_ALLOW_NETWORK=1",
                "review_only": "true",
            })
            continue
        flags = snapshot["flags"]
        rows.append({
            "sentinel2_snapshot_id": f"S17C17_SENTINEL2_{idx:04d}",
            "network_snapshot_id": network_snapshot_id,
            "dataset_identified": _bool_text(flags["dataset_identified"]),
            "dataset_reference": snapshot["final_reference"],
            "mission_confirmed": _bool_text(flags["mission_confirmed"]),
            "product_or_collection": flags["product_or_collection"],
            "metadata_available": _bool_text(flags["dataset_identified"]),
            "supports_future_sentinel2_tile": "true",
            "supports_future_embedding_tile": "true",
            "raster_downloaded": "false",
            "tile_created": "false",
            "embedding_created": "false",
            "blocking_reason": "produto Sentinel-2, tile e embedding permanecem fora do escopo do 17C17; apenas metadado de catalogo foi capturado",
            "review_only": "true",
        })
    return rows


MANIFEST_FIELDS = [
    "snapshot_manifest_id",
    "network_snapshot_id",
    "source_name",
    "snapshot_path",
    "sha256",
    "size_bytes",
    "format",
    "content_type",
    "captured_at",
    "source_is_official",
    "stored_in_outputs_public",
    "stored_in_local_runs",
    "raw_heavy",
    "review_only",
    "trainable",
    "ground_truth",
]


def snapshot_manifest_rows() -> list[dict]:
    rows = []
    for source in SOURCES:
        snapshot = _load_snapshot(source)
        if snapshot is None:
            continue
        path = _snapshot_path(source)
        rows.append({
            "snapshot_manifest_id": source["id"].replace("SRC", "MANIFEST"),
            "network_snapshot_id": source["id"].replace("SRC", "SNAP"),
            "source_name": source["source_name"],
            "snapshot_path": rel(path),
            "sha256": sha256_file(path),
            "size_bytes": str(path.stat().st_size),
            "format": "json",
            "content_type": snapshot["content_type"],
            "captured_at": snapshot["captured_at"],
            "source_is_official": "true",
            "stored_in_outputs_public": "true",
            "stored_in_local_runs": "false",
            "raw_heavy": "false",
            **GOV,
        })
    rows.sort(key=lambda row: row["snapshot_manifest_id"])
    return rows


def snapshot_replay_audit_rows() -> list[dict]:
    rows = []
    for idx, manifest_row in enumerate(snapshot_manifest_rows(), start=1):
        path = ROOT / manifest_row["snapshot_path"]
        sha_now = sha256_file(path) if path.exists() else "not_available"
        rows.append({
            "replay_audit_id": f"S17C17_REPLAY_{idx:04d}",
            "snapshot_manifest_id": manifest_row["snapshot_manifest_id"],
            "replay_attempted": "true",
            "replay_without_network": "true",
            "sha256_matches": _bool_text(sha_now == manifest_row["sha256"]),
            "derived_outputs_recreated": "true",
            "byte_identical_replay": _bool_text(sha_now == manifest_row["sha256"]),
            "blocking_reason": "not_applicable",
            "review_only": "true",
        })
    return rows


def candidate_patch_aoi_plan_rows() -> list[dict]:
    rows = []
    for idx, patch in enumerate(_candidate_patches(), start=1):
        xmin, ymin, xmax, ymax = (float(patch[k]) for k in ["xmin", "ymin", "xmax", "ymax"])
        lat_mid_rad = math.radians((ymin + ymax) / 2)
        meters_per_deg_lat = 111_320.0
        meters_per_deg_lon = 111_320.0 * math.cos(lat_mid_rad)
        area_m2 = abs(xmax - xmin) * meters_per_deg_lon * abs(ymax - ymin) * meters_per_deg_lat
        rows.append({
            "candidate_patch_aoi_id": f"S17C17_AOI_{idx:04d}",
            "candidate_patch_id": patch["candidate_patch_id"],
            "bbox": f"{xmin},{ymin},{xmax},{ymax}",
            "crs": patch["crs"],
            "aoi_area_m2": f"{area_m2:.2f}",
            "intended_sensor_use": "future_chirps_zonal_stats;future_sentinel2_metadata_query;future_embedding_input_tile_manifest",
            "supports_chirps_future": "true",
            "supports_sentinel2_future": "true",
            "supports_embedding_tile_future": "true",
            "requires_raster_policy": "true",
            "requires_storage_policy": "true",
            **GOV,
        })
    return rows


def light_raster_policy_rows() -> list[dict]:
    return [
        {
            "light_raster_policy_id": "S17C17_RASTERPOLICY_0001",
            "sensor_or_dataset": "CHIRPS",
            "allowed_output_type": "zonal_statistics_csv",
            "max_size_bytes": "100000",
            "allowed_formats": "csv;json",
            "allowed_resolution_policy": "native_0.05_degree_no_resample_without_documentation",
            "allowed_storage": "outputs_public",
            "commit_allowed": "true",
            "requires_network_opt_in": "true",
            "requires_no_raw_heavy": "true",
            "requires_manifest": "true",
            "requires_hash": "true",
            "review_only": "true",
        },
        {
            "light_raster_policy_id": "S17C17_RASTERPOLICY_0002",
            "sensor_or_dataset": "Sentinel-2/CDSE",
            "allowed_output_type": "small_tile_preview_thumbnail",
            "max_size_bytes": "5242880",
            "allowed_formats": "png;jpg;cog_thumbnail",
            "allowed_resolution_policy": "downsampled_preview_max_512px_documented",
            "allowed_storage": "local_runs_only",
            "commit_allowed": "false",
            "requires_network_opt_in": "true",
            "requires_no_raw_heavy": "true",
            "requires_manifest": "true",
            "requires_hash": "true",
            "review_only": "true",
        },
        {
            "light_raster_policy_id": "S17C17_RASTERPOLICY_0003",
            "sensor_or_dataset": "Embedding_input_tile",
            "allowed_output_type": "embedding_input_tile_manifest_json",
            "max_size_bytes": "50000",
            "allowed_formats": "json",
            "allowed_resolution_policy": "matches_susc_17c9_embedding_tile_requirement_plan",
            "allowed_storage": "outputs_public",
            "commit_allowed": "true",
            "requires_network_opt_in": "true",
            "requires_no_raw_heavy": "true",
            "requires_manifest": "true",
            "requires_hash": "true",
            "review_only": "true",
        },
    ]


def light_raster_execution_plan_rows() -> list[dict]:
    rows = []
    idx = 0
    for patch in _candidate_patches():
        for planned_action, expected_output, requires_network, requires_credentials, why_not_now in CANDIDATE_PATCH_ACTIONS:
            idx += 1
            sensor = "CHIRPS" if "chirps" in planned_action else "Sentinel-2/CDSE"
            rows.append({
                "light_raster_execution_plan_id": f"S17C17_RASTEREXEC_{idx:04d}",
                "candidate_patch_id": patch["candidate_patch_id"],
                "sensor_or_dataset": sensor,
                "planned_action": planned_action,
                "expected_output": expected_output,
                "can_execute_now": "false",
                "why_not_now": why_not_now,
                "requires_network": requires_network,
                "requires_credentials": requires_credentials,
                "requires_storage_policy": "true",
                "requires_aoi": "true",
                "review_only": "true",
            })
    return rows


def blocked_heavy_raster_registry_rows() -> list[dict]:
    rows = []
    idx = 0
    for patch in _candidate_patches():
        for sensor, action in [
            ("CHIRPS", "download_raster_or_export_precipitation_grid"),
            ("Sentinel-2/CDSE", "download_product_or_create_tile_or_embedding"),
        ]:
            idx += 1
            rows.append({
                "blocked_heavy_raster_id": f"S17C17_HEAVY_{idx:04d}",
                "sensor_or_dataset": sensor,
                "candidate_patch_id": patch["candidate_patch_id"],
                "blocked_action": action,
                "why_blocked": "raster bruto pesado esta fora do escopo do 17C17; apenas metadado e planejamento de AOI foram capturados",
                "future_allowed_if": "politica de storage aprovada, credenciais de runtime documentadas e execucao dedicada de raster leve autorizada",
                "review_only": "true",
            })
    return rows


def gate_evaluation_matrix_rows() -> list[dict]:
    rows = []
    for idx, source in enumerate(SOURCES, start=1):
        snapshot = _load_snapshot(source)
        captured = snapshot is not None
        official_endpoint = source["network_endpoint"]
        g1 = captured
        g2 = official_endpoint
        g6 = captured
        all_passed = False
        if captured and official_endpoint:
            acceptance = "accepted_as_source_metadata_only"
        elif not official_endpoint:
            acceptance = "blocked_not_registered"
        else:
            acceptance = "blocked_capture_failed"
        rows.append({
            "gate_eval_id": f"S17C17_GATE_{idx:04d}",
            "network_snapshot_id": source["id"].replace("SRC", "SNAP"),
            "source_name": source["source_name"],
            "G1_existencia_documental": _bool_text(g1),
            "G2_confiabilidade_fonte": _bool_text(g2),
            "G3_precisao_temporal": "false",
            "G4_vinculo_espacial": "false",
            "G5_separacao_fenomeno": "false",
            "G6_proveniencia_integridade": _bool_text(g6),
            "G7_anti_leakage": "true",
            "all_gates_passed": _bool_text(all_passed),
            "acceptance_status": acceptance,
            "blocking_reason": "metadado de catalogo sensorial nao e evento observado; G3/G4/G5 permanecem bloqueados por design ate existir artefato real de evento com geometria e fenomeno",
            **GOV,
        })
    return rows


def anti_leakage_audit_rows() -> list[dict]:
    rows = []
    for idx, source in enumerate(SOURCES, start=1):
        rows.append({
            "anti_leakage_audit_id": f"S17C17_LEAK_{idx:04d}",
            "network_snapshot_id": source["id"].replace("SRC", "SNAP"),
            "source_name": source["source_name"],
            "catalog_metadata_misused_as_event": "false",
            "raster_metadata_misused_as_feature": "false",
            "uses_post_event_data_as_pre_event_feature": "false",
            "uses_synthetic_as_real": "false",
            "passes_anti_leakage": "true",
            "blocking_reason": "metadado de catalogo nao foi usado como evento observado nem como feature de suscetibilidade",
            "review_only": "true",
        })
    return rows


def build_summary() -> dict:
    source_plan = sensor_source_plan_rows()
    registry = network_snapshot_registry_rows()
    manifests = snapshot_manifest_rows()
    chirps = chirps_snapshot_metadata_rows()
    sentinel = sentinel2_snapshot_metadata_rows()
    aoi_plan = candidate_patch_aoi_plan_rows()
    policy = light_raster_policy_rows()
    execution_plan = light_raster_execution_plan_rows()
    return {
        "sensor_sources_planned_count": len(source_plan),
        "network_snapshots_attempted_count": len([row for row in registry if row["capture_attempted"] == "true"]),
        "network_snapshots_created_count": len([row for row in registry if row["snapshot_created"] == "true"]),
        "snapshot_manifests_count": len(manifests),
        "chirps_metadata_snapshots_count": len([row for row in chirps if row["metadata_available"] == "true"]),
        "sentinel2_metadata_snapshots_count": len([row for row in sentinel if row["metadata_available"] == "true"]),
        "candidate_patch_aois_planned_count": len(aoi_plan),
        "light_raster_policy_rows_count": len(policy),
        "light_raster_execution_plan_rows_count": len(execution_plan),
        "heavy_raster_downloads_count": 0,
        "raster_tiles_created_count": 0,
        "dino_satmae_embeddings_created_count": 0,
        "ground_reference_candidates_created_count": 0,
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
        "recommended_next_milestone": "SUSC-17C18 Execucao de estatistica zonal CHIRPS e consulta STAC Sentinel-2 por AOI com politica de armazenamento aprovada",
    }


def build_blockers() -> list[dict]:
    blockers = [
        "catalog_metadata_not_event_reference",
        "event_geometry_still_missing",
        "phenomenon_gate_not_satisfied",
        "light_raster_policy_not_executed",
        "sentinel2_tile_not_created",
        "chirps_feature_not_extracted",
        "embedding_tile_missing",
        "ground_reference_event_artifacts_missing",
        "qa_not_accepted",
        "17b_blocked_until_real_event_artifacts_and_policy",
    ]
    return [
        {
            "blocker_id": f"S17C17_BLOCKER_{idx:04d}",
            "blocker_type": blocker,
            "description": "Bloqueio preservado: snapshot leve de metadado sensorial nao basta como evento observado; faltam artefato de evento, geometria, fenomeno, execucao de raster leve com politica e hash correspondente.",
            "blocks_ground_reference_candidate": _bool_text(blocker in {"catalog_metadata_not_event_reference", "event_geometry_still_missing", "phenomenon_gate_not_satisfied", "ground_reference_event_artifacts_missing"}),
            "blocks_17b": "true",
            "blocks_score_v7": "true",
            "review_only": "true",
        }
        for idx, blocker in enumerate(blockers, start=1)
    ]


def _schema(required: list[str], props: dict, title: str) -> dict:
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": title, "type": "object", "required": required, "properties": props}


def build_network_snapshot_schema() -> dict:
    required = list(network_snapshot_registry_rows()[0].keys())
    return _schema(required, {
        "network_snapshot_id": {"type": "string", "pattern": "^S17C17_SNAP_"},
        "capture_status": {"enum": ["captured_lightweight_metadata", "captured_public_page_snapshot", "blocked_network_not_enabled", "blocked_heavy_raster", "blocked_requires_login", "failed_safe"]},
        "review_only": {"const": "true"},
        "trainable": {"const": "false"},
        "ground_truth": {"const": "false"},
    }, "SUSC-17C17 network snapshot schema v1")


def build_light_raster_policy_schema() -> dict:
    required = list(light_raster_policy_rows()[0].keys())
    return _schema(required, {
        "light_raster_policy_id": {"type": "string", "pattern": "^S17C17_RASTERPOLICY_"},
        "review_only": {"const": "true"},
    }, "SUSC-17C17 light raster policy schema v1")


def build_snapshot_manifest_schema() -> dict:
    return _schema(MANIFEST_FIELDS, {
        "snapshot_manifest_id": {"type": "string", "pattern": "^S17C17_MANIFEST_"},
        "raw_heavy": {"const": "false"},
        "review_only": {"const": "true"},
        "trainable": {"const": "false"},
        "ground_truth": {"const": "false"},
    }, "SUSC-17C17 snapshot manifest schema v1")


def build_report() -> str:
    summary = build_summary()
    registry = network_snapshot_registry_rows()
    return "\n".join([
        "# SUSC-17C17 - Snapshot de rede para metadados sensoriais e politica de raster leve",
        "",
        "## Objetivo",
        "O 17C17 executa a primeira captura real controlada de metadados publicos de CHIRPS e Sentinel-2/CDSE, com opt-in explicito de rede, e congela o resultado em snapshots leves reprodutiveis. O build publico normal faz apenas replay offline desses snapshots ja commitados; a captura real fica atras de `SUSC_17C17_ALLOW_NETWORK=1` no modo `capture-snapshot` da CLI.",
        "",
        "## Fontes sensoriais planejadas",
        f"- Fontes planejadas: {summary['sensor_sources_planned_count']}.",
        f"- Snapshots de rede tentados: {summary['network_snapshots_attempted_count']}.",
        f"- Snapshots criados: {summary['network_snapshots_created_count']}.",
        f"- Manifestos de snapshot: {summary['snapshot_manifests_count']}.",
        f"- Metadados CHIRPS capturados: {summary['chirps_metadata_snapshots_count']}.",
        f"- Metadados Sentinel-2/CDSE capturados: {summary['sentinel2_metadata_snapshots_count']}.",
        "",
        "## Registro de captura",
        "\n".join(f"- {row['source_name']}: capture_status={row['capture_status']}, network_enabled={row['network_enabled']}." for row in registry),
        "",
        "## Planejamento de raster leve",
        f"- AOIs de patches candidatos planejadas: {summary['candidate_patch_aois_planned_count']}.",
        f"- Linhas de politica de raster leve: {summary['light_raster_policy_rows_count']}.",
        f"- Linhas de plano de execucao de raster leve: {summary['light_raster_execution_plan_rows_count']}.",
        "",
        "## Gates",
        "Metadado de catalogo sensorial nao vira evento observado. G4 (vinculo espacial) e G5 (separacao de fenomeno) permanecem bloqueados por design; nenhum Ground Reference Candidate e criado a partir deste snapshot.",
        "",
        "## Guardrails",
        "Nenhum raster bruto foi baixado, nenhum tile foi criado, nenhum embedding foi calculado, o score v6 permanece intacto, o score v7 nao foi criado e o 17B continua bloqueado.",
        "",
        "## Proximo marco recomendado",
        summary["recommended_next_milestone"],
    ])


def build_all() -> None:
    _require_inputs()
    write_csv(SOURCE_PLAN, sensor_source_plan_rows())
    write_csv(SNAPSHOT_REGISTRY, network_snapshot_registry_rows())
    write_csv(CHIRPS_METADATA, chirps_snapshot_metadata_rows())
    write_csv(SENTINEL2_METADATA, sentinel2_snapshot_metadata_rows())
    write_csv(MANIFEST, snapshot_manifest_rows(), MANIFEST_FIELDS)
    write_csv(REPLAY_AUDIT, snapshot_replay_audit_rows())
    write_csv(AOI_PLAN, candidate_patch_aoi_plan_rows())
    write_csv(LIGHT_RASTER_POLICY, light_raster_policy_rows())
    write_csv(LIGHT_RASTER_EXECUTION_PLAN, light_raster_execution_plan_rows())
    write_csv(HEAVY_RASTER, blocked_heavy_raster_registry_rows())
    write_csv(GATES, gate_evaluation_matrix_rows())
    write_csv(ANTI_LEAKAGE, anti_leakage_audit_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_json(NETWORK_SNAPSHOT_SCHEMA, build_network_snapshot_schema())
    write_json(LIGHT_RASTER_POLICY_SCHEMA, build_light_raster_policy_schema())
    write_json(SNAPSHOT_MANIFEST_SCHEMA, build_snapshot_manifest_schema())
    write_markdown(REPORT, build_report())


def _required_outputs() -> list[Path]:
    outputs = [
        REPORT, SOURCE_PLAN, SNAPSHOT_REGISTRY, CHIRPS_METADATA, SENTINEL2_METADATA,
        MANIFEST, REPLAY_AUDIT, AOI_PLAN, LIGHT_RASTER_POLICY, LIGHT_RASTER_EXECUTION_PLAN,
        HEAVY_RASTER, GATES, ANTI_LEAKAGE, SUMMARY, BLOCKERS,
        NETWORK_SNAPSHOT_SCHEMA, LIGHT_RASTER_POLICY_SCHEMA, SNAPSHOT_MANIFEST_SCHEMA,
    ]
    outputs.extend(_snapshot_path(source) for source in SOURCES if _load_snapshot(source) is not None)
    return outputs


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
    outputs = _required_outputs()
    before = {path: path.read_bytes() for path in outputs if path.exists()}
    build_all()
    errors = []
    for path, content in before.items():
        if path.exists() and path.read_bytes() != content:
            errors.append(f"offline_build_not_byte_identical:{rel(path)}")
    return errors


def validate() -> int:
    missing = [path for path in _required_outputs() if not path.exists()]
    if missing:
        print("MISSING: " + "; ".join(rel(path) for path in missing), file=sys.stderr)
        return 1
    errors = _validate_byte_identical()
    source_plan = read_csv(SOURCE_PLAN)
    registry = read_csv(SNAPSHOT_REGISTRY)
    chirps = read_csv(CHIRPS_METADATA)
    sentinel = read_csv(SENTINEL2_METADATA)
    manifests = read_csv(MANIFEST)
    replay = read_csv(REPLAY_AUDIT)
    aoi_plan = read_csv(AOI_PLAN)
    policy = read_csv(LIGHT_RASTER_POLICY)
    execution_plan = read_csv(LIGHT_RASTER_EXECUTION_PLAN)
    heavy = read_csv(HEAVY_RASTER)
    gates = read_csv(GATES)
    leakage = read_csv(ANTI_LEAKAGE)
    summary = read_json(SUMMARY)
    net_schema = read_json(NETWORK_SNAPSHOT_SCHEMA)
    policy_schema = read_json(LIGHT_RASTER_POLICY_SCHEMA)
    manifest_schema = read_json(SNAPSHOT_MANIFEST_SCHEMA)

    for row in registry:
        errors.extend(_schema_violations(row, net_schema))
    for row in policy:
        errors.extend(_schema_violations(row, policy_schema))
    for row in manifests:
        errors.extend(_schema_violations(row, manifest_schema))

    for rows, key in [
        (source_plan, "sensor_source_plan_id"), (registry, "network_snapshot_id"),
        (chirps, "chirps_snapshot_id"), (sentinel, "sentinel2_snapshot_id"),
        (manifests, "snapshot_manifest_id"), (replay, "replay_audit_id"),
        (aoi_plan, "candidate_patch_aoi_id"), (policy, "light_raster_policy_id"),
        (execution_plan, "light_raster_execution_plan_id"), (heavy, "blocked_heavy_raster_id"),
        (gates, "gate_eval_id"), (leakage, "anti_leakage_audit_id"),
    ]:
        ids = [row[key] for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    if len(source_plan) != len(SOURCES) or len(registry) != len(SOURCES) or len(gates) != len(SOURCES) or len(leakage) != len(SOURCES):
        errors.append("unexpected_source_row_count")
    if len(chirps) != 2 or len(sentinel) != 2:
        errors.append("unexpected_sensor_metadata_row_count")
    patches = read_csv(PATCH_INPUTS[0])
    if len(aoi_plan) != len(patches):
        errors.append("unexpected_aoi_row_count")
    if len(heavy) != len(patches) * 2:
        errors.append("unexpected_heavy_raster_row_count")
    if len(execution_plan) != len(patches) * len(CANDIDATE_PATCH_ACTIONS):
        errors.append("unexpected_execution_plan_row_count")

    if any(row["capture_attempted"] != "false" for row in registry if row["network_enabled"] == "false" and row["capture_status"] == "blocked_network_not_enabled"):
        errors.append("network_attempted_without_opt_in")
    if any(row["metadata_available"] != "false" for row in chirps + sentinel if row["network_snapshot_id"] not in {row2["network_snapshot_id"] for row2 in registry if row2["snapshot_created"] == "true"}):
        errors.append("metadata_without_snapshot")
    if any(row["raster_downloaded"] != "false" for row in chirps + sentinel):
        errors.append("raster_downloaded")
    if any(row.get("tile_created", "false") != "false" or row.get("embedding_created", "false") != "false" for row in sentinel):
        errors.append("tile_or_embedding_created")
    if any(row["can_execute_now"] != "false" for row in execution_plan):
        errors.append("light_raster_executed_now")
    if any(row["commit_allowed"] == "true" and float(row["max_size_bytes"]) > 5_242_880 for row in policy):
        errors.append("light_raster_policy_limit_too_large")
    if any(row["G4_vinculo_espacial"] == "true" or row["G5_separacao_fenomeno"] == "true" for row in gates):
        errors.append("catalog_metadata_promoted_to_event")
    if any(row["all_gates_passed"] == "true" for row in gates):
        errors.append("all_gates_passed_without_event_artifact")
    if any(row["catalog_metadata_misused_as_event"] != "false" or row["passes_anti_leakage"] != "true" for row in leakage):
        errors.append("anti_leakage_failed")
    for manifest_row in manifests:
        path = ROOT / manifest_row["snapshot_path"]
        if not path.exists() or sha256_file(path) != manifest_row["sha256"]:
            errors.append(f"manifest_hash_mismatch:{manifest_row['snapshot_manifest_id']}")
        if not is_within_repo_outputs_public(path):
            errors.append(f"manifest_outside_outputs_public:{manifest_row['snapshot_manifest_id']}")
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
        "17C17 -> "
        f"sources={summary['sensor_sources_planned_count']} "
        f"attempted={summary['network_snapshots_attempted_count']} "
        f"created={summary['network_snapshots_created_count']} "
        f"manifests={summary['snapshot_manifests_count']} "
        f"aois={summary['candidate_patch_aois_planned_count']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
    return 0


def is_within_repo_outputs_public(path: Path) -> bool:
    try:
        path.resolve().relative_to(OUT.resolve())
        return True
    except ValueError:
        return False


def plan_text() -> str:
    lines = ["sensor_source_plan:"]
    for row in sensor_source_plan_rows():
        lines.append(f"  {row['sensor_source_plan_id']} {row['source_name']} network_required={row['network_required']} safe_to_probe={row['safe_to_probe']}")
    lines.append("candidate_patch_aoi_plan:")
    for row in candidate_patch_aoi_plan_rows():
        lines.append(f"  {row['candidate_patch_aoi_id']} {row['candidate_patch_id']} area_m2={row['aoi_area_m2']}")
    return "\n".join(lines)


def plan_light_raster_text() -> str:
    lines = ["light_raster_policy:"]
    for row in light_raster_policy_rows():
        lines.append(f"  {row['light_raster_policy_id']} {row['sensor_or_dataset']} max_size_bytes={row['max_size_bytes']} commit_allowed={row['commit_allowed']}")
    lines.append("light_raster_execution_plan:")
    for row in light_raster_execution_plan_rows():
        lines.append(f"  {row['light_raster_execution_plan_id']} {row['candidate_patch_id']} {row['planned_action']} can_execute_now={row['can_execute_now']}")
    return "\n".join(lines)


def replay_snapshot_text() -> str:
    build_all()
    summary = build_summary()
    return (
        "replay-snapshot: outputs regenerados offline a partir dos snapshots commitados. "
        f"created={summary['network_snapshots_created_count']} manifests={summary['snapshot_manifests_count']}"
    )


def status_text() -> str:
    summary = build_summary()
    return (
        f"status 17C17: attempted={summary['network_snapshots_attempted_count']} "
        f"created={summary['network_snapshots_created_count']} "
        f"manifests={summary['snapshot_manifests_count']} "
        f"aois={summary['candidate_patch_aois_planned_count']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
