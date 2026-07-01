"""SUSC-17C18 estatistica zonal CHIRPS e consulta STAC/OData Sentinel-2 por AOI.

O build publico e offline e deterministico: ele apenas faz replay dos snapshots
de cena Sentinel-2 ja capturados e commitados em outputs_public e reafirma que
CHIRPS permanece bloqueado como raster pesado publico. A consulta real de rede
fica atras de SUSC_17C18_ALLOW_NETWORK=1 nos modos query-* da CLI; o download
leve de artefato adicional exige tambem SUSC_17C18_ALLOW_LIGHTWEIGHT_DOWNLOAD=1
e nunca e acionado nesta sprint. Nenhum produto Sentinel-2, tile, embedding,
feature real de chuva, Ground Reference Candidate, label, ground truth, score v7,
patch oficial ou patch-link oficial e criado.

Observacao metodologica: a raiz STAC publica do CDSE
(https://catalogue.dataspace.copernicus.eu/stac) nao expoe a colecao Sentinel-2,
entao a consulta espaco-temporal de cenas usa o catalogo OData oficial do CDSE
(https://catalogue.dataspace.copernicus.eu/odata/v1/Products) filtrando por AOI e
janela temporal. As cenas retornadas sao inventario de metadados leves, nunca
produto baixado.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.parse
from datetime import datetime, timedelta, timezone
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
LOCAL_STATE = ROOT / "local_runs" / "suscetibilidade" / "17c18_chirps_stac_aoi"
SCENE_SNAPSHOT_DIR = OUT / "susc_17c18_sentinel2_scenes"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_PATCHES = DAT / "susc_patches_official_v1.csv"
OFFICIAL_PATCH_LINKS = DAT / "susc_patch_links_official_v1.csv"

PATCH_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"
PATCH_LINKS = OUT / "susc_17c6_candidate_patch_links.csv"
REFERENCE_CANDIDATES = OUT / "susc_17c4_extracted_reference_candidates.csv"

C17_INPUTS = [
    OUT / "susc_17c17_sensor_source_plan.csv",
    OUT / "susc_17c17_network_snapshot_registry.csv",
    OUT / "susc_17c17_chirps_snapshot_metadata.csv",
    OUT / "susc_17c17_sentinel2_cdse_snapshot_metadata.csv",
    OUT / "susc_17c17_snapshot_manifest.csv",
    OUT / "susc_17c17_candidate_patch_aoi_plan.csv",
    OUT / "susc_17c17_light_raster_policy.csv",
    OUT / "susc_17c17_light_raster_execution_plan.csv",
    OUT / "susc_17c17_readiness_summary.json",
]
PATCH_INPUTS = [PATCH_GRID, OUT / "susc_17c6_candidate_patch_grid.geojson", PATCH_LINKS]
CONTRACT_INPUTS = [
    OUT / "susc_17c6_multimodal_feature_contract.csv",
    OUT / "susc_17c7_candidate_patch_feature_inventory.csv",
    OUT / "susc_17c7_embedding_input_readiness.csv",
    OUT / "susc_17c9_embedding_tile_requirement_plan.csv",
]
REQUIRED_INPUTS = [SCORE_V6, REFERENCE_CANDIDATES, *C17_INPUTS, *PATCH_INPUTS, *CONTRACT_INPUTS]

REPORT = OUT / "SUSC_17C18_ESTATISTICA_ZONAL_CHIRPS_E_CONSULTA_STAC_SENTINEL2_REPORT.md"
EXECUTION_PLAN = OUT / "susc_17c18_sensor_aoi_execution_plan.csv"
CHIRPS_SOURCE = OUT / "susc_17c18_chirps_source_resolution.csv"
CHIRPS_MANIFEST = OUT / "susc_17c18_chirps_lightweight_artifact_manifest.csv"
CHIRPS_ZONAL = OUT / "susc_17c18_chirps_zonal_stats.csv"
RAINFALL_FEATURES = OUT / "susc_17c18_rainfall_trigger_features.csv"
STAC_QUERY_PLAN = OUT / "susc_17c18_sentinel2_stac_query_plan.csv"
STAC_QUERY_RESULTS = OUT / "susc_17c18_sentinel2_stac_query_results.csv"
SCENE_REGISTRY = OUT / "susc_17c18_sentinel2_candidate_scene_registry.csv"
EMBEDDING_BLOCKERS = OUT / "susc_17c18_embedding_tile_blockers.csv"
FEATURE_PROVENANCE = OUT / "susc_17c18_sensor_feature_provenance.csv"
NO_LEAKAGE = OUT / "susc_17c18_no_leakage_audit.csv"
GATES = OUT / "susc_17c18_gate_evaluation_matrix.csv"
SUMMARY = OUT / "susc_17c18_readiness_summary.json"
BLOCKERS = OUT / "susc_17c18_promotion_blockers.csv"

CHIRPS_ZONAL_SCHEMA = SCHEMAS / "susc_17c18_chirps_zonal_schema_v1.json"
SENTINEL2_SCENE_SCHEMA = SCHEMAS / "susc_17c18_sentinel2_scene_schema_v1.json"
FEATURE_PROVENANCE_SCHEMA = SCHEMAS / "susc_17c18_sensor_feature_provenance_schema_v1.json"

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}

CHIRPS_WINDOWS = [("CHIRPS_3d", 3), ("CHIRPS_7d", 7), ("CHIRPS_30d", 30)]
CHIRPS_SOURCE_NAME = "CHIRPS_OFFICIAL_DATA_PORTAL"
CHIRPS_SOURCE_REFERENCE = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/"

SENTINEL2_ODATA_ENDPOINT = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
SENTINEL2_COLLECTION = "SENTINEL-2"
STAC_QUERY_PRE_DAYS = 30
STAC_QUERY_POST_DAYS = 15
STAC_TOP = 40
USER_AGENT = "REV-P-SUSC-17C18-review-only"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _network_enabled() -> bool:
    return os.environ.get("SUSC_17C18_ALLOW_NETWORK") == "1"


def _lightweight_download_enabled() -> bool:
    return os.environ.get("SUSC_17C18_ALLOW_LIGHTWEIGHT_DOWNLOAD") == "1"


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
    return read_csv(PATCH_GRID)


def _event_period(event_id: str) -> tuple[str, str] | None:
    for row in read_csv(REFERENCE_CANDIDATES):
        if row.get("event_id") == event_id:
            period = row.get("event_date_or_period", "")
            if "/" in period:
                start, end = period.split("/", 1)
                return start.strip(), end.strip()
    return None


def _patch_context() -> list[dict]:
    """AOI + evento + janela temporal por patch candidato (determinado offline)."""
    contexts = []
    for patch in _candidate_patches():
        event_id = patch["source_event_id"]
        period = _event_period(event_id)
        xmin, ymin, xmax, ymax = (float(patch[k]) for k in ["xmin", "ymin", "xmax", "ymax"])
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        if period is None:
            contexts.append({
                "candidate_patch_id": patch["candidate_patch_id"],
                "event_id": event_id,
                "bbox": f"{xmin},{ymin},{xmax},{ymax}",
                "crs": patch["crs"],
                "centroid": (cx, cy),
                "event_start": "",
                "event_end": "",
                "event_period": "",
                "blocking_reason": "blocked_missing_event_date",
            })
            continue
        contexts.append({
            "candidate_patch_id": patch["candidate_patch_id"],
            "event_id": event_id,
            "bbox": f"{xmin},{ymin},{xmax},{ymax}",
            "crs": patch["crs"],
            "centroid": (cx, cy),
            "event_start": period[0],
            "event_end": period[1],
            "event_period": f"{period[0]}/{period[1]}",
            "blocking_reason": "",
        })
    return contexts


def _chirps_window_dates(event_start: str, days: int) -> tuple[str, str]:
    end = datetime.strptime(event_start, "%Y-%m-%d").date()
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


def _stac_datetime_window(ctx: dict) -> str:
    start = datetime.strptime(ctx["event_start"], "%Y-%m-%d").date() - timedelta(days=STAC_QUERY_PRE_DAYS)
    end = datetime.strptime(ctx["event_end"], "%Y-%m-%d").date() + timedelta(days=STAC_QUERY_POST_DAYS)
    return f"{start.isoformat()}T00:00:00.000Z/{end.isoformat()}T23:59:59.999Z"


def _scene_snapshot_path(patch_id: str) -> Path:
    return SCENE_SNAPSHOT_DIR / f"{patch_id.lower()}.json"


def _load_scene_snapshot(patch_id: str) -> dict | None:
    path = _scene_snapshot_path(patch_id)
    if not path.exists():
        return None
    return read_json(path)


# ---------------------------------------------------------------------------
# Parte 1 - Plano de execucao sensorial por AOI
# ---------------------------------------------------------------------------

def sensor_aoi_execution_plan_rows() -> list[dict]:
    rows = []
    for idx, ctx in enumerate(_patch_context(), start=1):
        has_date = ctx["event_period"] != ""
        chirps_windows = ";".join(name for name, _ in CHIRPS_WINDOWS)
        rows.append({
            "sensor_aoi_plan_id": f"S17C18_AOIEXEC_{idx:04d}",
            "candidate_patch_id": ctx["candidate_patch_id"],
            "event_id": ctx["event_id"],
            "bbox": ctx["bbox"],
            "crs": ctx["crs"],
            "event_date_or_period": ctx["event_period"] if has_date else "not_available",
            "planned_chirps_windows": chirps_windows,
            "planned_sentinel2_window": _stac_datetime_window(ctx) if has_date else "not_available",
            "can_query_chirps": "true",
            "can_query_sentinel2_stac": _bool_text(has_date),
            "can_compute_chirps_zonal_now": "false",
            "can_create_sentinel2_tile_now": "false",
            "can_create_embedding_tile_now": "false",
            "blocking_reason": (
                "CHIRPS publico disponivel apenas como raster global pesado ou colecao GEE com runtime; "
                "estatistica zonal leve nao executavel nesta sprint. Sentinel-2 consultavel por AOI/data como metadado."
                if has_date
                else "blocked_missing_event_date: sem data/periodo verificavel para o evento do patch candidato"
            ),
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 2 - Resolucao de fonte CHIRPS
# ---------------------------------------------------------------------------

def chirps_source_resolution_rows() -> list[dict]:
    manifest_id = "S17C17_MANIFEST_0001"
    return [{
        "chirps_source_resolution_id": "S17C18_CHIRPSSRC_0001",
        "source_name": CHIRPS_SOURCE_NAME,
        "source_reference": CHIRPS_SOURCE_REFERENCE,
        "snapshot_manifest_id": manifest_id,
        "endpoint_or_artifact_found": "true",
        "endpoint_type": "public_heavy_raster",
        "supports_lightweight_access": "false",
        "requires_heavy_raster": "true",
        "requires_credentials": "false",
        "requires_network": "true",
        "can_compute_zonal_stats": "false",
        "blocking_reason": (
            "CHIRPS publico oficial serve apenas rasters globais diarios pesados (.tif) no diretorio de produtos; "
            "acesso leve por AOI exigiria runtime GEE ou download de raster global. Sem estatistica zonal nesta sprint."
        ),
        "review_only": "true",
    }, {
        "chirps_source_resolution_id": "S17C18_CHIRPSSRC_0002",
        "source_name": "CHIRPS_GEE_COLLECTION",
        "source_reference": "UCSB-CHG/CHIRPS/DAILY",
        "snapshot_manifest_id": "not_available",
        "endpoint_or_artifact_found": "false",
        "endpoint_type": "gee_collection_requires_runtime",
        "supports_lightweight_access": "false",
        "requires_heavy_raster": "false",
        "requires_credentials": "true",
        "requires_network": "true",
        "can_compute_zonal_stats": "false",
        "blocking_reason": "colecao CHIRPS no GEE exige runtime autenticado e nao esta registrada nesta cadeia SUSC-17C",
        "review_only": "true",
    }]


# ---------------------------------------------------------------------------
# Parte 3 - Manifesto CHIRPS leve (vazio: nenhum artefato leve obtido)
# ---------------------------------------------------------------------------

CHIRPS_MANIFEST_FIELDS = [
    "chirps_artifact_manifest_id",
    "source_name",
    "source_reference",
    "artifact_local_path",
    "sha256",
    "size_bytes",
    "format",
    "temporal_window",
    "stored_in_outputs_public",
    "stored_in_local_runs",
    "raw_heavy",
    "review_only",
    "trainable",
    "ground_truth",
]


def chirps_lightweight_artifact_manifest_rows() -> list[dict]:
    return []


# ---------------------------------------------------------------------------
# Parte 4 - Estatistica zonal CHIRPS (sem dado real -> bloqueado)
# ---------------------------------------------------------------------------

def chirps_zonal_stats_rows() -> list[dict]:
    rows = []
    idx = 0
    for ctx in _patch_context():
        has_date = ctx["event_period"] != ""
        for window_name, days in CHIRPS_WINDOWS:
            idx += 1
            if has_date:
                window_start, window_end = _chirps_window_dates(ctx["event_start"], days)
                blocking = "sem artefato CHIRPS leve com hash; estatistica zonal nao calculada (fonte CHIRPS resolvida como raster pesado)"
            else:
                window_start = window_end = "not_available"
                blocking = "blocked_missing_event_date"
            rows.append({
                "chirps_zonal_stat_id": f"S17C18_CHIRPSZONAL_{idx:04d}",
                "candidate_patch_id": ctx["candidate_patch_id"],
                "event_id": ctx["event_id"],
                "temporal_window": window_name,
                "window_start": window_start,
                "window_end": window_end,
                "stat_name": "precipitation_accumulation",
                "stat_value": "",
                "stat_unit": "mm",
                "source_artifact_manifest_id": "not_available",
                "aggregation_method": "zonal_sum_over_aoi_pending_source",
                "feature_value_real": "false",
                "usable_for_science_now": "false",
                "blocking_reason": blocking,
                **GOV,
            })
    return rows


# ---------------------------------------------------------------------------
# Parte 5 - Features de chuva/gatilho (sem dado real -> bloqueado)
# ---------------------------------------------------------------------------

def rainfall_trigger_feature_rows() -> list[dict]:
    rows = []
    idx = 0
    for ctx in _patch_context():
        has_date = ctx["event_period"] != ""
        for window_name, _ in CHIRPS_WINDOWS:
            idx += 1
            rows.append({
                "rainfall_trigger_feature_id": f"S17C18_RAINFEAT_{idx:04d}",
                "candidate_patch_id": ctx["candidate_patch_id"],
                "event_id": ctx["event_id"],
                "feature_name": window_name,
                "feature_value": "",
                "feature_unit": "mm",
                "temporal_window": window_name,
                "source_dataset": "CHIRPS",
                "source_manifest_id": "not_available",
                "feature_value_real": "false",
                "pre_event_only": "true",
                "temporal_leakage_risk": "false",
                "usable_for_science_now": "false",
                "blocking_reason": (
                    "sem artefato CHIRPS real com hash; feature de chuva antecedente nao calculada"
                    if has_date
                    else "blocked_missing_event_date"
                ),
                **GOV,
            })
    return rows


# ---------------------------------------------------------------------------
# Parte 6 - Plano de consulta STAC/OData Sentinel-2
# ---------------------------------------------------------------------------

def sentinel2_stac_query_plan_rows() -> list[dict]:
    rows = []
    for idx, ctx in enumerate(_patch_context(), start=1):
        has_date = ctx["event_period"] != ""
        rows.append({
            "stac_query_plan_id": f"S17C18_STACPLAN_{idx:04d}",
            "candidate_patch_id": ctx["candidate_patch_id"],
            "event_id": ctx["event_id"],
            "bbox": ctx["bbox"],
            "datetime_window": _stac_datetime_window(ctx) if has_date else "not_available",
            "collection_or_product": SENTINEL2_COLLECTION,
            "cloud_filter": "no_hard_filter_cloud_cover_recorded_per_scene",
            "query_endpoint": SENTINEL2_ODATA_ENDPOINT,
            "can_query_now": _bool_text(has_date),
            "requires_network": "true",
            "requires_credentials": "false",
            "download_product": "false",
            "create_tile": "false",
            "blocking_reason": (
                "consulta de metadado de cena por AOI/data permitida com opt-in de rede; nenhum produto baixado"
                if has_date
                else "blocked_missing_event_date"
            ),
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 7 - Resultados de consulta STAC/OData
# ---------------------------------------------------------------------------

def sentinel2_stac_query_results_rows() -> list[dict]:
    rows = []
    for idx, ctx in enumerate(_patch_context(), start=1):
        snapshot = _load_scene_snapshot(ctx["candidate_patch_id"])
        if snapshot is None:
            rows.append({
                "stac_query_result_id": f"S17C18_STACRESULT_{idx:04d}",
                "candidate_patch_id": ctx["candidate_patch_id"],
                "event_id": ctx["event_id"],
                "query_endpoint": SENTINEL2_ODATA_ENDPOINT,
                "query_attempted": "false",
                "query_success": "false",
                "features_returned_count": "0",
                "items_metadata_stored": "false",
                "raw_response_manifest_id": "not_available",
                "blocking_reason": "rede desativada no build; consulta real exige SUSC_17C18_ALLOW_NETWORK=1",
                **GOV,
            })
            continue
        rows.append({
            "stac_query_result_id": f"S17C18_STACRESULT_{idx:04d}",
            "candidate_patch_id": ctx["candidate_patch_id"],
            "event_id": ctx["event_id"],
            "query_endpoint": snapshot["query_endpoint"],
            "query_attempted": "true",
            "query_success": _bool_text(bool(snapshot.get("query_success"))),
            "features_returned_count": str(len(snapshot.get("scenes", []))),
            "items_metadata_stored": "true",
            "raw_response_manifest_id": f"S17C18_STACSNAP_{idx:04d}",
            "blocking_reason": "consulta de metadado bem sucedida; apenas inventario de cenas leve foi armazenado, nenhum produto baixado",
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 8 - Registro de cenas candidatas Sentinel-2
# ---------------------------------------------------------------------------

def _classify_scene(scene_datetime: str, ctx: dict) -> tuple[bool, bool]:
    scene_date = scene_datetime[:10]
    is_pre = scene_date < ctx["event_start"]
    is_post = scene_date > ctx["event_end"]
    return is_pre, is_post


def sentinel2_candidate_scene_registry_rows() -> list[dict]:
    rows = []
    idx = 0
    for ctx in _patch_context():
        snapshot = _load_scene_snapshot(ctx["candidate_patch_id"])
        if snapshot is None:
            continue
        for scene in snapshot.get("scenes", []):
            idx += 1
            is_pre, is_post = _classify_scene(scene["datetime"], ctx)
            rows.append({
                "candidate_scene_id": f"S17C18_SCENE_{idx:04d}",
                "candidate_patch_id": ctx["candidate_patch_id"],
                "event_id": ctx["event_id"],
                "scene_or_item_id": scene["scene_or_item_id"],
                "collection": scene["collection"],
                "datetime": scene["datetime"],
                "platform": scene["platform"],
                "product_type": scene["product_type"],
                "cloud_cover": scene["cloud_cover"],
                "bbox_intersects_patch": "true",
                "is_pre_event": _bool_text(is_pre),
                "is_post_event": _bool_text(is_post),
                "downloaded": "false",
                "tile_created": "false",
                "usable_for_future_tile": _bool_text(is_pre),
                "usable_for_pre_event_feature": _bool_text(is_pre),
                "blocking_reason": (
                    "cena pre-evento candidata para tile futuro; nenhum produto baixado, nenhum tile criado"
                    if is_pre
                    else "cena durante/pos-evento nao pode virar feature pre-evento; registrada apenas como contexto"
                ),
                **GOV,
            })
    return rows


# ---------------------------------------------------------------------------
# Parte 9 - Bloqueadores de tile embedding
# ---------------------------------------------------------------------------

def embedding_tile_blocker_rows() -> list[dict]:
    rows = []
    for idx, ctx in enumerate(_patch_context(), start=1):
        rows.append({
            "embedding_tile_blocker_id": f"S17C18_EMBBLOCK_{idx:04d}",
            "candidate_patch_id": ctx["candidate_patch_id"],
            "event_id": ctx["event_id"],
            "required_input": "real_sentinel2_pre_event_tile",
            "current_status": "candidate_scenes_inventoried_but_no_product_or_tile",
            "blocking_reason": "sem produto Sentinel-2 baixado nem tile criado; embedding DINO/SatMAE permanece bloqueado",
            "can_unlock_after_sentinel2_tile": "true",
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 10 - Proveniencia de feature sensorial
# ---------------------------------------------------------------------------

def sensor_feature_provenance_rows() -> list[dict]:
    rows = []
    idx = 0
    for ctx in _patch_context():
        snapshot = _load_scene_snapshot(ctx["candidate_patch_id"])
        if snapshot is None:
            continue
        idx += 1
        path = _scene_snapshot_path(ctx["candidate_patch_id"])
        rows.append({
            "sensor_feature_provenance_id": f"S17C18_PROV_{idx:04d}",
            "candidate_patch_id": ctx["candidate_patch_id"],
            "feature_name": "sentinel2_candidate_scene_inventory",
            "source_dataset": SENTINEL2_COLLECTION,
            "source_reference": snapshot["query_endpoint"],
            "source_manifest_id": f"S17C18_STACSNAP_{idx:04d}",
            "derived_output_path": rel(path),
            "sha256_or_manifest_ref": sha256_file(path),
            "processing_method": "odata_aoi_datetime_query_lightweight_scene_metadata_only",
            "reproducibility_status": "reproducible_offline_from_committed_scene_snapshot",
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 11 - Auditoria anti-leakage
# ---------------------------------------------------------------------------

def no_leakage_audit_rows() -> list[dict]:
    rows = []
    idx = 0
    # Uma linha por cena candidata + uma por feature de chuva.
    for scene in sentinel2_candidate_scene_registry_rows():
        idx += 1
        post_as_pre = scene["is_post_event"] == "true" and scene["usable_for_pre_event_feature"] == "true"
        rows.append({
            "no_leakage_audit_id": f"S17C18_LEAK_{idx:04d}",
            "candidate_patch_id": scene["candidate_patch_id"],
            "event_id": scene["event_id"],
            "feature_or_scene_id": scene["candidate_scene_id"],
            "uses_pre_event_data_only": _bool_text(scene["is_pre_event"] == "true"),
            "uses_post_event_as_pre_event_feature": _bool_text(post_as_pre),
            "uses_catalog_metadata_as_event": "false",
            "uses_risk_area_as_event": "false",
            "uses_synthetic_as_real": "false",
            "passes_no_leakage": _bool_text(not post_as_pre),
            "blocking_reason": "cena de metadado nao usada como evento; classificacao pre/pos-evento explicita e sem vazamento temporal",
            "review_only": "true",
        })
    for feature in rainfall_trigger_feature_rows():
        idx += 1
        rows.append({
            "no_leakage_audit_id": f"S17C18_LEAK_{idx:04d}",
            "candidate_patch_id": feature["candidate_patch_id"],
            "event_id": feature["event_id"],
            "feature_or_scene_id": feature["rainfall_trigger_feature_id"],
            "uses_pre_event_data_only": "true",
            "uses_post_event_as_pre_event_feature": "false",
            "uses_catalog_metadata_as_event": "false",
            "uses_risk_area_as_event": "false",
            "uses_synthetic_as_real": "false",
            "passes_no_leakage": "true",
            "blocking_reason": "feature CHIRPS planejada apenas para janela antecedente; nenhum valor real calculado nesta sprint",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 12 - Gates
# ---------------------------------------------------------------------------

def gate_evaluation_matrix_rows() -> list[dict]:
    rows = []
    idx = 0
    for scene in sentinel2_candidate_scene_registry_rows():
        idx += 1
        official = True
        rows.append({
            "gate_eval_id": f"S17C18_GATE_{idx:04d}",
            "object_id": scene["candidate_scene_id"],
            "object_type": "sentinel2_candidate_scene",
            "G1_existencia_documental": "true",
            "G2_confiabilidade_fonte": _bool_text(official),
            "G3_precisao_temporal": "true",
            "G4_vinculo_espacial_evento": "false",
            "G5_separacao_fenomeno": "false",
            "G6_proveniencia_integridade": "true",
            "G7_anti_leakage": _bool_text(scene["usable_for_pre_event_feature"] == "true" or scene["is_post_event"] == "true"),
            "all_gates_passed_for_ground_reference": "false",
            "usable_as_sensor_feature": "true",
            "acceptance_status": "accepted_as_candidate_scene_metadata_only",
            "blocking_reason": "cena Sentinel-2 e metadado orbital candidato, nao evento observado; G4/G5 bloqueados por design",
            **GOV,
        })
    for feature in rainfall_trigger_feature_rows():
        idx += 1
        rows.append({
            "gate_eval_id": f"S17C18_GATE_{idx:04d}",
            "object_id": feature["rainfall_trigger_feature_id"],
            "object_type": "chirps_rainfall_feature",
            "G1_existencia_documental": "true",
            "G2_confiabilidade_fonte": "true",
            "G3_precisao_temporal": "true",
            "G4_vinculo_espacial_evento": "false",
            "G5_separacao_fenomeno": "false",
            "G6_proveniencia_integridade": "false",
            "G7_anti_leakage": "true",
            "all_gates_passed_for_ground_reference": "false",
            "usable_as_sensor_feature": "false",
            "acceptance_status": "blocked_no_real_chirps_value",
            "blocking_reason": "feature CHIRPS sem artefato real com hash; proveniencia de integridade nao satisfeita",
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 13 - Resumo
# ---------------------------------------------------------------------------

def build_summary() -> dict:
    contexts = _patch_context()
    results = sentinel2_stac_query_results_rows()
    scenes = sentinel2_candidate_scene_registry_rows()
    rainfall = rainfall_trigger_feature_rows()
    zonal = chirps_zonal_stats_rows()
    return {
        "candidate_patch_count": len(contexts),
        "chirps_source_resolved": False,
        "chirps_lightweight_artifacts_count": len(chirps_lightweight_artifact_manifest_rows()),
        "chirps_zonal_stats_count": len([row for row in zonal if row["feature_value_real"] == "true"]),
        "rainfall_trigger_real_features_count": len([row for row in rainfall if row["feature_value_real"] == "true"]),
        "stac_queries_attempted_count": len([row for row in results if row["query_attempted"] == "true"]),
        "stac_queries_success_count": len([row for row in results if row["query_success"] == "true"]),
        "sentinel2_candidate_scenes_count": len(scenes),
        "pre_event_sentinel2_candidate_scenes_count": len([row for row in scenes if row["is_pre_event"] == "true"]),
        "post_event_sentinel2_candidate_scenes_count": len([row for row in scenes if row["is_post_event"] == "true"]),
        "sentinel2_products_downloaded_count": 0,
        "sentinel2_tiles_created_count": 0,
        "embedding_tiles_created_count": 0,
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
        "recommended_next_milestone": "SUSC-17C19 Politica de tile leve Sentinel-2 e runtime CHIRPS por AOI para primeira feature sensorial real reproduzivel",
    }


def build_blockers() -> list[dict]:
    blockers = [
        "ground_reference_event_artifacts_still_missing",
        "event_geometry_still_missing",
        "phenomenon_gate_not_satisfied",
        "candidate_patch_policy_missing",
        "sentinel2_product_not_downloaded",
        "sentinel2_tile_not_created",
        "embedding_tile_missing",
        "sar_runtime_missing",
        "qa_not_accepted",
        "17b_blocked_until_event_reference_and_policy",
    ]
    return [
        {
            "blocker_id": f"S17C18_BLOCKER_{idx:04d}",
            "blocker_type": blocker,
            "description": "Bloqueio preservado: inventario de cena Sentinel-2 e plano CHIRPS por AOI nao bastam como evento observado; faltam artefato de evento, geometria, fenomeno, feature CHIRPS real com hash, produto/tile Sentinel-2 e politica de patch candidato.",
            "blocks_ground_reference_candidate": _bool_text(blocker in {"ground_reference_event_artifacts_still_missing", "event_geometry_still_missing", "phenomenon_gate_not_satisfied", "candidate_patch_policy_missing"}),
            "blocks_17b": "true",
            "blocks_score_v7": "true",
            "review_only": "true",
        }
        for idx, blocker in enumerate(blockers, start=1)
    ]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

def _schema(required: list[str], props: dict, title: str) -> dict:
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": title, "type": "object", "required": required, "properties": props}


def build_chirps_zonal_schema() -> dict:
    # stat_value fica vazio quando nao ha dado CHIRPS real; nao pode ser exigido.
    required = [key for key in chirps_zonal_stats_rows()[0].keys() if key != "stat_value"]
    return _schema(required, {
        "chirps_zonal_stat_id": {"type": "string", "pattern": "^S17C18_CHIRPSZONAL_"},
        "feature_value_real": {"enum": ["true", "false"]},
        "review_only": {"const": "true"},
        "trainable": {"const": "false"},
        "ground_truth": {"const": "false"},
    }, "SUSC-17C18 CHIRPS zonal schema v1")


def build_sentinel2_scene_schema() -> dict:
    required = [
        "candidate_scene_id", "candidate_patch_id", "event_id", "scene_or_item_id",
        "collection", "datetime", "platform", "product_type", "cloud_cover",
        "bbox_intersects_patch", "is_pre_event", "is_post_event", "downloaded",
        "tile_created", "usable_for_future_tile", "usable_for_pre_event_feature",
        "blocking_reason", "review_only", "trainable", "ground_truth",
    ]
    return _schema(required, {
        "candidate_scene_id": {"type": "string", "pattern": "^S17C18_SCENE_"},
        "downloaded": {"const": "false"},
        "tile_created": {"const": "false"},
        "review_only": {"const": "true"},
        "trainable": {"const": "false"},
        "ground_truth": {"const": "false"},
    }, "SUSC-17C18 Sentinel-2 candidate scene schema v1")


def build_feature_provenance_schema() -> dict:
    required = [
        "sensor_feature_provenance_id", "candidate_patch_id", "feature_name",
        "source_dataset", "source_reference", "source_manifest_id",
        "derived_output_path", "sha256_or_manifest_ref", "processing_method",
        "reproducibility_status", "review_only", "trainable", "ground_truth",
    ]
    return _schema(required, {
        "sensor_feature_provenance_id": {"type": "string", "pattern": "^S17C18_PROV_"},
        "review_only": {"const": "true"},
        "trainable": {"const": "false"},
        "ground_truth": {"const": "false"},
    }, "SUSC-17C18 sensor feature provenance schema v1")


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------

def build_report() -> str:
    summary = build_summary()
    return "\n".join([
        "# SUSC-17C18 - Estatistica zonal CHIRPS e consulta STAC Sentinel-2 por AOI",
        "",
        "## Objetivo",
        "O 17C18 executa a primeira tentativa de extracao sensorial leve por AOI candidata para os 5 patches de Recife (evento REC_2022_05_24_30, periodo 2022-05-24/2022-05-30). CHIRPS e consultado como fonte de chuva antecedente e Sentinel-2/CDSE e consultado por AOI e janela temporal para inventario de cenas candidatas. O build publico e offline e deterministico; a consulta real de rede exige `SUSC_17C18_ALLOW_NETWORK=1`.",
        "",
        "## Separacao conceitual",
        "CHIRPS e Sentinel-2 sao camadas sensoriais/contextuais. Podem gerar feature de chuva antecedente (se houver dado verificavel), inventario de cenas e metadado de fonte. Nao geram evento observado, Ground Reference Candidate, ground truth, label, validacao 17B nem score v7.",
        "",
        "## CHIRPS",
        f"- Fonte CHIRPS resolvida para acesso leve por AOI: {summary['chirps_source_resolved']} (publico disponivel apenas como raster global pesado ou colecao GEE com runtime).",
        f"- Artefatos CHIRPS leves obtidos: {summary['chirps_lightweight_artifacts_count']}.",
        f"- Estatisticas zonais CHIRPS reais calculadas: {summary['chirps_zonal_stats_count']}.",
        f"- Features reais de chuva antecedente criadas: {summary['rainfall_trigger_real_features_count']}.",
        "",
        "## Sentinel-2 / CDSE",
        "A raiz STAC publica do CDSE nao expoe a colecao Sentinel-2; a consulta espaco-temporal de cenas usa o catalogo OData oficial do CDSE filtrando por AOI e janela temporal. As cenas retornadas sao metadado leve, nunca produto baixado.",
        f"- Consultas por AOI tentadas: {summary['stac_queries_attempted_count']}.",
        f"- Consultas bem sucedidas: {summary['stac_queries_success_count']}.",
        f"- Cenas Sentinel-2 candidatas registradas: {summary['sentinel2_candidate_scenes_count']}.",
        f"- Cenas pre-evento: {summary['pre_event_sentinel2_candidate_scenes_count']}.",
        f"- Cenas pos-evento: {summary['post_event_sentinel2_candidate_scenes_count']}.",
        f"- Produtos Sentinel-2 baixados: {summary['sentinel2_products_downloaded_count']}.",
        f"- Tiles criados: {summary['sentinel2_tiles_created_count']}.",
        f"- Embeddings criados: {summary['embedding_tiles_created_count']}.",
        "",
        "## Gates",
        "Feature CHIRPS e metadado de cena Sentinel-2 podem passar gates de fonte/tempo/proveniencia como camada sensorial candidata. Nenhum passa G4 (vinculo espacial de evento) ou G5 (separacao de fenomeno) como evento observado; nenhum vira Ground Reference Candidate.",
        "",
        "## Guardrails",
        "Nenhum produto Sentinel-2 foi baixado, nenhum tile foi criado, nenhum embedding foi calculado, nenhuma feature pos-evento virou pre-evento, o score v6 permanece intacto, o score v7 nao foi criado e o 17B continua bloqueado.",
        "",
        "## Proximo marco recomendado",
        summary["recommended_next_milestone"],
    ])


# ---------------------------------------------------------------------------
# Build / validacao
# ---------------------------------------------------------------------------

def build_all() -> None:
    _require_inputs()
    write_csv(EXECUTION_PLAN, sensor_aoi_execution_plan_rows())
    write_csv(CHIRPS_SOURCE, chirps_source_resolution_rows())
    write_csv(CHIRPS_MANIFEST, chirps_lightweight_artifact_manifest_rows(), CHIRPS_MANIFEST_FIELDS)
    write_csv(CHIRPS_ZONAL, chirps_zonal_stats_rows())
    write_csv(RAINFALL_FEATURES, rainfall_trigger_feature_rows())
    write_csv(STAC_QUERY_PLAN, sentinel2_stac_query_plan_rows())
    write_csv(STAC_QUERY_RESULTS, sentinel2_stac_query_results_rows())
    write_csv(SCENE_REGISTRY, sentinel2_candidate_scene_registry_rows())
    write_csv(EMBEDDING_BLOCKERS, embedding_tile_blocker_rows())
    write_csv(FEATURE_PROVENANCE, sensor_feature_provenance_rows())
    write_csv(NO_LEAKAGE, no_leakage_audit_rows())
    write_csv(GATES, gate_evaluation_matrix_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_json(CHIRPS_ZONAL_SCHEMA, build_chirps_zonal_schema())
    write_json(SENTINEL2_SCENE_SCHEMA, build_sentinel2_scene_schema())
    write_json(FEATURE_PROVENANCE_SCHEMA, build_feature_provenance_schema())
    write_markdown(REPORT, build_report())


def _required_outputs() -> list[Path]:
    outputs = [
        REPORT, EXECUTION_PLAN, CHIRPS_SOURCE, CHIRPS_MANIFEST, CHIRPS_ZONAL,
        RAINFALL_FEATURES, STAC_QUERY_PLAN, STAC_QUERY_RESULTS, SCENE_REGISTRY,
        EMBEDDING_BLOCKERS, FEATURE_PROVENANCE, NO_LEAKAGE, GATES, SUMMARY, BLOCKERS,
        CHIRPS_ZONAL_SCHEMA, SENTINEL2_SCENE_SCHEMA, FEATURE_PROVENANCE_SCHEMA,
    ]
    outputs.extend(
        _scene_snapshot_path(ctx["candidate_patch_id"])
        for ctx in _patch_context()
        if _load_scene_snapshot(ctx["candidate_patch_id"]) is not None
    )
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
    execution_plan = read_csv(EXECUTION_PLAN)
    chirps_source = read_csv(CHIRPS_SOURCE)
    chirps_manifest = read_csv(CHIRPS_MANIFEST)
    zonal = read_csv(CHIRPS_ZONAL)
    rainfall = read_csv(RAINFALL_FEATURES)
    query_plan = read_csv(STAC_QUERY_PLAN)
    query_results = read_csv(STAC_QUERY_RESULTS)
    scenes = read_csv(SCENE_REGISTRY)
    emb_blockers = read_csv(EMBEDDING_BLOCKERS)
    provenance = read_csv(FEATURE_PROVENANCE)
    leakage = read_csv(NO_LEAKAGE)
    gates = read_csv(GATES)
    summary = read_json(SUMMARY)
    zonal_schema = read_json(CHIRPS_ZONAL_SCHEMA)
    scene_schema = read_json(SENTINEL2_SCENE_SCHEMA)
    provenance_schema = read_json(FEATURE_PROVENANCE_SCHEMA)

    for row in zonal:
        errors.extend(_schema_violations(row, zonal_schema))
    for row in scenes:
        errors.extend(_schema_violations(row, scene_schema))
    for row in provenance:
        errors.extend(_schema_violations(row, provenance_schema))

    for rows, key in [
        (execution_plan, "sensor_aoi_plan_id"), (chirps_source, "chirps_source_resolution_id"),
        (zonal, "chirps_zonal_stat_id"), (rainfall, "rainfall_trigger_feature_id"),
        (query_plan, "stac_query_plan_id"), (query_results, "stac_query_result_id"),
        (scenes, "candidate_scene_id"), (emb_blockers, "embedding_tile_blocker_id"),
        (provenance, "sensor_feature_provenance_id"), (leakage, "no_leakage_audit_id"),
        (gates, "gate_eval_id"),
    ]:
        ids = [row[key] for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    patch_count = len(_candidate_patches())
    if len(execution_plan) != patch_count or len(query_plan) != patch_count or len(query_results) != patch_count or len(emb_blockers) != patch_count:
        errors.append("unexpected_patch_row_count")
    if len(zonal) != patch_count * len(CHIRPS_WINDOWS) or len(rainfall) != patch_count * len(CHIRPS_WINDOWS):
        errors.append("unexpected_chirps_row_count")

    # Guardrails de raster/tile/embedding/produto.
    if chirps_manifest:
        errors.append("chirps_lightweight_manifest_should_be_empty_without_real_artifact")
    if any(row["feature_value_real"] == "true" and row["source_artifact_manifest_id"] == "not_available" for row in zonal):
        errors.append("chirps_real_feature_without_manifest")
    if any(row["feature_value_real"] == "true" for row in rainfall + zonal):
        errors.append("chirps_real_feature_created_without_source")
    if any(row["stat_value"] != "" for row in zonal):
        errors.append("chirps_stat_value_without_real_source")
    if any(row["downloaded"] != "false" or row["tile_created"] != "false" for row in scenes):
        errors.append("sentinel2_product_or_tile_created")
    if any(row["download_product"] != "false" or row["create_tile"] != "false" for row in query_plan):
        errors.append("stac_plan_allows_download_or_tile")
    if any(row["is_post_event"] == "true" and row["usable_for_pre_event_feature"] == "true" for row in scenes):
        errors.append("post_event_scene_used_as_pre_event_feature")
    if any(row["G4_vinculo_espacial_evento"] == "true" or row["G5_separacao_fenomeno"] == "true" for row in gates):
        errors.append("sensor_object_promoted_to_event")
    if any(row["all_gates_passed_for_ground_reference"] == "true" for row in gates):
        errors.append("ground_reference_accepted_from_sensor")
    if any(row["passes_no_leakage"] != "true" for row in leakage):
        errors.append("no_leakage_audit_failed")
    if any(row["uses_post_event_as_pre_event_feature"] != "false" for row in leakage):
        errors.append("temporal_leakage_detected")

    # Proveniencia real deve ter hash coerente.
    for row in provenance:
        path = ROOT / row["derived_output_path"]
        if not path.exists() or sha256_file(path) != row["sha256_or_manifest_ref"]:
            errors.append(f"provenance_hash_mismatch:{row['sensor_feature_provenance_id']}")

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
    if summary["features_extracted_this_sprint"] != 0 or summary["ground_reference_candidates_created_count"] != 0:
        errors.append("unexpected_feature_or_ground_reference_creation")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "17C18 -> "
        f"patches={summary['candidate_patch_count']} "
        f"stac_success={summary['stac_queries_success_count']} "
        f"scenes={summary['sentinel2_candidate_scenes_count']} "
        f"pre={summary['pre_event_sentinel2_candidate_scenes_count']} "
        f"chirps_real={summary['rainfall_trigger_real_features_count']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
    return 0


# ---------------------------------------------------------------------------
# Rede (opt-in)
# ---------------------------------------------------------------------------

def _odata_query(ctx: dict) -> dict:
    cx, cy = ctx["centroid"]
    point = f"POINT({cx:.6f} {cy:.6f})"
    window = _stac_datetime_window(ctx)
    start_iso, end_iso = window.split("/", 1)
    flt = (
        f"Collection/Name eq '{SENTINEL2_COLLECTION}' "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{point}') "
        f"and ContentDate/Start gt {start_iso} "
        f"and ContentDate/Start lt {end_iso}"
    )
    url = (
        f"{SENTINEL2_ODATA_ENDPOINT}?$filter={urllib.parse.quote(flt)}"
        f"&$expand=Attributes&$top={STAC_TOP}&$orderby=ContentDate/Start"
    )
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=40) as response:
        raw = response.read()
        return {"http_status": str(response.status), "raw": raw, "url": url, "point": point, "window": window}


def _trim_scene(product: dict) -> dict:
    attrs = {a["Name"]: a.get("Value") for a in product.get("Attributes", [])}
    cloud = attrs.get("cloudCover")
    return {
        "scene_or_item_id": product.get("Name", "not_available"),
        "collection": SENTINEL2_COLLECTION,
        "datetime": (product.get("ContentDate") or {}).get("Start", "not_available"),
        "platform": attrs.get("platformShortName", "not_available"),
        "product_type": attrs.get("productType", "not_available"),
        "cloud_cover": f"{float(cloud):.4f}" if isinstance(cloud, (int, float)) else "not_available",
    }


def query_stac_sentinel2_text() -> tuple[int, str]:
    if not _network_enabled():
        return (2, "blocked_network_not_enabled: query-stac-sentinel2 exige SUSC_17C18_ALLOW_NETWORK=1.")
    ensure_dir(SCENE_SNAPSHOT_DIR)
    ensure_dir(LOCAL_STATE / "raw")
    created = 0
    for ctx in _patch_context():
        if ctx["event_period"] == "":
            continue
        try:
            result = _odata_query(ctx)
        except (HTTPError, URLError) as exc:
            (SCENE_SNAPSHOT_DIR / f"{ctx['candidate_patch_id'].lower()}_query_failed.txt").write_text(f"failed_safe: {exc}\n", encoding="utf-8")
            continue
        raw_path = LOCAL_STATE / "raw" / f"{ctx['candidate_patch_id'].lower()}.json"
        raw_path.write_bytes(result["raw"])
        obj = json.loads(result["raw"])
        scenes = [_trim_scene(product) for product in obj.get("value", [])]
        scenes.sort(key=lambda scene: scene["scene_or_item_id"])
        snapshot = {
            "candidate_patch_id": ctx["candidate_patch_id"],
            "event_id": ctx["event_id"],
            "query_endpoint": SENTINEL2_ODATA_ENDPOINT,
            "aoi_point": result["point"],
            "bbox": ctx["bbox"],
            "datetime_window": result["window"],
            "collection_or_product": SENTINEL2_COLLECTION,
            "http_status": result["http_status"],
            "query_success": True,
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "raw_local_path": rel(raw_path),
            "raw_sha256": sha256_bytes(result["raw"]),
            "scenes": scenes,
        }
        write_json(_scene_snapshot_path(ctx["candidate_patch_id"]), snapshot)
        created += 1
    return (0, f"query-stac-sentinel2: {created} inventarios de cena leves capturados e gravados em {rel(SCENE_SNAPSHOT_DIR)}.")


def query_chirps_text() -> tuple[int, str]:
    if not _network_enabled():
        return (2, "blocked_network_not_enabled: query-chirps exige SUSC_17C18_ALLOW_NETWORK=1.")
    ensure_dir(LOCAL_STATE / "raw")
    probe = {"source_reference": CHIRPS_SOURCE_REFERENCE, "endpoint_type": "public_heavy_raster"}
    try:
        req = Request(CHIRPS_SOURCE_REFERENCE, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=30) as response:
            probe["http_status"] = str(response.status)
            probe["content_type"] = response.headers.get("content-type", "not_available")
    except (HTTPError, URLError) as exc:
        probe["http_status"] = "failed_safe"
        probe["error"] = str(exc)
    (LOCAL_STATE / "chirps_source_probe.json").write_text(json.dumps(probe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    download_note = (
        "download leve habilitado, mas CHIRPS publico so oferece raster global pesado; nenhum arquivo leve baixado"
        if _lightweight_download_enabled()
        else "download leve nao habilitado (SUSC_17C18_ALLOW_LIGHTWEIGHT_DOWNLOAD=1 ausente)"
    )
    return (0, f"query-chirps: fonte CHIRPS resolvida como raster global pesado (endpoint_type=public_heavy_raster); {download_note}. Nenhuma estatistica zonal calculada.")


def compute_chirps_zonal_text() -> tuple[int, str]:
    return (
        0,
        "compute-chirps-zonal: nenhum artefato CHIRPS leve/local valido com hash disponivel; "
        "estatistica zonal nao calculada e nenhuma feature real de chuva criada.",
    )


def plan_text() -> str:
    lines = ["sensor_aoi_execution_plan:"]
    for row in sensor_aoi_execution_plan_rows():
        lines.append(
            f"  {row['sensor_aoi_plan_id']} {row['candidate_patch_id']} evento={row['event_date_or_period']} "
            f"chirps={row['planned_chirps_windows']} sentinel2_window={row['planned_sentinel2_window']}"
        )
    lines.append("sentinel2_stac_query_plan:")
    for row in sentinel2_stac_query_plan_rows():
        lines.append(f"  {row['stac_query_plan_id']} {row['candidate_patch_id']} can_query_now={row['can_query_now']} endpoint={row['query_endpoint']}")
    return "\n".join(lines)


def status_text() -> str:
    summary = build_summary()
    return (
        f"status 17C18: stac_success={summary['stac_queries_success_count']} "
        f"scenes={summary['sentinel2_candidate_scenes_count']} "
        f"pre_event={summary['pre_event_sentinel2_candidate_scenes_count']} "
        f"chirps_real_features={summary['rainfall_trigger_real_features_count']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
