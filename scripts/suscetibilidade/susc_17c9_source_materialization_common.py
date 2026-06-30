"""SUSC-17C9 materializacao de artefatos fonte para patches candidatos.

O pacote e review-only. Ele nao extrai features finais, nao baixa raster, nao
executa SAR/DINO/SatMAE e nao promove patches candidatos. O objetivo e tornar
verificavel quais insumos, manifests, adapters e pedidos externos faltam para
uma extracao real futura.
"""

from __future__ import annotations

import csv
import hashlib
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

AUDIT = OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C9.md"

C8_INPUTS = [
    OUT / "susc_17c8_extraction_run_registry.csv",
    OUT / "susc_17c8_physical_static_features.csv",
    OUT / "susc_17c8_urban_territorial_features.csv",
    OUT / "susc_17c8_rainfall_trigger_features.csv",
    OUT / "susc_17c8_candidate_feature_matrix_partial.csv",
    OUT / "susc_17c8_feature_quality_flags.csv",
    OUT / "susc_17c8_feature_provenance_manifest.csv",
    OUT / "susc_17c8_no_leakage_audit.csv",
    OUT / "susc_17c8_readiness_summary.json",
    OUT / "susc_17c8_promotion_blockers.csv",
]
C7_INPUTS = [
    OUT / "susc_17c7_candidate_patch_feature_inventory.csv",
    OUT / "susc_17c7_feature_source_mapping.csv",
    OUT / "susc_17c7_extraction_task_plan.csv",
    OUT / "susc_17c7_feature_missingness_matrix.csv",
    OUT / "susc_17c7_embedding_input_readiness.csv",
    OUT / "susc_17c7_no_leakage_policy.json",
    OUT / "susc_17c7_readiness_summary.json",
]
C6_INPUTS = [
    OUT / "susc_17c6_candidate_patch_grid.csv",
    OUT / "susc_17c6_candidate_patch_grid.geojson",
    OUT / "susc_17c6_candidate_patch_links.csv",
    OUT / "susc_17c6_multimodal_feature_contract.csv",
    OUT / "susc_17c6_multimodal_canary_matrix.csv",
    OUT / "susc_17c6_embedding_contract.json",
]

FEATURE_PROVENANCE = ROOT / "manifests" / "suscetibilidade" / "susc_features_provenance_manifest_v1.csv"
FEATURE_ARTIFACT_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_features_by_patch_v1_artifact_manifest.json"
SENTINEL_BINDING_ASSETS = ROOT / "outputs_public" / "mv2_sentinel_binding" / "mv2_13_asset_inventory_normalized.csv"
STAC_GATE = ROOT / "outputs_public" / "mv2_sentinel_binding" / "mv2_13_stac_gate_matrix.csv"
RASTER_CANARY_PLAN = ROOT / "outputs_public" / "mv2_data_api_raster_harness" / "mv2_data_02_raster_canary_plan.csv"
DINO_INPUT_MANIFEST = ROOT / "manifests" / "dino_inputs" / "revp_v1fu_dino_sentinel_input_manifest" / "dino_sentinel_input_manifest_v1fu.csv"
LOCAL_MULTIMODAL_FEATURES = ROOT / "local_runs" / "multimodal" / "v2bn" / "multimodal_feature_table_core_v2bn.csv"

REPORT = OUT / "SUSC_17C9_MATERIALIZACAO_ARTEFATOS_FONTE_PATCHES_CANDIDATOS_REPORT.md"
SOURCE_INVENTORY = OUT / "susc_17c9_source_artifact_inventory.csv"
PIPELINE_AUDIT = OUT / "susc_17c9_pipeline_reuse_audit.csv"
INPUT_REQUIREMENTS = OUT / "susc_17c9_candidate_patch_input_requirements.csv"
EXTERNAL_REQUESTS = OUT / "susc_17c9_external_artifact_request_plan.csv"
LIGHTWEIGHT_EXPORTS = OUT / "susc_17c9_lightweight_export_plan.csv"
EMBEDDING_TILE_PLAN = OUT / "susc_17c9_embedding_tile_requirement_plan.csv"
UNBLOCK_MATRIX = OUT / "susc_17c9_feature_extraction_unblock_matrix.csv"
NO_FAKE_POLICY = OUT / "susc_17c9_no_fake_feature_policy.json"
SUMMARY = OUT / "susc_17c9_readiness_summary.json"
BLOCKERS = OUT / "susc_17c9_promotion_blockers.csv"

SOURCE_SCHEMA = SCHEMAS / "susc_17c9_source_artifact_inventory_schema_v1.json"
PIPELINE_SCHEMA = SCHEMAS / "susc_17c9_pipeline_reuse_schema_v1.json"

REQUIRED_INPUTS = [FEATURES, SCORE_V6, FEATURE_PROVENANCE, FEATURE_ARTIFACT_MANIFEST, *C8_INPUTS, *C7_INPUTS, *C6_INPUTS]
REQUIRED_OUTPUTS = [
    AUDIT,
    REPORT,
    SOURCE_INVENTORY,
    PIPELINE_AUDIT,
    INPUT_REQUIREMENTS,
    EXTERNAL_REQUESTS,
    LIGHTWEIGHT_EXPORTS,
    EMBEDDING_TILE_PLAN,
    UNBLOCK_MATRIX,
    NO_FAKE_POLICY,
    SUMMARY,
    BLOCKERS,
    SOURCE_SCHEMA,
    PIPELINE_SCHEMA,
]

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}
EVENT_ID = "REC_2022_05_24_30"
EVENT_WINDOW = "2022-05-24_2022-05-30"

LAYERS = [
    "physical_static",
    "urban_territorial",
    "rainfall_trigger",
    "sentinel2_spectral",
    "embedding_representation",
]

FEATURES_BY_LAYER = {
    "physical_static": "HAND;elevation;slope;DEM;drainage;distance_to_water;flow_accumulation;TWI",
    "urban_territorial": "MapBiomas;urban_prop;vegetation_prop;water_prop;NDBI;imperviousness_proxy",
    "rainfall_trigger": "CHIRPS_3d;CHIRPS_7d;CHIRPS_30d;runoff_context_7d",
    "sentinel2_spectral": "B2;B3;B4;B8;B11;B12;NDVI;NDWI;MNDWI;NDBI",
    "embedding_representation": "DINOv2_tile;SatMAE_tile;Scale-MAE_tile",
}


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _run_git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _exists(path: Path) -> bool:
    return path.exists()


def _is_committed(path: Path) -> bool:
    return bool(_run_git(["ls-files", "--", rel(path)]))


def _is_gitignored(path: Path) -> bool:
    result = subprocess.run(["git", "check-ignore", "-q", rel(path)], cwd=ROOT, check=False)
    return result.returncode == 0


def _artifact_row(
    artifact_id: str,
    layer: str,
    feature_name: str,
    kind: str,
    path_or_ref: str,
    fmt: str,
    scope: str,
    covers_candidate: bool,
    covers_official: bool,
    can_use_candidate: bool,
    requires_download: bool,
    requires_credentials: bool,
    requires_manual: bool,
    blocking_reason: str,
    raw_heavy: bool = False,
) -> dict:
    path = ROOT / path_or_ref if not path_or_ref.startswith(("gee://", "stac://", "remote://")) else None
    exists = _exists(path) if path is not None else False
    return {
        "source_artifact_id": artifact_id,
        "feature_layer": layer,
        "feature_name": feature_name,
        "artifact_kind": kind,
        "artifact_path_or_reference": path_or_ref,
        "artifact_exists_local": _bool_text(exists),
        "artifact_is_committed": _bool_text(_is_committed(path) if path is not None and exists else False),
        "artifact_is_gitignored": _bool_text(_is_gitignored(path) if path is not None else False),
        "artifact_is_raw_heavy": _bool_text(raw_heavy),
        "artifact_format": fmt,
        "artifact_scope": scope,
        "covers_candidate_patch_area": _bool_text(covers_candidate),
        "covers_official_patch_area": _bool_text(covers_official),
        "can_be_used_for_candidate_patches": _bool_text(can_use_candidate),
        "requires_external_download": _bool_text(requires_download),
        "requires_runtime_credentials": _bool_text(requires_credentials),
        "requires_manual_preparation": _bool_text(requires_manual),
        "blocking_reason": blocking_reason,
        **GOV,
    }


def _candidate_rows() -> list[dict]:
    return read_csv(OUT / "susc_17c6_candidate_patch_grid.csv")


def _candidate_ids() -> list[str]:
    return [row["candidate_patch_id"] for row in _candidate_rows()]


def _touch_inputs() -> None:
    for path in C8_INPUTS + C7_INPUTS + C6_INPUTS:
        if path.suffix == ".json":
            read_json(path)
        else:
            read_csv(path)


def _require_inputs() -> None:
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))
    _touch_inputs()


def build_source_inventory() -> list[dict]:
    rows = [
        _artifact_row("S17C9_SRC_0001", "all_layers", "candidate_patch_geometry_csv", "local_committed_output", rel(OUT / "susc_17c6_candidate_patch_grid.csv"), "csv", "candidate_grid", True, False, True, False, False, False, "not_blocked_geometry_reference_only"),
        _artifact_row("S17C9_SRC_0002", "all_layers", "candidate_patch_geometry_geojson", "local_committed_output", rel(OUT / "susc_17c6_candidate_patch_grid.geojson"), "geojson", "candidate_grid", True, False, True, False, False, False, "not_blocked_geometry_reference_only"),
        _artifact_row("S17C9_SRC_0003", "all_layers", "official_feature_matrix", "local_committed_output", rel(FEATURES), "csv", "official_patch_matrix", False, True, False, False, False, False, "official_patch_values_cannot_be_reused_for_candidate_patches"),
        _artifact_row("S17C9_SRC_0004", "all_layers", "official_feature_provenance_manifest", "local_committed_output", rel(FEATURE_PROVENANCE), "csv", "official_patch_provenance", False, True, False, False, False, False, "provenance_describes_official_matrix_not_candidate_values"),
        _artifact_row("S17C9_SRC_0005", "physical_static", "DEM_HAND_coverage", "missing_required_artifact", "remote://dem_hand_candidate_grid_required", "raster_or_zonal_stats", "candidate_grid", False, False, False, True, False, True, "missing_candidate_specific_dem_hand_coverage"),
        _artifact_row("S17C9_SRC_0006", "physical_static", "drainage_network_coverage", "missing_required_artifact", "remote://drainage_candidate_grid_required", "vector_or_raster", "candidate_grid", False, False, False, True, False, True, "missing_candidate_specific_drainage_coverage"),
        _artifact_row("S17C9_SRC_0007", "physical_static", "flow_accumulation_twi_products", "pipeline_expected_input", "remote://dem_derived_flow_twi_required", "raster_or_zonal_stats", "candidate_grid", False, False, False, True, False, True, "missing_candidate_specific_dem_derivative_products"),
        _artifact_row("S17C9_SRC_0008", "urban_territorial", "MapBiomas_or_landcover_coverage", "missing_required_artifact", "remote://mapbiomas_candidate_grid_required", "raster_or_zonal_stats", "candidate_grid", False, False, False, True, False, True, "missing_candidate_specific_landcover_coverage"),
        _artifact_row("S17C9_SRC_0009", "rainfall_trigger", "CHIRPS_pre_event_window", "missing_required_artifact", "gee://UCSB-CHG/CHIRPS/DAILY/pre_event_REC_2022_05_24_30", "zonal_stats_csv", "candidate_grid_event_window", False, False, False, True, True, True, "missing_candidate_specific_chirps_pre_event_zonal_stats"),
        _artifact_row("S17C9_SRC_0010", "sentinel2_spectral", "Sentinel2_candidate_tile", "remote_asset_reference", "stac://sentinel-2_candidate_grid_pre_event_tile_required", "tile_metadata_or_small_clip", "candidate_grid", False, False, False, True, True, True, "missing_candidate_specific_sentinel2_tile"),
        _artifact_row("S17C9_SRC_0011", "sentinel2_spectral", "MV2_13_asset_inventory", "manifest_reference", rel(SENTINEL_BINDING_ASSETS), "csv", "other_patch_lineage", False, True, False, False, False, False, "manifest_exists_but_does_not_cover_17c6_candidate_grid"),
        _artifact_row("S17C9_SRC_0012", "sentinel2_spectral", "STAC_gate_reference", "manifest_reference", rel(STAC_GATE), "csv", "other_patch_lineage", False, True, False, False, False, False, "stac_gate_exists_but_not_bound_to_candidate_grid"),
        _artifact_row("S17C9_SRC_0013", "sentinel2_spectral", "raster_canary_plan_reference", "manifest_reference", rel(RASTER_CANARY_PLAN), "csv", "api_raster_canary_other_targets", False, True, False, False, True, False, "canary_plan_exists_but_download_is_blocked_and_targets_are_not_candidate_grid"),
        _artifact_row("S17C9_SRC_0014", "embedding_representation", "DINO_Sentinel_input_manifest", "manifest_reference", rel(DINO_INPUT_MANIFEST), "csv", "other_patch_embedding_inputs", False, True, False, False, False, False, "manifest_exists_for_other_patch_assets_not_candidate_tiles"),
        _artifact_row("S17C9_SRC_0015", "embedding_representation", "candidate_real_embedding_tile", "missing_required_artifact", "remote://candidate_grid_real_tile_for_dino_satmae_required", "tile_manifest_or_small_tile", "candidate_grid", False, False, False, True, True, True, "missing_real_candidate_tile_for_embedding"),
        _artifact_row("S17C9_SRC_0016", "all_layers", "local_multimodal_feature_table_reference", "local_gitignored_raw", rel(LOCAL_MULTIMODAL_FEATURES), "csv", "local_reference_not_candidate_grid", False, False, False, False, False, True, "local_runs_reference_is_gitignored_and_not_candidate_specific"),
    ]
    return rows


def build_pipeline_audit() -> list[dict]:
    specs = [
        ("physical_static", "HAND;elevation;slope;DEM;distance_to_water;flow_accumulation;TWI", "scripts/suscetibilidade/profile_susc_features_by_patch_v1.py", "official_feature_matrix", "profile_csv", True, False, True, True, False, True, False, False, "requires_candidate_zonal_stats_adapter", "adaptar leitura de geojson candidato e artefatos DEM/HAND reais"),
        ("urban_territorial", "MapBiomas;urban_prop;vegetation_prop;water_prop;NDBI", "scripts/suscetibilidade/profile_susc_features_by_patch_v1.py", "official_feature_matrix", "profile_csv", True, False, True, True, False, True, False, False, "requires_candidate_landcover_adapter", "adaptar estatistica zonal MapBiomas/Sentinel-2 para bbox candidato"),
        ("rainfall_trigger", "CHIRPS_3d;CHIRPS_7d;CHIRPS_30d;runoff_context_7d", "scripts/suscetibilidade/profile_susc_features_by_patch_v1.py", "official_feature_matrix", "profile_csv", True, False, True, True, False, True, False, False, "requires_candidate_chirps_adapter", "adaptar janela temporal pre-evento e estatistica zonal por patch candidato"),
        ("sentinel2_spectral", "Sentinel-2;NDVI;NDWI;MNDWI;NDBI", "scripts/mv2_13_run_sentinel_binding.py", "anchor_asset_patch_manifests", "binding_and_gate_csv", True, True, True, False, True, True, True, False, "requires_candidate_grid_binding_and_no_download_flag", "adapter de binding para S17C6 candidate geojson e STAC metadata-only"),
        ("sentinel2_spectral", "GEE_STAC_metadata", "scripts/mv2_data_01_run_gee_lineage_targets.py", "lineage_targets", "metadata_plan_csv", True, True, True, False, False, False, True, False, "requires_candidate_lineage_targets", "gerar targets metadata-only para bboxes candidatos"),
        ("embedding_representation", "DINOv2;SatMAE;Scale-MAE", "scripts/dino/revp_v1fu_dino_sentinel_input_manifest.py", "sentinel_tile_manifest", "dino_input_manifest", True, False, True, True, False, True, False, False, "requires_real_candidate_tile_manifest", "adapter para manifest de tile real candidato sem executar embedding"),
        ("sar_observational", "SAR_runtime", "scripts/suscetibilidade/build_susc_17c2_sar_footprint_execution.py", "sar_processing_plan", "review_plan", False, False, True, False, False, True, True, False, "sar_outside_17c9_scope_and_runtime_missing", "manter bloqueado ate marco especifico de SAR"),
    ]
    rows = []
    for i, spec in enumerate(specs, start=1):
        layer, feature, path, input_type, output_type, works, accepts_geojson, needs_adapter, needs_patch_id, bbox_only, raster, creds, safe, why, adapter = spec
        rows.append({
            "pipeline_audit_id": f"S17C9_PIPE_{i:04d}",
            "feature_layer": layer,
            "feature_name": feature,
            "script_or_pipeline_path": path,
            "pipeline_exists": _bool_text((ROOT / path).exists()),
            "pipeline_current_input_type": input_type,
            "pipeline_current_output_type": output_type,
            "works_for_official_patches": _bool_text(works),
            "can_accept_candidate_patch_geojson": _bool_text(accepts_geojson),
            "requires_adapter": _bool_text(needs_adapter),
            "requires_official_patch_id": _bool_text(needs_patch_id),
            "requires_patch_bbox_only": _bool_text(bbox_only),
            "requires_raster_input": _bool_text(raster),
            "requires_api_credentials": _bool_text(creds),
            "safe_to_run_now": _bool_text(safe),
            "why_not_safe": why if not safe else "metadata_only_parse_safe_but_not_executed_in_17c9",
            "recommended_adapter": adapter,
            **GOV,
        })
    return rows


def build_input_requirements() -> list[dict]:
    reqs = [
        ("physical_static", "DEM_HAND_raster", "raster_or_zonal_stats", "candidate_patch_bbox_plus_buffer", "static_or_pre_event", "csv_or_geotiff_local_only", "S17C9_SRC_0005", False, True, True, True, "missing_candidate_specific_dem_hand_coverage"),
        ("physical_static", "drainage_network", "vector_or_raster", "candidate_patch_bbox_plus_buffer", "static", "geojson_or_zonal_stats_csv", "S17C9_SRC_0006", False, True, True, True, "missing_candidate_specific_drainage_coverage"),
        ("urban_territorial", "MapBiomas_landcover", "raster_or_zonal_stats", "candidate_patch_bbox", "year_policy_documented", "csv_or_geotiff_local_only", "S17C9_SRC_0008", False, True, True, True, "missing_candidate_specific_landcover_coverage"),
        ("rainfall_trigger", "CHIRPS_pre_event_accumulation", "zonal_stats", "candidate_patch_bbox", EVENT_WINDOW, "csv", "S17C9_SRC_0009", False, False, False, True, "missing_chirps_pre_event_zonal_stats"),
        ("sentinel2_spectral", "Sentinel2_pre_event_tile", "tile_metadata_or_small_clip", "candidate_patch_bbox", "pre_event_cloud_policy_required", "csv_or_small_tile_metadata", "S17C9_SRC_0010", False, False, False, True, "missing_candidate_specific_sentinel2_tile"),
        ("embedding_representation", "DINO_SatMAE_real_tile", "embedding_input_tile", "candidate_patch_bbox_square_tile", "same_as_sentinel2_tile_policy", "embedding_input_manifest", "S17C9_SRC_0015", False, False, False, True, "missing_real_candidate_tile_for_embedding"),
    ]
    rows = []
    for candidate_id in _candidate_ids():
        for layer, name, typ, extent, temporal, fmt, source_id, available, no_download, no_creds, no_official, reason in reqs:
            rows.append({
                "candidate_patch_id": candidate_id,
                "feature_layer": layer,
                "required_input_name": name,
                "required_input_type": typ,
                "required_spatial_extent": extent,
                "required_temporal_window": temporal,
                "required_format": fmt,
                "available_now": _bool_text(available),
                "source_artifact_id": source_id,
                "can_materialize_without_download": _bool_text(no_download),
                "can_materialize_without_credentials": _bool_text(no_creds),
                "can_materialize_without_official_patch": _bool_text(no_official),
                "blocking_reason": reason,
                **GOV,
            })
    return rows


def build_external_requests() -> list[dict]:
    ids = ";".join(_candidate_ids())
    specs = [
        ("P0_DEM_HAND_COVERAGE_FOR_CANDIDATE_GRID", "physical_static", "DEM/HAND cobrindo a grade candidata", "GEE/DEM/HAND provider", "bbox;crs;resolution;source_dataset;zonal_stats_fields", "csv_or_geotiff_local_only", "P0", True, True, False, True),
        ("P0_DRAINAGE_COVERAGE_FOR_CANDIDATE_GRID", "physical_static", "hidrografia/drenagem cobrindo a grade candidata", "JRC/OSM/municipal_or_GEE", "bbox;crs;geometry_type;distance_to_water_method", "geojson_or_csv", "P0", True, True, False, True),
        ("P1_MAPBIOMAS_COVERAGE_FOR_CANDIDATE_GRID", "urban_territorial", "MapBiomas ou cobertura territorial por patch candidato", "MapBiomas/GEE", "year;class_mapping;urban;vegetation;water;zonal_stats", "csv", "P1", True, True, False, True),
        ("P1_CHIRPS_PRE_EVENT_WINDOW_FOR_REC_2022_05_24_30", "rainfall_trigger", "CHIRPS acumulado pre-evento", "CHIRPS/GEE", "event_window;3d;7d;30d;runoff_context_method", "csv", "P1", True, True, False, True),
        ("P1_SENTINEL2_TILE_FOR_CANDIDATE_GRID", "sentinel2_spectral", "tile Sentinel-2 real pre-evento", "STAC/CDSE/GEE", "product_id;scene_id;bands;cloud_mask;bbox;date", "metadata_csv_or_local_only_tile", "P1", True, True, True, True),
        ("P2_DINO_INPUT_TILE_EXPORT", "embedding_representation", "tile real para DINO/SatMAE", "local export from Sentinel-2 tile", "tile_path_hash;bands;resolution;normalization;cloud_policy", "embedding_input_manifest", "P2", True, False, True, True),
        ("P2_SAR_RUNTIME_CREDENTIALS", "sar_observational", "credenciais/runtime SAR", "runtime operator", "runtime;credentials_policy;no_post_event_as_pre_event_rule", "runbook", "P2", True, False, False, True),
    ]
    rows = []
    for i, (priority, layer, needed, provider, fields, fmt, prio, manual, blocks_feature, blocks_embedding, blocks17b) in enumerate(specs, start=1):
        rows.append({
            "request_id": f"S17C9_REQ_{i:04d}",
            "feature_layer": layer,
            "artifact_needed": priority,
            "provider_or_source": provider,
            "candidate_patch_ids_affected": ids,
            "why_needed": needed,
            "minimum_required_fields": fields,
            "expected_format": fmt,
            "priority": prio,
            "can_be_requested_manually": _bool_text(manual),
            "blocks_feature_extraction": _bool_text(blocks_feature),
            "blocks_embedding": _bool_text(blocks_embedding),
            "blocks_17b": _bool_text(blocks17b),
            "review_only": "true",
        })
    return rows


def build_lightweight_exports() -> list[dict]:
    specs = [
        ("physical_static", "zonal_stats_csv", "outputs_public/suscetibilidade/future/s17c10_physical_static_zonal_stats.csv", "small", False, True, True, False, False, "blocked_until_dem_hand_and_drainage_artifacts_exist"),
        ("urban_territorial", "zonal_stats_csv", "outputs_public/suscetibilidade/future/s17c10_urban_territorial_zonal_stats.csv", "small", False, True, True, False, False, "blocked_until_mapbiomas_or_landcover_artifact_exists"),
        ("rainfall_trigger", "zonal_stats_csv", "outputs_public/suscetibilidade/future/s17c10_chirps_pre_event_zonal_stats.csv", "small", False, True, True, True, False, "blocked_until_chirps_window_materialized"),
        ("sentinel2_spectral", "small_tile_metadata", "outputs_public/suscetibilidade/future/s17c10_sentinel2_tile_metadata.csv", "small", False, True, True, True, False, "blocked_until_sentinel2_scene_or_tile_selected"),
        ("embedding_representation", "embedding_input_manifest", "outputs_public/suscetibilidade/future/s17c10_embedding_input_manifest.csv", "small", False, True, True, True, False, "blocked_until_real_tile_exists"),
    ]
    rows = []
    for candidate_id in _candidate_ids():
        for layer, kind, target, size, heavy, commit, runtime, creds, can_run, reason in specs:
            rows.append({
                "export_plan_id": f"S17C9_EXPORT_{len(rows) + 1:05d}",
                "candidate_patch_id": candidate_id,
                "feature_layer": layer,
                "export_target": f"{target}#{candidate_id}",
                "export_kind": kind if can_run else "not_runnable_now",
                "expected_output_path": target,
                "expected_size_class": size,
                "raw_heavy_output": _bool_text(heavy),
                "commit_allowed": _bool_text(commit),
                "requires_runtime": _bool_text(runtime),
                "requires_credentials": _bool_text(creds),
                "can_run_now": _bool_text(can_run),
                "blocking_reason": reason,
                "review_only": "true",
            })
    return rows


def build_embedding_tile_plan() -> list[dict]:
    rows = []
    for candidate_id in _candidate_ids():
        for model, bands, resolution in [
            ("DINOv2", "RGB_or_B02_B03_B04_with_documented_transform", "10"),
            ("SatMAE", "B02;B03;B04;B08;B11;B12", "10_or_20_documented"),
            ("Scale-MAE", "B02;B03;B04;B08;B11;B12", "10_or_20_documented"),
        ]:
            rows.append({
                "embedding_tile_requirement_id": f"S17C9_EMBTILE_{len(rows) + 1:05d}",
                "candidate_patch_id": candidate_id,
                "target_model": model,
                "required_tile_type": "real_sentinel2_or_rgb_multiband_tile",
                "required_bands": bands,
                "required_resolution_m": resolution,
                "required_date_policy": "pre_event_or_documented_event_window_no_post_event_as_pre_event",
                "required_cloud_policy": "cloud_mask_or_cloud_risk_documented",
                "expected_preprocessing": "crop_or_resample_documented;normalization_policy_recorded;no_pixel_read_in_17c9",
                "tile_exists_now": "false",
                "can_create_now": "false",
                "requires_sentinel2_tile": "true",
                "requires_external_download": "true",
                "requires_runtime_credentials": "true",
                "blocks_real_embedding": "true",
                **GOV,
            })
    return rows


def build_unblock_matrix() -> list[dict]:
    specs = [
        ("physical_static", "blocked_no_candidate_specific_artifact", "DEM/HAND/drainage/flow/TWI artifacts", "materializar P0 DEM/HAND e adapter de estatistica zonal", "false", "high"),
        ("urban_territorial", "blocked_no_candidate_specific_artifact", "MapBiomas/landcover candidate coverage", "materializar cobertura territorial e adapter de estatistica zonal", "false", "medium"),
        ("rainfall_trigger", "blocked_external_download", "CHIRPS pre-event zonal stats", "criar pedido GEE/CHIRPS metadata-only ou export leve", "false", "medium"),
        ("sentinel2_spectral", "blocked_runtime_credentials", "Sentinel-2 tile or STAC/CDSE/GEE lineage", "resolver tile pre-evento e politica de nuvem", "false", "high"),
        ("embedding_representation", "blocked_external_download", "real DINO/SatMAE input tile", "preparar tile real apos Sentinel-2", "false", "high"),
    ]
    rows = []
    for candidate_id in _candidate_ids():
        for layer, status, missing, step, can_unlock, risk in specs:
            rows.append({
                "unblock_id": f"S17C9_UNBLOCK_{len(rows) + 1:05d}",
                "candidate_patch_id": candidate_id,
                "feature_layer": layer,
                "current_status": status,
                "missing_artifact_or_adapter": missing,
                "required_next_step": step,
                "can_unlock_in_next_sprint": can_unlock,
                "risk_level": risk,
                "blocks_scientific_use": "true",
                "blocks_17b": "true",
                "review_only": "true",
            })
    return rows


def build_no_fake_policy() -> dict:
    return {
        "policy_id": "SUSC_17C9_NO_FAKE_FEATURE_POLICY_V1",
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "features_extracted_this_sprint": 0,
        "blocked_uses": {
            "copy_neighbor_official_patch_value": True,
            "interpolation_without_method_and_provenance": True,
            "centroid_as_area_substitute": True,
            "protocolo_c_patch_as_susc_patch": True,
            "REC_00019_as_SUSC_patch": True,
            "charter_footprint_as_susceptibility_feature": True,
            "post_event_data_as_pre_event_feature": True,
            "synthetic_placeholder_as_real": True,
            "remote_artifact_not_downloaded_as_local": True,
            "existing_pipeline_as_executed_extraction": True,
            "promotion_to_17b_without_real_feature_provenance_qa": True,
        },
        "required_for_future_real_feature": [
            "candidate_specific_artifact",
            "provenance_manifest",
            "no_leakage_audit",
            "candidate_patch_policy_or_official_patch",
            "qa_acceptance",
        ],
    }


def build_blockers() -> list[dict]:
    codes = [
        "candidate_specific_artifacts_missing",
        "candidate_patch_policy_missing",
        "pipeline_adapters_missing",
        "official_patch_id_dependency",
        "sentinel2_tile_missing",
        "embedding_tile_missing",
        "sar_runtime_missing",
        "qa_not_accepted",
        "p0_official_packages_missing",
        "17b_blocked_until_real_features_and_policy",
    ]
    descriptions = {
        "candidate_specific_artifacts_missing": "faltam artefatos candidato-especificos para features reais",
        "candidate_patch_policy_missing": "nao ha politica aceita para usar grade candidata em score ou 17B",
        "pipeline_adapters_missing": "pipelines existentes exigem adapter para geojson/bbox candidato",
        "official_patch_id_dependency": "pipelines tabulares atuais dependem de patch oficial ou matriz oficial",
        "sentinel2_tile_missing": "tile Sentinel-2 real candidato-especifico nao existe",
        "embedding_tile_missing": "tile real para DINO/SatMAE nao existe",
        "sar_runtime_missing": "SAR continua fora de escopo e sem runtime/credencial",
        "qa_not_accepted": "links candidatos e evidencias ainda nao sao QA-accepted",
        "p0_official_packages_missing": "pacotes P0 de DEM/HAND/drenagem faltam",
        "17b_blocked_until_real_features_and_policy": "17B exige features reais, proveniencia, QA e politica",
    }
    return [
        {
            "blocker_id": f"S17C9_BLOCKER_{i:03d}",
            "blocker_code": code,
            "description": descriptions[code],
            "blocks_17b": "true",
            "blocks_score_v7": "true",
            "blocks_training": "true",
            **GOV,
        }
        for i, code in enumerate(codes, start=1)
    ]


def build_summary() -> dict:
    inventory = build_source_inventory()
    pipelines = build_pipeline_audit()
    exports = build_lightweight_exports()
    requests = build_external_requests()
    return {
        "milestone": "SUSC-17C9",
        "candidate_patch_count": len(_candidate_ids()),
        "source_artifacts_inventory_count": len(inventory),
        "local_artifacts_covering_candidate_grid_count": sum(1 for r in inventory if r["artifact_exists_local"] == "true" and r["covers_candidate_patch_area"] == "true"),
        "missing_required_artifacts_count": sum(1 for r in inventory if r["artifact_kind"] == "missing_required_artifact"),
        "pipelines_audited_count": len(pipelines),
        "pipelines_reusable_without_adapter_count": sum(1 for r in pipelines if r["requires_adapter"] == "false" and r["safe_to_run_now"] == "true"),
        "pipelines_requiring_adapter_count": sum(1 for r in pipelines if r["requires_adapter"] == "true"),
        "pipelines_blocked_official_patch_id_count": sum(1 for r in pipelines if r["requires_official_patch_id"] == "true"),
        "lightweight_exports_ready_count": sum(1 for r in exports if r["can_run_now"] == "true"),
        "embedding_tile_ready_count": 0,
        "external_requests_count": len(requests),
        "features_extracted_this_sprint": 0,
        "score_v6_changed": False,
        "score_v7_created": SCORE_V7.exists(),
        "official_patch_created": False,
        "official_patch_link_created": False,
        "raw_raster_committed": False,
        "eligible_for_17b_now": False,
        "eligible_for_score_v7": False,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "recommended_next_milestone": "SUSC-17C10 Pacote de Solicitacao Formal",
    }


def build_source_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17C9 source artifact inventory schema v1",
        "type": "object",
        "required": [
            "source_artifact_id", "feature_layer", "feature_name", "artifact_kind",
            "artifact_path_or_reference", "artifact_exists_local", "artifact_is_committed",
            "artifact_is_gitignored", "artifact_is_raw_heavy", "artifact_format",
            "artifact_scope", "covers_candidate_patch_area", "covers_official_patch_area",
            "can_be_used_for_candidate_patches", "requires_external_download",
            "requires_runtime_credentials", "requires_manual_preparation", "blocking_reason",
            "review_only", "trainable", "ground_truth",
        ],
        "properties": {
            "artifact_kind": {"enum": [
                "local_committed_output", "local_gitignored_raw", "manifest_reference",
                "pipeline_expected_input", "remote_asset_reference", "missing_required_artifact",
                "runtime_required_artifact", "not_found",
            ]},
            "review_only": {"const": "true"},
            "trainable": {"const": "false"},
            "ground_truth": {"const": "false"},
        },
    }


def build_pipeline_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17C9 pipeline reuse audit schema v1",
        "type": "object",
        "required": [
            "pipeline_audit_id", "feature_layer", "feature_name", "script_or_pipeline_path",
            "pipeline_exists", "pipeline_current_input_type", "pipeline_current_output_type",
            "works_for_official_patches", "can_accept_candidate_patch_geojson",
            "requires_adapter", "requires_official_patch_id", "requires_patch_bbox_only",
            "requires_raster_input", "requires_api_credentials", "safe_to_run_now",
            "why_not_safe", "recommended_adapter", "review_only", "trainable", "ground_truth",
        ],
        "properties": {
            "review_only": {"const": "true"},
            "trainable": {"const": "false"},
            "ground_truth": {"const": "false"},
        },
    }


def build_report() -> str:
    summary = build_summary()
    return "\n".join([
        "# SUSC-17C9 - Materializacao de artefatos fonte para patches candidatos",
        "",
        "O 17C8 revelou que a grade candidata tem geometria e contratos multimodais, mas nao tem artefatos fonte candidato-especificos para marcar features como reais. O resultado de 0 features reais foi metodologicamente correto: sem raster, tile, estatistica zonal, janela temporal e proveniencia por patch candidato, qualquer valor seria copia, inferencia ou placeholder.",
        "",
        "Este marco transforma esse bloqueio em plano operacional verificavel. Ele nao extrai feature final, nao executa SAR, nao executa DINO/SatMAE, nao baixa raster pesado e nao cria score v7.",
        "",
        "## Artefatos fonte existentes",
        "",
        "Existem a grade candidata 17C6 em CSV/GeoJSON, a matriz oficial de features, o manifesto de proveniencia da matriz oficial e manifests de referencia para Sentinel-2, STAC/GEE e DINO. Esses artefatos ajudam a desenhar adapters e requisitos, mas nao substituem artefatos reais por patch candidato.",
        "",
        "## Artefatos fonte ausentes",
        "",
        "Faltam DEM/HAND, drenagem, derivados de fluxo/TWI, MapBiomas ou cobertura territorial, CHIRPS pre-evento, tile Sentinel-2 real e tile real para DINO/SatMAE cobrindo os 5 patches candidatos.",
        "",
        "## Pipelines e adapters",
        "",
        "Os pipelines atuais conseguem auditar ou perfilar matrizes oficiais e manifests, mas exigem adapter para aceitar geojson/bbox candidato e para produzir estatistica zonal leve. Parte da cadeia Sentinel-2 aceita geometria/metadata, mas ainda precisa binding especifico da grade candidata e nao deve baixar raster neste marco.",
        "",
        "## Exports leves suficientes",
        "",
        "A proxima extracao deve preferir CSV de estatistica zonal, pequenos recortes vetoriais, metadata de tile e manifest de input de embedding. Raw raster pesado continua proibido no Git.",
        "",
        "## Bloqueios por modalidade",
        "",
        "- DINO/SatMAE: falta tile real candidato-especifico e politica de pre-processamento.",
        "- Sentinel-2: falta tile/metadata pre-evento com politica de nuvem.",
        "- Chuva/CHIRPS: falta janela pre-evento agregada por patch candidato.",
        "- Fisicas/urbanas: faltam DEM/HAND/drenagem/MapBiomas ou estatisticas zonais reais.",
        "",
        "## Prontidao",
        "",
        f"- Artefatos inventariados: {summary['source_artifacts_inventory_count']}",
        f"- Artefatos locais cobrindo a grade candidata: {summary['local_artifacts_covering_candidate_grid_count']}",
        f"- Artefatos obrigatorios ausentes: {summary['missing_required_artifacts_count']}",
        f"- Pipelines auditados: {summary['pipelines_audited_count']}",
        f"- Exports leves prontos: {summary['lightweight_exports_ready_count']}",
        "",
        "Score v6 permanece intacto. Score v7 e 17B continuam bloqueados porque nao ha features reais candidato-especificas, QA e politica de promocao.",
        "",
        f"Proximo marco recomendado: `{summary['recommended_next_milestone']}`.",
    ])


def run_all() -> None:
    _require_inputs()
    write_json(SOURCE_SCHEMA, build_source_schema())
    write_json(PIPELINE_SCHEMA, build_pipeline_schema())
    write_csv(SOURCE_INVENTORY, build_source_inventory())
    write_csv(PIPELINE_AUDIT, build_pipeline_audit())
    write_csv(INPUT_REQUIREMENTS, build_input_requirements())
    write_csv(EXTERNAL_REQUESTS, build_external_requests())
    write_csv(LIGHTWEIGHT_EXPORTS, build_lightweight_exports())
    write_csv(EMBEDDING_TILE_PLAN, build_embedding_tile_plan())
    write_csv(UNBLOCK_MATRIX, build_unblock_matrix())
    write_json(NO_FAKE_POLICY, build_no_fake_policy())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_markdown(REPORT, build_report())


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _schema_violations(row: dict, schema: dict) -> list[str]:
    errors = []
    for key in schema.get("required", []):
        if key not in row:
            errors.append(f"missing:{key}")
        elif row[key] == "":
            errors.append(f"empty:{key}")
    for key, rule in schema.get("properties", {}).items():
        if key not in row:
            continue
        if "const" in rule and row[key] != rule["const"]:
            errors.append(f"const:{key}")
        if "enum" in rule and row[key] not in rule["enum"]:
            errors.append(f"enum:{key}")
    return errors


def _assert(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate() -> int:
    errors: list[str] = []
    try:
        _require_inputs()
        run_all()
        first = {p: _sha(p) for p in REQUIRED_OUTPUTS}
        run_all()
        second = {p: _sha(p) for p in REQUIRED_OUTPUTS}
        _assert(first == second, "build nao e byte-identico", errors)

        for path in REQUIRED_OUTPUTS:
            _assert(path.exists(), f"artefato ausente: {rel(path)}", errors)

        source_schema = read_json(SOURCE_SCHEMA)
        pipeline_schema = read_json(PIPELINE_SCHEMA)
        inventory = read_csv(SOURCE_INVENTORY)
        pipelines = read_csv(PIPELINE_AUDIT)
        summary = read_json(SUMMARY)
        blockers = read_csv(BLOCKERS)
        policy = read_json(NO_FAKE_POLICY)

        for row in inventory:
            errors.extend(f"{row.get('source_artifact_id')}:{err}" for err in _schema_violations(row, source_schema))
            _assert(row["review_only"] == "true", "inventory nao review-only", errors)
            _assert(row["trainable"] == "false", "inventory trainable", errors)
            _assert(row["ground_truth"] == "false", "inventory ground_truth", errors)
            _assert(not (row["artifact_kind"] == "remote_asset_reference" and row["artifact_exists_local"] == "true"), "artefato remoto marcado como local", errors)
            _assert(row["artifact_is_raw_heavy"] != "true" or row["artifact_is_committed"] == "false", "raster bruto pesado commitado", errors)

        for row in pipelines:
            errors.extend(f"{row.get('pipeline_audit_id')}:{err}" for err in _schema_violations(row, pipeline_schema))
            _assert(row["review_only"] == "true", "pipeline nao review-only", errors)
            _assert(row["trainable"] == "false", "pipeline trainable", errors)
            _assert(row["ground_truth"] == "false", "pipeline ground_truth", errors)
            _assert(row["pipeline_current_output_type"] != "executed_candidate_feature_extraction", "pipeline marcado como extracao executada", errors)

        for path in [INPUT_REQUIREMENTS, EMBEDDING_TILE_PLAN, BLOCKERS]:
            for row in read_csv(path):
                _assert(row["review_only"] == "true", f"{rel(path)} nao review-only", errors)
                _assert(row["trainable"] == "false", f"{rel(path)} trainable", errors)
                _assert(row["ground_truth"] == "false", f"{rel(path)} ground_truth", errors)
        for path in [EXTERNAL_REQUESTS, LIGHTWEIGHT_EXPORTS, UNBLOCK_MATRIX]:
            for row in read_csv(path):
                _assert(row["review_only"] == "true", f"{rel(path)} nao review-only", errors)

        ids = [r["source_artifact_id"] for r in inventory]
        _assert(len(ids) == len(set(ids)), "source_artifact_id duplicado", errors)
        pipe_ids = [r["pipeline_audit_id"] for r in pipelines]
        _assert(len(pipe_ids) == len(set(pipe_ids)), "pipeline_audit_id duplicado", errors)
        _assert(pipe_ids == sorted(pipe_ids), "pipeline ids nao deterministicos", errors)

        blocked = policy["blocked_uses"]
        for key in [
            "copy_neighbor_official_patch_value",
            "interpolation_without_method_and_provenance",
            "centroid_as_area_substitute",
            "protocolo_c_patch_as_susc_patch",
            "REC_00019_as_SUSC_patch",
            "charter_footprint_as_susceptibility_feature",
            "post_event_data_as_pre_event_feature",
            "synthetic_placeholder_as_real",
            "remote_artifact_not_downloaded_as_local",
            "existing_pipeline_as_executed_extraction",
            "promotion_to_17b_without_real_feature_provenance_qa",
        ]:
            _assert(blocked.get(key) is True, f"politica nao bloqueia {key}", errors)

        candidate_ids = set(_candidate_ids())
        for path in [INPUT_REQUIREMENTS, LIGHTWEIGHT_EXPORTS, EMBEDDING_TILE_PLAN, UNBLOCK_MATRIX]:
            found = {r["candidate_patch_id"] for r in read_csv(path)}
            _assert(found == candidate_ids, f"{rel(path)} nao cobre todos candidatos", errors)
            _assert("REC_00019" not in found, "REC_00019 apareceu como patch SUSC", errors)

        _assert(summary["candidate_patch_count"] == len(candidate_ids), "summary candidate count incorreto", errors)
        _assert(summary["source_artifacts_inventory_count"] == len(inventory), "summary inventory count incorreto", errors)
        _assert(summary["local_artifacts_covering_candidate_grid_count"] == sum(1 for r in inventory if r["artifact_exists_local"] == "true" and r["covers_candidate_patch_area"] == "true"), "summary local coverage count incorreto", errors)
        _assert(summary["missing_required_artifacts_count"] == sum(1 for r in inventory if r["artifact_kind"] == "missing_required_artifact"), "summary missing count incorreto", errors)
        _assert(summary["pipelines_audited_count"] == len(pipelines), "summary pipeline count incorreto", errors)
        _assert(summary["features_extracted_this_sprint"] == 0, "features foram criadas nesta sprint", errors)
        _assert(summary["score_v6_changed"] is False, "summary indica score v6 alterado", errors)
        _assert(summary["score_v7_created"] is False, "summary indica score v7 criado", errors)
        _assert(summary["official_patch_created"] is False, "summary indica patch oficial criado", errors)
        _assert(summary["official_patch_link_created"] is False, "summary indica patch-link oficial criado", errors)
        _assert(summary["raw_raster_committed"] is False, "summary indica raster bruto commitado", errors)
        _assert(summary["eligible_for_17b_now"] is False, "17B elegivel indevidamente", errors)
        _assert(summary["eligible_for_score_v7"] is False, "score v7 elegivel indevidamente", errors)
        _assert(summary["review_only"] is True, "summary nao review-only", errors)
        _assert(summary["trainable"] is False, "summary trainable", errors)
        _assert(summary["ground_truth"] is False, "summary ground_truth", errors)
        _assert(len(blockers) > 0, "bloqueadores vazios", errors)

        _assert(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]) == "", "score v6 tem diff", errors)
        _assert(_run_git(["diff", "--name-only", "--", rel(FEATURES)]) == "", "dataset oficial de patches/features tem diff", errors)
        _assert(not SCORE_V7.exists(), "score v7 existe", errors)

        if errors:
            for err in errors:
                print(f"17C9 validation error: {err}", file=sys.stderr)
            return 1
        print(
            "17C9 -> "
            f"candidate_patches={summary['candidate_patch_count']} "
            f"source_artifacts={summary['source_artifacts_inventory_count']} "
            f"pipelines={summary['pipelines_audited_count']} "
            f"features_extracted={summary['features_extracted_this_sprint']} "
            "score_v7_created=False"
        )
        return 0
    except Exception as exc:
        print(f"17C9 validation exception: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    run_all()
