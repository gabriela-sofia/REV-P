"""SUSC-17C7 Plano de Extracao de Features para Patches Candidatos.

Creates a deterministic review-only extraction plan for the SUSC-17C6 candidate
patches. It does not extract rasters, execute SAR, run DINO/SatMAE, change score
v6, create score v7, official patches, official patch-links, labels, training
data or ground truth.
"""

from __future__ import annotations

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

AUDIT = OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C7.md"
FEATURES = DAT / "susc_features_by_patch_v1.csv"
SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

CANDIDATE_GRID_17C6 = OUT / "susc_17c6_candidate_patch_grid.csv"
CANDIDATE_GRID_GEOJSON_17C6 = OUT / "susc_17c6_candidate_patch_grid.geojson"
CANDIDATE_LINKS_17C6 = OUT / "susc_17c6_candidate_patch_links.csv"
FEATURE_CONTRACT_17C6 = OUT / "susc_17c6_multimodal_feature_contract.csv"
CANARY_MATRIX_17C6 = OUT / "susc_17c6_multimodal_canary_matrix.csv"
EMBEDDING_CONTRACT_17C6 = OUT / "susc_17c6_embedding_contract.json"
SUMMARY_17C6 = OUT / "susc_17c6_applicability_readiness_summary.json"
BLOCKERS_17C6 = OUT / "susc_17c6_promotion_blockers.csv"
SYNTHETIC_EMBEDDING_17C6 = OUT / "susc_17c6_synthetic_embedding_smoke_test.csv"

INVENTORY_SCHEMA = SCHEMAS / "susc_17c7_feature_inventory_schema_v1.json"
TASK_SCHEMA = SCHEMAS / "susc_17c7_extraction_task_schema_v1.json"

REPORT = OUT / "SUSC_17C7_PLANO_EXTRACAO_FEATURES_PATCHES_CANDIDATOS_REPORT.md"
FEATURE_INVENTORY = OUT / "susc_17c7_candidate_patch_feature_inventory.csv"
SOURCE_MAPPING = OUT / "susc_17c7_feature_source_mapping.csv"
TASK_PLAN = OUT / "susc_17c7_extraction_task_plan.csv"
MISSINGNESS = OUT / "susc_17c7_feature_missingness_matrix.csv"
EMBEDDING_READINESS = OUT / "susc_17c7_embedding_input_readiness.csv"
NO_LEAKAGE_POLICY = OUT / "susc_17c7_no_leakage_policy.json"
SUMMARY = OUT / "susc_17c7_readiness_summary.json"
BLOCKERS = OUT / "susc_17c7_promotion_blockers.csv"

MILESTONE_NAME = "SUSC-17C7 — Plano de Extração de Features para Patches Candidatos"
MILESTONE_CATEGORY = "Plano de Extração de Camadas Multimodais para Patches Candidatos"
EVENT_ID = "REC_2022_05_24_30"
REC_00019 = "REC_00019"

MULTIMODAL_LAYERS = [
    "physical_static",
    "urban_territorial",
    "sentinel2_spectral",
    "rainfall_trigger",
    "documentary_evidence",
    "sar_observational",
    "embedding_representation",
]

REQUIRED_INPUTS = [
    AUDIT, FEATURES, SCORE_V6,
    CANDIDATE_GRID_17C6, CANDIDATE_GRID_GEOJSON_17C6, CANDIDATE_LINKS_17C6,
    FEATURE_CONTRACT_17C6, CANARY_MATRIX_17C6, EMBEDDING_CONTRACT_17C6,
    SUMMARY_17C6, BLOCKERS_17C6, SYNTHETIC_EMBEDDING_17C6,
    INVENTORY_SCHEMA, TASK_SCHEMA,
]

REQUIRED_OUTPUTS = [
    REPORT, FEATURE_INVENTORY, SOURCE_MAPPING, TASK_PLAN, MISSINGNESS,
    EMBEDDING_READINESS, NO_LEAKAGE_POLICY, SUMMARY, BLOCKERS,
]


def _require_inputs() -> None:
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))


def _gov_str() -> dict:
    return {"review_only": "true", "trainable": "false", "ground_truth": "false"}


def _feature_cols() -> set[str]:
    rows = read_csv(FEATURES)
    return set(rows[0].keys()) if rows else set()


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _source_for_feature(feature: dict) -> dict:
    name = feature["feature_name"]
    layer = feature["layer"]
    official = feature["available_for_official_patches"] == "true"
    candidate = feature["available_for_candidate_patches"] == "true"
    if layer == "physical_static":
        return {
            "known_source_dataset": rel(FEATURES),
            "known_source_artifact": "DEM_HAND_slope_drainage_columns_in_official_patch_table",
            "known_script_or_pipeline": "scripts/suscetibilidade/profile_susc_features_by_patch_v1.py",
            "existing_for_official_patches": official,
            "can_apply_to_candidate_patches": True,
            "requires_patch_officialization": False,
            "requires_candidate_policy_only": True,
            "requires_raster_asset": True,
            "requires_api_or_credentials": False,
            "requires_manual_input": False,
            "expected_output_type": "numeric_patch_feature",
            "blocking_reason": "pipeline_exists_for_official_patches_but_not_run_for_candidate_grid",
        }
    if layer == "urban_territorial":
        return {
            "known_source_dataset": rel(FEATURES),
            "known_source_artifact": "MapBiomas_or_territorial_proxy_columns_in_official_patch_table",
            "known_script_or_pipeline": "scripts/suscetibilidade/profile_susc_features_by_patch_v1.py",
            "existing_for_official_patches": official,
            "can_apply_to_candidate_patches": True,
            "requires_patch_officialization": False,
            "requires_candidate_policy_only": True,
            "requires_raster_asset": True,
            "requires_api_or_credentials": False,
            "requires_manual_input": False,
            "expected_output_type": "numeric_patch_feature",
            "blocking_reason": "territorial_pipeline_not_run_for_candidate_grid",
        }
    if layer == "sentinel2_spectral":
        return {
            "known_source_dataset": rel(FEATURES),
            "known_source_artifact": "Sentinel-2_band_and_index_columns_for_official_patches",
            "known_script_or_pipeline": "scripts/mv2_13_run_sentinel_binding.py",
            "existing_for_official_patches": official,
            "can_apply_to_candidate_patches": True,
            "requires_patch_officialization": False,
            "requires_candidate_policy_only": True,
            "requires_raster_asset": True,
            "requires_api_or_credentials": True,
            "requires_manual_input": False,
            "expected_output_type": "numeric_patch_feature_and_tile_lineage",
            "blocking_reason": "candidate_patch_sentinel_tile_not_materialized",
        }
    if layer == "rainfall_trigger":
        return {
            "known_source_dataset": rel(FEATURES),
            "known_source_artifact": "CHIRPS_and_runoff_columns_for_official_patches",
            "known_script_or_pipeline": "scripts/suscetibilidade/profile_susc_features_by_patch_v1.py",
            "existing_for_official_patches": official,
            "can_apply_to_candidate_patches": True,
            "requires_patch_officialization": False,
            "requires_candidate_policy_only": True,
            "requires_raster_asset": True,
            "requires_api_or_credentials": False,
            "requires_manual_input": False,
            "expected_output_type": "numeric_patch_feature",
            "blocking_reason": "candidate_patch_rainfall_not_extracted",
        }
    if layer == "documentary_evidence":
        return {
            "known_source_dataset": rel(CANDIDATE_LINKS_17C6),
            "known_source_artifact": rel(CANDIDATE_GRID_17C6),
            "known_script_or_pipeline": "scripts/suscetibilidade/susc_17c6_multimodal_canary_common.py",
            "existing_for_official_patches": False,
            "can_apply_to_candidate_patches": candidate,
            "requires_patch_officialization": False,
            "requires_candidate_policy_only": False,
            "requires_raster_asset": False,
            "requires_api_or_credentials": False,
            "requires_manual_input": True,
            "expected_output_type": "review_only_candidate_link_and_qa_status",
            "blocking_reason": "qa_not_accepted_for_scientific_use",
        }
    if layer == "sar_observational":
        return {
            "known_source_dataset": rel(OUT / "susc_17c2_sar_processing_plan.csv"),
            "known_source_artifact": rel(OUT / "susc_17c2_candidate_footprints.geojson"),
            "known_script_or_pipeline": "scripts/suscetibilidade/build_susc_17c2_sar_footprint_execution.py",
            "existing_for_official_patches": False,
            "can_apply_to_candidate_patches": False,
            "requires_patch_officialization": False,
            "requires_candidate_policy_only": True,
            "requires_raster_asset": True,
            "requires_api_or_credentials": True,
            "requires_manual_input": True,
            "expected_output_type": "observational_footprint_review_package",
            "blocking_reason": "sar_runtime_unavailable",
        }
    return {
        "known_source_dataset": rel(EMBEDDING_CONTRACT_17C6),
        "known_source_artifact": rel(SYNTHETIC_EMBEDDING_17C6),
        "known_script_or_pipeline": "scripts/dino/revp_v1fu_dino_sentinel_input_manifest.py",
        "existing_for_official_patches": False,
        "can_apply_to_candidate_patches": False,
        "requires_patch_officialization": False,
        "requires_candidate_policy_only": True,
        "requires_raster_asset": True,
        "requires_api_or_credentials": False,
        "requires_manual_input": False,
        "expected_output_type": "real_tile_and_embedding_vector",
        "blocking_reason": "no_real_embedding_tile",
    }


INVENTORY_FIELDS = [
    "candidate_patch_id", "candidate_grid_id", "event_id", "feature_name",
    "feature_layer", "required_for_future_score", "required_for_future_embedding",
    "required_for_review_only_evaluation", "available_now",
    "available_from_existing_official_patch_pipeline",
    "available_from_existing_raw_artifact", "requires_new_extraction",
    "requires_external_download", "requires_runtime_credentials",
    "requires_raster_processing", "requires_human_review",
    "usable_for_science_now", "synthetic_placeholder", "blocking_reason",
    "review_only", "trainable", "ground_truth",
]


def build_feature_inventory() -> list[dict]:
    rows = []
    candidates = read_csv(CANDIDATE_GRID_17C6)
    contract = read_csv(FEATURE_CONTRACT_17C6)
    synthetic_features = {"embedding_dimension", "embedding_source", "embedding_status"}
    for patch in candidates:
        for feature in contract:
            source = _source_for_feature(feature)
            layer = feature["layer"]
            available_now = feature["available_for_candidate_patches"] == "true"
            synthetic = layer == "embedding_representation" and feature["feature_name"] in synthetic_features
            rows.append({
                "candidate_patch_id": patch["candidate_patch_id"],
                "candidate_grid_id": patch["candidate_grid_id"],
                "event_id": patch["source_event_id"],
                "feature_name": feature["feature_name"],
                "feature_layer": layer,
                "required_for_future_score": feature["required_for_future_score"],
                "required_for_future_embedding": feature["required_for_future_embedding"],
                "required_for_review_only_evaluation": feature["required_for_review_only_evaluation"],
                "available_now": _bool_text(available_now),
                "available_from_existing_official_patch_pipeline": _bool_text(source["existing_for_official_patches"]),
                "available_from_existing_raw_artifact": "true" if layer == "documentary_evidence" and available_now else "false",
                "requires_new_extraction": "false" if available_now else "true",
                "requires_external_download": _bool_text(layer in {"sentinel2_spectral", "sar_observational"}),
                "requires_runtime_credentials": _bool_text(layer in {"sentinel2_spectral", "sar_observational"}),
                "requires_raster_processing": _bool_text(source["requires_raster_asset"]),
                "requires_human_review": _bool_text(layer in {"documentary_evidence", "sar_observational"}),
                "usable_for_science_now": "false",
                "synthetic_placeholder": _bool_text(synthetic),
                "blocking_reason": feature["blocking_reason"] if not available_now else "available_review_only_but_qa_not_scientific",
                **_gov_str(),
            })
    rows.sort(key=lambda r: (r["candidate_patch_id"], r["feature_layer"], r["feature_name"]))
    return rows


SOURCE_MAPPING_FIELDS = [
    "feature_name", "feature_layer", "known_source_dataset", "known_source_artifact",
    "known_script_or_pipeline", "existing_for_official_patches",
    "can_apply_to_candidate_patches", "requires_patch_officialization",
    "requires_candidate_policy_only", "requires_raster_asset",
    "requires_api_or_credentials", "requires_manual_input", "expected_output_type",
    "blocking_reason",
]


def build_source_mapping() -> list[dict]:
    rows = []
    for feature in read_csv(FEATURE_CONTRACT_17C6):
        source = _source_for_feature(feature)
        rows.append({
            "feature_name": feature["feature_name"],
            "feature_layer": feature["layer"],
            "known_source_dataset": source["known_source_dataset"],
            "known_source_artifact": source["known_source_artifact"],
            "known_script_or_pipeline": source["known_script_or_pipeline"],
            "existing_for_official_patches": _bool_text(source["existing_for_official_patches"]),
            "can_apply_to_candidate_patches": _bool_text(source["can_apply_to_candidate_patches"]),
            "requires_patch_officialization": _bool_text(source["requires_patch_officialization"]),
            "requires_candidate_policy_only": _bool_text(source["requires_candidate_policy_only"]),
            "requires_raster_asset": _bool_text(source["requires_raster_asset"]),
            "requires_api_or_credentials": _bool_text(source["requires_api_or_credentials"]),
            "requires_manual_input": _bool_text(source["requires_manual_input"]),
            "expected_output_type": source["expected_output_type"],
            "blocking_reason": source["blocking_reason"],
        })
    rows.sort(key=lambda r: (r["feature_layer"], r["feature_name"]))
    return rows


TASK_FIELDS = [
    "extraction_task_id", "candidate_patch_id", "feature_layer", "task_name",
    "task_type", "input_required", "expected_output", "can_run_now",
    "why_not_now", "estimated_dependency", "must_not_commit_raw_raster",
    "review_only", "trainable", "ground_truth",
]


def _task_for_layer(patch_id: str, layer: str, idx: int) -> dict:
    configs = {
        "physical_static": ("adapt_existing_pipeline_to_candidate_patch", "candidate_patch_geometry;DEM_HAND_drainage_pipeline", "HAND_elevation_slope_drainage_features", "false", "candidate_grid_policy_and_pipeline_adaptation_required", "candidate_patch_policy"),
        "urban_territorial": ("adapt_existing_pipeline_to_candidate_patch", "candidate_patch_geometry;territorial_raster_pipeline", "MapBiomas_urban_vegetation_water_features", "false", "candidate_grid_policy_and_pipeline_adaptation_required", "candidate_patch_policy"),
        "sentinel2_spectral": ("requires_external_artifact", "candidate_patch_geometry;Sentinel-2_tile_or_scene_lineage", "NDVI_NDWI_MNDWI_NDBI_and_tile_metadata", "false", "real_sentinel_tile_not_materialized_for_candidate_patch", "sentinel2_scene_or_tile"),
        "rainfall_trigger": ("adapt_existing_pipeline_to_candidate_patch", "candidate_patch_geometry;CHIRPS_or_runoff_context", "CHIRPS_3d_7d_30d_and_runoff_context", "false", "candidate_patch_rainfall_pipeline_not_run", "candidate_patch_policy"),
        "documentary_evidence": ("requires_manual_review", "17C6_candidate_link_and_Charter758_reference", "QA_status_and_documentary_evidence_review", "false", "human_QA_not_accepted", "human_QA"),
        "sar_observational": ("requires_runtime_credentials", "Sentinel-1_pre_post_window;SAR_runtime", "review_only_SAR_footprint_candidate", "false", "sar_runtime_unavailable", "SAR_runtime_and_credentials"),
        "embedding_representation": ("requires_external_artifact", "real_tile;embedding_contract", "real_embedding_input_manifest_and_vector", "false", "no_real_tile_for_DINO_or_SatMAE", "real_tile_materialization"),
    }
    task_type, input_required, expected_output, can_run, why_not, dependency = configs[layer]
    return {
        "extraction_task_id": f"S17C7_TASK_{idx:05d}",
        "candidate_patch_id": patch_id,
        "feature_layer": layer,
        "task_name": f"planejar_{layer}_para_patch_candidato",
        "task_type": task_type,
        "input_required": input_required,
        "expected_output": expected_output,
        "can_run_now": can_run,
        "why_not_now": why_not,
        "estimated_dependency": dependency,
        "must_not_commit_raw_raster": "true",
        **_gov_str(),
    }


def build_task_plan() -> list[dict]:
    rows = []
    idx = 1
    for patch in read_csv(CANDIDATE_GRID_17C6):
        for layer in MULTIMODAL_LAYERS:
            rows.append(_task_for_layer(patch["candidate_patch_id"], layer, idx))
            idx += 1
    return rows


MISSINGNESS_FIELDS = [
    "candidate_patch_id", "has_physical_static", "has_urban_territorial",
    "has_sentinel2_spectral", "has_rainfall_trigger", "has_documentary_evidence",
    "has_sar_observational", "has_embedding_representation", "missing_layers",
    "available_layer_count", "missing_layer_count", "readiness_status",
]


def build_missingness_matrix() -> list[dict]:
    rows = []
    for row in read_csv(CANARY_MATRIX_17C6):
        missing = [x for x in row["missing_layers"].split(";") if x]
        available = [layer for layer in MULTIMODAL_LAYERS if layer not in missing]
        rows.append({
            "candidate_patch_id": row["candidate_patch_id"],
            "has_physical_static": row["has_physical_static_features"],
            "has_urban_territorial": row["has_urban_territorial_features"],
            "has_sentinel2_spectral": row["has_sentinel2_spectral_features"],
            "has_rainfall_trigger": row["has_rainfall_trigger_features"],
            "has_documentary_evidence": row["has_documentary_evidence"],
            "has_sar_observational": row["has_sar_observational_footprint"],
            "has_embedding_representation": "false",
            "missing_layers": ";".join(missing),
            "available_layer_count": str(len(available)),
            "missing_layer_count": str(len(missing)),
            "readiness_status": "PIPELINES_EXIST_BUT_NOT_RUN_FOR_CANDIDATES",
        })
    return rows


EMBEDDING_FIELDS = [
    "candidate_patch_id", "has_real_tile", "tile_source", "tile_date",
    "tile_modality", "tile_bands", "tile_resolution_m", "tile_cloud_status",
    "embedding_model_target", "embedding_input_ready", "embedding_can_run_now",
    "blocking_reason", "synthetic_embedding_exists_from_17c6",
    "synthetic_embedding_usable_for_science", "review_only", "trainable",
    "ground_truth",
]


def build_embedding_readiness() -> list[dict]:
    synthetic_ids = {row["candidate_patch_id"] for row in read_csv(SYNTHETIC_EMBEDDING_17C6)}
    rows = []
    for patch in read_csv(CANDIDATE_GRID_17C6):
        rows.append({
            "candidate_patch_id": patch["candidate_patch_id"],
            "has_real_tile": "false",
            "tile_source": "not_available",
            "tile_date": "not_available",
            "tile_modality": "not_available",
            "tile_bands": "not_available",
            "tile_resolution_m": "not_available",
            "tile_cloud_status": "not_available",
            "embedding_model_target": "DINOv2;SatMAE;Scale-MAE",
            "embedding_input_ready": "false",
            "embedding_can_run_now": "false",
            "blocking_reason": "no_real_tile_for_candidate_patch",
            "synthetic_embedding_exists_from_17c6": _bool_text(patch["candidate_patch_id"] in synthetic_ids),
            "synthetic_embedding_usable_for_science": "false",
            **_gov_str(),
        })
    return rows


def build_no_leakage_policy() -> dict:
    return {
        "policy_id": "SUSC_17C7_NO_LEAKAGE_POLICY_V1",
        "milestone_name": MILESTONE_NAME,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "blocked_uses": {
            "post_event_footprint_as_feature": True,
            "post_event_sar_as_pre_event_feature": True,
            "documentary_evidence_as_physical_susceptibility_proxy": True,
            "synthetic_placeholder_as_real_feature": True,
            "candidate_patch_as_official_patch": True,
            "REC_00019_as_SUSC_patch": True,
            "promotion_to_17b_without_qa_accepted": True,
            "score_v7_without_readiness": True,
            "training_without_future_ground_truth_policy": True,
        },
        "allowed_now": [
            "review_only_feature_inventory",
            "review_only_extraction_plan",
            "missingness_matrix",
            "embedding_input_readiness_without_model_execution",
        ],
        "raw_raster_policy": "raw_rasters_must_stay_local_only_and_must_not_be_committed",
    }


BLOCKER_FIELDS = [
    "blocker_id", "blocker_code", "description", "blocks_17b",
    "blocks_training", "blocks_score_v7", "required_resolution",
    "review_only", "trainable", "ground_truth",
]


def build_promotion_blockers() -> list[dict]:
    items = [
        ("candidate_patches_not_official", "Patches candidatos nao pertencem a grade oficial SUSC.", "official_patch_or_candidate_grid_policy"),
        ("features_not_extracted_for_candidate_patches", "Features fisicas, urbanas, espectrais e chuva ainda nao foram extraidas para candidatos.", "controlled_feature_extraction_for_candidate_patches"),
        ("patch_policy_missing", "Nao existe politica formal para uso cientifico de grade candidata.", "candidate_patch_policy"),
        ("no_real_embedding_tile", "Nao ha tile real para DINOv2 ou SatMAE.", "real_tile_materialization"),
        ("no_real_embedding_vector", "Nao ha embedding real para patches candidatos.", "real_model_execution_after_tile_readiness"),
        ("no_sar_runtime", "Runtime SAR segue indisponivel.", "SAR_runtime_and_credentials"),
        ("no_qa_accepted", "Vinculos candidatos ainda dependem de QA humana.", "human_QA_accepted"),
        ("score_policy_missing_for_candidate_grid", "Nao existe politica de score para grade candidata.", "separate_score_policy"),
        ("p0_official_packages_missing", "Pacotes formais P0 seguem ausentes.", "formal_packages_or_blocker_update"),
        ("17b_blocked_until_real_features_and_patch_policy", "17B exige politica, QA aceito e features reais.", "patch_policy_QA_and_real_features"),
    ]
    rows = []
    for idx, (code, desc, resolution) in enumerate(items, start=1):
        rows.append({
            "blocker_id": f"S17C7_BLOCKER_{idx:03d}",
            "blocker_code": code,
            "description": desc,
            "blocks_17b": "true",
            "blocks_training": "true",
            "blocks_score_v7": "true",
            "required_resolution": resolution,
            **_gov_str(),
        })
    return rows


def build_summary(inventory: list[dict], tasks: list[dict], missingness: list[dict], embedding: list[dict], blockers: list[dict]) -> dict:
    return {
        "milestone_name": MILESTONE_NAME,
        "milestone_category": MILESTONE_CATEGORY,
        "candidate_patch_count": len(read_csv(CANDIDATE_GRID_17C6)),
        "feature_rows_count": len(inventory),
        "features_available_now_count": sum(1 for row in inventory if row["available_now"] == "true"),
        "features_requiring_existing_pipeline_adaptation_count": sum(1 for row in inventory if row["available_from_existing_official_patch_pipeline"] == "true" and row["requires_new_extraction"] == "true"),
        "features_requiring_external_download_count": sum(1 for row in inventory if row["requires_external_download"] == "true"),
        "features_requiring_runtime_credentials_count": sum(1 for row in inventory if row["requires_runtime_credentials"] == "true"),
        "features_blocked_by_patch_policy_count": sum(1 for row in inventory if row["available_from_existing_official_patch_pipeline"] == "true" and row["available_now"] == "false"),
        "candidate_patches_ready_for_real_feature_extraction_count": 0,
        "candidate_patches_documentary_only_count": sum(1 for row in missingness if row["has_documentary_evidence"] == "true" and row["available_layer_count"] == "1"),
        "real_embedding_input_ready_count": sum(1 for row in embedding if row["embedding_input_ready"] == "true"),
        "synthetic_embedding_only_count": sum(1 for row in embedding if row["synthetic_embedding_exists_from_17c6"] == "true" and row["has_real_tile"] == "false"),
        "eligible_for_17b_now": False,
        "eligible_for_training": False,
        "eligible_for_score_v7": False,
        "score_v6_changed": False,
        "score_v7_created": False,
        "official_patch_created": False,
        "official_patch_link_created": False,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "promotion_blocker_count": len(blockers),
        "recommended_next_milestone": "SUSC-17C8 Extração Real Controlada de Features para Patches Candidatos",
    }


def build_report(summary: dict) -> None:
    text = f"""# SUSC-17C7 — Plano de Extração de Features para Patches Candidatos

Status: somente para revisão. `review_only=true`; `trainable=false`; `ground_truth=false`; `score_v6_changed=false`; `score_v7_created=false`; `official_patch_created=false`; `official_patch_link_created=false`; `eligible_for_17b_now=false`.

## O que o 17C6 demonstrou

O SUSC-17C6 demonstrou que 5 patches candidatos podem ser representados por uma grade candidata separada e por vínculos candidatos patch-evento, sem alterar a grade oficial SUSC. O 17C6 também mostrou que apenas a camada `documentary_evidence` existe para os candidatos; as demais camadas ficaram ausentes ou dependentes de extração futura.

## Por que o 17C7 é plano, não extração completa

Este marco não extrai raster, não baixa Sentinel, não executa SAR, não executa DINOv2/SatMAE e não preenche valores físicos, espectrais ou chuvosos sintéticos. Ele apenas mapeia quais tarefas seriam necessárias para converter os patches candidatos em unidades multimodais reais.

## Camadas existentes para patches oficiais

O dataset oficial `datasets/suscetibilidade/susc_features_by_patch_v1.csv` contém colunas para camadas físicas, urbanas, espectrais Sentinel-2 e chuva/runoff em patches oficiais. Isso indica pipeline ou produto existente para a grade oficial, mas não significa que essas features existam para os patches candidatos.

## Camadas ausentes para patches candidatos

Para os candidatos, seguem ausentes `physical_static`, `urban_territorial`, `sentinel2_spectral`, `rainfall_trigger`, `sar_observational` e `embedding_representation`. O plano registrou `{summary['feature_rows_count']}` linhas de inventário de feature para `{summary['candidate_patch_count']}` patches candidatos.

## Pipelines adaptáveis

As camadas físicas, urbanas e de chuva podem ser planejadas como adaptação de pipeline existente para a grade candidata. Essa adaptação ainda depende de política para candidatos e execução controlada, sem promover patch candidato a patch oficial.

## Dependências de raster, API ou credencial

Sentinel-2 depende de tile/cena real e lineage de aquisição. SAR depende de runtime e credenciais. Embedding DINOv2/SatMAE depende primeiro de tile real e política de normalização, composição de bandas e nuvem.

## DINO/SatMAE real

Não há tile real pronto para os 5 patches candidatos. Portanto `embedding_input_ready=false`, `embedding_can_run_now=false` e nenhum embedding real foi executado. O smoke test sintético herdado do 17C6 continua não científico.

## SAR real

SAR permanece bloqueado por runtime indisponível e por ausência de execução controlada pré/pós evento para a grade candidata. Footprint pós-evento não pode ser usado como feature de suscetibilidade pré-evento.

## Uso científico futuro

Para tornar os patches candidatos cientificamente utilizáveis, ainda faltam política de grade candidata, QA humano, extração real de features, tile real, embedding real, controle anti-leakage e decisão explícita sobre score.

## Score v6, score v7 e 17B

O score v6 não foi alterado. O score v7 continua inexistente. O 17B segue bloqueado porque não há patch oficial ou política aprovada, QA aceito, features reais completas nem evidência observacional suficiente.

## Próximo marco recomendado

`{summary['recommended_next_milestone']}`. Se o maior bloqueio operacional for embedding, a alternativa é `SUSC-17C8 Preparação de Tile Real / Entrada DINO`. Se o maior bloqueio institucional continuar sendo P0, a alternativa é `SUSC-17C8 Pacote de Solicitação Formal`.
"""
    write_markdown(REPORT, text)


def run_all() -> int:
    _require_inputs()
    inventory = build_feature_inventory()
    write_csv(FEATURE_INVENTORY, inventory, INVENTORY_FIELDS)
    mapping = build_source_mapping()
    write_csv(SOURCE_MAPPING, mapping, SOURCE_MAPPING_FIELDS)
    tasks = build_task_plan()
    write_csv(TASK_PLAN, tasks, TASK_FIELDS)
    missing = build_missingness_matrix()
    write_csv(MISSINGNESS, missing, MISSINGNESS_FIELDS)
    embedding = build_embedding_readiness()
    write_csv(EMBEDDING_READINESS, embedding, EMBEDDING_FIELDS)
    write_json(NO_LEAKAGE_POLICY, build_no_leakage_policy())
    blockers = build_promotion_blockers()
    write_csv(BLOCKERS, blockers, BLOCKER_FIELDS)
    summary = build_summary(inventory, tasks, missing, embedding, blockers)
    write_json(SUMMARY, summary)
    build_report(summary)
    print(
        "17C7 -> "
        f"candidate_patches={summary['candidate_patch_count']} "
        f"feature_rows={summary['feature_rows_count']} "
        f"embedding_ready={summary['real_embedding_input_ready_count']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
    return 0


def _schema_violations(row: dict, schema: dict) -> list[str]:
    out = []
    props = schema.get("properties", {})
    for field in schema.get("required", []):
        if field not in row or not str(row.get(field, "")).strip():
            out.append(f"required empty: {field}")
    for field, spec in props.items():
        if field not in row:
            continue
        value = row.get(field, "")
        if "const" in spec and value != spec["const"]:
            out.append(f"{field}={value} must be {spec['const']}")
        if "enum" in spec and value not in spec["enum"]:
            out.append(f"{field}={value} not in enum")
    return out


def _gov_violations(row: dict, label: str) -> list[str]:
    out = []
    if row.get("review_only") != "true":
        out.append(f"{label}: review_only not true")
    if row.get("trainable") != "false":
        out.append(f"{label}: trainable not false")
    if row.get("ground_truth") != "false":
        out.append(f"{label}: ground_truth not false")
    return out


def _git_diff_name_only(path: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", rel(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip()


def _file_hashes(paths: list[Path]) -> dict[str, str]:
    return {rel(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def validate() -> int:
    errors: list[str] = []
    for path in [*REQUIRED_INPUTS, *REQUIRED_OUTPUTS]:
        if not path.exists():
            errors.append(f"missing: {rel(path)}")
    if errors:
        print("FAILED: SUSC-17C7 validation")
        for err in errors:
            print("  ERROR:", err)
        return 1

    inventory_schema = read_json(INVENTORY_SCHEMA)
    task_schema = read_json(TASK_SCHEMA)
    inventory = read_csv(FEATURE_INVENTORY)
    mapping = read_csv(SOURCE_MAPPING)
    tasks = read_csv(TASK_PLAN)
    missing = read_csv(MISSINGNESS)
    embedding = read_csv(EMBEDDING_READINESS)
    blockers = read_csv(BLOCKERS)
    summary = read_json(SUMMARY)
    policy = read_json(NO_LEAKAGE_POLICY)
    candidate_ids = [row["candidate_patch_id"] for row in read_csv(CANDIDATE_GRID_17C6)]

    for row_no, row in enumerate(inventory, start=2):
        for err in _schema_violations(row, inventory_schema):
            errors.append(f"{FEATURE_INVENTORY.name}:{row_no}: {err}")
        errors.extend(_gov_violations(row, FEATURE_INVENTORY.name))
        if row["available_now"] == "true" and row["feature_layer"] != "documentary_evidence":
            errors.append("non-documentary candidate feature marked available_now")
        if row["synthetic_placeholder"] == "true" and row["usable_for_science_now"] != "false":
            errors.append("synthetic placeholder marked scientific")
        if row["candidate_patch_id"] == REC_00019:
            errors.append("REC_00019 appears as candidate patch id")

    for row_no, row in enumerate(tasks, start=2):
        for err in _schema_violations(row, task_schema):
            errors.append(f"{TASK_PLAN.name}:{row_no}: {err}")
        errors.extend(_gov_violations(row, TASK_PLAN.name))
        if row["must_not_commit_raw_raster"] != "true":
            errors.append("task allows committing raw raster")
        if row["task_type"] == "reuse_existing_pipeline" and row["can_run_now"] != "false":
            errors.append("task claims direct reuse can run now")

    if sorted({row["candidate_patch_id"] for row in missing}) != sorted(candidate_ids):
        errors.append("missingness matrix does not contain all 17C6 candidate patches")
    for row in missing:
        listed = {x for x in row["missing_layers"].split(";") if x}
        if row["has_documentary_evidence"] != "true":
            errors.append("documentary evidence missing in missingness matrix")
        for layer in ("physical_static", "urban_territorial", "sentinel2_spectral", "rainfall_trigger", "sar_observational", "embedding_representation"):
            if layer not in listed:
                errors.append(f"missingness does not list {layer}")

    for row in embedding:
        errors.extend(_gov_violations(row, EMBEDDING_READINESS.name))
        if row["has_real_tile"] != "false":
            errors.append("embedding readiness found unexpected real tile")
        if row["embedding_input_ready"] != "false" or row["embedding_can_run_now"] != "false":
            errors.append("embedding readiness not blocked without real tile")
        if row["synthetic_embedding_usable_for_science"] != "false":
            errors.append("synthetic embedding marked scientific")

    if {row["feature_layer"] for row in mapping} != set(MULTIMODAL_LAYERS):
        errors.append("source mapping missing mandatory layers")
    if len({(row["candidate_patch_id"], row["feature_name"]) for row in inventory}) != len(inventory):
        errors.append("feature inventory ids not unique by patch+feature")
    if len({row["extraction_task_id"] for row in tasks}) != len(tasks):
        errors.append("task ids not unique")

    blocked = policy.get("blocked_uses", {})
    for key in (
        "post_event_footprint_as_feature",
        "post_event_sar_as_pre_event_feature",
        "documentary_evidence_as_physical_susceptibility_proxy",
        "synthetic_placeholder_as_real_feature",
        "candidate_patch_as_official_patch",
        "REC_00019_as_SUSC_patch",
        "promotion_to_17b_without_qa_accepted",
        "score_v7_without_readiness",
        "training_without_future_ground_truth_policy",
    ):
        if blocked.get(key) is not True:
            errors.append(f"anti-leakage policy does not block {key}")

    expected_blockers = {
        "candidate_patches_not_official",
        "features_not_extracted_for_candidate_patches",
        "patch_policy_missing",
        "no_real_embedding_tile",
        "no_real_embedding_vector",
        "no_sar_runtime",
        "no_qa_accepted",
        "score_policy_missing_for_candidate_grid",
        "p0_official_packages_missing",
        "17b_blocked_until_real_features_and_patch_policy",
    }
    if {row["blocker_code"] for row in blockers} != expected_blockers:
        errors.append("promotion blockers set mismatch")
    for row in blockers:
        errors.extend(_gov_violations(row, BLOCKERS.name))

    if summary.get("candidate_patch_count") != len(candidate_ids):
        errors.append("summary candidate patch count mismatch")
    if summary.get("feature_rows_count") != len(inventory):
        errors.append("summary feature row count mismatch")
    for key in (
        "eligible_for_17b_now", "eligible_for_training", "eligible_for_score_v7",
        "score_v6_changed", "score_v7_created", "official_patch_created",
        "official_patch_link_created", "trainable", "ground_truth",
    ):
        if summary.get(key) is not False:
            errors.append(f"summary {key} not false")
    if summary.get("review_only") is not True:
        errors.append("summary review_only not true")

    if SCORE_V7.exists():
        errors.append("score v7 artifact exists")
    if _git_diff_name_only(SCORE_V6):
        errors.append("score v6 has git diff")
    if _git_diff_name_only(FEATURES):
        errors.append("official patch dataset has git diff")

    before = _file_hashes(REQUIRED_OUTPUTS)
    run_all()
    after = _file_hashes(REQUIRED_OUTPUTS)
    if before != after:
        errors.append("build is not byte-identical across two runs")

    report_text = REPORT.read_text(encoding="utf-8")
    for phrase in (
        "# SUSC-17C7 — Plano de Extração de Features para Patches Candidatos",
        "O score v6 não foi alterado",
        "O score v7 continua inexistente",
        "não executa DINOv2/SatMAE",
    ):
        if phrase not in report_text:
            errors.append(f"report missing phrase: {phrase}")

    if errors:
        print("FAILED: SUSC-17C7 validation")
        for err in errors:
            print("  ERROR:", err)
        return 1
    print("PASSED: SUSC-17C7 validations passed")
    return 0
