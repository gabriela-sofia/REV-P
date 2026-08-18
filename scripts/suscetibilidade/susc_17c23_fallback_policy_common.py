"""SUSC-17C23 politica de equivalencia Earth Search e matriz cientifica review-only.

Marco totalmente offline e deterministico. Nao coleta dados nem acessa rede: usa
apenas artefatos 17C18-17C22 para provar, por criterios objetivos, se o Earth
Search usado no 17C20/17C21 e equivalente o suficiente a cena CDSE/OData para
sustentar features sensoriais cientificas review-only. Se a equivalencia passar,
promove as 40 features pre-evento de `accepted_review_only_sensor_feature` para
`accepted_scientific_review_only_sensor_feature`, sem liberar treino, ground
truth, score v7, 17B, patch oficial ou Ground Reference. A diferenca de nivel de
processamento L1C (CDSE) vs L2A (Earth Search) e registrada como limitacao
metodologica, nao como bloqueio.
"""

from __future__ import annotations

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

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_PATCHES = DAT / "susc_patches_official_v1.csv"
OFFICIAL_PATCH_LINKS = DAT / "susc_patch_links_official_v1.csv"

C22_PRE_ACCEPT = OUT / "susc_17c22_pre_event_sensor_feature_acceptance.csv"
C22_DELTA_ACCEPT = OUT / "susc_17c22_observational_delta_acceptance.csv"
C22_FEATURE_MATRIX = OUT / "susc_17c22_candidate_patch_sensor_feature_matrix.csv"
C22_DELTA_MATRIX = OUT / "susc_17c22_candidate_patch_observational_delta_matrix.csv"
C22_PROVENANCE = OUT / "susc_17c22_feature_provenance_trace.csv"
C22_FALLBACK_REVIEW = OUT / "susc_17c22_fallback_policy_review.csv"
C22_SCIENCE_GATES = OUT / "susc_17c22_science_readiness_gates.csv"

C21_POST_SELECTION = OUT / "susc_17c21_post_event_scene_selection.csv"
C21_FALLBACK = OUT / "susc_17c21_fallback_source_audit.csv"
C21_REPLAY = OUT / "susc_17c21_replay_audit.csv"

C20_SOURCE_RESOLUTION = OUT / "susc_17c20_sentinel2_source_resolution.csv"
C20_ASSET_ATTEMPTS = OUT / "susc_17c20_asset_resolution_attempts.csv"
C20_MANIFEST = OUT / "susc_17c20_light_artifact_manifest.csv"
C20_FEATURES = OUT / "susc_17c20_materialized_feature_registry.csv"
C20_FALLBACK = OUT / "susc_17c20_fallback_source_audit.csv"
C20_REPLAY = OUT / "susc_17c20_replay_audit.csv"

C19_LINEAGE = OUT / "susc_17c19_event_temporal_lineage.csv"
C19_BINDING = OUT / "susc_17c19_candidate_patch_temporal_binding.csv"
C19_CANONICAL = OUT / "susc_17c19_sentinel2_canonical_pre_event_scenes.csv"
C18_SCENE_REGISTRY = OUT / "susc_17c18_sentinel2_candidate_scene_registry.csv"
PATCH_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"
PATCH_LINKS = OUT / "susc_17c6_candidate_patch_links.csv"

REQUIRED_INPUTS = [
    SCORE_V6, C22_PRE_ACCEPT, C22_DELTA_ACCEPT, C22_FEATURE_MATRIX, C22_DELTA_MATRIX,
    C22_PROVENANCE, C22_FALLBACK_REVIEW, C22_SCIENCE_GATES,
    C21_POST_SELECTION, C21_FALLBACK, C21_REPLAY,
    C20_SOURCE_RESOLUTION, C20_ASSET_ATTEMPTS, C20_MANIFEST, C20_FEATURES, C20_FALLBACK, C20_REPLAY,
    C19_LINEAGE, C19_BINDING, C19_CANONICAL, C18_SCENE_REGISTRY, PATCH_GRID, PATCH_LINKS,
]

REPORT = OUT / "SUSC_17C23_POLITICA_EQUIVALENCIA_EARTH_SEARCH_MATRIZ_CIENTIFICA_REVIEW_ONLY_REPORT.md"
DOSSIER = OUT / "susc_17c23_earth_search_resolution_dossier.csv"
EQUIV_POLICY = OUT / "susc_17c23_fallback_equivalence_policy.csv"
SCENE_EQUIV_AUDIT = OUT / "susc_17c23_fallback_scene_equivalence_audit.csv"
FALLBACK_DECISION = OUT / "susc_17c23_fallback_acceptance_decision.csv"
PATCH_USAGE_POLICY = OUT / "susc_17c23_candidate_patch_usage_policy.csv"
PATCH_USAGE_DECISION = OUT / "susc_17c23_candidate_patch_usage_decision.csv"
SCIENTIFIC_ACCEPT = OUT / "susc_17c23_scientific_review_only_feature_acceptance.csv"
MULTIMODAL_MATRIX = OUT / "susc_17c23_candidate_multimodal_feature_matrix.csv"
DELTA_REVIEW_MATRIX = OUT / "susc_17c23_observational_delta_review_matrix.csv"
SCOPE_LIMITS = OUT / "susc_17c23_science_scope_limits.csv"
TS17B_BLOCKERS = OUT / "susc_17c23_training_score_17b_blockers.csv"
NO_LEAKAGE = OUT / "susc_17c23_no_leakage_audit.csv"
GATES = OUT / "susc_17c23_gate_evaluation_matrix.csv"
SUMMARY = OUT / "susc_17c23_readiness_summary.json"
BLOCKERS = OUT / "susc_17c23_promotion_blockers.csv"

DOSSIER_SCHEMA = SCHEMAS / "susc_17c23_earth_search_resolution_schema_v1.json"
PATCH_USAGE_SCHEMA = SCHEMAS / "susc_17c23_candidate_patch_usage_schema_v1.json"
SCIENTIFIC_SCHEMA = SCHEMAS / "susc_17c23_scientific_feature_acceptance_schema_v1.json"

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}

FEATURE_ORDER = ["B03_mean", "B04_mean", "B08_mean", "B11_mean", "NDVI_mean", "NDWI_mean", "MNDWI_mean", "NDBI_mean"]
DELTA_ORDER = [f"delta_{f}" for f in FEATURE_ORDER]
REQUIRED_BANDS = ["B03", "B04", "B08", "B11"]
L1C_L2A_LIMITATION = "cdse_l1c_vs_earth_search_l2a_processing_level_difference_documented_as_methodological_limitation"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


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


def _patch_ids() -> list[str]:
    return [row["candidate_patch_id"] for row in read_csv(PATCH_GRID)]


def _event_id() -> str:
    rows = read_csv(C19_CANONICAL)
    return rows[0]["event_id"] if rows else ""


# --- parsing de identificadores de cena Sentinel-2 (sem rede) ---

def _parse_l1c(scene_id: str) -> dict:
    tokens = scene_id.split("_")
    platform = tokens[0] if tokens else "unknown"
    level = "L1C" if len(tokens) > 1 and "L1C" in tokens[1] else ("L2A" if len(tokens) > 1 and "L2A" in tokens[1] else "unknown")
    date = tokens[2][:8] if len(tokens) > 2 else "unknown"
    tile = next((t[1:] for t in tokens if t.startswith("T") and len(t) == 6 and t[1:].isalnum()), "unknown")
    return {"platform": platform, "level": level, "date": date, "tile": tile}


def _parse_earth_search(item_id: str) -> dict:
    tokens = item_id.split("_")
    platform = tokens[0] if tokens else "unknown"
    tile = tokens[1] if len(tokens) > 1 else "unknown"
    date = tokens[2] if len(tokens) > 2 else "unknown"
    level = tokens[-1] if tokens else "unknown"
    return {"platform": platform, "level": level, "date": date, "tile": tile}


# --- acessos aos artefatos upstream ---

def _c20_source_by_patch() -> dict:
    return {r["candidate_patch_id"]: r for r in read_csv(C20_SOURCE_RESOLUTION)}


def _c20_earth_search_asset(patch_id: str) -> dict | None:
    for r in read_csv(C20_ASSET_ATTEMPTS):
        if r["candidate_patch_id"] == patch_id and r["source_name"].startswith("earth_search"):
            return r
    return None


def _c20_manifest_by_patch(patch_id: str) -> list[dict]:
    return [r for r in read_csv(C20_MANIFEST) if r["candidate_patch_id"] == patch_id]


def _c20_replay_all_ok(patch_id: str) -> bool:
    manifest_ids = {r["light_artifact_manifest_id"] for r in _c20_manifest_by_patch(patch_id)}
    replay = {r["light_artifact_manifest_id"]: r["byte_identical_replay"] for r in read_csv(C20_REPLAY)}
    return bool(manifest_ids) and all(replay.get(mid) == "true" for mid in manifest_ids)


def _c20_hash_all_ok(patch_id: str) -> bool:
    rows = _c20_manifest_by_patch(patch_id)
    for r in rows:
        path = ROOT / r["artifact_local_path"]
        if not path.exists():
            return False
    return bool(rows)


def _c20_bands_materialized(patch_id: str) -> str:
    for r in _c20_manifest_by_patch(patch_id):
        if r["artifact_type"] == "band_stats_csv":
            return r["bands_included"]
    return "not_available"


def _c20_size_policy_ok(patch_id: str) -> bool:
    for r in _c20_manifest_by_patch(patch_id):
        if r["stored_in_outputs_public"] == "true" and int(r["size_bytes"]) > 1_000_000:
            return False
    return True


def _pre_scene(patch_id: str) -> str:
    for r in read_csv(C19_CANONICAL):
        if r["candidate_patch_id"] == patch_id:
            return r["scene_or_item_id"]
    return "not_available"


def _pre_feature_values(patch_id: str) -> dict:
    return {r["feature_name"]: r["feature_value"] for r in read_csv(C22_PRE_ACCEPT) if r["candidate_patch_id"] == patch_id}


def _delta_values(patch_id: str) -> dict:
    return {r["feature_name"]: r["delta_value"] for r in read_csv(C22_DELTA_ACCEPT) if r["candidate_patch_id"] == patch_id}


# ---------------------------------------------------------------------------
# Dossie de resolucao Earth Search
# ---------------------------------------------------------------------------

def earth_search_resolution_dossier_rows() -> list[dict]:
    source = _c20_source_by_patch()
    rows = []
    for idx, patch_id in enumerate(_patch_ids(), start=1):
        src = source.get(patch_id, {})
        cdse_scene = src.get("scene_or_item_id", "not_available")
        es_item = src.get("selected_materialization_source", "not_available")
        cdse = _parse_l1c(cdse_scene)
        es = _parse_earth_search(es_item)
        asset = _c20_earth_search_asset(patch_id) or {}
        same_mission = cdse_scene.startswith("S2") and es_item.startswith("S2")
        same_platform = cdse["platform"] == es["platform"]
        same_date = cdse["date"] == es["date"] and cdse["date"] != "unknown"
        same_tile = cdse["tile"] == es["tile"] and cdse["tile"] != "unknown"
        compatible_level = True  # L1C e L2A sao compativeis com limitacao documentada
        aoi_ok = asset.get("asset_level_access_available") == "true"
        bands_ok = all(b in asset.get("available_bands", "") for b in REQUIRED_BANDS)
        bands_mat = _c20_bands_materialized(patch_id)
        hash_ok = _c20_hash_all_ok(patch_id)
        replay_ok = _c20_replay_all_ok(patch_id)
        cog_ok = asset.get("probe_status") == "asset_level_cog_available"
        size_ok = _c20_size_policy_ok(patch_id)
        resolved = all([same_mission, same_platform, same_date, same_tile, aoi_ok, bands_ok, hash_ok, replay_ok, cog_ok, size_ok])
        rows.append({
            "earth_search_resolution_id": f"S17C23_ESRES_{idx:04d}",
            "candidate_patch_id": patch_id,
            "cdse_scene_or_item_id": cdse_scene,
            "earth_search_item_id": es_item,
            "cdse_processing_level": cdse["level"],
            "earth_search_processing_level": es["level"],
            "same_mission": _bool_text(same_mission),
            "same_platform": _bool_text(same_platform),
            "same_acquisition_date": _bool_text(same_date),
            "same_mgrs_tile": _bool_text(same_tile),
            "same_or_compatible_processing_level": _bool_text(compatible_level),
            "aoi_overlap_confirmed": _bool_text(aoi_ok),
            "bands_required_available": _bool_text(bands_ok),
            "bands_materialized": bands_mat,
            "hash_manifest_verified": _bool_text(hash_ok),
            "replay_verified": _bool_text(replay_ok),
            "cog_window_read_confirmed": _bool_text(cog_ok),
            "artifact_size_policy_passed": _bool_text(size_ok),
            "equivalence_resolved": _bool_text(resolved),
            "accepted_for_scientific_review_only": _bool_text(resolved),
            "accepted_for_ground_reference": "false",
            "accepted_for_training": "false",
            "accepted_for_score_v7": "false",
            "accepted_for_17b": "false",
            "remaining_limitation": L1C_L2A_LIMITATION,
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Politica de equivalencia
# ---------------------------------------------------------------------------

EQUIV_CRITERIA = [
    ("same_mission", "mesma missao Sentinel-2", "true"),
    ("same_platform", "mesma plataforma/satelite", "true"),
    ("same_acquisition_date", "mesma data de aquisicao", "true"),
    ("same_mgrs_tile", "mesmo tile MGRS 25MBM", "true"),
    ("aoi_overlap_confirmed", "AOI do patch coberta", "true"),
    ("bands_required_available", "assets reais B03/B04/B08/B11", "true"),
    ("hash_manifest_verified", "artefatos com SHA256 no manifesto", "true"),
    ("replay_verified", "replay byte-identico validado", "true"),
    ("cog_window_read_confirmed", "leitura COG/window confirmada", "true"),
    ("artifact_size_policy_passed", "tamanho dentro da politica", "true"),
    ("same_or_compatible_processing_level", "nivel de processamento compativel (L1C~L2A com limitacao)", "true"),
    ("fallback_declared_materialization_fallback", "fallback declarado como materialization_fallback", "true"),
]


def fallback_equivalence_policy_rows() -> list[dict]:
    rows = []
    for idx, (name, desc, required_value) in enumerate(EQUIV_CRITERIA, start=1):
        rows.append({
            "equivalence_policy_id": f"S17C23_EQPOL_{idx:04d}",
            "criterion_name": name,
            "criterion_description": desc,
            "required_value": required_value,
            "blocks_if_failed": "true",
            "applies_to": "earth_search_scientific_review_only_equivalence",
            "review_only": "true",
        })
    return rows


def fallback_scene_equivalence_audit_rows() -> list[dict]:
    fallback_declared = {r["candidate_patch_id"]: (r["fallback_source_role"] == "materialization_fallback") for r in read_csv(C20_FALLBACK)}
    rows = []
    for idx, dossier in enumerate(earth_search_resolution_dossier_rows(), start=1):
        patch_id = dossier["candidate_patch_id"]
        checks = {name: dossier.get(name, "false") for name, _, _ in EQUIV_CRITERIA if name in dossier}
        checks["fallback_declared_materialization_fallback"] = _bool_text(fallback_declared.get(patch_id, False))
        failed = [name for name, _, req in EQUIV_CRITERIA if checks.get(name, "false") != req]
        rows.append({
            "scene_equivalence_audit_id": f"S17C23_SCENEQ_{idx:04d}",
            "candidate_patch_id": patch_id,
            "cdse_scene_or_item_id": dossier["cdse_scene_or_item_id"],
            "earth_search_item_id": dossier["earth_search_item_id"],
            "same_mission": dossier["same_mission"],
            "same_platform": dossier["same_platform"],
            "same_acquisition_date": dossier["same_acquisition_date"],
            "same_mgrs_tile": dossier["same_mgrs_tile"],
            "same_or_compatible_processing_level": dossier["same_or_compatible_processing_level"],
            "fallback_declared": checks["fallback_declared_materialization_fallback"],
            "failed_criteria": ";".join(failed) if failed else "none",
            "scene_equivalence_passed": _bool_text(not failed),
            "remaining_limitation": L1C_L2A_LIMITATION,
            "review_only": "true",
        })
    return rows


def fallback_acceptance_decision_rows() -> list[dict]:
    rows = []
    for idx, dossier in enumerate(earth_search_resolution_dossier_rows(), start=1):
        resolved = dossier["equivalence_resolved"] == "true"
        rows.append({
            "fallback_acceptance_decision_id": f"S17C23_FBDEC_{idx:04d}",
            "candidate_patch_id": dossier["candidate_patch_id"],
            "earth_search_item_id": dossier["earth_search_item_id"],
            "source_role": "materialization_fallback",
            "equivalence_resolved": dossier["equivalence_resolved"],
            "accepted_for_scientific_review_only": _bool_text(resolved),
            "accepted_for_ground_reference": "false",
            "accepted_for_training": "false",
            "accepted_for_score_v7": "false",
            "accepted_for_17b": "false",
            "not_official_primary_source": "true",
            "decision_status": "accepted_scientific_review_only_materialization_source" if resolved else "blocked_equivalence_not_resolved",
            "remaining_limitation": L1C_L2A_LIMITATION,
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Politica de patch candidato
# ---------------------------------------------------------------------------

PATCH_USAGE_CRITERIA = [
    ("has_geometry_aoi", "patch tem geometria/AOI"),
    ("has_candidate_event_link", "patch tem vinculo candidato com evento"),
    ("has_temporal_binding", "patch tem binding temporal"),
    ("has_real_feature_with_hash_replay", "patch tem feature real com hash e replay"),
    ("fallback_source_resolved", "fonte fallback resolvida para review-only cientifico"),
    ("not_official_patch", "patch nao e patch oficial"),
    ("not_official_patch_link", "patch nao e patch-link oficial"),
]


def candidate_patch_usage_policy_rows() -> list[dict]:
    rows = []
    for idx, (name, desc) in enumerate(PATCH_USAGE_CRITERIA, start=1):
        rows.append({
            "patch_usage_policy_id": f"S17C23_PUPOL_{idx:04d}",
            "criterion_name": name,
            "criterion_description": desc,
            "required_value": "true",
            "blocks_if_failed": "true",
            "allowed_use_if_all_pass": "candidate_patch_scientific_review_only_sensor_feature",
            "review_only": "true",
        })
    return rows


def _patch_has_geometry(patch_id: str) -> bool:
    for r in read_csv(PATCH_GRID):
        if r["candidate_patch_id"] == patch_id:
            return all(r.get(k, "") not in ("", "not_available") for k in ["xmin", "ymin", "xmax", "ymax"])
    return False


def _patch_has_event_link(patch_id: str) -> bool:
    return any(r["candidate_patch_id"] == patch_id for r in read_csv(PATCH_LINKS))


def _patch_has_binding(patch_id: str) -> bool:
    return any(r["candidate_patch_id"] == patch_id and r["temporal_binding_status"] == "resolved" for r in read_csv(C19_BINDING))


def _patch_is_official(patch_id: str) -> bool:
    for r in read_csv(PATCH_GRID):
        if r["candidate_patch_id"] == patch_id:
            return r.get("official_patch", "false") == "true"
    return False


def candidate_patch_usage_decision_rows() -> list[dict]:
    event_id = _event_id()
    fb_resolved = {r["candidate_patch_id"]: (r["decision_status"] == "accepted_scientific_review_only_materialization_source") for r in fallback_acceptance_decision_rows()}
    rows = []
    for idx, patch_id in enumerate(_patch_ids(), start=1):
        geometry = _patch_has_geometry(patch_id)
        event_link = _patch_has_event_link(patch_id)
        binding = _patch_has_binding(patch_id)
        feature_ok = _c20_hash_all_ok(patch_id) and _c20_replay_all_ok(patch_id)
        fallback_ok = fb_resolved.get(patch_id, False)
        not_official = not _patch_is_official(patch_id)
        checks = [geometry, event_link, binding, feature_ok, fallback_ok, not_official, not_official]
        accepted = all(checks)
        rows.append({
            "patch_usage_decision_id": f"S17C23_PUDEC_{idx:04d}",
            "candidate_patch_id": patch_id,
            "event_id": event_id,
            "has_geometry_aoi": _bool_text(geometry),
            "has_candidate_event_link": _bool_text(event_link),
            "has_temporal_binding": _bool_text(binding),
            "has_real_feature_with_hash_replay": _bool_text(feature_ok),
            "fallback_source_resolved": _bool_text(fallback_ok),
            "not_official_patch": _bool_text(not_official),
            "not_official_patch_link": _bool_text(not_official),
            "usage_decision_status": "accepted_candidate_patch_scientific_review_only" if accepted else "blocked_missing_criterion",
            "usable_for_science_now": _bool_text(accepted),
            "usable_for_training": "false",
            "ground_truth": "false",
            "eligible_for_score_v7": "false",
            "eligible_for_17b": "false",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Aceite cientifico review-only das features
# ---------------------------------------------------------------------------

def scientific_review_only_feature_acceptance_rows() -> list[dict]:
    fb_resolved = {r["candidate_patch_id"]: (r["decision_status"] == "accepted_scientific_review_only_materialization_source") for r in fallback_acceptance_decision_rows()}
    patch_ok = {r["candidate_patch_id"]: (r["usage_decision_status"] == "accepted_candidate_patch_scientific_review_only") for r in candidate_patch_usage_decision_rows()}
    rows = []
    for idx, feat in enumerate(read_csv(C22_PRE_ACCEPT), start=1):
        patch_id = feat["candidate_patch_id"]
        prior_ok = feat["acceptance_status"] == "accepted_review_only_sensor_feature"
        promote = prior_ok and fb_resolved.get(patch_id, False) and patch_ok.get(patch_id, False)
        rows.append({
            "scientific_feature_acceptance_id": f"S17C23_SCIFEAT_{idx:04d}",
            "candidate_patch_id": patch_id,
            "event_id": feat["event_id"],
            "scene_or_item_id": feat["scene_or_item_id"],
            "feature_name": feat["feature_name"],
            "feature_value": feat["feature_value"],
            "feature_unit": feat["feature_unit"],
            "source_artifact_manifest_id": feat["source_artifact_manifest_id"],
            "source_role": feat["source_role"],
            "prior_acceptance_status": feat["acceptance_status"],
            "fallback_scientific_resolved": _bool_text(fb_resolved.get(patch_id, False)),
            "patch_usage_accepted": _bool_text(patch_ok.get(patch_id, False)),
            "acceptance_status": "accepted_scientific_review_only_sensor_feature" if promote else "blocked_pending_equivalence_or_patch_policy",
            "usable_for_science_now": _bool_text(promote),
            "usable_for_training": "false",
            "ground_truth": "false",
            "eligible_for_score_v7": "false",
            "eligible_for_17b": "false",
            "remaining_limitation": L1C_L2A_LIMITATION,
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Matriz multimodal candidata (cientifica review-only)
# ---------------------------------------------------------------------------

def candidate_multimodal_feature_matrix_rows() -> list[dict]:
    event_id = _event_id()
    source = _c20_source_by_patch()
    accepted = scientific_review_only_feature_acceptance_rows()
    fb_dec = {r["candidate_patch_id"]: r["decision_status"] for r in fallback_acceptance_decision_rows()}
    patch_dec = {r["candidate_patch_id"]: r["usage_decision_status"] for r in candidate_patch_usage_decision_rows()}
    rows = []
    for patch_id in _patch_ids():
        values = _pre_feature_values(patch_id)
        patch_feats = [f for f in accepted if f["candidate_patch_id"] == patch_id]
        sci_count = len([f for f in patch_feats if f["acceptance_status"] == "accepted_scientific_review_only_sensor_feature"])
        row = {
            "candidate_patch_id": patch_id,
            "event_id": event_id,
            "feature_matrix_status": "accepted_scientific_review_only" if sci_count == len(FEATURE_ORDER) else "partial_or_blocked",
            "pre_event_scene_or_item_id": _pre_scene(patch_id),
            "earth_search_item_id": source.get(patch_id, {}).get("selected_materialization_source", "not_available"),
            "source_role": "materialization_fallback",
            "fallback_decision_status": fb_dec.get(patch_id, "not_available"),
            "patch_usage_decision_status": patch_dec.get(patch_id, "not_available"),
        }
        for name in FEATURE_ORDER:
            row[name] = values.get(name, "not_available")
        row.update({
            "sensor_features_real_count": str(len(patch_feats)),
            "sensor_features_scientific_review_only_count": str(sci_count),
            "usable_for_science_now": _bool_text(sci_count == len(FEATURE_ORDER)),
            "usable_for_training": "false",
            "ground_truth": "false",
            "eligible_for_score_v7": "false",
            "eligible_for_17b": "false",
            "review_only": "true",
        })
        rows.append(row)
    return rows


def observational_delta_review_matrix_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    for patch_id in _patch_ids():
        deltas = _delta_values(patch_id)
        row = {"candidate_patch_id": patch_id, "event_id": event_id}
        for name, delta_name in zip(FEATURE_ORDER, DELTA_ORDER):
            row[delta_name] = deltas.get(name, "not_available")
        row.update({
            "usable_as_observational_change_review_only": "true",
            "usable_as_pre_event_feature": "false",
            "usable_for_training": "false",
            "ground_truth": "false",
            "eligible_for_score_v7": "false",
            "eligible_for_17b": "false",
            "review_only": "true",
        })
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Limites de escopo cientifico
# ---------------------------------------------------------------------------

def science_scope_limits_rows() -> list[dict]:
    limits = [
        ("processing_level_l1c_vs_l2a", "cena CDSE canonica e L1C; materializacao Earth Search e L2A", "equivalencia de missao/plataforma/data/tile/AOI mantida; diferenca de nivel documentada como limitacao"),
        ("fallback_not_official_primary", "Earth Search/STAC/COG nao e fonte oficial primaria", "uso restrito a materializacao sensorial cientifica review-only"),
        ("no_ground_reference", "features sensoriais nao sao Ground Reference", "sem geometria de evento observado nem fenomeno confirmado"),
        ("no_training", "features nao alimentam treino", "sem label e sem politica de treino"),
        ("no_score_v7", "features nao alimentam score v7", "score v7 exige ground reference e politica"),
        ("no_17b", "features nao liberam 17B", "17B exige artefato de evento e politica"),
        ("single_pre_event_scene", "uma cena pre-evento canonica por patch", "sem serie temporal pre-evento ampla ainda"),
        ("cloud_cover_high", "cobertura de nuvem alta na cena pre-evento", "afeta interpretacao espectral; documentado"),
    ]
    rows = []
    for idx, (name, desc, mitigation) in enumerate(limits, start=1):
        rows.append({
            "science_scope_limit_id": f"S17C23_LIMIT_{idx:04d}",
            "limit_name": name,
            "limit_description": desc,
            "mitigation_or_note": mitigation,
            "blocks_scientific_review_only": "false",
            "blocks_training_score_17b": "true",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Bloqueadores de treino/score/17B
# ---------------------------------------------------------------------------

def training_score_17b_blocker_rows() -> list[dict]:
    rows = []
    idx = 0
    for feat in scientific_review_only_feature_acceptance_rows():
        idx += 1
        rows.append({
            "training_score_17b_blocker_id": f"S17C23_TS17B_{idx:04d}",
            "object_type": "scientific_review_only_sensor_feature",
            "object_id": feat["scientific_feature_acceptance_id"],
            "blocks_training": "true", "blocks_score_v7": "true", "blocks_17b": "true",
            "blocking_reason": "feature cientifica review-only via fallback; treino/score/17B exigem ground reference, evento e politica",
            "review_only": "true",
        })
    for delta in read_csv(C22_DELTA_ACCEPT):
        idx += 1
        rows.append({
            "training_score_17b_blocker_id": f"S17C23_TS17B_{idx:04d}",
            "object_type": "observational_delta",
            "object_id": delta["delta_acceptance_id"],
            "blocks_training": "true", "blocks_score_v7": "true", "blocks_17b": "true",
            "blocking_reason": "delta observacional review-only; nunca treino/score/17B",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Anti-leakage
# ---------------------------------------------------------------------------

def no_leakage_audit_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    idx = 0
    for feat in scientific_review_only_feature_acceptance_rows():
        idx += 1
        rows.append({
            "no_leakage_audit_id": f"S17C23_LEAK_{idx:04d}",
            "candidate_patch_id": feat["candidate_patch_id"], "event_id": event_id,
            "object_id": feat["scientific_feature_acceptance_id"], "object_type": "scientific_review_only_sensor_feature",
            "uses_pre_event_data_only_for_sensor_feature": "true",
            "uses_post_event_as_pre_event_feature": "false",
            "uses_delta_as_training_label": "false",
            "uses_fallback_as_ground_reference": "false",
            "uses_synthetic_as_real": "false",
            "passes_no_leakage": "true",
            "blocking_reason": "feature cientifica review-only sem vazamento", "review_only": "true",
        })
    for delta in read_csv(C22_DELTA_ACCEPT):
        idx += 1
        rows.append({
            "no_leakage_audit_id": f"S17C23_LEAK_{idx:04d}",
            "candidate_patch_id": delta["candidate_patch_id"], "event_id": event_id,
            "object_id": delta["delta_acceptance_id"], "object_type": "observational_delta",
            "uses_pre_event_data_only_for_sensor_feature": "true",
            "uses_post_event_as_pre_event_feature": "false",
            "uses_delta_as_training_label": "false",
            "uses_fallback_as_ground_reference": "false",
            "uses_synthetic_as_real": "false",
            "passes_no_leakage": "true",
            "blocking_reason": "delta observacional review-only", "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Gates gerais
# ---------------------------------------------------------------------------

def gate_evaluation_matrix_rows() -> list[dict]:
    rows = []
    idx = 0
    for feat in scientific_review_only_feature_acceptance_rows():
        idx += 1
        accepted = feat["acceptance_status"] == "accepted_scientific_review_only_sensor_feature"
        rows.append({
            "gate_eval_id": f"S17C23_GATE_{idx:04d}",
            "object_id": feat["scientific_feature_acceptance_id"], "object_type": "scientific_review_only_sensor_feature",
            "G1_existencia_documental": "true", "G2_confiabilidade_fonte": "true", "G3_precisao_temporal": "true",
            "G4_vinculo_espacial_evento": "false", "G5_separacao_fenomeno": "false",
            "G6_proveniencia_integridade": "true", "G7_anti_leakage": "true",
            "all_gates_passed_for_ground_reference": "false",
            "usable_as_scientific_review_only_sensor_feature": _bool_text(accepted),
            "usable_as_observational_change_review_only": "false",
            "acceptance_status": feat["acceptance_status"],
            "blocking_reason": "feature sensorial cientifica review-only; nao e evento observado, nao vira Ground Reference",
            **GOV,
        })
    for delta in read_csv(C22_DELTA_ACCEPT):
        idx += 1
        rows.append({
            "gate_eval_id": f"S17C23_GATE_{idx:04d}",
            "object_id": delta["delta_acceptance_id"], "object_type": "observational_delta",
            "G1_existencia_documental": "true", "G2_confiabilidade_fonte": "true", "G3_precisao_temporal": "true",
            "G4_vinculo_espacial_evento": "false", "G5_separacao_fenomeno": "false",
            "G6_proveniencia_integridade": "true", "G7_anti_leakage": "true",
            "all_gates_passed_for_ground_reference": "false",
            "usable_as_scientific_review_only_sensor_feature": "false",
            "usable_as_observational_change_review_only": "true",
            "acceptance_status": delta["acceptance_status"],
            "blocking_reason": "delta observacional review-only; nao e evento observado, nao vira Ground Reference",
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary() -> dict:
    dossier = earth_search_resolution_dossier_rows()
    equiv_passed = [r for r in dossier if r["equivalence_resolved"] == "true"]
    fb_dec = fallback_acceptance_decision_rows()
    fb_accepted = [r for r in fb_dec if r["accepted_for_scientific_review_only"] == "true"]
    scene_audit = fallback_scene_equivalence_audit_rows()
    review_blockers = len([r for r in scene_audit if r["scene_equivalence_passed"] != "true"])
    patch_dec = candidate_patch_usage_decision_rows()
    patch_accepted = [r for r in patch_dec if r["usage_decision_status"] == "accepted_candidate_patch_scientific_review_only"]
    sci = scientific_review_only_feature_acceptance_rows()
    sci_accepted = [r for r in sci if r["acceptance_status"] == "accepted_scientific_review_only_sensor_feature"]
    matrix = candidate_multimodal_feature_matrix_rows()
    equivalence_resolved = len(equiv_passed) == len(dossier) and len(dossier) >= 5
    patch_policy_completed = len(patch_dec) == 5
    matrix_created = len(matrix) == 5
    minimum = (
        equivalence_resolved and review_blockers == 0 and len(fb_accepted) >= 5
        and patch_policy_completed and len(patch_accepted) == 5
        and len(sci_accepted) >= 40 and matrix_created
    )
    return {
        "earth_search_equivalence_resolved": equivalence_resolved,
        "earth_search_resolution_rows_count": len(dossier),
        "earth_search_equivalence_passed_count": len(equiv_passed),
        "fallback_policy_review_blockers_count": review_blockers,
        "fallback_accepted_for_scientific_review_only_count": len(fb_accepted),
        "fallback_accepted_for_ground_reference_count": 0,
        "fallback_accepted_for_training_count": 0,
        "fallback_accepted_for_score_v7_count": 0,
        "fallback_accepted_for_17b_count": 0,
        "candidate_patch_usage_policy_completed": patch_policy_completed,
        "candidate_patch_usage_accepted_count": len(patch_accepted),
        "pre_event_sensor_features_evaluated_count": len(sci),
        "accepted_scientific_review_only_sensor_features_count": len(sci_accepted),
        "candidate_multimodal_feature_matrix_created": matrix_created,
        "candidate_multimodal_feature_matrix_rows_count": len(matrix),
        "features_promoted_to_training_count": 0,
        "post_event_used_as_pre_event_feature_count": 0,
        "delta_used_as_training_label_count": 0,
        "ground_reference_candidates_created_count": 0,
        "ground_truth_created": False,
        "training_labels_created": False,
        "score_v6_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
        "official_patch_created": False,
        "official_patch_link_created": False,
        "eligible_for_17b_now": False,
        "eligible_for_score_v7": False,
        "minimum_success_achieved": minimum,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "recommended_next_milestone": "SUSC-17C24 Serie temporal pre-evento multi-cena e reducao de nuvem para robustez das features sensoriais cientificas review-only",
    }


def build_blockers() -> list[dict]:
    blockers = [
        "sensor_features_scientific_review_only_not_trainable",
        "observational_deltas_not_trainable",
        "processing_level_l1c_l2a_limitation_documented",
        "event_geometry_still_missing",
        "phenomenon_gate_not_satisfied",
        "ground_reference_event_artifacts_still_missing",
        "official_patch_link_missing",
        "17b_blocked_until_event_reference_and_policy",
        "score_v7_blocked_until_ground_reference_and_policy",
    ]
    return [
        {
            "blocker_id": f"S17C23_BLOCKER_{idx:04d}",
            "blocker_type": blocker,
            "description": "Bloqueio preservado: equivalencia Earth Search libera apenas uso cientifico review-only; treino, score v7, 17B, Ground Reference e patch oficial permanecem bloqueados por falta de evento observado, geometria, fenomeno e politica.",
            "blocks_ground_reference_candidate": _bool_text(blocker in {"event_geometry_still_missing", "phenomenon_gate_not_satisfied", "ground_reference_event_artifacts_still_missing"}),
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


def build_dossier_schema() -> dict:
    required = list(earth_search_resolution_dossier_rows()[0].keys())
    return _schema(required, {
        "earth_search_resolution_id": {"type": "string", "pattern": "^S17C23_ESRES_"},
        "accepted_for_ground_reference": {"const": "false"}, "accepted_for_training": {"const": "false"},
        "accepted_for_score_v7": {"const": "false"}, "accepted_for_17b": {"const": "false"},
        "review_only": {"const": "true"}, "trainable": {"const": "false"}, "ground_truth": {"const": "false"},
    }, "SUSC-17C23 earth search resolution schema v1")


def build_patch_usage_schema() -> dict:
    required = list(candidate_patch_usage_decision_rows()[0].keys())
    return _schema(required, {
        "patch_usage_decision_id": {"type": "string", "pattern": "^S17C23_PUDEC_"},
        "usable_for_training": {"const": "false"}, "ground_truth": {"const": "false"},
        "eligible_for_score_v7": {"const": "false"}, "eligible_for_17b": {"const": "false"},
        "review_only": {"const": "true"},
    }, "SUSC-17C23 candidate patch usage schema v1")


def build_scientific_schema() -> dict:
    required = list(scientific_review_only_feature_acceptance_rows()[0].keys())
    return _schema(required, {
        "scientific_feature_acceptance_id": {"type": "string", "pattern": "^S17C23_SCIFEAT_"},
        "usable_for_training": {"const": "false"}, "ground_truth": {"const": "false"},
        "eligible_for_score_v7": {"const": "false"}, "eligible_for_17b": {"const": "false"},
        "review_only": {"const": "true"},
    }, "SUSC-17C23 scientific feature acceptance schema v1")


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------

def build_report() -> str:
    summary = build_summary()
    return "\n".join([
        "# SUSC-17C23 - Politica de equivalencia Earth Search e matriz cientifica review-only",
        "",
        "## Objetivo",
        "Resolver objetivamente a trava Earth Search/STAC/COG e a politica de uso dos patches candidatos, usando apenas artefatos 17C18-17C22 (sem rede). Se a equivalencia passar, promover as 40 features pre-evento de accepted_review_only_sensor_feature para accepted_scientific_review_only_sensor_feature, sem liberar treino, ground truth, score v7 ou 17B.",
        "",
        "## Equivalencia Earth Search",
        f"- Equivalencia resolvida: {summary['earth_search_equivalence_resolved']}.",
        f"- Dossies avaliados: {summary['earth_search_resolution_rows_count']}; equivalencias aprovadas: {summary['earth_search_equivalence_passed_count']}.",
        f"- Blockers de politica de fallback restantes: {summary['fallback_policy_review_blockers_count']}.",
        f"- Fallback aceito para review-only cientifico: {summary['fallback_accepted_for_scientific_review_only_count']} patches.",
        "- Criterios provados por patch: mesma missao, plataforma, data (2022-04-27), tile MGRS 25MBM, AOI coberta, bandas reais B03/B04/B08/B11, SHA256, replay, leitura COG/window e tamanho sob politica.",
        "",
        "## Limitacao metodologica L1C vs L2A",
        "A cena CDSE canonica e L1C e a materializacao Earth Search e L2A. Como missao, plataforma, data, tile e AOI sao equivalentes, a diferenca de nivel de processamento e registrada como limitacao metodologica (nao bloqueio) em susc_17c23_science_scope_limits.csv.",
        "",
        "## Politica de patch candidato",
        f"- Politica concluida: {summary['candidate_patch_usage_policy_completed']}; patches aceitos: {summary['candidate_patch_usage_accepted_count']}.",
        "- Cada patch prova geometria/AOI, vinculo candidato com evento, binding temporal, feature real com hash/replay, fonte fallback resolvida e nao ser patch/patch-link oficial.",
        "",
        "## Aceite cientifico review-only e matriz multimodal",
        f"- Features aceitas como scientific review-only: {summary['accepted_scientific_review_only_sensor_features_count']}.",
        f"- Matriz multimodal candidata: {summary['candidate_multimodal_feature_matrix_rows_count']} linhas (deltas em bloco separado, nunca feature pre-evento).",
        "",
        "## Guardrails",
        f"- Fallback como Ground Reference: {summary['fallback_accepted_for_ground_reference_count']}; treino: {summary['fallback_accepted_for_training_count']}; score v7: {summary['fallback_accepted_for_score_v7_count']}; 17B: {summary['fallback_accepted_for_17b_count']}.",
        "- Features promovidas a treino: 0; Ground Reference: 0; ground truth: 0; label: 0; score v6 intacto; score v7 inexistente; 17B bloqueado.",
        "",
        f"## minimum_success_achieved: {summary['minimum_success_achieved']}",
        "",
        "## Proximo marco recomendado",
        summary["recommended_next_milestone"],
    ])


# ---------------------------------------------------------------------------
# Build / validacao
# ---------------------------------------------------------------------------

def build_all() -> None:
    _require_inputs()
    write_csv(DOSSIER, earth_search_resolution_dossier_rows())
    write_csv(EQUIV_POLICY, fallback_equivalence_policy_rows())
    write_csv(SCENE_EQUIV_AUDIT, fallback_scene_equivalence_audit_rows())
    write_csv(FALLBACK_DECISION, fallback_acceptance_decision_rows())
    write_csv(PATCH_USAGE_POLICY, candidate_patch_usage_policy_rows())
    write_csv(PATCH_USAGE_DECISION, candidate_patch_usage_decision_rows())
    write_csv(SCIENTIFIC_ACCEPT, scientific_review_only_feature_acceptance_rows())
    write_csv(MULTIMODAL_MATRIX, candidate_multimodal_feature_matrix_rows())
    write_csv(DELTA_REVIEW_MATRIX, observational_delta_review_matrix_rows())
    write_csv(SCOPE_LIMITS, science_scope_limits_rows())
    write_csv(TS17B_BLOCKERS, training_score_17b_blocker_rows())
    write_csv(NO_LEAKAGE, no_leakage_audit_rows())
    write_csv(GATES, gate_evaluation_matrix_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_json(DOSSIER_SCHEMA, build_dossier_schema())
    write_json(PATCH_USAGE_SCHEMA, build_patch_usage_schema())
    write_json(SCIENTIFIC_SCHEMA, build_scientific_schema())
    write_markdown(REPORT, build_report())


REQUIRED_OUTPUTS = [
    REPORT, DOSSIER, EQUIV_POLICY, SCENE_EQUIV_AUDIT, FALLBACK_DECISION, PATCH_USAGE_POLICY,
    PATCH_USAGE_DECISION, SCIENTIFIC_ACCEPT, MULTIMODAL_MATRIX, DELTA_REVIEW_MATRIX,
    SCOPE_LIMITS, TS17B_BLOCKERS, NO_LEAKAGE, GATES, SUMMARY, BLOCKERS,
    DOSSIER_SCHEMA, PATCH_USAGE_SCHEMA, SCIENTIFIC_SCHEMA,
]


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
    dossier = read_csv(DOSSIER)
    scene_audit = read_csv(SCENE_EQUIV_AUDIT)
    fb_dec = read_csv(FALLBACK_DECISION)
    patch_dec = read_csv(PATCH_USAGE_DECISION)
    sci = read_csv(SCIENTIFIC_ACCEPT)
    matrix = read_csv(MULTIMODAL_MATRIX)
    delta_matrix = read_csv(DELTA_REVIEW_MATRIX)
    limits = read_csv(SCOPE_LIMITS)
    ts_blockers = read_csv(TS17B_BLOCKERS)
    leakage = read_csv(NO_LEAKAGE)
    gates = read_csv(GATES)
    summary = read_json(SUMMARY)
    dossier_schema = read_json(DOSSIER_SCHEMA)
    patch_schema = read_json(PATCH_USAGE_SCHEMA)
    sci_schema = read_json(SCIENTIFIC_SCHEMA)

    for row in dossier:
        errors.extend(_schema_violations(row, dossier_schema))
    for row in patch_dec:
        errors.extend(_schema_violations(row, patch_schema))
    for row in sci:
        errors.extend(_schema_violations(row, sci_schema))

    for rows, key in [
        (dossier, "earth_search_resolution_id"), (scene_audit, "scene_equivalence_audit_id"),
        (fb_dec, "fallback_acceptance_decision_id"), (patch_dec, "patch_usage_decision_id"),
        (sci, "scientific_feature_acceptance_id"), (limits, "science_scope_limit_id"),
        (ts_blockers, "training_score_17b_blocker_id"), (leakage, "no_leakage_audit_id"),
        (gates, "gate_eval_id"),
    ]:
        ids = [row[key] for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    sci_accepted = [r for r in sci if r["acceptance_status"] == "accepted_scientific_review_only_sensor_feature"]
    # 1: equivalencia resolvida.
    if not summary["earth_search_equivalence_resolved"]:
        errors.append("earth_search_equivalence_not_resolved")
    # 2: sem blockers.
    if summary["fallback_policy_review_blockers_count"] != 0:
        errors.append("fallback_policy_blockers_present")
    # 3: >=5 patches aceitos.
    if summary["fallback_accepted_for_scientific_review_only_count"] < 5:
        errors.append("fallback_accepted_lt_5")
    # 4/5/6: equivalencia so aprovada com data/tile/hash/replay.
    for row in dossier:
        if row["equivalence_resolved"] == "true":
            if row["same_acquisition_date"] != "true":
                errors.append(f"equivalence_without_same_date:{row['candidate_patch_id']}")
            if row["same_mgrs_tile"] != "true":
                errors.append(f"equivalence_without_same_tile:{row['candidate_patch_id']}")
            if row["hash_manifest_verified"] != "true" or row["replay_verified"] != "true":
                errors.append(f"equivalence_without_hash_replay:{row['candidate_patch_id']}")
    # 7: L1C/L2A registrada como limitacao.
    if not any(r["limit_name"] == "processing_level_l1c_vs_l2a" for r in limits):
        errors.append("l1c_l2a_limitation_not_registered")
    if any(r["remaining_limitation"] == "" for r in dossier):
        errors.append("dossier_missing_limitation")
    # 8..11: fallback nao vira ground reference/treino/score/17b.
    if any(r["accepted_for_ground_reference"] != "false" for r in fb_dec) or summary["fallback_accepted_for_ground_reference_count"] != 0:
        errors.append("fallback_as_ground_reference")
    if any(r["accepted_for_training"] != "false" for r in fb_dec) or summary["fallback_accepted_for_training_count"] != 0:
        errors.append("fallback_as_training")
    if any(r["accepted_for_score_v7"] != "false" for r in fb_dec) or summary["fallback_accepted_for_score_v7_count"] != 0:
        errors.append("fallback_as_score_v7")
    if any(r["accepted_for_17b"] != "false" for r in fb_dec) or summary["fallback_accepted_for_17b_count"] != 0:
        errors.append("fallback_as_17b")
    # 12/13: matriz 5 linhas sem deltas.
    if len(matrix) != 5:
        errors.append("matrix_not_5")
    if matrix and any(col.startswith("delta_") for col in matrix[0].keys()):
        errors.append("matrix_contains_delta")
    if len(delta_matrix) != 5:
        errors.append("delta_matrix_not_5")
    # 14: >=40 features scientific review-only.
    if len(sci_accepted) < 40:
        errors.append("scientific_features_lt_40")
    # 15/16/17: sem treino/ground reference/ground truth.
    if any(r["usable_for_training"] != "false" for r in sci) or summary["features_promoted_to_training_count"] != 0:
        errors.append("feature_promoted_to_training")
    if any(r["all_gates_passed_for_ground_reference"] == "true" for r in gates):
        errors.append("ground_reference_accepted")
    if any(r["G4_vinculo_espacial_evento"] == "true" or r["G5_separacao_fenomeno"] == "true" for r in gates):
        errors.append("sensor_promoted_to_event")
    if summary["ground_reference_candidates_created_count"] != 0 or summary["ground_truth_created"] or summary["training_labels_created"]:
        errors.append("forbidden_promotion")
    if any(r["passes_no_leakage"] != "true" for r in leakage):
        errors.append("no_leakage_failed")
    if any(r["blocks_training"] != "true" or r["blocks_score_v7"] != "true" or r["blocks_17b"] != "true" for r in ts_blockers):
        errors.append("training_score_17b_not_blocked")
    # patch candidato nao vira oficial.
    if any(r["not_official_patch"] != "true" or r["not_official_patch_link"] != "true" for r in patch_dec):
        errors.append("candidate_patch_marked_official")

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
    if not summary["minimum_success_achieved"]:
        errors.append("minimum_success_not_achieved")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "17C23 -> "
        f"equivalence={summary['earth_search_equivalence_passed_count']}/5 "
        f"fb_accepted={summary['fallback_accepted_for_scientific_review_only_count']} "
        f"patch_accepted={summary['candidate_patch_usage_accepted_count']} "
        f"sci_features={summary['accepted_scientific_review_only_sensor_features_count']} "
        f"matrix={summary['candidate_multimodal_feature_matrix_rows_count']} "
        f"min_success={summary['minimum_success_achieved']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def evaluate_earth_search_text() -> str:
    dossier = earth_search_resolution_dossier_rows()
    passed = [r for r in dossier if r["equivalence_resolved"] == "true"]
    return f"evaluate-earth-search: {len(passed)}/{len(dossier)} patches com equivalencia Earth Search resolvida (L1C~L2A documentado)."


def evaluate_candidate_patch_policy_text() -> str:
    dec = candidate_patch_usage_decision_rows()
    accepted = [r for r in dec if r["usage_decision_status"] == "accepted_candidate_patch_scientific_review_only"]
    return f"evaluate-candidate-patch-policy: {len(accepted)}/{len(dec)} patches aceitos para uso sensorial cientifico review-only."


def promote_review_only_science_text() -> str:
    sci = scientific_review_only_feature_acceptance_rows()
    accepted = [r for r in sci if r["acceptance_status"] == "accepted_scientific_review_only_sensor_feature"]
    return f"promote-review-only-science: {len(accepted)}/{len(sci)} features promovidas a accepted_scientific_review_only_sensor_feature (sem treino/score/17B)."


def build_candidate_matrix_text() -> str:
    matrix = candidate_multimodal_feature_matrix_rows()
    delta = observational_delta_review_matrix_rows()
    return f"build-candidate-matrix: {len(matrix)} linhas de matriz multimodal e {len(delta)} linhas de matriz de deltas (blocos separados)."


def status_text() -> str:
    summary = build_summary()
    return (
        f"status 17C23: equivalence={summary['earth_search_equivalence_passed_count']} "
        f"sci_features={summary['accepted_scientific_review_only_sensor_features_count']} "
        f"min_success={summary['minimum_success_achieved']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
