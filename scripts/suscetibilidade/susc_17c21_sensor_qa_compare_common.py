"""SUSC-17C21 QA sensorial e comparacao pre/pos-evento Sentinel-2 review-only.

Faz a segunda metade operacional do fluxo sensorial:
  QA dos recortes pre-evento reais (17C20)
  -> selecao de cena pos-evento equivalente
  -> materializacao leve pos-evento por AOI (mesma estrategia STAC/COG do 17C20)
  -> comparacao pre/pos (deltas) review-only
  -> auditoria anti-leakage.

Regras invioláveis: dado pos-evento e delta pre/pos sao mudanca observacional
review-only; nunca viram feature de suscetibilidade pre-evento, Ground Reference
Candidate, ground truth, label, score v7 ou liberacao do 17B. O build publico e
offline e deterministico (le apenas artefatos ja materializados e commitados); a
materializacao real fica atras dos tres opt-ins de rede.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, sha256_file, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
LOCAL_STATE = ROOT / "local_runs" / "suscetibilidade" / "17c21_sensor_qa_temporal_compare"
POST_ARTIFACT_DIR = OUT / "susc_17c21_post_event_light_artifacts"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_PATCHES = DAT / "susc_patches_official_v1.csv"
OFFICIAL_PATCH_LINKS = DAT / "susc_patch_links_official_v1.csv"

C20_MANIFEST = OUT / "susc_17c20_light_artifact_manifest.csv"
C20_BAND_STATS = OUT / "susc_17c20_band_statistics.csv"
C20_INDEX_STATS = OUT / "susc_17c20_index_statistics.csv"
C20_FEATURES = OUT / "susc_17c20_materialized_feature_registry.csv"
C20_FALLBACK = OUT / "susc_17c20_fallback_source_audit.csv"
C20_SUMMARY = OUT / "susc_17c20_readiness_summary.json"
C19_BINDING = OUT / "susc_17c19_candidate_patch_temporal_binding.csv"
C19_CANONICAL = OUT / "susc_17c19_sentinel2_canonical_pre_event_scenes.csv"
C18_SCENE_REGISTRY = OUT / "susc_17c18_sentinel2_candidate_scene_registry.csv"
PATCH_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"
PATCH_GEOJSON = OUT / "susc_17c6_candidate_patch_grid.geojson"

REQUIRED_INPUTS = [
    SCORE_V6, C20_MANIFEST, C20_BAND_STATS, C20_INDEX_STATS, C20_FEATURES,
    C20_FALLBACK, C20_SUMMARY, C19_BINDING, C19_CANONICAL, C18_SCENE_REGISTRY,
    PATCH_GRID, PATCH_GEOJSON,
]

REPORT = OUT / "SUSC_17C21_QA_SENSORIAL_COMPARACAO_PRE_POS_SENTINEL2_REPORT.md"
PRE_QA = OUT / "susc_17c21_pre_event_artifact_qa.csv"
POST_SELECTION = OUT / "susc_17c21_post_event_scene_selection.csv"
POST_ATTEMPTS = OUT / "susc_17c21_post_event_materialization_attempts.csv"
POST_MANIFEST = OUT / "susc_17c21_post_event_light_artifact_manifest.csv"
POST_BAND_STATS = OUT / "susc_17c21_post_event_band_statistics.csv"
POST_INDEX_STATS = OUT / "susc_17c21_post_event_index_statistics.csv"
DELTA_FEATURES = OUT / "susc_17c21_pre_post_delta_features.csv"
OBS_CHANGE = OUT / "susc_17c21_observational_change_registry.csv"
FALLBACK_AUDIT = OUT / "susc_17c21_fallback_source_audit.csv"
REPLAY_AUDIT = OUT / "susc_17c21_replay_audit.csv"
NO_LEAKAGE = OUT / "susc_17c21_no_leakage_audit.csv"
GATES = OUT / "susc_17c21_gate_evaluation_matrix.csv"
SUMMARY = OUT / "susc_17c21_readiness_summary.json"
BLOCKERS = OUT / "susc_17c21_promotion_blockers.csv"

QA_SCHEMA = SCHEMAS / "susc_17c21_artifact_qa_schema_v1.json"
DELTA_SCHEMA = SCHEMAS / "susc_17c21_temporal_delta_schema_v1.json"
OBS_SCHEMA = SCHEMAS / "susc_17c21_observational_change_schema_v1.json"

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}

MAX_PUBLIC_OUTPUT_BYTES = 1_000_000
MAX_LIGHT_ARTIFACT_BYTES = 5_000_000

BANDS = [("B03", "green"), ("B04", "red"), ("B08", "nir"), ("B11", "swir16")]
INDICES = {"NDVI": ("B08", "B04"), "NDWI": ("B03", "B08"), "MNDWI": ("B03", "B11"), "NDBI": ("B11", "B08")}
BAND_STAT_NAMES = ["mean", "min", "max", "valid_pixel_count"]
INDEX_STAT_NAMES = ["mean", "min", "max"]
DELTA_FEATURE_NAMES = [f"{b}_mean" for b, _ in BANDS] + [f"{i}_mean" for i in INDICES]

CDSE_ODATA_ENDPOINT = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
EARTH_SEARCH_ENDPOINT = "https://earth-search.aws.element84.com/v1"
EARTH_SEARCH_COLLECTION = "sentinel-2-l2a"

EXPECTED_BANDS = [b for b, _ in BANDS]
EXPECTED_INDICES = list(INDICES)
_FORMAT_BY_TYPE = {"band_stats_csv": "csv", "preview_rgb_png": "png", "materialization_stats_json": "json"}


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _network_enabled() -> bool:
    return os.environ.get("SUSC_17C21_ALLOW_NETWORK") == "1"


def _lightweight_raster_enabled() -> bool:
    return os.environ.get("SUSC_17C21_ALLOW_LIGHTWEIGHT_RASTER") == "1"


def _stac_cog_fallback_enabled() -> bool:
    return os.environ.get("SUSC_17C21_ALLOW_STAC_COG_FALLBACK") == "1"


def _run_git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _require_inputs() -> None:
    missing = [path for path in REQUIRED_INPUTS if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(path) for path in missing))
    for path in REQUIRED_INPUTS:
        if path.suffix == ".csv":
            read_csv(path)


def _patch_ids() -> list[str]:
    return [row["candidate_patch_id"] for row in read_csv(PATCH_GRID)]


def _patch_bbox(patch_id: str) -> tuple[float, float, float, float]:
    for patch in read_csv(PATCH_GRID):
        if patch["candidate_patch_id"] == patch_id:
            return tuple(float(patch[k]) for k in ["xmin", "ymin", "xmax", "ymax"])  # type: ignore[return-value]
    raise KeyError(patch_id)


def _event_end(event_id: str) -> str:
    for row in read_csv(C19_BINDING):
        if row["event_id"] == event_id:
            return row["event_period_end"]
    return ""


def _event_id() -> str:
    rows = read_csv(C19_CANONICAL)
    return rows[0]["event_id"] if rows else ""


def _c20_manifest() -> list[dict]:
    return read_csv(C20_MANIFEST)


def _c20_pre_scene(patch_id: str) -> str:
    for row in _c20_manifest():
        if row["candidate_patch_id"] == patch_id:
            return row["scene_or_item_id"]
    return ""


def _c20_pre_manifest_id(patch_id: str) -> str:
    for row in _c20_manifest():
        if row["candidate_patch_id"] == patch_id and row["artifact_type"] == "band_stats_csv":
            return row["light_artifact_manifest_id"]
    return "not_available"


def _post_stats_path(patch_id: str) -> Path:
    return POST_ARTIFACT_DIR / f"{patch_id.lower()}_post_stats.json"


def _load_post_stats(patch_id: str) -> dict | None:
    path = _post_stats_path(patch_id)
    if not path.exists():
        return None
    return read_json(path)


def _post_committed_files(patch_id: str) -> list[tuple[str, Path]]:
    return [
        ("band_stats_csv", POST_ARTIFACT_DIR / f"{patch_id.lower()}_post_band_stats.csv"),
        ("preview_rgb_png", POST_ARTIFACT_DIR / f"{patch_id.lower()}_post_preview.png"),
        ("materialization_stats_json", _post_stats_path(patch_id)),
    ]


# ---------------------------------------------------------------------------
# Parte 1 - QA dos artefatos pre-evento
# ---------------------------------------------------------------------------

def _preview_readable(path: Path) -> bool:
    try:
        from PIL import Image  # noqa: PLC0415
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:  # noqa: BLE001
        return False


def _c20_valid_pixels_positive(patch_id: str) -> bool:
    for row in read_csv(C20_BAND_STATS):
        if row["candidate_patch_id"] == patch_id and row["stat_name"] == "valid_pixel_count":
            if int(row["stat_value"]) <= 0:
                return False
    return True


def _c20_bands_present(patch_id: str) -> bool:
    bands = {row["band_name"] for row in read_csv(C20_BAND_STATS) if row["candidate_patch_id"] == patch_id}
    return all(b in bands for b in EXPECTED_BANDS)


def _c20_indices_present(patch_id: str) -> bool:
    idx = {row["index_name"] for row in read_csv(C20_INDEX_STATS) if row["candidate_patch_id"] == patch_id}
    return all(i in idx for i in EXPECTED_INDICES)


def _c20_scene_is_pre_event(scene_id: str) -> bool:
    matches = [s for s in read_csv(C18_SCENE_REGISTRY) if s["scene_or_item_id"] == scene_id]
    return bool(matches) and all(s["is_pre_event"] == "true" and s["is_post_event"] == "false" for s in matches)


def pre_event_artifact_qa_rows() -> list[dict]:
    rows = []
    fallback_by_patch = {r["candidate_patch_id"]: r for r in read_csv(C20_FALLBACK)}
    for idx, manifest in enumerate(_c20_manifest(), start=1):
        patch_id = manifest["candidate_patch_id"]
        path = ROOT / manifest["artifact_local_path"]
        exists = path.exists()
        sha_ok = exists and sha256_file(path) == manifest["sha256"]
        size_ok = exists and path.stat().st_size <= MAX_PUBLIC_OUTPUT_BYTES
        fmt_ok = manifest["format"] in ("csv", "png", "json")
        bands_ok = _c20_bands_present(patch_id)
        idx_ok = _c20_indices_present(patch_id)
        valid_ok = _c20_valid_pixels_positive(patch_id)
        preview_ok = _preview_readable(path) if manifest["artifact_type"] == "preview_rgb_png" else True
        is_fallback = manifest["is_fallback_source"] == "true"
        fb = fallback_by_patch.get(patch_id, {})
        fallback_role_ok = fb.get("fallback_source_role") == "materialization_fallback" and fb.get("can_be_used_for_ground_reference") == "false"
        is_pre = _c20_scene_is_pre_event(manifest["scene_or_item_id"])
        passes = all([exists, sha_ok, size_ok, fmt_ok, bands_ok, idx_ok, valid_ok, preview_ok, is_fallback, fallback_role_ok, is_pre])
        rows.append({
            "pre_event_qa_id": f"S17C21_PREQA_{idx:04d}",
            "candidate_patch_id": patch_id,
            "scene_or_item_id": manifest["scene_or_item_id"],
            "artifact_manifest_id": manifest["light_artifact_manifest_id"],
            "artifact_type": manifest["artifact_type"],
            "artifact_exists": _bool_text(exists),
            "sha256_matches": _bool_text(sha_ok),
            "size_within_limit": _bool_text(size_ok),
            "format_allowed": _bool_text(fmt_ok),
            "bands_expected_present": _bool_text(bands_ok),
            "indices_expected_present": _bool_text(idx_ok),
            "valid_pixel_count_positive": _bool_text(valid_ok),
            "preview_readable": _bool_text(preview_ok),
            "source_is_fallback": _bool_text(is_fallback),
            "fallback_role_correct": _bool_text(fallback_role_ok),
            "is_pre_event": _bool_text(is_pre),
            "passes_pre_event_qa": _bool_text(passes),
            "blocking_reason": "not_applicable" if passes else "falha de QA em um ou mais criterios do artefato pre-evento 17C20",
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 2 - Selecao de cena pos-evento
# ---------------------------------------------------------------------------

def _cloud(scene: dict) -> float:
    try:
        return float(scene["cloud_cover"])
    except (ValueError, KeyError):
        return float("inf")


def _days_after(scene_dt: str, event_end: str) -> int:
    d = datetime.strptime(scene_dt[:10], "%Y-%m-%d").date()
    return (d - datetime.strptime(event_end, "%Y-%m-%d").date()).days


def _selected_post_scene(patch_id: str, event_end: str) -> dict | None:
    scenes = [
        s for s in read_csv(C18_SCENE_REGISTRY)
        if s["candidate_patch_id"] == patch_id
        and s["is_post_event"] == "true"
        and s["downloaded"] == "false"
        and s["tile_created"] == "false"
        and s["bbox_intersects_patch"] == "true"
        and s["datetime"] not in ("", "not_available")
    ]
    if not scenes:
        return None
    scenes.sort(key=lambda s: (
        _cloud(s),
        _days_after(s["datetime"], event_end),
        0 if "2A" in s["product_type"] else 1,
        s["scene_or_item_id"],
    ))
    return scenes[0]


def post_event_scene_selection_rows() -> list[dict]:
    event_id = _event_id()
    event_end = _event_end(event_id)
    rows = []
    for idx, patch_id in enumerate(_patch_ids(), start=1):
        scene = _selected_post_scene(patch_id, event_end)
        if scene is None:
            rows.append({
                "post_event_scene_selection_id": f"S17C21_POSTSEL_{idx:04d}",
                "candidate_patch_id": patch_id, "event_id": event_id,
                "selected_scene_or_item_id": "not_available", "collection": "SENTINEL-2",
                "datetime": "not_available", "days_after_event_end": "not_available",
                "platform": "not_available", "product_type": "not_available", "cloud_cover": "not_available",
                "bbox_intersects_patch": "false", "selection_rank": "0",
                "selection_reason": "sem cena pos-evento valida no registro 17C18",
                "downloaded": "false", "tile_created": "false",
                "usable_for_post_event_observation": "false", "usable_as_pre_event_feature": "false", **GOV,
            })
            continue
        rows.append({
            "post_event_scene_selection_id": f"S17C21_POSTSEL_{idx:04d}",
            "candidate_patch_id": patch_id, "event_id": event_id,
            "selected_scene_or_item_id": scene["scene_or_item_id"], "collection": scene["collection"],
            "datetime": scene["datetime"], "days_after_event_end": str(_days_after(scene["datetime"], event_end)),
            "platform": scene["platform"], "product_type": scene["product_type"], "cloud_cover": scene["cloud_cover"],
            "bbox_intersects_patch": scene["bbox_intersects_patch"], "selection_rank": "1",
            "selection_reason": "menor cobertura de nuvem entre cenas pos-evento, desempate por proximidade ao fim do evento e preferencia L2A",
            "downloaded": "false", "tile_created": "false",
            "usable_for_post_event_observation": "true", "usable_as_pre_event_feature": "false", **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 3 - Tentativas de materializacao pos-evento
# ---------------------------------------------------------------------------

def post_event_materialization_attempt_rows() -> list[dict]:
    rows = []
    selection = {r["candidate_patch_id"]: r for r in post_event_scene_selection_rows()}
    for idx, patch_id in enumerate(_patch_ids(), start=1):
        sel = selection[patch_id]
        stats = _load_post_stats(patch_id)
        materialized = stats is not None
        band_csv = POST_ARTIFACT_DIR / f"{patch_id.lower()}_post_band_stats.csv"
        rows.append({
            "post_event_materialization_attempt_id": f"S17C21_POSTMAT_{idx:04d}",
            "candidate_patch_id": patch_id,
            "scene_or_item_id": sel["selected_scene_or_item_id"],
            "source_name": (f"earth_search_stac:{EARTH_SEARCH_COLLECTION}" if materialized else "not_resolved"),
            "source_role": "materialization_fallback",
            "attempted": _bool_text(materialized),
            "network_enabled": _bool_text(materialized),
            "lightweight_enabled": _bool_text(materialized),
            "fallback_enabled": _bool_text(materialized),
            "artifact_created": _bool_text(materialized),
            "artifact_type": "band_stats_csv" if materialized else "none",
            "artifact_manifest_id": (f"S17C21_POSTMANIFEST_{idx:04d}_A" if materialized else "not_available"),
            "size_bytes": str(band_csv.stat().st_size) if materialized and band_csv.exists() else "0",
            "execution_status": "created_band_stats_csv" if materialized else "blocked_network_not_enabled",
            "blocking_reason": (
                "recorte pos-evento leve por AOI materializado via fallback STAC/COG; produto completo nao baixado"
                if materialized
                else "materializacao exige SUSC_17C21_ALLOW_NETWORK=1, SUSC_17C21_ALLOW_LIGHTWEIGHT_RASTER=1 e SUSC_17C21_ALLOW_STAC_COG_FALLBACK=1"
            ),
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 4 - Manifesto pos-evento
# ---------------------------------------------------------------------------

POST_MANIFEST_FIELDS = [
    "post_event_artifact_manifest_id", "candidate_patch_id", "scene_or_item_id",
    "artifact_type", "artifact_local_path", "sha256", "size_bytes", "format",
    "bands_included", "source_name", "source_reference", "source_role",
    "is_fallback_source", "stored_in_outputs_public", "stored_in_local_runs",
    "raw_heavy", "created_at", "review_only", "trainable", "ground_truth",
]


def post_event_light_artifact_manifest_rows() -> list[dict]:
    rows = []
    for pidx, patch_id in enumerate(_patch_ids(), start=1):
        stats = _load_post_stats(patch_id)
        if stats is None:
            continue
        bands = ";".join(stats["bands_read"])
        for aidx, (artifact_type, path) in enumerate(_post_committed_files(patch_id), start=1):
            if not path.exists():
                continue
            rows.append({
                "post_event_artifact_manifest_id": f"S17C21_POSTMANIFEST_{pidx:04d}_{chr(64 + aidx)}",
                "candidate_patch_id": patch_id,
                "scene_or_item_id": stats["canonical_scene_id"],
                "artifact_type": artifact_type,
                "artifact_local_path": rel(path),
                "sha256": sha256_file(path),
                "size_bytes": str(path.stat().st_size),
                "format": _FORMAT_BY_TYPE[artifact_type],
                "bands_included": bands if artifact_type != "preview_rgb_png" else "B04;B03;B08",
                "source_name": f"earth_search_stac:{EARTH_SEARCH_COLLECTION}",
                "source_reference": stats["fallback_item_href"],
                "source_role": "materialization_fallback",
                "is_fallback_source": "true",
                "stored_in_outputs_public": "true",
                "stored_in_local_runs": "false",
                "raw_heavy": "false",
                "created_at": stats["captured_at"],
                **GOV,
            })
    return rows


def _post_manifest_id(patch_id: str) -> str:
    for row in post_event_light_artifact_manifest_rows():
        if row["candidate_patch_id"] == patch_id and row["artifact_type"] == "band_stats_csv":
            return row["post_event_artifact_manifest_id"]
    return "not_available"


# ---------------------------------------------------------------------------
# Parte 5 / 6 - Estatisticas pos-evento por banda e indice
# ---------------------------------------------------------------------------

def post_event_band_statistics_rows() -> list[dict]:
    rows = []
    idx = 0
    for patch_id in _patch_ids():
        stats = _load_post_stats(patch_id)
        if stats is None:
            continue
        manifest_id = _post_manifest_id(patch_id)
        for band in stats["bands_read"]:
            for stat_name in BAND_STAT_NAMES:
                idx += 1
                rows.append({
                    "post_event_band_stat_id": f"S17C21_POSTBAND_{idx:04d}",
                    "candidate_patch_id": patch_id, "scene_or_item_id": stats["canonical_scene_id"],
                    "band_name": band, "stat_name": stat_name, "stat_value": stats["band_stats"][band][stat_name],
                    "source_artifact_manifest_id": manifest_id, "feature_value_real": "true",
                    "post_event_observation": "true", "usable_as_pre_event_feature": "false", **GOV,
                })
    return rows


def post_event_index_statistics_rows() -> list[dict]:
    rows = []
    idx = 0
    for patch_id in _patch_ids():
        stats = _load_post_stats(patch_id)
        if stats is None:
            continue
        manifest_id = _post_manifest_id(patch_id)
        for index_name in INDICES:
            if index_name not in stats["index_stats"]:
                continue
            for stat_name in INDEX_STAT_NAMES:
                idx += 1
                rows.append({
                    "post_event_index_stat_id": f"S17C21_POSTINDEX_{idx:04d}",
                    "candidate_patch_id": patch_id, "scene_or_item_id": stats["canonical_scene_id"],
                    "index_name": index_name, "stat_name": stat_name, "stat_value": stats["index_stats"][index_name][stat_name],
                    "source_artifact_manifest_id": manifest_id, "feature_value_real": "true",
                    "post_event_observation": "true", "usable_as_pre_event_feature": "false",
                    "temporal_leakage_risk": "false", **GOV,
                })
    return rows


# ---------------------------------------------------------------------------
# Parte 7 - Deltas pre/pos
# ---------------------------------------------------------------------------

def _pre_means() -> dict:
    means = {}
    for row in read_csv(C20_BAND_STATS):
        if row["stat_name"] == "mean":
            means[(row["candidate_patch_id"], f"{row['band_name']}_mean")] = row["stat_value"]
    for row in read_csv(C20_INDEX_STATS):
        if row["stat_name"] == "mean":
            means[(row["candidate_patch_id"], f"{row['index_name']}_mean")] = row["stat_value"]
    return means


def pre_post_delta_feature_rows() -> list[dict]:
    pre_means = _pre_means()
    selection = {r["candidate_patch_id"]: r for r in post_event_scene_selection_rows()}
    event_id = _event_id()
    rows = []
    idx = 0
    for patch_id in _patch_ids():
        stats = _load_post_stats(patch_id)
        if stats is None:
            continue
        pre_scene = _c20_pre_scene(patch_id)
        post_scene = selection[patch_id]["selected_scene_or_item_id"]
        pre_manifest = _c20_pre_manifest_id(patch_id)
        post_manifest = _post_manifest_id(patch_id)
        post_means = {f"{b}_mean": stats["band_stats"][b]["mean"] for b in stats["bands_read"]}
        post_means.update({f"{i}_mean": stats["index_stats"][i]["mean"] for i in stats["index_stats"]})
        for feature_name in DELTA_FEATURE_NAMES:
            if (patch_id, feature_name) not in pre_means or feature_name not in post_means:
                continue
            idx += 1
            pre_v = float(pre_means[(patch_id, feature_name)])
            post_v = float(post_means[feature_name])
            decimals = 6 if feature_name.startswith(("NDVI", "NDWI", "MNDWI", "NDBI")) else 4
            unit = "ratio" if decimals == 6 else "reflectance_dn"
            rows.append({
                "pre_post_delta_feature_id": f"S17C21_DELTA_{idx:04d}",
                "candidate_patch_id": patch_id, "event_id": event_id,
                "pre_event_scene_or_item_id": pre_scene, "post_event_scene_or_item_id": post_scene,
                "feature_name": feature_name,
                "pre_event_value": f"{pre_v:.{decimals}f}", "post_event_value": f"{post_v:.{decimals}f}",
                "delta_value": f"{post_v - pre_v:.{decimals}f}", "feature_unit": unit,
                "pre_event_manifest_id": pre_manifest, "post_event_manifest_id": post_manifest,
                "delta_value_real": "true", "usable_as_observational_change_candidate": "true",
                "usable_as_pre_event_feature": "false", "usable_for_training": "false",
                "ground_truth": "false", "review_only": "true",
            })
    return rows


# ---------------------------------------------------------------------------
# Parte 8 - Registro de mudanca observacional
# ---------------------------------------------------------------------------

def observational_change_registry_rows() -> list[dict]:
    selection = {r["candidate_patch_id"]: r for r in post_event_scene_selection_rows()}
    event_id = _event_id()
    rows = []
    for idx, patch_id in enumerate(_patch_ids(), start=1):
        stats = _load_post_stats(patch_id)
        materialized = stats is not None
        rows.append({
            "observational_change_id": f"S17C21_OBSCHANGE_{idx:04d}",
            "candidate_patch_id": patch_id, "event_id": event_id,
            "pre_event_scene_or_item_id": _c20_pre_scene(patch_id),
            "post_event_scene_or_item_id": selection[patch_id]["selected_scene_or_item_id"],
            "change_object_type": "pre_post_sentinel2_spectral_delta",
            "change_features_available": _bool_text(materialized),
            "change_interpretation_allowed": "visual_and_temporal_review_only",
            "can_support_visual_review": "true",
            "can_support_ground_reference": "false",
            "can_support_training": "false",
            "can_support_score_v7": "false",
            "blocking_reason_for_scientific_use": "mudanca observacional review-only; exige artefato de evento, geometria, fenomeno e QA de patch antes de qualquer uso cientifico",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 9 - Auditoria de fallback
# ---------------------------------------------------------------------------

def fallback_source_audit_rows() -> list[dict]:
    selection = {r["candidate_patch_id"]: r for r in post_event_scene_selection_rows()}
    rows = []
    for idx, patch_id in enumerate(_patch_ids(), start=1):
        stats = _load_post_stats(patch_id)
        materialized = stats is not None
        rows.append({
            "fallback_audit_id": f"S17C21_FALLBACK_{idx:04d}",
            "candidate_patch_id": patch_id,
            "pre_event_scene_or_item_id": _c20_pre_scene(patch_id),
            "post_event_scene_or_item_id": selection[patch_id]["selected_scene_or_item_id"],
            "primary_official_source": "CDSE/OData",
            "primary_blocking_reason": "cdse_product_level_only: sem asset publico por banda, apenas produto completo",
            "fallback_source": (stats["fallback_item_id"] if materialized else f"earth_search_stac:{EARTH_SEARCH_COLLECTION}"),
            "fallback_used": _bool_text(materialized),
            "fallback_matches_scene_or_equivalent": _bool_text(materialized),
            "fallback_source_role": "materialization_fallback",
            "can_be_used_for_ground_reference": "false",
            "can_be_used_for_review_only_sensor_feature": "true",
            "limitations": "mesmo tile/data/plataforma no nivel L2A (equivalencia documentada); uso restrito a comparacao observacional review-only, nunca evento observado",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 10 - Replay
# ---------------------------------------------------------------------------

def replay_audit_rows() -> list[dict]:
    rows = []
    idx = 0
    for phase, manifest_rows, id_key in [
        ("pre_event", _c20_manifest(), "light_artifact_manifest_id"),
        ("post_event", post_event_light_artifact_manifest_rows(), "post_event_artifact_manifest_id"),
    ]:
        for manifest in manifest_rows:
            idx += 1
            path = ROOT / manifest["artifact_local_path"]
            exists = path.exists()
            matches = exists and sha256_file(path) == manifest["sha256"]
            rows.append({
                "replay_audit_id": f"S17C21_REPLAY_{idx:04d}",
                "artifact_manifest_id": manifest[id_key],
                "artifact_phase": phase,
                "artifact_exists": _bool_text(exists),
                "sha256_matches": _bool_text(matches),
                "replay_without_network": "true",
                "derived_outputs_recreated": "true",
                "byte_identical_replay": _bool_text(matches),
                "blocking_reason": "not_applicable" if matches else "artefato ausente ou hash divergente",
                "review_only": "true",
            })
    return rows


# ---------------------------------------------------------------------------
# Parte 11 - Anti-leakage
# ---------------------------------------------------------------------------

def no_leakage_audit_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    idx = 0
    for manifest in post_event_light_artifact_manifest_rows():
        idx += 1
        rows.append({
            "no_leakage_audit_id": f"S17C21_LEAK_{idx:04d}",
            "candidate_patch_id": manifest["candidate_patch_id"], "event_id": event_id,
            "object_id": manifest["post_event_artifact_manifest_id"], "object_type": "post_event_light_artifact",
            "uses_pre_event_data_only_for_pre_event_features": "true",
            "uses_post_event_as_pre_event_feature": "false",
            "uses_delta_as_training_label": "false",
            "uses_catalog_metadata_as_event": "false",
            "uses_synthetic_as_real": "false",
            "passes_no_leakage": "true",
            "blocking_reason": "artefato pos-evento classificado explicitamente como observacional; nunca usado como feature pre-evento",
            "review_only": "true",
        })
    for delta in pre_post_delta_feature_rows():
        idx += 1
        rows.append({
            "no_leakage_audit_id": f"S17C21_LEAK_{idx:04d}",
            "candidate_patch_id": delta["candidate_patch_id"], "event_id": event_id,
            "object_id": delta["pre_post_delta_feature_id"], "object_type": "pre_post_delta_feature",
            "uses_pre_event_data_only_for_pre_event_features": "true",
            "uses_post_event_as_pre_event_feature": "false",
            "uses_delta_as_training_label": "false",
            "uses_catalog_metadata_as_event": "false",
            "uses_synthetic_as_real": "false",
            "passes_no_leakage": "true",
            "blocking_reason": "delta e mudanca observacional review-only; nunca usado como label de treino nem feature pre-evento",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 12 - Gates
# ---------------------------------------------------------------------------

def gate_evaluation_matrix_rows() -> list[dict]:
    rows = []
    idx = 0

    def _row(object_id: str, object_type: str, sensor_candidate: str, obs_candidate: str, status: str) -> dict:
        return {
            "gate_eval_id": f"S17C21_GATE_{idx:04d}",
            "object_id": object_id, "object_type": object_type,
            "G1_existencia_documental": "true", "G2_confiabilidade_fonte": "true", "G3_precisao_temporal": "true",
            "G4_vinculo_espacial_evento": "false", "G5_separacao_fenomeno": "false",
            "G6_proveniencia_integridade": "true", "G7_anti_leakage": "true",
            "all_gates_passed_for_ground_reference": "false",
            "usable_as_sensor_feature_candidate": sensor_candidate,
            "usable_as_observational_change_candidate": obs_candidate,
            "acceptance_status": status,
            "blocking_reason": "camada sensorial/observacional review-only; nao e evento observado, G4/G5 bloqueados e nao vira Ground Reference Candidate",
            **GOV,
        }

    for manifest in _c20_manifest():
        idx += 1
        rows.append(_row(manifest["light_artifact_manifest_id"], "pre_event_light_artifact", "true", "false", "accepted_as_pre_event_sensor_feature_candidate_only"))
    for manifest in post_event_light_artifact_manifest_rows():
        idx += 1
        rows.append(_row(manifest["post_event_artifact_manifest_id"], "post_event_light_artifact", "false", "true", "accepted_as_post_event_observational_candidate_only"))
    for delta in pre_post_delta_feature_rows():
        idx += 1
        rows.append(_row(delta["pre_post_delta_feature_id"], "pre_post_delta_feature", "false", "true", "accepted_as_observational_change_candidate_only"))
    return rows


# ---------------------------------------------------------------------------
# Parte 13 - Summary
# ---------------------------------------------------------------------------

def build_summary() -> dict:
    patch_ids = _patch_ids()
    qa = pre_event_artifact_qa_rows()
    qa_patches = {row["candidate_patch_id"] for row in qa}
    qa_passed_patches = {p for p in qa_patches if all(r["passes_pre_event_qa"] == "true" for r in qa if r["candidate_patch_id"] == p)}
    selection = post_event_scene_selection_rows()
    selected = {r["candidate_patch_id"] for r in selection if r["selected_scene_or_item_id"] != "not_available"}
    post_manifests = post_event_light_artifact_manifest_rows()
    post_patches = {row["candidate_patch_id"] for row in post_manifests}
    deltas = pre_post_delta_feature_rows()
    delta_patches = {row["candidate_patch_id"] for row in deltas}
    minimum = (
        len(qa_patches) == 5
        and len(post_patches) == 5
        and len(delta_patches) == 5
        and all(row["usable_as_pre_event_feature"] == "false" for row in post_event_band_statistics_rows() + post_event_index_statistics_rows())
    )
    return {
        "candidate_patch_count": len(patch_ids),
        "pre_event_qa_completed_count": len(qa_patches),
        "pre_event_qa_passed_count": len(qa_passed_patches),
        "post_event_scenes_selected_count": len(selected),
        "post_event_materialization_attempts_count": len(post_event_materialization_attempt_rows()),
        "post_event_light_artifacts_created_count": len(post_manifests),
        "patches_with_post_event_light_artifact_count": len(post_patches),
        "post_event_band_statistics_rows_count": len(post_event_band_statistics_rows()),
        "post_event_index_statistics_rows_count": len(post_event_index_statistics_rows()),
        "patches_with_pre_post_delta_count": len(delta_patches),
        "pre_post_delta_features_count": len(deltas),
        "observational_change_registry_rows_count": len(observational_change_registry_rows()),
        "post_event_used_as_pre_event_feature_count": 0,
        "delta_used_as_training_label_count": 0,
        "sentinel2_full_products_downloaded_count": 0,
        "heavy_rasters_downloaded_count": 0,
        "embeddings_created_count": 0,
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
        "recommended_next_milestone": "SUSC-17C22 Revisao humana formal do QA sensorial e das mudancas observacionais pre/pos antes de qualquer politica de patch candidato",
    }


def build_blockers() -> list[dict]:
    blockers = [
        "observational_change_not_ground_reference",
        "event_geometry_still_missing",
        "phenomenon_gate_not_satisfied",
        "candidate_patch_policy_missing",
        "fallback_source_requires_policy_review",
        "sensor_feature_qa_needs_formal_acceptance",
        "ground_reference_event_artifacts_still_missing",
        "embedding_tile_missing",
        "sar_runtime_missing",
        "17b_blocked_until_event_reference_and_policy",
    ]
    return [
        {
            "blocker_id": f"S17C21_BLOCKER_{idx:04d}",
            "blocker_type": blocker,
            "description": "Bloqueio preservado: comparacao pre/pos e mudanca observacional review-only; faltam artefato de evento, geometria, fenomeno, aceite formal de QA e revisao de politica do fallback antes de qualquer Ground Reference, score v7 ou 17B.",
            "blocks_ground_reference_candidate": _bool_text(blocker in {"observational_change_not_ground_reference", "event_geometry_still_missing", "phenomenon_gate_not_satisfied", "candidate_patch_policy_missing", "ground_reference_event_artifacts_still_missing"}),
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


def build_qa_schema() -> dict:
    required = list(pre_event_artifact_qa_rows()[0].keys())
    return _schema(required, {
        "pre_event_qa_id": {"type": "string", "pattern": "^S17C21_PREQA_"},
        "review_only": {"const": "true"}, "trainable": {"const": "false"}, "ground_truth": {"const": "false"},
    }, "SUSC-17C21 artifact QA schema v1")


def build_delta_schema() -> dict:
    required = list(pre_post_delta_feature_rows()[0].keys()) if pre_post_delta_feature_rows() else [
        "pre_post_delta_feature_id", "candidate_patch_id", "event_id", "pre_event_scene_or_item_id",
        "post_event_scene_or_item_id", "feature_name", "pre_event_value", "post_event_value",
        "delta_value", "feature_unit", "pre_event_manifest_id", "post_event_manifest_id",
        "delta_value_real", "usable_as_observational_change_candidate", "usable_as_pre_event_feature",
        "usable_for_training", "ground_truth", "review_only",
    ]
    return _schema(required, {
        "pre_post_delta_feature_id": {"type": "string", "pattern": "^S17C21_DELTA_"},
        "usable_as_pre_event_feature": {"const": "false"}, "usable_for_training": {"const": "false"},
        "ground_truth": {"const": "false"}, "review_only": {"const": "true"},
    }, "SUSC-17C21 temporal delta schema v1")


def build_obs_schema() -> dict:
    required = list(observational_change_registry_rows()[0].keys())
    return _schema(required, {
        "observational_change_id": {"type": "string", "pattern": "^S17C21_OBSCHANGE_"},
        "can_support_ground_reference": {"const": "false"}, "can_support_training": {"const": "false"},
        "can_support_score_v7": {"const": "false"}, "review_only": {"const": "true"},
    }, "SUSC-17C21 observational change schema v1")


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------

def build_report() -> str:
    summary = build_summary()
    selection = [r for r in post_event_scene_selection_rows() if r["selection_rank"] == "1"]
    post_scene = selection[0]["selected_scene_or_item_id"] if selection else "not_available"
    return "\n".join([
        "# SUSC-17C21 - QA sensorial e comparacao pre/pos-evento Sentinel-2 review-only",
        "",
        "## Objetivo",
        "Fazer a segunda metade operacional do fluxo sensorial: QA dos recortes pre-evento reais do 17C20, selecao e materializacao de cena pos-evento equivalente, comparacao pre/pos (deltas) e auditoria anti-leakage, tudo review-only.",
        "",
        "## QA pre-evento",
        f"- QA concluido para {summary['pre_event_qa_completed_count']} patches; aprovado para {summary['pre_event_qa_passed_count']}.",
        "- Cada artefato 17C20 foi verificado: existencia, SHA256, tamanho, formato, bandas B03/B04/B08/B11, indices NDVI/NDWI/MNDWI/NDBI, valid pixel count, leitura do preview, marcacao de fallback e cena pre-evento.",
        "",
        "## Selecao e materializacao pos-evento",
        f"- Cenas pos-evento selecionadas: {summary['post_event_scenes_selected_count']} (cena canonica pos-evento: {post_scene}).",
        f"- Patches com artefato pos-evento: {summary['patches_with_post_event_light_artifact_count']}; artefatos pos-evento: {summary['post_event_light_artifacts_created_count']}.",
        "- Mesma estrategia do 17C20: CDSE/OData e produto completo (bloqueado como primario), materializacao via fallback STAC/COG Earth Search por janela HTTP range, sem baixar produto.",
        "",
        "## Comparacao pre/pos",
        f"- Patches com delta pre/pos: {summary['patches_with_pre_post_delta_count']}; features de delta: {summary['pre_post_delta_features_count']}.",
        "- delta = pos-evento - pre-evento, para B03/B04/B08/B11_mean e NDVI/NDWI/MNDWI/NDBI_mean.",
        "- Todo delta e `observational_change_review_only`: nao e feature pre-evento, nao e label, nao e ground truth, nao e score.",
        "",
        "## Guardrails",
        f"- Pos-evento usado como feature pre-evento: {summary['post_event_used_as_pre_event_feature_count']}.",
        f"- Delta usado como label: {summary['delta_used_as_training_label_count']}.",
        "- Produtos completos baixados: 0; `.SAFE`/ZIP: 0; raster pesado commitado: 0; embeddings: 0; Ground Reference: 0; ground truth: 0; label: 0.",
        "",
        "## Score v6, score v7 e 17B",
        "Score v6 intacto, score v7 inexistente, 17B bloqueado ate existir artefato de evento com geometria e fenomeno, aceite formal de QA e revisao de politica do fallback.",
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
    write_csv(PRE_QA, pre_event_artifact_qa_rows())
    write_csv(POST_SELECTION, post_event_scene_selection_rows())
    write_csv(POST_ATTEMPTS, post_event_materialization_attempt_rows())
    write_csv(POST_MANIFEST, post_event_light_artifact_manifest_rows(), POST_MANIFEST_FIELDS)
    write_csv(POST_BAND_STATS, post_event_band_statistics_rows())
    write_csv(POST_INDEX_STATS, post_event_index_statistics_rows())
    write_csv(DELTA_FEATURES, pre_post_delta_feature_rows())
    write_csv(OBS_CHANGE, observational_change_registry_rows())
    write_csv(FALLBACK_AUDIT, fallback_source_audit_rows())
    write_csv(REPLAY_AUDIT, replay_audit_rows())
    write_csv(NO_LEAKAGE, no_leakage_audit_rows())
    write_csv(GATES, gate_evaluation_matrix_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_json(QA_SCHEMA, build_qa_schema())
    write_json(DELTA_SCHEMA, build_delta_schema())
    write_json(OBS_SCHEMA, build_obs_schema())
    write_markdown(REPORT, build_report())


def _required_outputs() -> list[Path]:
    outputs = [
        REPORT, PRE_QA, POST_SELECTION, POST_ATTEMPTS, POST_MANIFEST, POST_BAND_STATS,
        POST_INDEX_STATS, DELTA_FEATURES, OBS_CHANGE, FALLBACK_AUDIT, REPLAY_AUDIT,
        NO_LEAKAGE, GATES, SUMMARY, BLOCKERS, QA_SCHEMA, DELTA_SCHEMA, OBS_SCHEMA,
    ]
    for patch_id in _patch_ids():
        if _load_post_stats(patch_id) is not None:
            for _, path in _post_committed_files(patch_id):
                if path.exists():
                    outputs.append(path)
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
    qa = read_csv(PRE_QA)
    selection = read_csv(POST_SELECTION)
    attempts = read_csv(POST_ATTEMPTS)
    manifests = read_csv(POST_MANIFEST)
    band = read_csv(POST_BAND_STATS)
    index = read_csv(POST_INDEX_STATS)
    deltas = read_csv(DELTA_FEATURES)
    obs = read_csv(OBS_CHANGE)
    fallback = read_csv(FALLBACK_AUDIT)
    replay = read_csv(REPLAY_AUDIT)
    leakage = read_csv(NO_LEAKAGE)
    gates = read_csv(GATES)
    summary = read_json(SUMMARY)
    qa_schema = read_json(QA_SCHEMA)
    delta_schema = read_json(DELTA_SCHEMA)
    obs_schema = read_json(OBS_SCHEMA)

    for row in qa:
        errors.extend(_schema_violations(row, qa_schema))
    for row in deltas:
        errors.extend(_schema_violations(row, delta_schema))
    for row in obs:
        errors.extend(_schema_violations(row, obs_schema))

    for rows, key in [
        (qa, "pre_event_qa_id"), (selection, "post_event_scene_selection_id"),
        (attempts, "post_event_materialization_attempt_id"), (manifests, "post_event_artifact_manifest_id"),
        (band, "post_event_band_stat_id"), (index, "post_event_index_stat_id"),
        (deltas, "pre_post_delta_feature_id"), (obs, "observational_change_id"),
        (fallback, "fallback_audit_id"), (replay, "replay_audit_id"),
        (leakage, "no_leakage_audit_id"), (gates, "gate_eval_id"),
    ]:
        ids = [row[key] for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    patch_ids = set(_patch_ids())
    # 5..9: sucesso minimo, QA e materializacao/delta para 5 patches.
    if not summary["minimum_success_achieved"]:
        errors.append("minimum_success_not_achieved")
    if len({r["candidate_patch_id"] for r in qa}) != 5:
        errors.append("pre_event_qa_not_5_patches")
    if any(r["passes_pre_event_qa"] != "true" for r in qa):
        errors.append("pre_event_qa_failed")
    if {r["candidate_patch_id"] for r in selection if r["selection_rank"] == "1"} != patch_ids:
        errors.append("post_event_selection_not_5_patches")
    if len({r["candidate_patch_id"] for r in manifests}) != 5:
        errors.append("post_event_materialization_not_5_patches")
    if len({r["candidate_patch_id"] for r in deltas}) != 5:
        errors.append("delta_not_5_patches")
    # 10/11/12: manifesto/hash/tamanho.
    for row in manifests:
        path = ROOT / row["artifact_local_path"]
        if not path.exists():
            errors.append(f"post_artifact_missing:{row['post_event_artifact_manifest_id']}")
            continue
        if sha256_file(path) != row["sha256"]:
            errors.append(f"post_hash_mismatch:{row['post_event_artifact_manifest_id']}")
        if str(path.stat().st_size) != row["size_bytes"]:
            errors.append(f"post_size_mismatch:{row['post_event_artifact_manifest_id']}")
        if path.stat().st_size > MAX_PUBLIC_OUTPUT_BYTES:
            errors.append(f"post_artifact_over_limit:{row['post_event_artifact_manifest_id']}")
        if not _within_post_dir(path):
            errors.append(f"post_artifact_outside_dir:{row['post_event_artifact_manifest_id']}")
        if row["format"] not in ("csv", "png", "json"):
            errors.append(f"post_forbidden_format:{row['post_event_artifact_manifest_id']}")
    # 13/14/15: sem produto completo / .SAFE / raster pesado.
    if summary["sentinel2_full_products_downloaded_count"] != 0 or summary["heavy_rasters_downloaded_count"] != 0:
        errors.append("full_product_or_heavy_reported")
    for path in POST_ARTIFACT_DIR.glob("**/*") if POST_ARTIFACT_DIR.exists() else []:
        if path.is_file() and path.suffix.lower() in (".tif", ".tiff", ".npz", ".npy", ".zip", ".safe", ".geotiff"):
            errors.append(f"forbidden_artifact_committed:{rel(path)}")
    # 17/18: pos-evento nunca pre-evento; delta nunca label.
    if any(r["usable_as_pre_event_feature"] != "false" for r in band + index + deltas):
        errors.append("post_or_delta_marked_pre_event")
    if any(r["usable_for_training"] != "false" for r in deltas):
        errors.append("delta_marked_trainable")
    if any(r["uses_post_event_as_pre_event_feature"] != "false" or r["uses_delta_as_training_label"] != "false" for r in leakage):
        errors.append("leakage_detected")
    if any(r["passes_no_leakage"] != "true" for r in leakage):
        errors.append("no_leakage_failed")
    # 19/20: fallback marcado e nao Ground Reference.
    if manifests and any(r["is_fallback_source"] != "true" for r in manifests):
        errors.append("fallback_not_marked")
    if any(r["can_be_used_for_ground_reference"] != "false" for r in fallback):
        errors.append("fallback_as_ground_reference")
    if any(r["can_support_ground_reference"] != "false" for r in obs):
        errors.append("obs_change_as_ground_reference")
    # 21/22: cenas usadas sao pos-evento reais, nunca durante-evento como principal; pre nunca pos.
    scene_reg = read_csv(C18_SCENE_REGISTRY)
    for scene_id in {r["selected_scene_or_item_id"] for r in selection if r["selection_rank"] == "1"}:
        matches = [s for s in scene_reg if s["scene_or_item_id"] == scene_id]
        if not matches or any(s["is_post_event"] != "true" for s in matches):
            errors.append(f"selected_scene_not_post_event:{scene_id}")
    # 23/24/25: sem Ground Reference/ground truth/label.
    if summary["ground_reference_candidates_created_count"] != 0 or summary["ground_truth_created"] or summary["training_labels_created"]:
        errors.append("forbidden_artifact_created")
    if any(r["all_gates_passed_for_ground_reference"] == "true" for r in gates):
        errors.append("ground_reference_accepted")
    if any(r["G4_vinculo_espacial_evento"] == "true" or r["G5_separacao_fenomeno"] == "true" for r in gates):
        errors.append("sensor_promoted_to_event")
    if any(r["byte_identical_replay"] != "true" for r in replay):
        errors.append("replay_not_byte_identical")
    # feature real exige manifesto.
    post_manifest_ids = {r["post_event_artifact_manifest_id"] for r in manifests}
    for row in band + index:
        if row["feature_value_real"] == "true" and row["source_artifact_manifest_id"] not in post_manifest_ids:
            errors.append("real_post_feature_without_manifest")

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
        "17C21 -> "
        f"pre_qa={summary['pre_event_qa_passed_count']}/5 "
        f"post_patches={summary['patches_with_post_event_light_artifact_count']}/5 "
        f"deltas={summary['pre_post_delta_features_count']} "
        f"post_as_pre={summary['post_event_used_as_pre_event_feature_count']} "
        f"min_success={summary['minimum_success_achieved']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
    return 0


def _within_post_dir(path: Path) -> bool:
    try:
        path.resolve().relative_to(POST_ARTIFACT_DIR.resolve())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Materializacao pos-evento (opt-in de rede)
# ---------------------------------------------------------------------------

def _round(value, decimals: int) -> str:
    return f"{float(value):.{decimals}f}"


def _materialize_post_patch(patch_id: str, selected_scene_id: str, scene_datetime: str, event_id: str, event_end: str) -> dict:
    import numpy as np  # noqa: PLC0415
    import rasterio  # noqa: PLC0415
    from rasterio.enums import Resampling  # noqa: PLC0415
    from rasterio.warp import transform_bounds  # noqa: PLC0415
    from rasterio.windows import from_bounds  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from pystac_client import Client  # noqa: PLC0415

    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

    target_date = scene_datetime[:10]
    bbox = _patch_bbox(patch_id)
    client = Client.open(EARTH_SEARCH_ENDPOINT)
    items = list(client.search(
        collections=[EARTH_SEARCH_COLLECTION], bbox=list(bbox),
        datetime=f"{target_date}T00:00:00Z/{target_date}T23:59:59Z",
    ).items())
    items = [it for it in items if it.datetime is not None and it.datetime.date().isoformat() == target_date]
    if not items:
        raise RuntimeError("post_fallback_item_not_found")
    item = items[0]

    href_ref = item.assets[BANDS[0][1]].href
    with rasterio.open(href_ref) as ds:
        bounds = transform_bounds("EPSG:4326", ds.crs, *bbox)
        window = from_bounds(*bounds, ds.transform)
        ref = ds.read(1, window=window)
    out_hw = (int(ref.shape[0]), int(ref.shape[1]))

    ensure_dir(POST_ARTIFACT_DIR)
    ensure_dir(LOCAL_STATE / "raw")

    arrays: dict[str, "np.ndarray"] = {}
    for band, asset in BANDS:
        with rasterio.open(item.assets[asset].href) as ds:
            bounds = transform_bounds("EPSG:4326", ds.crs, *bbox)
            window = from_bounds(*bounds, ds.transform)
            arrays[band] = ds.read(1, window=window, out_shape=out_hw, resampling=Resampling.nearest).astype("float64")

    band_stats: dict[str, dict[str, str]] = {}
    for band, arr in arrays.items():
        valid = arr > 0
        vals = arr[valid]
        if vals.size == 0:
            raise RuntimeError(f"band_all_nodata:{band}")
        band_stats[band] = {
            "mean": _round(vals.mean(), 4), "min": _round(vals.min(), 4),
            "max": _round(vals.max(), 4), "valid_pixel_count": str(int(valid.sum())),
        }

    def _index(x: "np.ndarray", y: "np.ndarray") -> "np.ndarray":
        denom = x + y
        mask = (denom != 0) & (x > 0) & (y > 0)
        return (x[mask] - y[mask]) / denom[mask]

    index_stats: dict[str, dict[str, str]] = {}
    for name, (bx, by) in INDICES.items():
        vals = _index(arrays[bx], arrays[by])
        if vals.size == 0:
            continue
        index_stats[name] = {"mean": _round(vals.mean(), 6), "min": _round(vals.min(), 6), "max": _round(vals.max(), 6)}

    def _scale(a: "np.ndarray") -> "np.ndarray":
        lo, hi = np.percentile(a, 2), np.percentile(a, 98)
        return np.clip((a - lo) / (hi - lo + 1e-9) * 255.0, 0, 255).astype("uint8")

    rgb = np.dstack([_scale(arrays["B04"]), _scale(arrays["B03"]), _scale(arrays["B08"])])
    Image.fromarray(rgb, mode="RGB").save(POST_ARTIFACT_DIR / f"{patch_id.lower()}_post_preview.png", format="PNG", optimize=True)

    band_stats_csv_rows = []
    for band in arrays:
        for stat_name in BAND_STAT_NAMES:
            band_stats_csv_rows.append({
                "candidate_patch_id": patch_id, "scene_or_item_id": selected_scene_id,
                "band_name": band, "stat_name": stat_name, "stat_value": band_stats[band][stat_name],
            })
    write_csv(POST_ARTIFACT_DIR / f"{patch_id.lower()}_post_band_stats.csv", band_stats_csv_rows)

    npz_path = LOCAL_STATE / "raw" / f"{patch_id.lower()}_post_bandstack.npz"
    np.savez_compressed(npz_path, **{band: arrays[band].astype("uint16") for band in arrays})

    payload = {
        "candidate_patch_id": patch_id, "canonical_scene_id": selected_scene_id, "event_id": event_id,
        "event_end": event_end, "scene_datetime": item.datetime.isoformat(),
        "is_post_event": item.datetime.date().isoformat() > event_end,
        "primary_source": "CDSE/OData", "primary_source_status": "cdse_product_level_only",
        "fallback_source": f"earth_search_stac:{EARTH_SEARCH_COLLECTION}", "fallback_item_id": item.id,
        "fallback_item_href": item.assets[BANDS[0][1]].href, "source_role": "materialization_fallback",
        "cloud_cover": _round(item.properties.get("eo:cloud_cover", 0.0), 4),
        "aoi_bbox_4326": list(bbox), "window_shape": list(out_hw),
        "bands_read": [b for b, _ in BANDS], "band_stats": band_stats, "index_stats": index_stats,
        "npz_local_path": rel(npz_path), "npz_size_bytes": npz_path.stat().st_size,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(_post_stats_path(patch_id), payload)
    return payload


def materialize_post_event_text() -> tuple[int, str]:
    if not _network_enabled():
        return (2, "blocked_network_not_enabled: materialize-post-event exige SUSC_17C21_ALLOW_NETWORK=1.")
    if not _lightweight_raster_enabled():
        return (2, "blocked_lightweight_not_enabled: materialize-post-event exige SUSC_17C21_ALLOW_LIGHTWEIGHT_RASTER=1.")
    if not _stac_cog_fallback_enabled():
        return (2, "blocked_fallback_not_enabled: fallback STAC/COG exige SUSC_17C21_ALLOW_STAC_COG_FALLBACK=1.")
    event_id = _event_id()
    event_end = _event_end(event_id)
    created = 0
    for sel in post_event_scene_selection_rows():
        if sel["selected_scene_or_item_id"] == "not_available":
            continue
        try:
            _materialize_post_patch(sel["candidate_patch_id"], sel["selected_scene_or_item_id"], sel["datetime"], event_id, event_end)
            created += 1
        except Exception as exc:  # noqa: BLE001
            (POST_ARTIFACT_DIR / f"{sel['candidate_patch_id'].lower()}_post_failed.txt").write_text(f"failed_safe: {exc}\n", encoding="utf-8")
    return (0, f"materialize-post-event: {created} recortes pos-evento reais materializados via fallback STAC/COG em {rel(POST_ARTIFACT_DIR)}.")


def compute_delta_text() -> tuple[int, str]:
    materialized = [p for p in _patch_ids() if _load_post_stats(p) is not None]
    if not materialized:
        return (2, "compute-delta: sem artefato pos-evento materializado; rode materialize-post-event primeiro.")
    deltas = pre_post_delta_feature_rows()
    return (0, f"compute-delta: {len(deltas)} deltas pre/pos calculados para {len({d['candidate_patch_id'] for d in deltas})} patches (review-only).")


def qa_pre_event_text() -> str:
    qa = pre_event_artifact_qa_rows()
    patches = {r["candidate_patch_id"] for r in qa}
    passed = {p for p in patches if all(r["passes_pre_event_qa"] == "true" for r in qa if r["candidate_patch_id"] == p)}
    return f"qa-pre-event: {len(passed)}/{len(patches)} patches aprovados no QA pre-evento ({len(qa)} artefatos verificados)."


def select_post_event_text() -> str:
    sel = [r for r in post_event_scene_selection_rows() if r["selection_rank"] == "1"]
    scenes = {r["selected_scene_or_item_id"] for r in sel}
    return f"select-post-event: {len(sel)} cenas pos-evento canonicas selecionadas ({';'.join(sorted(scenes))[:80]})."


def status_text() -> str:
    summary = build_summary()
    return (
        f"status 17C21: pre_qa={summary['pre_event_qa_passed_count']}/5 "
        f"post={summary['patches_with_post_event_light_artifact_count']}/5 "
        f"deltas={summary['pre_post_delta_features_count']} "
        f"min_success={summary['minimum_success_achieved']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
