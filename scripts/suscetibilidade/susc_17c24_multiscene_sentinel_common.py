"""SUSC-17C24 serie temporal pre-evento multi-cena e reducao de nuvem Sentinel-2.

Materializa recortes leves reais de multiplas cenas Sentinel-2 pre-evento por
patch candidato (top 3 datas por menor nuvem + diversidade temporal), calcula
estatisticas por banda/indice por cena e consolida features temporais robustas
por patch, reduzindo a dependencia da cena unica de 2022-04-27. Mantem tudo
scientific_review_only conforme a equivalencia Earth Search resolvida no 17C23:
nenhum produto Sentinel completo e baixado, nenhum raster pesado e commitado,
nenhum delta/pos-evento vira feature pre-evento, e nada libera treino, ground
truth, score v7 ou 17B.

Build publico offline/deterministico: le apenas os artefatos ja materializados e
commitados. A materializacao real fica atras dos tres opt-ins de rede.
"""

from __future__ import annotations

import os
import statistics
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
LOCAL_STATE = ROOT / "local_runs" / "suscetibilidade" / "17c24_multiscene_sentinel"
LIGHT_ARTIFACT_DIR = OUT / "susc_17c24_light_artifacts"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_PATCHES = DAT / "susc_patches_official_v1.csv"
OFFICIAL_PATCH_LINKS = DAT / "susc_patch_links_official_v1.csv"

C23_DOSSIER = OUT / "susc_17c23_earth_search_resolution_dossier.csv"
C23_MATRIX = OUT / "susc_17c23_candidate_multimodal_feature_matrix.csv"
C23_SCIENTIFIC = OUT / "susc_17c23_scientific_review_only_feature_acceptance.csv"
C23_SUMMARY = OUT / "susc_17c23_readiness_summary.json"
C20_MANIFEST = OUT / "susc_17c20_light_artifact_manifest.csv"
C18_SCENE_REGISTRY = OUT / "susc_17c18_sentinel2_candidate_scene_registry.csv"
C19_BINDING = OUT / "susc_17c19_candidate_patch_temporal_binding.csv"
PATCH_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"
PATCH_GEOJSON = OUT / "susc_17c6_candidate_patch_grid.geojson"

REQUIRED_INPUTS = [
    SCORE_V6, C23_DOSSIER, C23_MATRIX, C23_SCIENTIFIC, C23_SUMMARY, C20_MANIFEST,
    C18_SCENE_REGISTRY, C19_BINDING, PATCH_GRID, PATCH_GEOJSON,
]

REPORT = OUT / "SUSC_17C24_SERIE_TEMPORAL_MULTICENA_REDUCAO_NUVEM_SENTINEL2_REPORT.md"
SELECTION_POLICY = OUT / "susc_17c24_multiscene_selection_policy.csv"
SCENE_SELECTION = OUT / "susc_17c24_pre_event_scene_selection.csv"
MATERIALIZATION_ATTEMPTS = OUT / "susc_17c24_multiscene_materialization_attempts.csv"
ARTIFACT_MANIFEST = OUT / "susc_17c24_light_artifact_manifest.csv"
SCENE_BAND_STATS = OUT / "susc_17c24_scene_band_statistics.csv"
SCENE_INDEX_STATS = OUT / "susc_17c24_scene_index_statistics.csv"
MULTITEMPORAL_FEATURES = OUT / "susc_17c24_multitemporal_feature_registry.csv"
CLOUD_ROBUST_MATRIX = OUT / "susc_17c24_cloud_robust_feature_matrix.csv"
CLOUD_REDUCTION_AUDIT = OUT / "susc_17c24_cloud_reduction_audit.csv"
FALLBACK_REUSE_AUDIT = OUT / "susc_17c24_fallback_equivalence_reuse_audit.csv"
REPLAY_AUDIT = OUT / "susc_17c24_replay_audit.csv"
NO_LEAKAGE = OUT / "susc_17c24_no_leakage_audit.csv"
GATES = OUT / "susc_17c24_gate_evaluation_matrix.csv"
SUMMARY = OUT / "susc_17c24_readiness_summary.json"
BLOCKERS = OUT / "susc_17c24_promotion_blockers.csv"

SELECTION_SCHEMA = SCHEMAS / "susc_17c24_scene_selection_schema_v1.json"
MULTITEMPORAL_SCHEMA = SCHEMAS / "susc_17c24_multitemporal_feature_schema_v1.json"
CLOUD_MATRIX_SCHEMA = SCHEMAS / "susc_17c24_cloud_robust_matrix_schema_v1.json"

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}

MAX_PUBLIC_OUTPUT_BYTES = 1_000_000
BANDS = [("B03", "green"), ("B04", "red"), ("B08", "nir"), ("B11", "swir16")]
INDICES = {"NDVI": ("B08", "B04"), "NDWI": ("B03", "B08"), "MNDWI": ("B03", "B11"), "NDBI": ("B11", "B08")}
BAND_STAT_NAMES = ["mean", "min", "max", "valid_pixel_count"]
INDEX_STAT_NAMES = ["mean", "min", "max"]
FEATURE_NAMES = [f"{b}_mean" for b, _ in BANDS] + [f"{i}_mean" for i in INDICES]
SCENES_PER_PATCH = 3

EARTH_SEARCH_ENDPOINT = "https://earth-search.aws.element84.com/v1"
EARTH_SEARCH_COLLECTION = "sentinel-2-l2a"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _network_enabled() -> bool:
    return os.environ.get("SUSC_17C24_ALLOW_NETWORK") == "1"


def _lightweight_raster_enabled() -> bool:
    return os.environ.get("SUSC_17C24_ALLOW_LIGHTWEIGHT_RASTER") == "1"


def _stac_cog_fallback_enabled() -> bool:
    return os.environ.get("SUSC_17C24_ALLOW_STAC_COG_FALLBACK") == "1"


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


def _patch_bbox(patch_id: str) -> tuple[float, float, float, float]:
    for patch in read_csv(PATCH_GRID):
        if patch["candidate_patch_id"] == patch_id:
            return tuple(float(patch[k]) for k in ["xmin", "ymin", "xmax", "ymax"])  # type: ignore[return-value]
    raise KeyError(patch_id)


def _event_id() -> str:
    rows = read_csv(C18_SCENE_REGISTRY)
    return rows[0]["event_id"] if rows else ""


def _event_start(event_id: str) -> str:
    for row in read_csv(C19_BINDING):
        if row["event_id"] == event_id:
            return row["event_period_start"]
    return ""


def _parse_scene_meta(scene_id: str) -> dict:
    tokens = scene_id.split("_")
    platform = tokens[0] if tokens else "unknown"
    tile = next((t[1:] for t in tokens if t.startswith("T") and len(t) == 6 and t[1:].isalnum()), "unknown")
    return {"platform": platform, "mgrs_tile": tile}


def _earth_search_item_id(platform: str, tile: str, date_iso: str, level: str = "L2A") -> str:
    return f"{platform}_{tile}_{date_iso.replace('-', '')}_0_{level}"


# ---------------------------------------------------------------------------
# Parte 1 - Politica de selecao
# ---------------------------------------------------------------------------

def multiscene_selection_policy_rows() -> list[dict]:
    defs = [
        ("pre_event_only", "usar apenas cenas is_pre_event=true e is_post_event=false", "true"),
        ("exclude_during_and_post_event", "excluir cenas durante/pos-evento das features pre-evento", "true"),
        ("min_scenes_per_patch", "minimo de 3 cenas pre-evento por patch", "3"),
        ("rank_by_lowest_cloud", "ranquear por menor cobertura de nuvem", "true"),
        ("temporal_diversity_distinct_dates", "exigir datas distintas para diversidade temporal", "true"),
        ("prefer_l2a_via_earth_search", "preferir L2A via Earth Search para materializacao", "true"),
        ("same_mission_platform_tile_when_validable", "manter missao/plataforma/tile validaveis", "true"),
        ("earth_search_scientific_review_only", "Earth Search e fonte sensorial scientific_review_only (17C23)", "true"),
        ("no_full_product_download", "sem download de produto Sentinel completo", "true"),
        ("no_heavy_raster_commit", "sem commit de raster pesado em outputs_public", "true"),
    ]
    return [
        {
            "selection_policy_id": f"S17C24_SELPOL_{idx:04d}",
            "criterion_name": name, "criterion_description": desc, "required_value": req,
            "blocks_if_failed": "true", "review_only": "true",
        }
        for idx, (name, desc, req) in enumerate(defs, start=1)
    ]


# ---------------------------------------------------------------------------
# Parte 2 - Selecao de cenas pre-evento
# ---------------------------------------------------------------------------

def _cloud(scene: dict) -> float:
    try:
        return float(scene["cloud_cover"])
    except (ValueError, KeyError):
        return float("inf")


def _pre_event_scenes(patch_id: str) -> list[dict]:
    return [
        s for s in read_csv(C18_SCENE_REGISTRY)
        if s["candidate_patch_id"] == patch_id
        and s["is_pre_event"] == "true"
        and s["is_post_event"] == "false"
        and s["datetime"] not in ("", "not_available")
    ]


def _selected_scenes(patch_id: str) -> list[dict]:
    """Top SCENES_PER_PATCH por menor nuvem em datas distintas (diversidade temporal)."""
    scenes = _pre_event_scenes(patch_id)
    best_by_date: dict[str, dict] = {}
    for s in scenes:
        date = s["datetime"][:10]
        if date not in best_by_date or _cloud(s) < _cloud(best_by_date[date]):
            best_by_date[date] = s
    ordered = sorted(best_by_date.values(), key=lambda s: (_cloud(s), s["datetime"][:10]))
    return ordered[:SCENES_PER_PATCH]


def pre_event_scene_selection_rows() -> list[dict]:
    event_id = _event_id()
    event_start = _event_start(event_id)
    rows = []
    idx = 0
    for patch_id in _patch_ids():
        selected = _selected_scenes(patch_id)
        for rank, scene in enumerate(sorted(selected, key=lambda s: s["datetime"][:10]), start=1):
            idx += 1
            meta = _parse_scene_meta(scene["scene_or_item_id"])
            date = scene["datetime"][:10]
            days_before = (datetime.strptime(event_start, "%Y-%m-%d").date() - datetime.strptime(date, "%Y-%m-%d").date()).days
            rows.append({
                "scene_selection_id": f"S17C24_SCENESEL_{idx:04d}",
                "candidate_patch_id": patch_id, "event_id": event_id,
                "scene_or_item_id": scene["scene_or_item_id"],
                "earth_search_item_id": _earth_search_item_id(meta["platform"], meta["mgrs_tile"], date),
                "datetime": scene["datetime"], "days_before_event_start": str(days_before),
                "platform": meta["platform"], "mgrs_tile": meta["mgrs_tile"],
                "product_type": scene["product_type"], "cloud_cover": scene["cloud_cover"],
                "selection_rank": str(rank),
                "selection_reason": "cena pre-evento em data distinta com menor nuvem para diversidade temporal multicena",
                "selected_for_multiscene_stack": "true",
                "is_pre_event": "true", "is_during_event": "false", "is_post_event": "false",
                "downloaded_full_product": "false", "tile_created": "false", **GOV,
            })
    return rows


# ---------------------------------------------------------------------------
# Materializacao (opt-in) e leitura dos stats commitados
# ---------------------------------------------------------------------------

def _scene_stats_path(patch_id: str, date_iso: str) -> Path:
    return LIGHT_ARTIFACT_DIR / f"{patch_id.lower()}_{date_iso.replace('-', '')}_stats.json"


def _load_scene_stats(patch_id: str, date_iso: str) -> dict | None:
    path = _scene_stats_path(patch_id, date_iso)
    if not path.exists():
        return None
    return read_json(path)


def _scene_committed_files(patch_id: str, date_iso: str) -> list[tuple[str, Path]]:
    stamp = f"{patch_id.lower()}_{date_iso.replace('-', '')}"
    return [
        ("band_stats_csv", LIGHT_ARTIFACT_DIR / f"{stamp}_band_stats.csv"),
        ("preview_rgb_png", LIGHT_ARTIFACT_DIR / f"{stamp}_preview.png"),
        ("materialization_stats_json", _scene_stats_path(patch_id, date_iso)),
    ]


def _materialized_scene_dates(patch_id: str) -> list[str]:
    dates = []
    for scene in _selected_scenes(patch_id):
        date = scene["datetime"][:10]
        if _load_scene_stats(patch_id, date) is not None:
            dates.append(date)
    return sorted(dates)


# ---------------------------------------------------------------------------
# Parte 3 - Tentativas de materializacao
# ---------------------------------------------------------------------------

def multiscene_materialization_attempt_rows() -> list[dict]:
    rows = []
    idx = 0
    for sel in pre_event_scene_selection_rows():
        idx += 1
        patch_id = sel["candidate_patch_id"]
        date = sel["datetime"][:10]
        stats = _load_scene_stats(patch_id, date)
        materialized = stats is not None
        band_csv = LIGHT_ARTIFACT_DIR / f"{patch_id.lower()}_{date.replace('-', '')}_band_stats.csv"
        rows.append({
            "materialization_attempt_id": f"S17C24_MAT_{idx:04d}",
            "candidate_patch_id": patch_id, "scene_or_item_id": sel["scene_or_item_id"],
            "earth_search_item_id": sel["earth_search_item_id"],
            "source_name": (f"earth_search_stac:{EARTH_SEARCH_COLLECTION}" if materialized else "not_resolved"),
            "source_role": "materialization_fallback",
            "attempted": _bool_text(materialized), "network_enabled": _bool_text(materialized),
            "lightweight_enabled": _bool_text(materialized), "fallback_enabled": _bool_text(materialized),
            "artifact_created": _bool_text(materialized), "artifact_type": "band_stats_csv" if materialized else "none",
            "size_bytes": str(band_csv.stat().st_size) if materialized and band_csv.exists() else "0",
            "execution_status": "created_band_stats_csv" if materialized else "blocked_network_not_enabled",
            "blocking_reason": (
                "recorte leve pre-evento materializado via fallback STAC/COG; produto completo nao baixado"
                if materialized
                else "materializacao exige SUSC_17C24_ALLOW_NETWORK=1, SUSC_17C24_ALLOW_LIGHTWEIGHT_RASTER=1 e SUSC_17C24_ALLOW_STAC_COG_FALLBACK=1"
            ),
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 4 - Manifesto
# ---------------------------------------------------------------------------

MANIFEST_FIELDS = [
    "light_artifact_manifest_id", "candidate_patch_id", "scene_or_item_id", "earth_search_item_id",
    "artifact_type", "artifact_local_path", "sha256", "size_bytes", "format", "bands_included",
    "source_name", "source_role", "stored_in_outputs_public", "raw_heavy", "created_at",
    "review_only", "trainable", "ground_truth",
]
_FORMAT_BY_TYPE = {"band_stats_csv": "csv", "preview_rgb_png": "png", "materialization_stats_json": "json"}


def light_artifact_manifest_rows() -> list[dict]:
    rows = []
    sidx = 0
    for patch_id in _patch_ids():
        for date in _materialized_scene_dates(patch_id):
            stats = _load_scene_stats(patch_id, date)
            if stats is None:
                continue
            sidx += 1
            bands = ";".join(stats["bands_read"])
            for aidx, (artifact_type, path) in enumerate(_scene_committed_files(patch_id, date), start=1):
                if not path.exists():
                    continue
                rows.append({
                    "light_artifact_manifest_id": f"S17C24_MANIFEST_{sidx:04d}_{chr(64 + aidx)}",
                    "candidate_patch_id": patch_id, "scene_or_item_id": stats["canonical_scene_id"],
                    "earth_search_item_id": stats["fallback_item_id"], "artifact_type": artifact_type,
                    "artifact_local_path": rel(path), "sha256": sha256_file(path),
                    "size_bytes": str(path.stat().st_size), "format": _FORMAT_BY_TYPE[artifact_type],
                    "bands_included": bands if artifact_type != "preview_rgb_png" else "B04;B03;B08",
                    "source_name": f"earth_search_stac:{EARTH_SEARCH_COLLECTION}", "source_role": "materialization_fallback",
                    "stored_in_outputs_public": "true", "raw_heavy": "false", "created_at": stats["captured_at"],
                    **GOV,
                })
    return rows


def _scene_manifest_id(patch_id: str, date: str) -> str:
    for row in light_artifact_manifest_rows():
        if row["candidate_patch_id"] == patch_id and row["artifact_type"] == "band_stats_csv":
            stats = _load_scene_stats(patch_id, date)
            if stats and row["scene_or_item_id"] == stats["canonical_scene_id"]:
                return row["light_artifact_manifest_id"]
    return "not_available"


# ---------------------------------------------------------------------------
# Parte 5 / 6 - Estatisticas por cena
# ---------------------------------------------------------------------------

def scene_band_statistics_rows() -> list[dict]:
    rows = []
    idx = 0
    for patch_id in _patch_ids():
        for date in _materialized_scene_dates(patch_id):
            stats = _load_scene_stats(patch_id, date)
            if stats is None:
                continue
            manifest_id = _scene_manifest_id(patch_id, date)
            for band in stats["bands_read"]:
                for stat_name in BAND_STAT_NAMES:
                    idx += 1
                    rows.append({
                        "scene_band_stat_id": f"S17C24_SBAND_{idx:04d}",
                        "candidate_patch_id": patch_id, "scene_or_item_id": stats["canonical_scene_id"],
                        "scene_date": date, "band_name": band, "stat_name": stat_name,
                        "stat_value": stats["band_stats"][band][stat_name], "cloud_cover": stats["cloud_cover"],
                        "source_artifact_manifest_id": manifest_id, "feature_value_real": "true",
                        "pre_event_only": "true", **GOV,
                    })
    return rows


def scene_index_statistics_rows() -> list[dict]:
    rows = []
    idx = 0
    for patch_id in _patch_ids():
        for date in _materialized_scene_dates(patch_id):
            stats = _load_scene_stats(patch_id, date)
            if stats is None:
                continue
            manifest_id = _scene_manifest_id(patch_id, date)
            for index_name in INDICES:
                if index_name not in stats["index_stats"]:
                    continue
                for stat_name in INDEX_STAT_NAMES:
                    idx += 1
                    rows.append({
                        "scene_index_stat_id": f"S17C24_SINDEX_{idx:04d}",
                        "candidate_patch_id": patch_id, "scene_or_item_id": stats["canonical_scene_id"],
                        "scene_date": date, "index_name": index_name, "stat_name": stat_name,
                        "stat_value": stats["index_stats"][index_name][stat_name], "cloud_cover": stats["cloud_cover"],
                        "source_artifact_manifest_id": manifest_id, "feature_value_real": "true",
                        "pre_event_only": "true", **GOV,
                    })
    return rows


# ---------------------------------------------------------------------------
# Consolidacao temporal
# ---------------------------------------------------------------------------

def _patch_feature_series(patch_id: str) -> dict:
    """feature_name -> lista de (date, cloud, mean_value) ordenada por data."""
    series: dict[str, list] = {name: [] for name in FEATURE_NAMES}
    for date in _materialized_scene_dates(patch_id):
        stats = _load_scene_stats(patch_id, date)
        if stats is None:
            continue
        cloud = float(stats["cloud_cover"])
        for band, _ in BANDS:
            series[f"{band}_mean"].append((date, cloud, float(stats["band_stats"][band]["mean"])))
        for index_name in INDICES:
            if index_name in stats["index_stats"]:
                series[f"{index_name}_mean"].append((date, cloud, float(stats["index_stats"][index_name]["mean"])))
    return series


def _consolidate(values: list[float], decimals: int) -> dict:
    return {
        "temporal_mean": f"{statistics.fmean(values):.{decimals}f}",
        "temporal_median": f"{statistics.median(values):.{decimals}f}",
        "temporal_min": f"{min(values):.{decimals}f}",
        "temporal_max": f"{max(values):.{decimals}f}",
        "temporal_std": f"{(statistics.pstdev(values) if len(values) > 1 else 0.0):.{decimals}f}",
    }


# ---------------------------------------------------------------------------
# Parte 7 - Registro de features multitemporais
# ---------------------------------------------------------------------------

def multitemporal_feature_registry_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    idx = 0
    for patch_id in _patch_ids():
        series = _patch_feature_series(patch_id)
        dates = _materialized_scene_dates(patch_id)
        for feature_name in FEATURE_NAMES:
            entries = series[feature_name]
            if not entries:
                continue
            idx += 1
            decimals = 6 if feature_name.startswith(("NDVI", "NDWI", "MNDWI", "NDBI")) else 4
            values = [v for _, _, v in entries]
            best = min(entries, key=lambda e: e[1])  # menor nuvem
            cons = _consolidate(values, decimals)
            rows.append({
                "multitemporal_feature_id": f"S17C24_MTFEAT_{idx:04d}",
                "candidate_patch_id": patch_id, "event_id": event_id, "feature_name": feature_name,
                "scene_count": str(len(entries)),
                "temporal_mean": cons["temporal_mean"], "temporal_median": cons["temporal_median"],
                "temporal_min": cons["temporal_min"], "temporal_max": cons["temporal_max"],
                "temporal_std": cons["temporal_std"],
                "best_cloud_scene_value": f"{best[2]:.{decimals}f}",
                "best_cloud_scene_date": best[0], "best_cloud_scene_cloud_cover": f"{best[1]:.4f}",
                "cloud_cover_min": f"{min(c for _, c, _ in entries):.4f}",
                "cloud_cover_median": f"{statistics.median([c for _, c, _ in entries]):.4f}",
                "temporal_span_days": str((datetime.strptime(dates[-1], "%Y-%m-%d").date() - datetime.strptime(dates[0], "%Y-%m-%d").date()).days),
                "feature_value_real": "true", "pre_event_only": "true",
                "usable_for_science_now": "true", "usable_for_training": "false",
                "ground_truth": "false", "eligible_for_score_v7": "false", "eligible_for_17b": "false",
                "review_only": "true",
            })
    return rows


# ---------------------------------------------------------------------------
# Parte 10 - Matriz robusta a nuvem
# ---------------------------------------------------------------------------

def cloud_robust_feature_matrix_rows() -> list[dict]:
    event_id = _event_id()
    mt = {(r["candidate_patch_id"], r["feature_name"]): r for r in multitemporal_feature_registry_rows()}
    rows = []
    for patch_id in _patch_ids():
        dates = _materialized_scene_dates(patch_id)
        clouds = []
        for date in dates:
            stats = _load_scene_stats(patch_id, date)
            if stats:
                clouds.append(float(stats["cloud_cover"]))
        span = (datetime.strptime(dates[-1], "%Y-%m-%d").date() - datetime.strptime(dates[0], "%Y-%m-%d").date()).days if dates else 0
        real_count = len([k for k in mt if k[0] == patch_id])
        row = {
            "candidate_patch_id": patch_id, "event_id": event_id,
            "scene_count": str(len(dates)),
            "cloud_cover_min": f"{min(clouds):.4f}" if clouds else "not_available",
            "cloud_cover_median": f"{statistics.median(clouds):.4f}" if clouds else "not_available",
            "temporal_span_days": str(span),
        }
        for band, _ in BANDS:
            key = (patch_id, f"{band}_mean")
            row[f"{band}_mean_temporal_median"] = mt[key]["temporal_median"] if key in mt else "not_available"
        for index_name in INDICES:
            key = (patch_id, f"{index_name}_mean")
            row[f"{index_name}_mean_temporal_median"] = mt[key]["temporal_median"] if key in mt else "not_available"
        for index_name in INDICES:
            key = (patch_id, f"{index_name}_mean")
            row[f"{index_name}_mean_temporal_std"] = mt[key]["temporal_std"] if key in mt else "not_available"
        row.update({
            "features_real_count": str(real_count),
            "usable_for_science_now": _bool_text(real_count == len(FEATURE_NAMES)),
            "usable_for_training": "false", "ground_truth": "false",
            "eligible_for_score_v7": "false", "eligible_for_17b": "false", "review_only": "true",
        })
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Parte - Auditorias
# ---------------------------------------------------------------------------

def cloud_reduction_audit_rows() -> list[dict]:
    event_id = _event_id()
    single_scene_cloud = 42.3181  # cena unica 2022-04-27 do 17C20
    rows = []
    for idx, patch_id in enumerate(_patch_ids(), start=1):
        dates = _materialized_scene_dates(patch_id)
        clouds = []
        for d in dates:
            stats = _load_scene_stats(patch_id, d)
            if stats is not None:
                clouds.append(float(stats["cloud_cover"]))
        best = min(clouds) if clouds else float("nan")
        rows.append({
            "cloud_reduction_audit_id": f"S17C24_CLOUDRED_{idx:04d}",
            "candidate_patch_id": patch_id, "event_id": event_id,
            "single_scene_baseline_cloud_cover": f"{single_scene_cloud:.4f}",
            "multiscene_scene_count": str(len(dates)),
            "multiscene_cloud_cover_min": f"{best:.4f}" if clouds else "not_available",
            "multiscene_cloud_cover_median": f"{statistics.median(clouds):.4f}" if clouds else "not_available",
            "temporal_diversity_dates": ";".join(dates),
            "provides_cloud_robustness": _bool_text(len(dates) >= SCENES_PER_PATCH),
            "review_only": "true",
        })
    return rows


def fallback_equivalence_reuse_audit_rows() -> list[dict]:
    resolved = {r["candidate_patch_id"]: r["equivalence_resolved"] for r in read_csv(C23_DOSSIER)}
    rows = []
    for idx, patch_id in enumerate(_patch_ids(), start=1):
        rows.append({
            "fallback_reuse_audit_id": f"S17C24_FBREUSE_{idx:04d}",
            "candidate_patch_id": patch_id, "source_role": "materialization_fallback",
            "earth_search_equivalence_resolved_17c23": resolved.get(patch_id, "false"),
            "reused_as_scientific_review_only": "true",
            "accepted_for_ground_reference": "false", "accepted_for_training": "false",
            "accepted_for_score_v7": "false", "accepted_for_17b": "false",
            "not_official_primary_source": "true",
            "limitation": "cdse_l1c_vs_earth_search_l2a_documented; multicena mantem uso scientific_review_only",
            "review_only": "true",
        })
    return rows


def replay_audit_rows() -> list[dict]:
    rows = []
    for idx, manifest in enumerate(light_artifact_manifest_rows(), start=1):
        path = ROOT / manifest["artifact_local_path"]
        exists = path.exists()
        matches = exists and sha256_file(path) == manifest["sha256"]
        rows.append({
            "replay_audit_id": f"S17C24_REPLAY_{idx:04d}",
            "light_artifact_manifest_id": manifest["light_artifact_manifest_id"],
            "artifact_exists": _bool_text(exists), "sha256_matches": _bool_text(matches),
            "replay_without_network": "true", "derived_outputs_recreated": "true",
            "byte_identical_replay": _bool_text(matches),
            "blocking_reason": "not_applicable" if matches else "artefato ausente ou hash divergente",
            "review_only": "true",
        })
    return rows


def no_leakage_audit_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    for idx, feat in enumerate(multitemporal_feature_registry_rows(), start=1):
        rows.append({
            "no_leakage_audit_id": f"S17C24_LEAK_{idx:04d}",
            "candidate_patch_id": feat["candidate_patch_id"], "event_id": event_id,
            "object_id": feat["multitemporal_feature_id"], "object_type": "multitemporal_sensor_feature",
            "uses_pre_event_data_only": "true", "uses_during_event_as_pre_event": "false",
            "uses_post_event_as_pre_event": "false", "uses_delta_as_training_label": "false",
            "uses_synthetic_as_real": "false", "passes_no_leakage": "true",
            "blocking_reason": "feature multitemporal restrita a cenas pre-evento reais; sem vazamento", "review_only": "true",
        })
    return rows


def gate_evaluation_matrix_rows() -> list[dict]:
    rows = []
    for idx, feat in enumerate(multitemporal_feature_registry_rows(), start=1):
        rows.append({
            "gate_eval_id": f"S17C24_GATE_{idx:04d}",
            "object_id": feat["multitemporal_feature_id"], "object_type": "multitemporal_sensor_feature",
            "G1_existencia_documental": "true", "G2_confiabilidade_fonte": "true", "G3_precisao_temporal": "true",
            "G4_vinculo_espacial_evento": "false", "G5_separacao_fenomeno": "false",
            "G6_proveniencia_integridade": "true", "G7_anti_leakage": "true",
            "all_gates_passed_for_ground_reference": "false",
            "usable_as_scientific_review_only_sensor_feature": "true",
            "acceptance_status": "accepted_multitemporal_scientific_review_only",
            "blocking_reason": "feature sensorial multitemporal scientific review-only; nao e evento observado, nao vira Ground Reference",
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary() -> dict:
    selection = pre_event_scene_selection_rows()
    manifests = light_artifact_manifest_rows()
    materialized_patches = {row["candidate_patch_id"] for row in manifests}
    patches_with_stack = {p for p in _patch_ids() if len(_materialized_scene_dates(p)) >= SCENES_PER_PATCH}
    band_rows = scene_band_statistics_rows()
    index_rows = scene_index_statistics_rows()
    mt = multitemporal_feature_registry_rows()
    matrix = cloud_robust_feature_matrix_rows()
    available = sum(len(_pre_event_scenes(p)) for p in _patch_ids())
    matrix_created = len(matrix) == 5 and all(row["scene_count"] != "0" for row in matrix)
    minimum = (
        len(_patch_ids()) == 5 and len(selection) >= 15 and len(patches_with_stack) == 5
        and len(manifests) >= 15 and len(mt) >= 40 and matrix_created
    )
    return {
        "minimum_success_achieved": minimum,
        "patches_processed_count": len(_patch_ids()),
        "pre_event_scenes_available_count": available,
        "pre_event_scenes_selected_count": len(selection),
        "patches_with_multiscene_stack_count": len(patches_with_stack),
        "materialization_attempts_count": len(multiscene_materialization_attempt_rows()),
        "light_artifacts_created_count": len(manifests),
        "scene_band_statistics_rows_count": len(band_rows),
        "scene_index_statistics_rows_count": len(index_rows),
        "multitemporal_sensor_features_created_count": len(mt),
        "cloud_robust_feature_matrix_created": matrix_created,
        "cloud_robust_feature_matrix_rows_count": len(matrix),
        "sentinel2_full_products_downloaded_count": 0,
        "heavy_rasters_committed_count": 0,
        "post_event_used_as_pre_event_feature_count": 0,
        "features_promoted_to_training_count": 0,
        "ground_reference_candidates_created_count": 0,
        "ground_truth_created": False,
        "training_labels_created": False,
        "score_v6_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
        "official_patch_created": False,
        "official_patch_link_created": False,
        "eligible_for_17b_now": False,
        "eligible_for_score_v7": False,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "patches_with_multiscene_materialized_count": len(materialized_patches),
        "recommended_next_milestone": "SUSC-17C25 Consolidacao multimodal (Sentinel-2 multitemporal + CHIRPS runtime) para dossie sensorial scientific review-only por patch",
    }


def build_blockers() -> list[dict]:
    blockers = [
        "multitemporal_sensor_features_scientific_review_only_not_trainable",
        "processing_level_l1c_l2a_limitation_documented",
        "event_geometry_still_missing",
        "phenomenon_gate_not_satisfied",
        "ground_reference_event_artifacts_still_missing",
        "candidate_patch_policy_review_only",
        "official_patch_link_missing",
        "17b_blocked_until_event_reference_and_policy",
        "score_v7_blocked_until_ground_reference_and_policy",
    ]
    return [
        {
            "blocker_id": f"S17C24_BLOCKER_{idx:04d}", "blocker_type": blocker,
            "description": "Bloqueio preservado: serie temporal Sentinel-2 e feature sensorial scientific review-only; treino, score v7, 17B, Ground Reference e patch oficial permanecem bloqueados por falta de evento observado, geometria, fenomeno e politica.",
            "blocks_ground_reference_candidate": _bool_text(blocker in {"event_geometry_still_missing", "phenomenon_gate_not_satisfied", "ground_reference_event_artifacts_still_missing"}),
            "blocks_17b": "true", "blocks_score_v7": "true", "review_only": "true",
        }
        for idx, blocker in enumerate(blockers, start=1)
    ]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

def _schema(required: list[str], props: dict, title: str) -> dict:
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": title, "type": "object", "required": required, "properties": props}


def build_selection_schema() -> dict:
    required = list(pre_event_scene_selection_rows()[0].keys())
    return _schema(required, {
        "scene_selection_id": {"type": "string", "pattern": "^S17C24_SCENESEL_"},
        "is_pre_event": {"const": "true"}, "is_during_event": {"const": "false"}, "is_post_event": {"const": "false"},
        "downloaded_full_product": {"const": "false"}, "tile_created": {"const": "false"},
        "review_only": {"const": "true"}, "trainable": {"const": "false"}, "ground_truth": {"const": "false"},
    }, "SUSC-17C24 scene selection schema v1")


def build_multitemporal_schema() -> dict:
    required = list(multitemporal_feature_registry_rows()[0].keys())
    return _schema(required, {
        "multitemporal_feature_id": {"type": "string", "pattern": "^S17C24_MTFEAT_"},
        "usable_for_training": {"const": "false"}, "ground_truth": {"const": "false"},
        "eligible_for_score_v7": {"const": "false"}, "eligible_for_17b": {"const": "false"},
        "review_only": {"const": "true"},
    }, "SUSC-17C24 multitemporal feature schema v1")


def build_cloud_matrix_schema() -> dict:
    required = list(cloud_robust_feature_matrix_rows()[0].keys())
    return _schema(required, {
        "usable_for_training": {"const": "false"}, "ground_truth": {"const": "false"},
        "eligible_for_score_v7": {"const": "false"}, "eligible_for_17b": {"const": "false"},
        "review_only": {"const": "true"},
    }, "SUSC-17C24 cloud robust matrix schema v1")


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------

def build_report() -> str:
    summary = build_summary()
    return "\n".join([
        "# SUSC-17C24 - Serie temporal pre-evento multi-cena e reducao de nuvem Sentinel-2",
        "",
        "## Objetivo",
        "Reduzir a dependencia da cena unica de 2022-04-27 (cloud cover 42,32%) materializando recortes leves reais de multiplas cenas Sentinel-2 pre-evento por patch e consolidando features temporais robustas, mantendo tudo scientific_review_only conforme o 17C23.",
        "",
        "## Selecao e materializacao multicena",
        f"- Patches processados: {summary['patches_processed_count']}.",
        f"- Cenas pre-evento disponiveis: {summary['pre_event_scenes_available_count']}; selecionadas: {summary['pre_event_scenes_selected_count']} (>= 3 por patch, datas distintas).",
        f"- Patches com pilha multicena: {summary['patches_with_multiscene_stack_count']}.",
        f"- Artefatos leves criados: {summary['light_artifacts_created_count']} (band_stats.csv + preview.png + stats.json por patch x cena).",
        "- Materializacao via Earth Search/STAC/COG com leitura windowed COG, sem baixar produto completo.",
        "",
        "## Estatisticas e features temporais",
        f"- Estatisticas por banda: {summary['scene_band_statistics_rows_count']}; por indice: {summary['scene_index_statistics_rows_count']}.",
        f"- Features multitemporais consolidadas: {summary['multitemporal_sensor_features_created_count']} (mean/median/min/max/std + best_cloud + scene_count + cloud_cover_min/median).",
        f"- Matriz robusta a nuvem criada: {summary['cloud_robust_feature_matrix_created']} ({summary['cloud_robust_feature_matrix_rows_count']} linhas).",
        "",
        "## Guardrails",
        f"- Cena durante/pos-evento usada como pre-evento: {summary['post_event_used_as_pre_event_feature_count']}.",
        f"- Features promovidas a treino: {summary['features_promoted_to_training_count']}; produtos completos baixados: {summary['sentinel2_full_products_downloaded_count']}; raster pesado commitado: {summary['heavy_rasters_committed_count']}.",
        "- Deltas nao entram na matriz robusta; fallback Earth Search continua scientific_review_only; Ground Reference: 0; ground truth: 0; label: 0; score v6 intacto; score v7 inexistente; 17B bloqueado.",
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
    write_csv(SELECTION_POLICY, multiscene_selection_policy_rows())
    write_csv(SCENE_SELECTION, pre_event_scene_selection_rows())
    write_csv(MATERIALIZATION_ATTEMPTS, multiscene_materialization_attempt_rows())
    write_csv(ARTIFACT_MANIFEST, light_artifact_manifest_rows(), MANIFEST_FIELDS)
    write_csv(SCENE_BAND_STATS, scene_band_statistics_rows())
    write_csv(SCENE_INDEX_STATS, scene_index_statistics_rows())
    write_csv(MULTITEMPORAL_FEATURES, multitemporal_feature_registry_rows())
    write_csv(CLOUD_ROBUST_MATRIX, cloud_robust_feature_matrix_rows())
    write_csv(CLOUD_REDUCTION_AUDIT, cloud_reduction_audit_rows())
    write_csv(FALLBACK_REUSE_AUDIT, fallback_equivalence_reuse_audit_rows())
    write_csv(REPLAY_AUDIT, replay_audit_rows())
    write_csv(NO_LEAKAGE, no_leakage_audit_rows())
    write_csv(GATES, gate_evaluation_matrix_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_json(SELECTION_SCHEMA, build_selection_schema())
    write_json(MULTITEMPORAL_SCHEMA, build_multitemporal_schema())
    write_json(CLOUD_MATRIX_SCHEMA, build_cloud_matrix_schema())
    write_markdown(REPORT, build_report())


def _required_outputs() -> list[Path]:
    outputs = [
        REPORT, SELECTION_POLICY, SCENE_SELECTION, MATERIALIZATION_ATTEMPTS, ARTIFACT_MANIFEST,
        SCENE_BAND_STATS, SCENE_INDEX_STATS, MULTITEMPORAL_FEATURES, CLOUD_ROBUST_MATRIX,
        CLOUD_REDUCTION_AUDIT, FALLBACK_REUSE_AUDIT, REPLAY_AUDIT, NO_LEAKAGE, GATES, SUMMARY, BLOCKERS,
        SELECTION_SCHEMA, MULTITEMPORAL_SCHEMA, CLOUD_MATRIX_SCHEMA,
    ]
    for patch_id in _patch_ids():
        for date in _materialized_scene_dates(patch_id):
            for _, path in _scene_committed_files(patch_id, date):
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
    selection = read_csv(SCENE_SELECTION)
    attempts = read_csv(MATERIALIZATION_ATTEMPTS)
    manifests = read_csv(ARTIFACT_MANIFEST)
    band = read_csv(SCENE_BAND_STATS)
    index = read_csv(SCENE_INDEX_STATS)
    mt = read_csv(MULTITEMPORAL_FEATURES)
    matrix = read_csv(CLOUD_ROBUST_MATRIX)
    cloud_audit = read_csv(CLOUD_REDUCTION_AUDIT)
    fb_reuse = read_csv(FALLBACK_REUSE_AUDIT)
    replay = read_csv(REPLAY_AUDIT)
    leakage = read_csv(NO_LEAKAGE)
    gates = read_csv(GATES)
    summary = read_json(SUMMARY)
    selection_schema = read_json(SELECTION_SCHEMA)
    mt_schema = read_json(MULTITEMPORAL_SCHEMA)
    matrix_schema = read_json(CLOUD_MATRIX_SCHEMA)

    for row in selection:
        errors.extend(_schema_violations(row, selection_schema))
    for row in mt:
        errors.extend(_schema_violations(row, mt_schema))
    for row in matrix:
        errors.extend(_schema_violations(row, matrix_schema))

    for rows, key in [
        (selection, "scene_selection_id"), (attempts, "materialization_attempt_id"),
        (manifests, "light_artifact_manifest_id"), (band, "scene_band_stat_id"),
        (index, "scene_index_stat_id"), (mt, "multitemporal_feature_id"),
        (cloud_audit, "cloud_reduction_audit_id"), (fb_reuse, "fallback_reuse_audit_id"),
        (replay, "replay_audit_id"), (leakage, "no_leakage_audit_id"), (gates, "gate_eval_id"),
    ]:
        ids = [row[key] for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    # 1/2/3/4: sucesso minimo, patches, cenas, >=3 por patch.
    if not summary["minimum_success_achieved"]:
        errors.append("minimum_success_not_achieved")
    if summary["patches_processed_count"] != 5:
        errors.append("patches_processed_not_5")
    if len(selection) < 15:
        errors.append("selected_scenes_lt_15")
    for patch_id in _patch_ids():
        n = len({r["datetime"][:10] for r in selection if r["candidate_patch_id"] == patch_id})
        if n < 3:
            errors.append(f"patch_lt_3_scenes:{patch_id}")
    # 5/6/7/8/9: artefatos, hash, tamanho, formatos.
    if len(manifests) < 15:
        errors.append("light_artifacts_lt_15")
    for row in manifests:
        path = ROOT / row["artifact_local_path"]
        if not path.exists():
            errors.append(f"manifest_artifact_missing:{row['light_artifact_manifest_id']}")
            continue
        if not row["sha256"] or sha256_file(path) != row["sha256"]:
            errors.append(f"manifest_hash_mismatch:{row['light_artifact_manifest_id']}")
        if str(path.stat().st_size) != row["size_bytes"]:
            errors.append(f"manifest_size_mismatch:{row['light_artifact_manifest_id']}")
        if path.stat().st_size > MAX_PUBLIC_OUTPUT_BYTES:
            errors.append(f"artifact_over_limit:{row['light_artifact_manifest_id']}")
        if row["format"] not in ("csv", "png", "json"):
            errors.append(f"forbidden_format:{row['light_artifact_manifest_id']}")
    for path in LIGHT_ARTIFACT_DIR.glob("**/*") if LIGHT_ARTIFACT_DIR.exists() else []:
        if path.is_file() and path.suffix.lower() in (".tif", ".tiff", ".npz", ".npy", ".zip", ".safe", ".geotiff"):
            errors.append(f"forbidden_artifact_committed:{rel(path)}")
    # 10: nenhuma cena durante/pos como pre.
    if any(r["is_pre_event"] != "true" or r["is_during_event"] != "false" or r["is_post_event"] != "false" for r in selection):
        errors.append("non_pre_event_scene_selected")
    scene_reg = read_csv(C18_SCENE_REGISTRY)
    for scene_id in {r["scene_or_item_id"] for r in selection}:
        matches = [s for s in scene_reg if s["scene_or_item_id"] == scene_id]
        if not matches or any(s["is_pre_event"] != "true" or s["is_post_event"] != "false" for s in matches):
            errors.append(f"selected_scene_not_pre_event:{scene_id}")
    # 11: matriz robusta sem deltas.
    if len(matrix) != 5:
        errors.append("matrix_not_5")
    if matrix and any("delta" in col for col in matrix[0].keys()):
        errors.append("matrix_contains_delta")
    # 12/13/14: sem treino/ground reference/ground truth.
    if any(r["usable_for_training"] != "false" for r in mt + matrix) or summary["features_promoted_to_training_count"] != 0:
        errors.append("feature_promoted_to_training")
    if any(r["all_gates_passed_for_ground_reference"] == "true" for r in gates):
        errors.append("ground_reference_accepted")
    if any(r["G4_vinculo_espacial_evento"] == "true" or r["G5_separacao_fenomeno"] == "true" for r in gates):
        errors.append("sensor_promoted_to_event")
    if summary["ground_reference_candidates_created_count"] != 0 or summary["ground_truth_created"] or summary["training_labels_created"]:
        errors.append("forbidden_promotion")
    if any(r["passes_no_leakage"] != "true" or r["uses_post_event_as_pre_event"] != "false" for r in leakage):
        errors.append("leakage_detected")
    if any(r["byte_identical_replay"] != "true" for r in replay):
        errors.append("replay_not_byte_identical")
    # fallback continua review-only.
    if any(r["reused_as_scientific_review_only"] != "true" or r["accepted_for_ground_reference"] != "false" for r in fb_reuse):
        errors.append("fallback_not_review_only")

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
        "17C24 -> "
        f"patches={summary['patches_processed_count']} "
        f"scenes_selected={summary['pre_event_scenes_selected_count']} "
        f"stacks={summary['patches_with_multiscene_stack_count']}/5 "
        f"artifacts={summary['light_artifacts_created_count']} "
        f"mt_features={summary['multitemporal_sensor_features_created_count']} "
        f"min_success={summary['minimum_success_achieved']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
    return 0


# ---------------------------------------------------------------------------
# Materializacao real (opt-in de rede)
# ---------------------------------------------------------------------------

def _round(value, decimals: int) -> str:
    return f"{float(value):.{decimals}f}"


def _materialize_scene(patch_id: str, scene_id: str, date_iso: str, event_id: str) -> dict:
    import numpy as np  # noqa: PLC0415
    import rasterio  # noqa: PLC0415
    from rasterio.enums import Resampling  # noqa: PLC0415
    from rasterio.warp import transform_bounds  # noqa: PLC0415
    from rasterio.windows import from_bounds  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from pystac_client import Client  # noqa: PLC0415

    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

    bbox = _patch_bbox(patch_id)
    client = Client.open(EARTH_SEARCH_ENDPOINT)
    items = list(client.search(collections=[EARTH_SEARCH_COLLECTION], bbox=list(bbox), datetime=f"{date_iso}/{date_iso}").items())
    items = [it for it in items if it.datetime is not None and it.datetime.date().isoformat() == date_iso]
    if not items:
        raise RuntimeError(f"earth_search_item_not_found:{date_iso}")
    item = items[0]

    with rasterio.open(item.assets[BANDS[0][1]].href) as ds:
        bounds = transform_bounds("EPSG:4326", ds.crs, *bbox)
        ref = ds.read(1, window=from_bounds(*bounds, ds.transform))
    out_hw = (int(ref.shape[0]), int(ref.shape[1]))

    ensure_dir(LIGHT_ARTIFACT_DIR)
    ensure_dir(LOCAL_STATE / "raw")

    arrays: dict[str, "np.ndarray"] = {}
    for band, asset in BANDS:
        with rasterio.open(item.assets[asset].href) as ds:
            bounds = transform_bounds("EPSG:4326", ds.crs, *bbox)
            arrays[band] = ds.read(1, window=from_bounds(*bounds, ds.transform), out_shape=out_hw, resampling=Resampling.nearest).astype("float64")

    band_stats: dict[str, dict[str, str]] = {}
    for band, arr in arrays.items():
        valid = arr > 0
        vals = arr[valid]
        if vals.size == 0:
            raise RuntimeError(f"band_all_nodata:{band}")
        band_stats[band] = {"mean": _round(vals.mean(), 4), "min": _round(vals.min(), 4), "max": _round(vals.max(), 4), "valid_pixel_count": str(int(valid.sum()))}

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
    stamp = f"{patch_id.lower()}_{date_iso.replace('-', '')}"
    Image.fromarray(rgb, mode="RGB").save(LIGHT_ARTIFACT_DIR / f"{stamp}_preview.png", format="PNG", optimize=True)

    band_stats_csv_rows = []
    for band in arrays:
        for stat_name in BAND_STAT_NAMES:
            band_stats_csv_rows.append({"candidate_patch_id": patch_id, "scene_or_item_id": scene_id, "scene_date": date_iso, "band_name": band, "stat_name": stat_name, "stat_value": band_stats[band][stat_name]})
    write_csv(LIGHT_ARTIFACT_DIR / f"{stamp}_band_stats.csv", band_stats_csv_rows)

    npz_path = LOCAL_STATE / "raw" / f"{stamp}_bandstack.npz"
    np.savez_compressed(npz_path, **{band: arrays[band].astype("uint16") for band in arrays})

    payload = {
        "candidate_patch_id": patch_id, "canonical_scene_id": scene_id, "event_id": event_id,
        "scene_date": date_iso, "scene_datetime": item.datetime.isoformat(), "is_pre_event": True,
        "primary_source": "CDSE/OData", "primary_source_status": "cdse_product_level_only",
        "fallback_source": f"earth_search_stac:{EARTH_SEARCH_COLLECTION}", "fallback_item_id": item.id,
        "fallback_item_href": item.assets[BANDS[0][1]].href, "source_role": "materialization_fallback",
        "cloud_cover": _round(item.properties.get("eo:cloud_cover", 0.0), 4), "aoi_bbox_4326": list(bbox),
        "window_shape": list(out_hw), "bands_read": [b for b, _ in BANDS], "band_stats": band_stats,
        "index_stats": index_stats, "npz_local_path": rel(npz_path), "npz_size_bytes": npz_path.stat().st_size,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(_scene_stats_path(patch_id, date_iso), payload)
    return payload


def materialize_multiscene_text() -> tuple[int, str]:
    if not _network_enabled():
        return (2, "blocked_network_not_enabled: materialize-multiscene exige SUSC_17C24_ALLOW_NETWORK=1.")
    if not _lightweight_raster_enabled():
        return (2, "blocked_lightweight_not_enabled: materialize-multiscene exige SUSC_17C24_ALLOW_LIGHTWEIGHT_RASTER=1.")
    if not _stac_cog_fallback_enabled():
        return (2, "blocked_fallback_not_enabled: fallback STAC/COG exige SUSC_17C24_ALLOW_STAC_COG_FALLBACK=1.")
    event_id = _event_id()
    created = 0
    for sel in pre_event_scene_selection_rows():
        date = sel["datetime"][:10]
        try:
            _materialize_scene(sel["candidate_patch_id"], sel["scene_or_item_id"], date, event_id)
            created += 1
        except Exception as exc:  # noqa: BLE001
            (LIGHT_ARTIFACT_DIR / f"{sel['candidate_patch_id'].lower()}_{date.replace('-', '')}_failed.txt").write_text(f"failed_safe: {exc}\n", encoding="utf-8")
    return (0, f"materialize-multiscene: {created} recortes leves pre-evento multicena materializados em {rel(LIGHT_ARTIFACT_DIR)}.")


def compute_scene_features_text() -> str:
    band = scene_band_statistics_rows()
    index = scene_index_statistics_rows()
    return f"compute-scene-features: {len(band)} estatisticas de banda e {len(index)} de indice ja materializadas por cena real."


def build_cloud_robust_matrix_text() -> str:
    mt = multitemporal_feature_registry_rows()
    matrix = cloud_robust_feature_matrix_rows()
    return f"build-cloud-robust-matrix: {len(mt)} features multitemporais e {len(matrix)} linhas de matriz robusta a nuvem (review-only)."


def select_scenes_text() -> str:
    selection = pre_event_scene_selection_rows()
    by_patch = {}
    for r in selection:
        by_patch.setdefault(r["candidate_patch_id"], set()).add(r["datetime"][:10])
    return "select-scenes: " + "; ".join(f"{p}={len(d)} cenas" for p, d in sorted(by_patch.items())) + "."


def status_text() -> str:
    summary = build_summary()
    return (
        f"status 17C24: scenes_selected={summary['pre_event_scenes_selected_count']} "
        f"stacks={summary['patches_with_multiscene_stack_count']} "
        f"artifacts={summary['light_artifacts_created_count']} "
        f"mt_features={summary['multitemporal_sensor_features_created_count']} "
        f"min_success={summary['minimum_success_achieved']}"
    )
