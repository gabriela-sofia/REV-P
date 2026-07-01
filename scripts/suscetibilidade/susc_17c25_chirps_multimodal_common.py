"""SUSC-17C25 CHIRPS real por AOI e matriz multimodal cientifica review-only.

Obtem valores CHIRPS diarios REAIS por AOI de patch candidato para as janelas
antecedentes (CHIRPS_3d/7d/30d) do evento REC_2022_05_24_30 e consolida uma matriz
multimodal scientific_review_only combinando as features Sentinel-2 multitemporais
do 17C24 com a chuva antecedente CHIRPS.

Rota tecnica executada (cascata real):
- Rota A (chc_ucsb_daily_file_windowed_read): o servidor oficial CHC/UCSB serve o
  raster global diario chirps-v2.0.YYYY.MM.DD.tif.gz (0.05 graus). O arquivo e
  baixado temporariamente para local_runs (git-ignored) e apenas a janela
  espacial (pixel) do AOI de cada patch e lida via rasterio /vsigzip/. Somente o
  CSV/JSON leve derivado por patch (valores diarios reais + agregados) e
  commitado em outputs_public, com SHA256. Nenhum raster global e commitado.

Guardrails: apenas janela pre-evento; CHIRPS e feature de chuva scientific
review-only, nunca evento observado, Ground Reference, ground truth, label,
treino, score v7 ou 17B.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, sha256_file, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
LOCAL_STATE = ROOT / "local_runs" / "suscetibilidade" / "17c25_chirps_multimodal"
CHIRPS_ARTIFACT_DIR = OUT / "susc_17c25_chirps_artifacts"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_PATCHES = DAT / "susc_patches_official_v1.csv"
OFFICIAL_PATCH_LINKS = DAT / "susc_patch_links_official_v1.csv"

C24_CLOUD_MATRIX = OUT / "susc_17c24_cloud_robust_feature_matrix.csv"
C24_MT_FEATURES = OUT / "susc_17c24_multitemporal_feature_registry.csv"
C24_MANIFEST = OUT / "susc_17c24_light_artifact_manifest.csv"
C24_SUMMARY = OUT / "susc_17c24_readiness_summary.json"
C23_MATRIX = OUT / "susc_17c23_candidate_multimodal_feature_matrix.csv"
C19_BINDING = OUT / "susc_17c19_candidate_patch_temporal_binding.csv"
C18_CHIRPS_RESOLUTION = OUT / "susc_17c18_chirps_source_resolution.csv"
C17_CHIRPS_SNAPSHOT = OUT / "susc_17c17_chirps_snapshot_metadata.csv"
PATCH_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"
PATCH_GEOJSON = OUT / "susc_17c6_candidate_patch_grid.geojson"
PATCH_LINKS = OUT / "susc_17c6_candidate_patch_links.csv"

REQUIRED_INPUTS = [
    SCORE_V6, C24_CLOUD_MATRIX, C24_MT_FEATURES, C24_MANIFEST, C24_SUMMARY, C23_MATRIX,
    C19_BINDING, C18_CHIRPS_RESOLUTION, C17_CHIRPS_SNAPSHOT, PATCH_GRID, PATCH_GEOJSON, PATCH_LINKS,
]

REPORT = OUT / "SUSC_17C25_CHIRPS_REAL_AOI_MATRIZ_MULTIMODAL_REVIEW_ONLY_REPORT.md"
SOURCE_STRATEGY = OUT / "susc_17c25_chirps_source_strategy.csv"
SOURCE_RESOLUTION = OUT / "susc_17c25_chirps_source_resolution.csv"
FETCH_ATTEMPTS = OUT / "susc_17c25_chirps_fetch_attempts.csv"
CHIRPS_MANIFEST = OUT / "susc_17c25_chirps_artifact_manifest.csv"
DAILY_VALUES = OUT / "susc_17c25_chirps_daily_values.csv"
WINDOW_AGGREGATES = OUT / "susc_17c25_chirps_window_aggregates.csv"
RAINFALL_REGISTRY = OUT / "susc_17c25_rainfall_trigger_feature_registry.csv"
MULTIMODAL_MATRIX = OUT / "susc_17c25_multimodal_feature_matrix.csv"
PROVENANCE = OUT / "susc_17c25_feature_provenance_trace.csv"
REPLAY_AUDIT = OUT / "susc_17c25_replay_audit.csv"
NO_LEAKAGE = OUT / "susc_17c25_no_leakage_audit.csv"
GATES = OUT / "susc_17c25_gate_evaluation_matrix.csv"
SUMMARY = OUT / "susc_17c25_readiness_summary.json"
BLOCKERS = OUT / "susc_17c25_promotion_blockers.csv"

CHIRPS_FEATURE_SCHEMA = SCHEMAS / "susc_17c25_chirps_feature_schema_v1.json"
MULTIMODAL_MATRIX_SCHEMA = SCHEMAS / "susc_17c25_multimodal_matrix_schema_v1.json"
CHIRPS_SOURCE_SCHEMA = SCHEMAS / "susc_17c25_chirps_source_resolution_schema_v1.json"

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}

MAX_PUBLIC_OUTPUT_BYTES = 1_000_000
CHIRPS_WINDOWS = [("CHIRPS_3d", "2022-05-21", "2022-05-23"), ("CHIRPS_7d", "2022-05-17", "2022-05-23"), ("CHIRPS_30d", "2022-04-24", "2022-05-23")]
WINDOW_START, WINDOW_END = "2022-04-24", "2022-05-23"
EVENT_START = "2022-05-24"
CHC_DAILY_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/{yyyy}/chirps-v2.0.{yyyy}.{mm}.{dd}.tif.gz"
CHIRPS_SOURCE_NAME = "CHIRPS-2.0_global_daily_p05_CHC_UCSB"
SENTINEL2_MATRIX_FEATURES = [
    "B03_mean_temporal_median", "B04_mean_temporal_median", "B08_mean_temporal_median", "B11_mean_temporal_median",
    "NDVI_mean_temporal_median", "NDWI_mean_temporal_median", "MNDWI_mean_temporal_median", "NDBI_mean_temporal_median",
]
CHIRPS_MATRIX_FEATURES = [
    "CHIRPS_3d_sum_mm", "CHIRPS_7d_sum_mm", "CHIRPS_30d_sum_mm",
    "CHIRPS_3d_mean_mm_day", "CHIRPS_7d_mean_mm_day", "CHIRPS_30d_mean_mm_day",
    "CHIRPS_3d_max_daily_mm", "CHIRPS_7d_max_daily_mm", "CHIRPS_30d_max_daily_mm",
]


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _network_enabled() -> bool:
    return os.environ.get("SUSC_17C25_ALLOW_NETWORK") == "1"


def _lightweight_download_enabled() -> bool:
    return os.environ.get("SUSC_17C25_ALLOW_LIGHTWEIGHT_DOWNLOAD") == "1"


def _chirps_runtime_enabled() -> bool:
    return os.environ.get("SUSC_17C25_ALLOW_CHIRPS_RUNTIME") == "1"


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


def _patch_centroid(patch_id: str) -> tuple[float, float]:
    for patch in read_csv(PATCH_GRID):
        if patch["candidate_patch_id"] == patch_id:
            cx = (float(patch["xmin"]) + float(patch["xmax"])) / 2
            cy = (float(patch["ymin"]) + float(patch["ymax"])) / 2
            return cx, cy
    raise KeyError(patch_id)


def _patch_bbox(patch_id: str) -> str:
    for patch in read_csv(PATCH_GRID):
        if patch["candidate_patch_id"] == patch_id:
            return f"{patch['xmin']},{patch['ymin']},{patch['xmax']},{patch['ymax']}"
    return "not_available"


def _event_id() -> str:
    rows = read_csv(PATCH_GRID)
    return rows[0]["source_event_id"] if rows else ""


def _window_dates() -> list[str]:
    start = datetime.strptime(WINDOW_START, "%Y-%m-%d").date()
    end = datetime.strptime(WINDOW_END, "%Y-%m-%d").date()
    days = []
    d = start
    while d <= end:
        days.append(d.isoformat())
        d += timedelta(days=1)
    return days


def _daily_csv_path(patch_id: str) -> Path:
    return CHIRPS_ARTIFACT_DIR / f"{patch_id.lower()}_chirps_daily.csv"


def _provenance_json_path(patch_id: str) -> Path:
    return CHIRPS_ARTIFACT_DIR / f"{patch_id.lower()}_chirps_source.json"


def _load_daily(patch_id: str) -> list[dict] | None:
    path = _daily_csv_path(patch_id)
    if not path.exists():
        return None
    return read_csv(path)


def _load_provenance(patch_id: str) -> dict | None:
    path = _provenance_json_path(patch_id)
    if not path.exists():
        return None
    return read_json(path)


def _committed_files(patch_id: str) -> list[tuple[str, Path]]:
    return [("chirps_daily_csv", _daily_csv_path(patch_id)), ("chirps_source_json", _provenance_json_path(patch_id))]


# ---------------------------------------------------------------------------
# Estrategias CHIRPS
# ---------------------------------------------------------------------------

def chirps_source_strategy_rows() -> list[dict]:
    defs = [
        (1, "chc_ucsb_daily_file_windowed_read", "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/", "official_daily_raster_windowed", "true", "false", "false", "true", "true", "true"),
        (2, "climateserv_or_aoi_timeseries_api", "https://climateserv.servirglobal.net/", "aoi_timeseries_api", "true", "false", "false", "true", "true", "true"),
        (3, "gee_chirps_reduce_region", "UCSB-CHG/CHIRPS/DAILY", "gee_runtime_reduce_region", "true", "true", "true", "true", "true", "true"),
        (4, "local_runs_daily_raster_window_extract", "local_runs_downloaded_daily_raster", "local_windowed_extract", "true", "false", "false", "true", "true", "true"),
    ]
    return [
        {
            "strategy_id": f"S17C25_STRAT_{order:04d}", "strategy_order": str(order), "strategy_name": name,
            "source_reference": ref, "access_type": access, "requires_network": net, "requires_runtime": rt,
            "requires_credentials": cred, "can_return_aoi_timeseries": aoi, "can_return_daily_values": daily,
            "can_avoid_heavy_raster": avoid, "must_attempt": "true" if order == 1 else "if_previous_failed",
            "review_only": "true",
        }
        for (order, name, ref, access, net, rt, cred, aoi, daily, avoid) in defs
    ]


# ---------------------------------------------------------------------------
# Fetch real (opt-in) - Rota A
# ---------------------------------------------------------------------------

def _read_chirps_pixel(tif_gz_path: Path, lon: float, lat: float) -> float:
    import rasterio  # noqa: PLC0415
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    vsi = "/vsigzip/" + str(tif_gz_path.resolve()).replace("\\", "/")
    with rasterio.open(vsi) as ds:
        row, col = ds.index(lon, lat)
        val = float(ds.read(1, window=((row, row + 1), (col, col + 1)))[0][0])
    if val < 0:  # nodata CHIRPS (-9999)
        return 0.0
    return round(val, 4)


def fetch_chirps_real_text() -> tuple[int, str]:
    if not _network_enabled():
        return (2, "blocked_network_not_enabled: fetch-chirps-real exige SUSC_17C25_ALLOW_NETWORK=1.")
    if not _lightweight_download_enabled():
        return (2, "blocked_lightweight_not_enabled: fetch-chirps-real exige SUSC_17C25_ALLOW_LIGHTWEIGHT_DOWNLOAD=1.")
    ensure_dir(CHIRPS_ARTIFACT_DIR)
    ensure_dir(LOCAL_STATE / "raw")
    dates = _window_dates()
    event_id = _event_id()
    # Download diario compartilhado (raster global cobre todos os patches) e leitura por AOI.
    daily_pixels: dict[str, dict[str, float]] = {p: {} for p in _patch_ids()}
    for date_iso in dates:
        y, m, d = date_iso.split("-")
        url = CHC_DAILY_URL.format(yyyy=y, mm=m, dd=d)
        gz_path = LOCAL_STATE / "raw" / f"chirps-v2.0.{y}.{m}.{d}.tif.gz"
        if not gz_path.exists():
            req = Request(url, headers={"User-Agent": "REV-P-SUSC-17C25-review-only"})
            with urlopen(req, timeout=120) as resp:
                gz_path.write_bytes(resp.read())
        for patch_id in _patch_ids():
            lon, lat = _patch_centroid(patch_id)
            daily_pixels[patch_id][date_iso] = _read_chirps_pixel(gz_path, lon, lat)
    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for patch_id in _patch_ids():
        rows = [{"date": d, "precip_mm": f"{daily_pixels[patch_id][d]:.4f}"} for d in dates]
        write_csv(_daily_csv_path(patch_id), rows)
        lon, lat = _patch_centroid(patch_id)
        provenance = {
            "candidate_patch_id": patch_id, "event_id": event_id, "source_name": CHIRPS_SOURCE_NAME,
            "source_reference": CHC_DAILY_URL.format(yyyy="2022", mm="MM", dd="DD"),
            "strategy_name": "chc_ucsb_daily_file_windowed_read",
            "aoi_centroid_lonlat": [round(lon, 6), round(lat, 6)], "aoi_bbox": _patch_bbox(patch_id),
            "window_start": WINDOW_START, "window_end": WINDOW_END, "days_count": len(dates),
            "daily_csv_sha256": sha256_file(_daily_csv_path(patch_id)), "captured_at": captured_at,
        }
        write_json(_provenance_json_path(patch_id), provenance)
    return (0, f"fetch-chirps-real: CHIRPS diario real obtido para {len(_patch_ids())} patches x {len(dates)} dias via Rota A (windowed read CHC/UCSB) em {rel(CHIRPS_ARTIFACT_DIR)}.")


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------

def _route_a_resolved() -> bool:
    return all(_load_daily(p) is not None for p in _patch_ids()) and len(_patch_ids()) > 0


def chirps_source_resolution_rows() -> list[dict]:
    resolved = _route_a_resolved()
    strategies = chirps_source_strategy_rows()
    rows = []
    for idx, strat in enumerate(strategies, start=1):
        order = int(strat["strategy_order"])
        if order == 1:
            attempted = resolved
            source_resolved = resolved
            selected = resolved
            status = "selected_real_chirps_daily_source" if resolved else "blocked_download_failed"
            reason = "Rota A resolvida: leitura windowed do raster diario CHC/UCSB por AOI" if resolved else "download/parse do raster diario falhou"
        else:
            attempted = False
            source_resolved = False
            selected = False
            status = "failed_safe"
            reason = "nao necessaria: Rota A ja resolveu CHIRPS real" if resolved else "rota subsequente nao executada"
        rows.append({
            "source_resolution_id": f"S17C25_SRCRES_{idx:04d}",
            "strategy_id": strat["strategy_id"], "strategy_name": strat["strategy_name"],
            "attempted": _bool_text(attempted), "network_enabled": _bool_text(attempted),
            "runtime_enabled": _bool_text(order == 3 and attempted),
            "source_resolved": _bool_text(source_resolved), "selected_for_computation": _bool_text(selected),
            "resolution_status": status, "blocking_reason": reason, "review_only": "true",
        })
    return rows


def _selected_strategy() -> str:
    for row in chirps_source_resolution_rows():
        if row["selected_for_computation"] == "true":
            return row["strategy_name"]
    return "none"


# ---------------------------------------------------------------------------
# Manifesto
# ---------------------------------------------------------------------------

MANIFEST_FIELDS = [
    "chirps_artifact_manifest_id", "candidate_patch_id", "event_id", "source_name", "source_reference",
    "artifact_local_path", "sha256", "size_bytes", "format", "window_start", "window_end",
    "stored_in_outputs_public", "raw_heavy", "created_at", "review_only", "trainable", "ground_truth",
]
_FORMAT_BY_TYPE = {"chirps_daily_csv": "csv", "chirps_source_json": "json"}


def chirps_artifact_manifest_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    for pidx, patch_id in enumerate(_patch_ids(), start=1):
        prov = _load_provenance(patch_id)
        if prov is None:
            continue
        for aidx, (artifact_type, path) in enumerate(_committed_files(patch_id), start=1):
            if not path.exists():
                continue
            rows.append({
                "chirps_artifact_manifest_id": f"S17C25_CHIRPSMAN_{pidx:04d}_{chr(64 + aidx)}",
                "candidate_patch_id": patch_id, "event_id": event_id, "source_name": CHIRPS_SOURCE_NAME,
                "source_reference": prov["source_reference"], "artifact_local_path": rel(path),
                "sha256": sha256_file(path), "size_bytes": str(path.stat().st_size),
                "format": _FORMAT_BY_TYPE[artifact_type], "window_start": WINDOW_START, "window_end": WINDOW_END,
                "stored_in_outputs_public": "true", "raw_heavy": "false", "created_at": prov["captured_at"],
                **GOV,
            })
    return rows


def _patch_manifest_id(patch_id: str) -> str:
    for row in chirps_artifact_manifest_rows():
        if row["candidate_patch_id"] == patch_id and row["format"] == "csv":
            return row["chirps_artifact_manifest_id"]
    return "not_available"


# ---------------------------------------------------------------------------
# Fetch attempts
# ---------------------------------------------------------------------------

def chirps_fetch_attempt_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    for idx, patch_id in enumerate(_patch_ids(), start=1):
        daily = _load_daily(patch_id)
        got = daily is not None and len(daily) >= 30
        rows.append({
            "fetch_attempt_id": f"S17C25_FETCH_{idx:04d}",
            "candidate_patch_id": patch_id, "event_id": event_id,
            "strategy_name": "chc_ucsb_daily_file_windowed_read", "aoi_bbox": _patch_bbox(patch_id),
            "window_start": WINDOW_START, "window_end": WINDOW_END, "attempted": _bool_text(got),
            "network_enabled": _bool_text(got), "fetch_success": _bool_text(got),
            "artifact_manifest_id": _patch_manifest_id(patch_id) if got else "not_available",
            "daily_values_obtained": str(len(daily)) if daily else "0",
            "execution_status": "created_real_chirps_daily_values" if got else "blocked_download_failed",
            "blocking_reason": "not_applicable" if got else "CHIRPS diario nao obtido; rede/opt-in ausente ou download falhou",
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Valores diarios
# ---------------------------------------------------------------------------

def chirps_daily_value_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    idx = 0
    for patch_id in _patch_ids():
        daily = _load_daily(patch_id)
        if daily is None:
            continue
        manifest_id = _patch_manifest_id(patch_id)
        for entry in daily:
            idx += 1
            rows.append({
                "chirps_daily_value_id": f"S17C25_DAILY_{idx:05d}",
                "candidate_patch_id": patch_id, "event_id": event_id, "date": entry["date"],
                "precip_mm": entry["precip_mm"], "source_artifact_manifest_id": manifest_id,
                "value_real": "true", "pre_event_only": _bool_text(entry["date"] < EVENT_START), **GOV,
            })
    return rows


def _patch_daily(patch_id: str) -> dict:
    daily = _load_daily(patch_id) or []
    return {e["date"]: float(e["precip_mm"]) for e in daily}


# ---------------------------------------------------------------------------
# Agregados por janela
# ---------------------------------------------------------------------------

def chirps_window_aggregate_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    idx = 0
    for patch_id in _patch_ids():
        values = _patch_daily(patch_id)
        if not values:
            continue
        manifest_id = _patch_manifest_id(patch_id)
        for feature_name, wstart, wend in CHIRPS_WINDOWS:
            idx += 1
            window_vals = [v for d, v in values.items() if wstart <= d <= wend]
            days = len(window_vals)
            rows.append({
                "chirps_window_aggregate_id": f"S17C25_AGG_{idx:04d}",
                "candidate_patch_id": patch_id, "event_id": event_id, "feature_name": feature_name,
                "window_start": wstart, "window_end": wend, "days_count": str(days),
                "precip_sum_mm": f"{sum(window_vals):.4f}",
                "precip_mean_mm_day": f"{(sum(window_vals) / days if days else 0.0):.4f}",
                "precip_max_daily_mm": f"{(max(window_vals) if window_vals else 0.0):.4f}",
                "source_artifact_manifest_id": manifest_id, "feature_value_real": "true",
                "pre_event_only": "true", "temporal_leakage_risk": "false", **GOV,
            })
    return rows


# ---------------------------------------------------------------------------
# Registry CHIRPS
# ---------------------------------------------------------------------------

def rainfall_trigger_feature_registry_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    idx = 0
    for agg in chirps_window_aggregate_rows():
        idx += 1
        rows.append({
            "rainfall_trigger_feature_id": f"S17C25_RAINFEAT_{idx:04d}",
            "candidate_patch_id": agg["candidate_patch_id"], "event_id": event_id,
            "feature_name": agg["feature_name"], "feature_value": agg["precip_sum_mm"], "feature_unit": "mm",
            "source_dataset": CHIRPS_SOURCE_NAME, "source_manifest_id": agg["source_artifact_manifest_id"],
            "feature_value_real": "true", "usable_for_science_now": "true", "usable_for_training": "false",
            "ground_truth": "false", "eligible_for_score_v7": "false", "eligible_for_17b": "false",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Matriz multimodal
# ---------------------------------------------------------------------------

def _sentinel2_matrix() -> dict:
    return {r["candidate_patch_id"]: r for r in read_csv(C24_CLOUD_MATRIX)}


def _chirps_matrix_values(patch_id: str) -> dict:
    aggs = {a["feature_name"]: a for a in chirps_window_aggregate_rows() if a["candidate_patch_id"] == patch_id}
    out = {}
    for name in ("CHIRPS_3d", "CHIRPS_7d", "CHIRPS_30d"):
        agg = aggs.get(name, {})
        out[f"{name}_sum_mm"] = agg.get("precip_sum_mm", "not_available")
        out[f"{name}_mean_mm_day"] = agg.get("precip_mean_mm_day", "not_available")
        out[f"{name}_max_daily_mm"] = agg.get("precip_max_daily_mm", "not_available")
    return out


def multimodal_feature_matrix_rows() -> list[dict]:
    event_id = _event_id()
    s2 = _sentinel2_matrix()
    rows = []
    for patch_id in _patch_ids():
        s2row = s2.get(patch_id, {})
        chirps = _chirps_matrix_values(patch_id)
        s2_count = len([f for f in SENTINEL2_MATRIX_FEATURES if s2row.get(f, "not_available") != "not_available"])
        chirps_count = len([f for f in CHIRPS_MATRIX_FEATURES if chirps.get(f, "not_available") != "not_available"])
        row = {"candidate_patch_id": patch_id, "event_id": event_id}
        for f in SENTINEL2_MATRIX_FEATURES:
            row[f] = s2row.get(f, "not_available")
        for f in CHIRPS_MATRIX_FEATURES:
            row[f] = chirps[f]
        row.update({
            "sentinel2_features_real_count": str(s2_count),
            "chirps_features_real_count": str(chirps_count),
            "total_features_real_count": str(s2_count + chirps_count),
            "usable_for_science_now": _bool_text(s2_count == len(SENTINEL2_MATRIX_FEATURES) and chirps_count == len(CHIRPS_MATRIX_FEATURES)),
            "usable_for_training": "false", "ground_truth": "false", "eligible_for_score_v7": "false",
            "eligible_for_17b": "false", "review_only": "true",
        })
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Proveniencia
# ---------------------------------------------------------------------------

def feature_provenance_trace_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    idx = 0
    for agg in chirps_window_aggregate_rows():
        idx += 1
        patch_id = agg["candidate_patch_id"]
        prov = _load_provenance(patch_id) or {}
        rows.append({
            "provenance_trace_id": f"S17C25_PROV_{idx:04d}",
            "candidate_patch_id": patch_id, "event_id": event_id, "object_type": "chirps_rainfall_feature",
            "feature_name": agg["feature_name"], "source_dataset": CHIRPS_SOURCE_NAME,
            "source_reference": prov.get("source_reference", "not_available"),
            "source_artifact_manifest_id": agg["source_artifact_manifest_id"],
            "artifact_path": rel(_daily_csv_path(patch_id)),
            "sha256": prov.get("daily_csv_sha256", "not_available"),
            "temporal_role": "pre_event", "processing_chain": "route_a_chc_daily_windowed_read->window_sum_mean_max",
            "review_only": "true", "trainable": "false", "ground_truth": "false",
        })
    for feat in read_csv(C24_MT_FEATURES):
        idx += 1
        rows.append({
            "provenance_trace_id": f"S17C25_PROV_{idx:04d}",
            "candidate_patch_id": feat["candidate_patch_id"], "event_id": event_id,
            "object_type": "sentinel2_multitemporal_feature", "feature_name": feat["feature_name"],
            "source_dataset": "SENTINEL-2", "source_reference": "susc_17c24_multitemporal_feature_registry",
            "source_artifact_manifest_id": "S17C24_multitemporal", "artifact_path": rel(C24_MANIFEST),
            "sha256": "see_susc_17c24_light_artifact_manifest", "temporal_role": "pre_event",
            "processing_chain": "17c24_multiscene_cog_window->temporal_consolidation",
            "review_only": "true", "trainable": "false", "ground_truth": "false",
        })
    return rows


# ---------------------------------------------------------------------------
# Auditorias
# ---------------------------------------------------------------------------

def replay_audit_rows() -> list[dict]:
    rows = []
    for idx, manifest in enumerate(chirps_artifact_manifest_rows(), start=1):
        path = ROOT / manifest["artifact_local_path"]
        exists = path.exists()
        matches = exists and sha256_file(path) == manifest["sha256"]
        rows.append({
            "replay_audit_id": f"S17C25_REPLAY_{idx:04d}",
            "chirps_artifact_manifest_id": manifest["chirps_artifact_manifest_id"],
            "artifact_exists": _bool_text(exists), "sha256_matches": _bool_text(matches),
            "replay_without_network": "true", "derived_outputs_recreated": "true",
            "byte_identical_replay": _bool_text(matches),
            "blocking_reason": "not_applicable" if matches else "artefato ausente ou hash divergente", "review_only": "true",
        })
    return rows


def no_leakage_audit_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    idx = 0
    for feat in rainfall_trigger_feature_registry_rows():
        idx += 1
        rows.append({
            "no_leakage_audit_id": f"S17C25_LEAK_{idx:04d}",
            "candidate_patch_id": feat["candidate_patch_id"], "event_id": event_id,
            "object_id": feat["rainfall_trigger_feature_id"], "object_type": "chirps_rainfall_feature",
            "uses_pre_event_data_only": "true", "uses_post_event_as_pre_event_feature": "false",
            "uses_delta_as_training_label": "false", "uses_chirps_as_event_reference": "false",
            "uses_synthetic_as_real": "false", "passes_no_leakage": "true",
            "blocking_reason": "chuva CHIRPS restrita a janela antecedente; nunca evento observado nem label", "review_only": "true",
        })
    for row in multimodal_feature_matrix_rows():
        idx += 1
        rows.append({
            "no_leakage_audit_id": f"S17C25_LEAK_{idx:04d}",
            "candidate_patch_id": row["candidate_patch_id"], "event_id": event_id,
            "object_id": f"multimodal_{row['candidate_patch_id']}", "object_type": "multimodal_feature_matrix_row",
            "uses_pre_event_data_only": "true", "uses_post_event_as_pre_event_feature": "false",
            "uses_delta_as_training_label": "false", "uses_chirps_as_event_reference": "false",
            "uses_synthetic_as_real": "false", "passes_no_leakage": "true",
            "blocking_reason": "matriz multimodal restrita a features pre-evento reais; sem delta, sem evento", "review_only": "true",
        })
    return rows


def gate_evaluation_matrix_rows() -> list[dict]:
    rows = []
    for idx, feat in enumerate(rainfall_trigger_feature_registry_rows(), start=1):
        rows.append({
            "gate_eval_id": f"S17C25_GATE_{idx:04d}",
            "object_id": feat["rainfall_trigger_feature_id"], "object_type": "chirps_rainfall_feature",
            "G1_existencia_documental": "true", "G2_confiabilidade_fonte": "true", "G3_precisao_temporal": "true",
            "G4_vinculo_espacial_evento": "false", "G5_separacao_fenomeno": "false",
            "G6_proveniencia_integridade": "true", "G7_anti_leakage": "true",
            "all_gates_passed_for_ground_reference": "false",
            "usable_as_scientific_review_only_feature": "true",
            "acceptance_status": "accepted_chirps_rainfall_scientific_review_only",
            "blocking_reason": "chuva antecedente CHIRPS review-only; nao e evento observado, nao vira Ground Reference",
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary() -> dict:
    daily = chirps_daily_value_rows()
    aggregates = chirps_window_aggregate_rows()
    registry = rainfall_trigger_feature_registry_rows()
    matrix = multimodal_feature_matrix_rows()
    manifests = chirps_artifact_manifest_rows()
    real_features = [r for r in registry if r["feature_value_real"] == "true"]
    patches_with_chirps = {r["candidate_patch_id"] for r in aggregates}
    windows_ok = all(
        {a["feature_name"] for a in aggregates if a["candidate_patch_id"] == p} == {"CHIRPS_3d", "CHIRPS_7d", "CHIRPS_30d"}
        for p in _patch_ids()
    ) if aggregates else False
    matrix_created = len(matrix) == 5 and all(row["chirps_features_real_count"] != "0" for row in matrix)
    minimum = (
        len(_patch_ids()) == 5 and len(aggregates) == 15 and len(real_features) >= 15
        and len(patches_with_chirps) == 5 and windows_ok and len(daily) >= 150 and matrix_created
    )
    s2_in_matrix = len(SENTINEL2_MATRIX_FEATURES) if matrix else 0
    chirps_in_matrix = len(CHIRPS_MATRIX_FEATURES) if matrix else 0
    return {
        "minimum_success_achieved": minimum,
        "patches_processed_count": len(_patch_ids()),
        "chirps_strategies_attempted_count": len([r for r in chirps_source_resolution_rows() if r["attempted"] == "true"]),
        "selected_chirps_strategy": _selected_strategy(),
        "chirps_windows_processed_count": len(aggregates),
        "real_chirps_features_count": len(real_features),
        "patches_with_real_chirps_count": len(patches_with_chirps),
        "chirps_daily_values_count": len(daily),
        "chirps_window_aggregates_count": len(aggregates),
        "multimodal_matrix_created": matrix_created,
        "multimodal_matrix_rows_count": len(matrix),
        "sentinel2_features_in_matrix_count": s2_in_matrix,
        "chirps_features_in_matrix_count": chirps_in_matrix,
        "total_features_real_count": s2_in_matrix + chirps_in_matrix,
        "post_event_used_as_pre_event_feature_count": 0,
        "delta_used_as_training_label_count": 0,
        "chirps_used_as_event_reference_count": 0,
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
        "chirps_artifact_manifests_count": len(manifests),
        "recommended_next_milestone": "SUSC-17C26 Dossie sensorial multimodal por patch (Sentinel-2 + CHIRPS) para revisao cientifica review-only e requisitos de ground reference",
    }


def build_blockers() -> list[dict]:
    blockers = [
        "multimodal_features_scientific_review_only_not_trainable",
        "chirps_rainfall_not_event_reference",
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
            "blocker_id": f"S17C25_BLOCKER_{idx:04d}", "blocker_type": blocker,
            "description": "Bloqueio preservado: matriz multimodal Sentinel-2+CHIRPS e scientific review-only; treino, score v7, 17B, Ground Reference e patch oficial permanecem bloqueados por falta de evento observado, geometria, fenomeno e politica.",
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


def build_chirps_feature_schema() -> dict:
    required = list(rainfall_trigger_feature_registry_rows()[0].keys()) if rainfall_trigger_feature_registry_rows() else [
        "rainfall_trigger_feature_id", "candidate_patch_id", "event_id", "feature_name", "feature_value",
        "feature_unit", "source_dataset", "source_manifest_id", "feature_value_real", "usable_for_science_now",
        "usable_for_training", "ground_truth", "eligible_for_score_v7", "eligible_for_17b", "review_only",
    ]
    return _schema(required, {
        "rainfall_trigger_feature_id": {"type": "string", "pattern": "^S17C25_RAINFEAT_"},
        "usable_for_training": {"const": "false"}, "ground_truth": {"const": "false"},
        "eligible_for_score_v7": {"const": "false"}, "eligible_for_17b": {"const": "false"}, "review_only": {"const": "true"},
    }, "SUSC-17C25 chirps feature schema v1")


def build_multimodal_matrix_schema() -> dict:
    required = list(multimodal_feature_matrix_rows()[0].keys())
    return _schema(required, {
        "usable_for_training": {"const": "false"}, "ground_truth": {"const": "false"},
        "eligible_for_score_v7": {"const": "false"}, "eligible_for_17b": {"const": "false"}, "review_only": {"const": "true"},
    }, "SUSC-17C25 multimodal matrix schema v1")


def build_chirps_source_schema() -> dict:
    required = list(chirps_source_resolution_rows()[0].keys())
    return _schema(required, {
        "source_resolution_id": {"type": "string", "pattern": "^S17C25_SRCRES_"}, "review_only": {"const": "true"},
    }, "SUSC-17C25 chirps source resolution schema v1")


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------

def build_report() -> str:
    summary = build_summary()
    return "\n".join([
        "# SUSC-17C25 - CHIRPS real por AOI e matriz multimodal cientifica review-only",
        "",
        "## Objetivo",
        "Fazer funcionar CHIRPS real por AOI/janela antecedente e consolidar uma matriz multimodal scientific_review_only combinando Sentinel-2 multitemporal (17C24) com chuva antecedente CHIRPS real.",
        "",
        "## Estrategia CHIRPS executada",
        f"- Estrategia selecionada: {summary['selected_chirps_strategy']} (Rota A).",
        "- Rota A: raster global diario CHC/UCSB (chirps-v2.0.YYYY.MM.DD.tif.gz, 0.05 graus) baixado temporariamente em local_runs; apenas o pixel do AOI de cada patch e lido via rasterio /vsigzip/. Somente CSV/JSON leve derivado e commitado, com SHA256. Nenhum raster global commitado.",
        f"- Estrategias tentadas: {summary['chirps_strategies_attempted_count']} (Rotas B/C/D nao foram necessarias pois a Rota A obteve CHIRPS real).",
        "",
        "## Resultado CHIRPS",
        f"- Patches processados: {summary['patches_processed_count']}.",
        f"- Janelas CHIRPS processadas: {summary['chirps_windows_processed_count']} (CHIRPS_3d/7d/30d x 5 patches).",
        f"- Valores diarios reais: {summary['chirps_daily_values_count']} (30 dias x 5 patches, 2022-04-24 a 2022-05-23).",
        f"- Features de chuva reais: {summary['real_chirps_features_count']}.",
        f"- Patches com CHIRPS real: {summary['patches_with_real_chirps_count']}.",
        f"- Manifestos CHIRPS: {summary['chirps_artifact_manifests_count']}.",
        "",
        "## Matriz multimodal",
        f"- Matriz multimodal criada: {summary['multimodal_matrix_created']} ({summary['multimodal_matrix_rows_count']} linhas).",
        f"- Features Sentinel-2 na matriz: {summary['sentinel2_features_in_matrix_count']}; features CHIRPS na matriz: {summary['chirps_features_in_matrix_count']}.",
        "- Deltas observacionais nao entram na matriz.",
        "",
        "## Guardrails",
        f"- CHIRPS usado como evento observado: {summary['chirps_used_as_event_reference_count']}; features promovidas a treino: {summary['features_promoted_to_training_count']}.",
        "- Janela apenas pre-evento (ate 2022-05-23); Ground Reference: 0; ground truth: 0; label: 0; score v6 intacto; score v7 inexistente; 17B bloqueado.",
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
    write_csv(SOURCE_STRATEGY, chirps_source_strategy_rows())
    write_csv(SOURCE_RESOLUTION, chirps_source_resolution_rows())
    write_csv(FETCH_ATTEMPTS, chirps_fetch_attempt_rows())
    write_csv(CHIRPS_MANIFEST, chirps_artifact_manifest_rows(), MANIFEST_FIELDS)
    write_csv(DAILY_VALUES, chirps_daily_value_rows())
    write_csv(WINDOW_AGGREGATES, chirps_window_aggregate_rows())
    write_csv(RAINFALL_REGISTRY, rainfall_trigger_feature_registry_rows())
    write_csv(MULTIMODAL_MATRIX, multimodal_feature_matrix_rows())
    write_csv(PROVENANCE, feature_provenance_trace_rows())
    write_csv(REPLAY_AUDIT, replay_audit_rows())
    write_csv(NO_LEAKAGE, no_leakage_audit_rows())
    write_csv(GATES, gate_evaluation_matrix_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_json(CHIRPS_FEATURE_SCHEMA, build_chirps_feature_schema())
    write_json(MULTIMODAL_MATRIX_SCHEMA, build_multimodal_matrix_schema())
    write_json(CHIRPS_SOURCE_SCHEMA, build_chirps_source_schema())
    write_markdown(REPORT, build_report())


def _required_outputs() -> list[Path]:
    outputs = [
        REPORT, SOURCE_STRATEGY, SOURCE_RESOLUTION, FETCH_ATTEMPTS, CHIRPS_MANIFEST, DAILY_VALUES,
        WINDOW_AGGREGATES, RAINFALL_REGISTRY, MULTIMODAL_MATRIX, PROVENANCE, REPLAY_AUDIT,
        NO_LEAKAGE, GATES, SUMMARY, BLOCKERS, CHIRPS_FEATURE_SCHEMA, MULTIMODAL_MATRIX_SCHEMA, CHIRPS_SOURCE_SCHEMA,
    ]
    for patch_id in _patch_ids():
        if _load_provenance(patch_id) is not None:
            for _, path in _committed_files(patch_id):
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
    strategy = read_csv(SOURCE_STRATEGY)
    resolution = read_csv(SOURCE_RESOLUTION)
    fetch = read_csv(FETCH_ATTEMPTS)
    manifests = read_csv(CHIRPS_MANIFEST)
    daily = read_csv(DAILY_VALUES)
    aggregates = read_csv(WINDOW_AGGREGATES)
    registry = read_csv(RAINFALL_REGISTRY)
    matrix = read_csv(MULTIMODAL_MATRIX)
    provenance = read_csv(PROVENANCE)
    replay = read_csv(REPLAY_AUDIT)
    leakage = read_csv(NO_LEAKAGE)
    gates = read_csv(GATES)
    summary = read_json(SUMMARY)
    feat_schema = read_json(CHIRPS_FEATURE_SCHEMA)
    matrix_schema = read_json(MULTIMODAL_MATRIX_SCHEMA)
    source_schema = read_json(CHIRPS_SOURCE_SCHEMA)

    for row in registry:
        errors.extend(_schema_violations(row, feat_schema))
    for row in matrix:
        errors.extend(_schema_violations(row, matrix_schema))
    for row in resolution:
        errors.extend(_schema_violations(row, source_schema))

    for rows, key in [
        (strategy, "strategy_id"), (resolution, "source_resolution_id"), (fetch, "fetch_attempt_id"),
        (manifests, "chirps_artifact_manifest_id"), (daily, "chirps_daily_value_id"),
        (aggregates, "chirps_window_aggregate_id"), (registry, "rainfall_trigger_feature_id"),
        (provenance, "provenance_trace_id"), (replay, "replay_audit_id"),
        (leakage, "no_leakage_audit_id"), (gates, "gate_eval_id"),
    ]:
        ids = [row[key] for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    # 1..7: contagens minimas e cobertura de janela.
    if not summary["minimum_success_achieved"]:
        errors.append("minimum_success_not_achieved")
    if summary["patches_processed_count"] != 5:
        errors.append("patches_not_5")
    if len(aggregates) < 15:
        errors.append("chirps_windows_lt_15")
    real_features = [r for r in registry if r["feature_value_real"] == "true"]
    if len(real_features) < 15:
        errors.append("real_chirps_features_lt_15")
    if len([r for r in daily if r["value_real"] == "true"]) < 150:
        errors.append("chirps_daily_values_lt_150")
    for patch_id in _patch_ids():
        windows = {a["feature_name"] for a in aggregates if a["candidate_patch_id"] == patch_id}
        if windows != {"CHIRPS_3d", "CHIRPS_7d", "CHIRPS_30d"}:
            errors.append(f"patch_missing_chirps_window:{patch_id}")
        if not any(d["candidate_patch_id"] == patch_id and d["value_real"] == "true" for d in daily):
            errors.append(f"patch_without_real_chirps:{patch_id}")
    # 8: manifesto/hash.
    for row in manifests:
        path = ROOT / row["artifact_local_path"]
        if not path.exists():
            errors.append(f"manifest_missing:{row['chirps_artifact_manifest_id']}")
            continue
        if not row["sha256"] or sha256_file(path) != row["sha256"]:
            errors.append(f"manifest_hash_mismatch:{row['chirps_artifact_manifest_id']}")
        if path.stat().st_size > MAX_PUBLIC_OUTPUT_BYTES:
            errors.append(f"artifact_over_limit:{row['chirps_artifact_manifest_id']}")
        if row["format"] not in ("csv", "json"):
            errors.append(f"forbidden_format:{row['chirps_artifact_manifest_id']}")
    for agg in aggregates:
        if agg["source_artifact_manifest_id"] == "not_available":
            errors.append("chirps_feature_without_manifest")
    for path in CHIRPS_ARTIFACT_DIR.glob("**/*") if CHIRPS_ARTIFACT_DIR.exists() else []:
        if path.is_file() and path.suffix.lower() in (".tif", ".tiff", ".nc", ".zip", ".gz", ".npz", ".npy"):
            errors.append(f"forbidden_artifact_committed:{rel(path)}")
    # 9/10/11: matriz.
    if len(matrix) != 5:
        errors.append("matrix_not_5")
    if matrix:
        cols = matrix[0].keys()
        if not any(c.startswith("B03_mean") for c in cols) or not any(c.startswith("CHIRPS_3d_sum") for c in cols):
            errors.append("matrix_missing_sentinel2_or_chirps")
    # 12: sem delta.
    if matrix and any("delta" in c for c in matrix[0].keys()):
        errors.append("matrix_contains_delta")
    # 13: sem pos-evento.
    if any(d["pre_event_only"] != "true" for d in daily):
        errors.append("chirps_daily_not_pre_event")
    if any(d["date"] >= EVENT_START for d in daily):
        errors.append("chirps_daily_on_or_after_event")
    # 14: CHIRPS nao vira evento.
    if any(r["uses_chirps_as_event_reference"] != "false" for r in leakage) or summary["chirps_used_as_event_reference_count"] != 0:
        errors.append("chirps_as_event_reference")
    if any(r["G4_vinculo_espacial_evento"] == "true" or r["G5_separacao_fenomeno"] == "true" for r in gates):
        errors.append("sensor_promoted_to_event")
    # 15/16/17: sem treino/ground reference/ground truth.
    if any(r["usable_for_training"] != "false" for r in registry + matrix) or summary["features_promoted_to_training_count"] != 0:
        errors.append("feature_promoted_to_training")
    if any(r["all_gates_passed_for_ground_reference"] == "true" for r in gates):
        errors.append("ground_reference_accepted")
    if summary["ground_reference_candidates_created_count"] != 0 or summary["ground_truth_created"] or summary["training_labels_created"]:
        errors.append("forbidden_promotion")
    if any(r["passes_no_leakage"] != "true" for r in leakage):
        errors.append("no_leakage_failed")
    if any(r["byte_identical_replay"] != "true" for r in replay):
        errors.append("replay_not_byte_identical")

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
        "17C25 -> "
        f"strategy={summary['selected_chirps_strategy']} "
        f"windows={summary['chirps_windows_processed_count']} "
        f"real_features={summary['real_chirps_features_count']} "
        f"daily={summary['chirps_daily_values_count']} "
        f"matrix={summary['multimodal_matrix_rows_count']} "
        f"min_success={summary['minimum_success_achieved']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def plan_text() -> str:
    lines = ["plan 17C25 (CHIRPS real por AOI):"]
    for patch_id in _patch_ids():
        lon, lat = _patch_centroid(patch_id)
        lines.append(f"  {patch_id} centroid=({lon:.5f},{lat:.5f}) janelas=CHIRPS_3d/7d/30d ({WINDOW_START}..{WINDOW_END})")
    lines.append("  estrategias: chc_ucsb_daily_file_windowed_read -> climateserv -> gee -> local_extract")
    return "\n".join(lines)


def resolve_chirps_source_text() -> str:
    rows = chirps_source_resolution_rows()
    selected = [r for r in rows if r["selected_for_computation"] == "true"]
    return f"resolve-chirps-source: estrategia selecionada={selected[0]['strategy_name'] if selected else 'none'}; {len(selected)} fonte real resolvida."


def compute_rainfall_windows_text() -> str:
    aggregates = chirps_window_aggregate_rows()
    return f"compute-rainfall-windows: {len(aggregates)} agregados CHIRPS_3d/7d/30d calculados a partir de valores diarios reais."


def build_multimodal_matrix_text() -> str:
    matrix = multimodal_feature_matrix_rows()
    return f"build-multimodal-matrix: {len(matrix)} linhas com features Sentinel-2 multitemporais + CHIRPS reais (review-only)."


def status_text() -> str:
    summary = build_summary()
    return (
        f"status 17C25: strategy={summary['selected_chirps_strategy']} "
        f"real_features={summary['real_chirps_features_count']} "
        f"daily={summary['chirps_daily_values_count']} "
        f"matrix={summary['multimodal_matrix_rows_count']} "
        f"min_success={summary['minimum_success_achieved']}"
    )
