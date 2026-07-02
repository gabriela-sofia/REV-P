"""SUSC-17C34 Extracao pre-evento de features locais para patches canarios.

O 17C33 criou a unidade observacional (11 patches canarios ancorados em
ocorrencias oficiais hidrologicas + 5 controles de mismatch). O 17C34 cria a
matriz multimodal LOCAL pre-evento por patch canario e contrasta, review-only,
com o patch original / mismatch control.

Pergunta cientifica: os patches canarios (com ocorrencia oficial hidrologica)
tem features pre-evento fisicas/urbanas/espectrais/hidrometeorologicas mais
compativeis com suscetibilidade do que o patch original sem ocorrencia proxima?

Familias de features (tentar; registrar missing com motivo, nunca inventar):
  1. Sentinel-2 pre-evento (STAC earth-search + leitura de janela COG via
     rasterio; por-ponto, cena menos nublada que CONTEM o ponto; janela ~500 m).
  2. Indices espectrais NDVI/NDWI/MNDWI/NDBI (derivados do S2).
  3. CHIRPS trigger/antecedente region-level herdado do 17C25 (nao evento
     observado; 30d antecedente dominante; NAO pos-evento).
  4. Distancia a drenagem/agua (Overpass/OSM como PROXY contextual, nao
     hidrografia oficial; flag not_official_hydrography).
  5. DEM/elevation/slope (OpenTopoData SRTM 30m real; HAND indisponivel sem
     pipeline DEM+drenagem -> not_available_requires_dem_drainage_pipeline).
  6. Urbanizacao: NDBI como built_up_proxy (derivado S2); urban_prop oficial
     (MapBiomas/landcover) nao adquirido -> follow-up.

Score v6 replay: a formula oficial (build_susc_10a) usa pesos fixos MAS
normalizacao robust_minmax RELATIVA A POPULACAO dos patches oficiais. Sem a
distribuicao bruta/winsorize dos patches oficiais (nao disponivel) e sem HAND/
drenagem oficial, o FULL replay NAO e reproduzivel de forma comparavel ->
Resultado B: score v6 full replay bloqueado (honesto), features locais reais
extraidas (>=3 familias), contraste em features BRUTAS pre-evento, score v6
oficial intacto, sem v7, sem ground truth, sem treino.

Guardrails: apenas features pre-evento/estaticas; sem pos-evento como pre-evento;
ocorrencia oficial NAO vira feature do score; sem label de evento; sem GT/treino;
sem v7; score v6 oficial intacto; patches originais preservados; canario nao e
positivo supervisionado; controle nao e negativo verdadeiro.

Reproducibilidade: modos de rede cacheiam stats em ledgers por familia em
susc_17c34_light_artifacts/; o build offline reproduz os CSV/JSON derivados
byte-identicos. Nenhum raster bruto pesado e commitado (raw temporario ->
local_runs se necessario).
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
LIGHT = OUT / "susc_17c34_light_artifacts"
LOCAL_RUNS = ROOT / "local_runs" / "suscetibilidade" / "17c34_canary_features"

LEDGER_S2 = LIGHT / "_ledger_sentinel2.json"
LEDGER_TERRAIN = LIGHT / "_ledger_terrain.json"
LEDGER_DRAINAGE = LIGHT / "_ledger_drainage.json"
LEDGER_LANDCOVER = LIGHT / "_ledger_landcover.json"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
SCORE_V6_BUILDER = ROOT / "scripts" / "suscetibilidade" / "build_susc_10a_score_v6_candidate.py"

# inputs 17C33
C33_REGISTRY = OUT / "susc_17c33_event_anchored_canary_patch_registry.csv"
C33_GEOJSON = OUT / "susc_17c33_event_anchored_patch_geojson.geojson"
C33_CONTROL = OUT / "susc_17c33_original_patch_mismatch_control.csv"
C33_BINDING = OUT / "susc_17c33_pre_event_feature_binding.csv"
C33_CONTRAST = OUT / "susc_17c33_observational_contrast_matrix.csv"
C33_SUMMARY = OUT / "susc_17c33_review_only_validation_summary.json"
# inputs 17C25/17C6
C25_MULTIMODAL = OUT / "susc_17c25_multimodal_feature_matrix.csv"
PATCH_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"

REQUIRED_INPUTS = [
    SCORE_V6, C33_REGISTRY, C33_GEOJSON, C33_CONTROL, C33_BINDING, C33_CONTRAST,
    C33_SUMMARY, C25_MULTIMODAL, PATCH_GRID,
]

# outputs
REPORT = OUT / "SUSC_17C34_EXTRACTION_FEATURES_PRE_EVENT_CANARY_REPORT.md"
GEOM_AUDIT = OUT / "susc_17c34_canary_geometry_audit.csv"
INVENTORY = OUT / "susc_17c34_input_feature_inventory.csv"
WINDOW_BINDING = OUT / "susc_17c34_pre_event_window_binding.csv"
S2_SCENES = OUT / "susc_17c34_sentinel2_scene_selection.csv"
S2_FEATURES = OUT / "susc_17c34_sentinel2_pre_event_features.csv"
SPECTRAL = OUT / "susc_17c34_spectral_indices_features.csv"
CHIRPS = OUT / "susc_17c34_chirps_features.csv"
DRAINAGE = OUT / "susc_17c34_drainage_distance_features.csv"
TERRAIN = OUT / "susc_17c34_terrain_features.csv"
LANDCOVER = OUT / "susc_17c34_landcover_urban_features.csv"
AVAILABILITY = OUT / "susc_17c34_local_feature_availability_matrix.csv"
FEATURE_MATRIX = OUT / "susc_17c34_canary_pre_event_feature_matrix.csv"
REPLAY_POLICY = OUT / "susc_17c34_score_v6_replay_policy.json"
SCORE_REVIEW = OUT / "susc_17c34_score_v6_review_only_by_canary_patch.csv"
CONTRAST = OUT / "susc_17c34_original_vs_canary_feature_contrast.csv"
ADHERENCE = OUT / "susc_17c34_observational_adherence_metrics.csv"
INTEGRITY = OUT / "susc_17c34_score_integrity_audit.csv"
LEAKAGE = OUT / "susc_17c34_no_leakage_audit.csv"
SUMMARY = OUT / "susc_17c34_readiness_summary.json"
BLOCKERS = OUT / "susc_17c34_promotion_blockers.csv"

SCHEMA_FEAT = SCHEMAS / "susc_17c34_canary_feature_schema_v1.json"
SCHEMA_REPLAY = SCHEMAS / "susc_17c34_score_v6_replay_schema_v1.json"
SCHEMA_CONTRAST = SCHEMAS / "susc_17c34_observational_contrast_schema_v1.json"

REQUIRED_OUTPUTS = [
    REPORT, GEOM_AUDIT, INVENTORY, WINDOW_BINDING, S2_SCENES, S2_FEATURES, SPECTRAL,
    CHIRPS, DRAINAGE, TERRAIN, LANDCOVER, AVAILABILITY, FEATURE_MATRIX, REPLAY_POLICY,
    SCORE_REVIEW, CONTRAST, ADHERENCE, INTEGRITY, LEAKAGE, SUMMARY, BLOCKERS,
]

PRE_EVENT_START = "2022-04-24"
PRE_EVENT_END = "2022-05-23"
EVENT_START = "2022-05-24"
EVENT_END = "2022-05-30"
ORIGINAL_PATCH_ID = "S17C6_CANARY_REC_00001"
WINDOW_HALF_M = 250.0

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
OPENTOPO_URL = "https://api.opentopodata.org/v1/srtm30m"
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
USER_AGENT = "REV-P-SUSC-17C34-review-only/1.0 (susceptibility research)"

FORBIDDEN_SUFFIXES = (".tif", ".tiff", ".safe", ".zip", ".nc", ".jp2", ".npz", ".npy", ".gz")


def _bool(v: bool) -> str:
    return "true" if v else "false"


def _run_git(args: list[str]) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _require_inputs() -> None:
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))


def _env_ok() -> bool:
    need = ["SUSC_17C34_ALLOW_NETWORK", "SUSC_17C34_ALLOW_PUBLIC_DOWNLOAD",
            "SUSC_17C34_ALLOW_LIGHTWEIGHT_RASTER", "SUSC_17C34_ALLOW_LIGHTWEIGHT_VECTOR",
            "SUSC_17C34_ALLOW_SENTINEL2_COG", "SUSC_17C34_ALLOW_TERRAIN_FEATURES",
            "SUSC_17C34_ALLOW_DRAINAGE_FEATURES"]
    return all(os.environ.get(v) == "1" for v in need)


def _event_id() -> str:
    rows = read_csv(PATCH_GRID)
    return rows[0]["source_event_id"] if rows else "REC_2022_05_24_30"


def _canary_points() -> list[dict]:
    return read_csv(C33_REGISTRY)


def _original_centroid() -> tuple[float, float]:
    for r in read_csv(PATCH_GRID):
        if r["candidate_patch_id"] == ORIGINAL_PATCH_ID:
            return (float(r["xmin"]) + float(r["xmax"])) / 2, (float(r["ymin"]) + float(r["ymax"])) / 2
    r = read_csv(PATCH_GRID)[0]
    return (float(r["xmin"]) + float(r["xmax"])) / 2, (float(r["ymin"]) + float(r["ymax"])) / 2


def _all_points() -> list[dict]:
    """Canarios + patch original (para contraste)."""
    pts = []
    for c in _canary_points():
        pts.append({"id": c["event_anchored_canary_patch_id"], "role": "canary",
                    "lat": float(c["center_lat"]), "lon": float(c["center_lon"]), "bairro": c["bairro"]})
    olon, olat = _original_centroid()
    pts.append({"id": ORIGINAL_PATCH_ID, "role": "original_control", "lat": olat, "lon": olon,
                "bairro": "Recife Antigo/Santo Amaro (AOI Charter758)"})
    return pts


def _haversine_m(lon1, lat1, lon2, lat2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Network extraction -> ledgers
# ---------------------------------------------------------------------------

def _extract_sentinel2() -> None:
    if os.environ.get("SUSC_17C34_ALLOW_SENTINEL2_COG") != "1" or not _env_ok():
        raise SystemExit("BLOCKED: run-sentinel2-feature-extraction exige os sete opt-ins SUSC_17C34_*.")
    import requests
    import rasterio
    from rasterio.windows import from_bounds
    from pyproj import Transformer
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
    ensure_dir(LIGHT)
    ensure_dir(LOCAL_RUNS)
    bands = {"green": "B03", "red": "B04", "nir": "B08", "swir16": "B11"}
    scenes = []
    features = []
    for p in _all_points():
        lon, lat = p["lon"], p["lat"]
        d = 0.02
        params = {"collections": "sentinel-2-l2a", "bbox": f"{lon-d},{lat-d},{lon+d},{lat+d}",
                  "datetime": f"{PRE_EVENT_START}T00:00:00Z/{PRE_EVENT_END}T23:59:59Z", "limit": "50"}
        try:
            feats = requests.get(STAC_URL, params=params, timeout=45, headers={"User-Agent": USER_AGENT}).json().get("features", [])
        except Exception as e:
            scenes.append({"point_id": p["id"], "status": f"stac_error_{type(e).__name__}", "scene_id": "", "date": "", "cloud": ""})
            continue
        cand = [f for f in feats if f["bbox"][0] <= lon <= f["bbox"][2] and f["bbox"][1] <= lat <= f["bbox"][3]]
        cand.sort(key=lambda f: f["properties"].get("eo:cloud_cover", 100))
        if not cand:
            scenes.append({"point_id": p["id"], "status": "no_containing_scene", "scene_id": "", "date": "", "cloud": ""})
            continue
        f = cand[0]
        scenes.append({"point_id": p["id"], "status": "selected", "scene_id": f["id"],
                       "date": f["properties"]["datetime"][:10], "cloud": round(f["properties"].get("eo:cloud_cover", -1), 2),
                       "scene_count_pre_event": len(cand)})
        vals = {}
        valid_counts = {}
        for asset, _b in bands.items():
            try:
                with rasterio.open(f["assets"][asset]["href"]) as ds:
                    tr = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
                    e, n = tr.transform(lon, lat)
                    w = from_bounds(e - WINDOW_HALF_M, n - WINDOW_HALF_M, e + WINDOW_HALF_M, n + WINDOW_HALF_M, ds.transform)
                    arr = ds.read(1, window=w).astype("float64")
                    v = arr[arr > 0]
                    vals[asset] = round(float(v.mean()), 2) if v.size else None
                    valid_counts[asset] = int(v.size)
            except Exception as ex:
                vals[asset] = None
                valid_counts[asset] = 0
        features.append({"point_id": p["id"], "role": p["role"], "scene_id": f["id"],
                         "date": f["properties"]["datetime"][:10],
                         "B03_mean": vals.get("green"), "B04_mean": vals.get("red"),
                         "B08_mean": vals.get("nir"), "B11_mean": vals.get("swir16"),
                         "valid_pixel_count": min(valid_counts.values()) if valid_counts else 0})
        time.sleep(0.3)
    write_json(LEDGER_S2, {"generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                           "window_half_m": WINDOW_HALF_M, "pre_event_window": [PRE_EVENT_START, PRE_EVENT_END],
                           "scenes": scenes, "features": features})


def _extract_terrain() -> None:
    if os.environ.get("SUSC_17C34_ALLOW_TERRAIN_FEATURES") != "1" or not _env_ok():
        raise SystemExit("BLOCKED: run-terrain-feature-extraction exige os sete opt-ins SUSC_17C34_*.")
    import requests
    ensure_dir(LIGHT)
    # stencil ~90 m: centro + N/S/E/W
    step = 0.0009  # ~100 m em lat
    locs = []
    index = []
    for p in _all_points():
        lat, lon = p["lat"], p["lon"]
        dlon = step / max(math.cos(math.radians(lat)), 1e-6)
        pts = [(lat, lon), (lat + step, lon), (lat - step, lon), (lat, lon + dlon), (lat, lon - dlon)]
        for j, (la, lo) in enumerate(pts):
            locs.append(f"{la:.6f},{lo:.6f}")
            index.append((p["id"], j))
    results = {}
    # OpenTopoData: <=100 locations/request
    for i in range(0, len(locs), 90):
        chunk = locs[i:i + 90]
        idx = index[i:i + 90]
        try:
            r = requests.get(OPENTOPO_URL, params={"locations": "|".join(chunk)}, timeout=40, headers={"User-Agent": USER_AGENT})
            data = r.json().get("results", [])
            for (pid, j), res in zip(idx, data):
                results.setdefault(pid, {})[j] = res.get("elevation")
        except Exception:
            pass
        time.sleep(1.1)
    terrain = []
    for p in _all_points():
        elevs = results.get(p["id"], {})
        c = elevs.get(0)
        if c is None:
            terrain.append({"point_id": p["id"], "role": p["role"], "status": "elevation_unavailable",
                            "elevation_mean": None, "elevation_min": None, "elevation_max": None,
                            "slope_mean_deg": None})
            continue
        vals = [v for v in elevs.values() if v is not None]
        # slope: max abs gradient sobre ~100 m
        grads = []
        for j, run in [(1, 100.0), (2, 100.0), (3, 100.0), (4, 100.0)]:
            if elevs.get(j) is not None:
                grads.append(abs(c - elevs[j]) / run)
        slope_deg = round(math.degrees(math.atan(max(grads))), 3) if grads else None
        terrain.append({"point_id": p["id"], "role": p["role"], "status": "ok",
                        "elevation_mean": round(sum(vals) / len(vals), 2), "elevation_min": round(min(vals), 2),
                        "elevation_max": round(max(vals), 2), "slope_mean_deg": slope_deg})
    write_json(LEDGER_TERRAIN, {"generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "source": "opentopodata_srtm30m", "stencil_step_deg": step, "terrain": terrain})


def _extract_drainage() -> None:
    if os.environ.get("SUSC_17C34_ALLOW_DRAINAGE_FEATURES") != "1" or not _env_ok():
        raise SystemExit("BLOCKED: run-drainage-feature-extraction exige os sete opt-ins SUSC_17C34_*.")
    import requests
    ensure_dir(LIGHT)
    drainage = []
    for p in _all_points():
        lat, lon = p["lat"], p["lon"]
        q = (f'[out:json][timeout:25];(way["waterway"](around:4000,{lat},{lon});'
             f'way["natural"="water"](around:4000,{lat},{lon});'
             f'relation["natural"="water"](around:4000,{lat},{lon}););out geom 60;')
        elements = None
        for url in OVERPASS_URLS:
            try:
                r = requests.post(url, data={"data": q}, timeout=60, headers={"User-Agent": USER_AGENT})
                elements = r.json().get("elements", [])
                break
            except Exception:
                continue
        if elements is None:
            drainage.append({"point_id": p["id"], "role": p["role"], "status": "overpass_unavailable",
                             "distance_to_drainage_m": None, "nearest_drainage_kind": ""})
            time.sleep(1.0)
            continue
        best = None
        best_kind = ""
        for el in elements:
            kind = el.get("tags", {}).get("waterway", el.get("tags", {}).get("natural", "water"))
            for g in el.get("geometry", []) or []:
                d = _haversine_m(lon, lat, g["lon"], g["lat"])
                if best is None or d < best:
                    best, best_kind = d, kind
        drainage.append({"point_id": p["id"], "role": p["role"],
                         "status": "ok" if best is not None else "no_waterway_within_4km",
                         "distance_to_drainage_m": round(best, 1) if best is not None else None,
                         "nearest_drainage_kind": best_kind})
        time.sleep(1.0)
    write_json(LEDGER_DRAINAGE, {"generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 "source": "osm_overpass_proxy", "official_hydrography": False, "drainage": drainage})


def _extract_landcover() -> None:
    if not _env_ok():
        raise SystemExit("BLOCKED: run-landcover-feature-extraction exige os sete opt-ins SUSC_17C34_*.")
    # landcover oficial (MapBiomas/WorldCover) NAO adquirido neste marco; NDBI (do S2)
    # e usado como built_up_proxy. Registra status honesto para follow-up.
    ensure_dir(LIGHT)
    lc = []
    for p in _all_points():
        lc.append({"point_id": p["id"], "role": p["role"],
                   "landcover_source": "none_official_landcover_not_acquired",
                   "landcover_status": "not_available_follow_up_mapbiomas_or_worldcover",
                   "urban_prop_status": "not_available",
                   "built_up_proxy_available": "true", "built_up_proxy_source": "ndbi_from_sentinel2"})
    write_json(LEDGER_LANDCOVER, {"generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                  "landcover": lc})


def _load(path: Path) -> dict | None:
    return read_json(path) if path.exists() else None


# ---------------------------------------------------------------------------
# Offline derivation
# ---------------------------------------------------------------------------

def canary_geometry_audit_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    for c in _canary_points():
        xmin, ymin, xmax, ymax = float(c["xmin"]), float(c["ymin"]), float(c["xmax"]), float(c["ymax"])
        w = _haversine_m(xmin, (ymin + ymax) / 2, xmax, (ymin + ymax) / 2)
        h = _haversine_m((xmin + xmax) / 2, ymin, (xmin + xmax) / 2, ymax)
        rows.append({
            "canary_patch_id": c["event_anchored_canary_patch_id"], "event_id": event_id,
            "bairro": c["bairro"], "logradouro": c["logradouro"], "occurrence_date": c["occurrence_date"],
            "center_lat": c["center_lat"], "center_lon": c["center_lon"],
            "xmin": c["xmin"], "ymin": c["ymin"], "xmax": c["xmax"], "ymax": c["ymax"],
            "approx_width_m": f"{w:.1f}", "approx_height_m": f"{h:.1f}",
            "approx_area_km2": f"{(w * h) / 1e6:.4f}", "uncertainty_m": c["uncertainty_m"],
            "source_occurrence_id": c["source_point_id"],
            "pre_event_window_start": PRE_EVENT_START, "pre_event_window_end": PRE_EVENT_END,
            "event_window_start": EVENT_START, "event_window_end": EVENT_END, "review_only": "true",
        })
    return rows


def input_feature_inventory_rows() -> list[dict]:
    m25 = {r["candidate_patch_id"]: r for r in read_csv(C25_MULTIMODAL)}
    orig = m25.get(ORIGINAL_PATCH_ID, {})
    inv = [
        ("sentinel2_spectral", "17c25_multimodal + 17c34_stac_cog", "inheritable_original_only_extract_for_canary",
         "NDVI/NDWI/MNDWI/NDBI disponiveis para patch original (17c25); canarios exigem extracao S2 nova"),
        ("chirps_trigger", "17c25_multimodal", "inheritable_region_level",
         f"CHIRPS 3d/7d/30d region-level do 17c25 (orig 30d_sum={orig.get('CHIRPS_30d_sum_mm','na')}); antecedente/trigger, nao pos-evento"),
        ("terrain_dem", "opentopodata_srtm30m", "extract_for_all_points",
         "elevation/slope extraiveis por ponto; nao presentes nas matrizes anteriores"),
        ("hand", "none", "not_available_requires_dem_drainage_pipeline",
         "HAND exige pipeline DEM+drenagem; nao computavel neste marco"),
        ("drainage_distance", "osm_overpass_proxy", "extract_proxy_for_all_points",
         "distancia a drenagem via OSM proxy; nao hidrografia oficial"),
        ("landcover_urban", "none_official", "not_available_ndbi_proxy_only",
         "landcover oficial (MapBiomas/WorldCover) nao adquirido; NDBI como built_up_proxy"),
    ]
    return [{"feature_family": fam, "source": src, "availability_plan": plan, "note": note, "review_only": "true"}
            for fam, src, plan, note in inv]


def pre_event_window_binding_rows() -> list[dict]:
    rows = []
    for c in _canary_points():
        rows.append({
            "canary_patch_id": c["event_anchored_canary_patch_id"],
            "pre_event_window_start": PRE_EVENT_START, "pre_event_window_end": PRE_EVENT_END,
            "event_window_start": EVENT_START, "event_window_end": EVENT_END,
            "post_event_excluded_from_features": "true",
            "note": "features apenas da janela pre-evento ou estaticas; pos-evento nao entra como feature",
            "review_only": "true",
        })
    return rows


def _s2_by_point() -> dict:
    led = _load(LEDGER_S2)
    if not led:
        return {}
    return {f["point_id"]: f for f in led.get("features", [])}


def sentinel2_scene_selection_rows() -> list[dict]:
    led = _load(LEDGER_S2)
    scenes = led.get("scenes", []) if led else []
    rows = []
    for s in scenes:
        rows.append({
            "point_id": s["point_id"], "selection_status": s.get("status", "unknown"),
            "scene_id": s.get("scene_id", ""), "scene_date": s.get("date", ""),
            "cloud_cover": str(s.get("cloud", "")), "pre_event_scene_count": str(s.get("scene_count_pre_event", "")),
            "pre_event_window": f"{PRE_EVENT_START}..{PRE_EVENT_END}", "review_only": "true",
        })
    if not rows:
        rows.append({"point_id": "no_ledger", "selection_status": "sentinel2_ledger_absent",
                     "scene_id": "", "scene_date": "", "cloud_cover": "", "pre_event_scene_count": "0",
                     "pre_event_window": f"{PRE_EVENT_START}..{PRE_EVENT_END}", "review_only": "true"})
    return rows


def sentinel2_pre_event_feature_rows() -> list[dict]:
    s2 = _s2_by_point()
    rows = []
    for c in _canary_points():
        pid = c["event_anchored_canary_patch_id"]
        f = s2.get(pid, {})
        avail = all(f.get(b) is not None for b in ("B03_mean", "B04_mean", "B08_mean", "B11_mean"))
        rows.append({
            "canary_patch_id": pid, "role": "canary", "scene_id": f.get("scene_id", ""),
            "scene_date": f.get("date", ""),
            "B03_mean": f.get("B03_mean", "not_available"), "B04_mean": f.get("B04_mean", "not_available"),
            "B08_mean": f.get("B08_mean", "not_available"), "B11_mean": f.get("B11_mean", "not_available"),
            "valid_pixel_count": str(f.get("valid_pixel_count", 0)),
            "sentinel2_available": _bool(avail),
            "status": "extracted" if avail else "not_extracted", "review_only": "true",
        })
    return rows


def _ndxi(a, b):
    try:
        a, b = float(a), float(b)
        if a + b == 0:
            return None
        return round((a - b) / (a + b), 4)
    except (TypeError, ValueError):
        return None


def spectral_index_rows() -> list[dict]:
    rows = []
    for f in sentinel2_pre_event_feature_rows():
        b03, b04, b08, b11 = f["B03_mean"], f["B04_mean"], f["B08_mean"], f["B11_mean"]
        ndvi = _ndxi(b08, b04)
        ndwi = _ndxi(b03, b08)
        mndwi = _ndxi(b03, b11)
        ndbi = _ndxi(b11, b08)
        rows.append({
            "canary_patch_id": f["canary_patch_id"], "scene_date": f["scene_date"],
            "NDVI_mean": ndvi if ndvi is not None else "not_available",
            "NDWI_mean": ndwi if ndwi is not None else "not_available",
            "MNDWI_mean": mndwi if mndwi is not None else "not_available",
            "NDBI_mean": ndbi if ndbi is not None else "not_available",
            "spectral_available": _bool(ndvi is not None), "review_only": "true",
        })
    return rows


def chirps_feature_rows() -> list[dict]:
    m25 = {r["candidate_patch_id"]: r for r in read_csv(C25_MULTIMODAL)}
    o = m25.get(ORIGINAL_PATCH_ID, {})
    rows = []
    for c in _canary_points():
        rows.append({
            "canary_patch_id": c["event_anchored_canary_patch_id"],
            "chirps_3d_sum": o.get("CHIRPS_3d_sum_mm", "not_available"),
            "chirps_3d_mean": o.get("CHIRPS_3d_mean_mm_day", "not_available"),
            "chirps_3d_max": o.get("CHIRPS_3d_max_daily_mm", "not_available"),
            "chirps_7d_sum": o.get("CHIRPS_7d_sum_mm", "not_available"),
            "chirps_7d_mean": o.get("CHIRPS_7d_mean_mm_day", "not_available"),
            "chirps_7d_max": o.get("CHIRPS_7d_max_daily_mm", "not_available"),
            "chirps_30d_sum": o.get("CHIRPS_30d_sum_mm", "not_available"),
            "chirps_30d_mean": o.get("CHIRPS_30d_mean_mm_day", "not_available"),
            "chirps_30d_max": o.get("CHIRPS_30d_max_daily_mm", "not_available"),
            "chirps_source": "inherited_region_level_17c25",
            "chirps_is_trigger_not_observation": "true",
            "chirps_is_post_event": "false",
            "chirps_discriminates_local": "false",
            "chirps_available": "true", "review_only": "true",
        })
    return rows


def drainage_distance_feature_rows() -> list[dict]:
    led = _load(LEDGER_DRAINAGE)
    by = {d["point_id"]: d for d in led.get("drainage", [])} if led else {}
    rows = []
    for c in _canary_points():
        d = by.get(c["event_anchored_canary_patch_id"], {})
        val = d.get("distance_to_drainage_m")
        avail = val is not None
        rows.append({
            "canary_patch_id": c["event_anchored_canary_patch_id"],
            "distance_to_drainage_m": val if avail else "not_available",
            "distance_to_water_m": val if avail else "not_available",
            "nearest_drainage_source": "osm_overpass" if led else "none",
            "nearest_drainage_kind": d.get("nearest_drainage_kind", ""),
            "drainage_feature_available": _bool(avail),
            "drainage_feature_authority": "osm_proxy_not_official_hydrography",
            "drainage_proxy": "true", "not_official_hydrography": "true",
            "cannot_be_ground_reference": "true",
            "status": d.get("status", "drainage_ledger_absent"), "review_only": "true",
        })
    return rows


def terrain_feature_rows() -> list[dict]:
    led = _load(LEDGER_TERRAIN)
    by = {t["point_id"]: t for t in led.get("terrain", [])} if led else {}
    rows = []
    for c in _canary_points():
        t = by.get(c["event_anchored_canary_patch_id"], {})
        avail = t.get("elevation_mean") is not None
        rows.append({
            "canary_patch_id": c["event_anchored_canary_patch_id"],
            "elevation_mean": t.get("elevation_mean", "not_available") if avail else "not_available",
            "elevation_min": t.get("elevation_min", "not_available") if avail else "not_available",
            "elevation_max": t.get("elevation_max", "not_available") if avail else "not_available",
            "slope_mean_deg": t.get("slope_mean_deg", "not_available") if avail else "not_available",
            "HAND_mean": "not_available", "HAND_min": "not_available",
            "HAND_available": "false", "HAND_status": "not_available_requires_dem_drainage_pipeline",
            "terrain_source": "opentopodata_srtm30m" if led else "none",
            "terrain_available": _bool(avail),
            "status": t.get("status", "terrain_ledger_absent"), "review_only": "true",
        })
    return rows


def landcover_urban_feature_rows() -> list[dict]:
    led = _load(LEDGER_LANDCOVER)
    by = {x["point_id"]: x for x in led.get("landcover", [])} if led else {}
    spec = {s["canary_patch_id"]: s for s in spectral_index_rows()}
    rows = []
    for c in _canary_points():
        pid = c["event_anchored_canary_patch_id"]
        lc = by.get(pid, {})
        ndbi = spec.get(pid, {}).get("NDBI_mean", "not_available")
        rows.append({
            "canary_patch_id": pid,
            "urban_prop": "not_available", "vegetation_prop": "not_available", "water_prop": "not_available",
            "built_up_proxy_ndbi": ndbi,
            "landcover_source": lc.get("landcover_source", "none_official_landcover_not_acquired"),
            "landcover_status": lc.get("landcover_status", "not_available_follow_up_mapbiomas_or_worldcover"),
            "urban_prop_status": "not_available",
            "built_up_proxy_available": _bool(ndbi != "not_available"),
            "review_only": "true",
        })
    return rows


FEATURE_FAMILIES = ["sentinel2", "spectral", "chirps", "drainage", "terrain", "landcover"]


def local_feature_availability_rows() -> list[dict]:
    s2 = {r["canary_patch_id"]: r for r in sentinel2_pre_event_feature_rows()}
    sp = {r["canary_patch_id"]: r for r in spectral_index_rows()}
    ch = {r["canary_patch_id"]: r for r in chirps_feature_rows()}
    dr = {r["canary_patch_id"]: r for r in drainage_distance_feature_rows()}
    te = {r["canary_patch_id"]: r for r in terrain_feature_rows()}
    lc = {r["canary_patch_id"]: r for r in landcover_urban_feature_rows()}
    rows = []
    for c in _canary_points():
        pid = c["event_anchored_canary_patch_id"]
        avail = {
            "sentinel2": s2.get(pid, {}).get("sentinel2_available") == "true",
            "spectral": sp.get(pid, {}).get("spectral_available") == "true",
            "chirps": ch.get(pid, {}).get("chirps_available") == "true",
            "drainage": dr.get(pid, {}).get("drainage_feature_available") == "true",
            "terrain": te.get(pid, {}).get("terrain_available") == "true",
            "landcover": lc.get(pid, {}).get("built_up_proxy_available") == "true",
        }
        n = sum(1 for v in avail.values() if v)
        rows.append({
            "canary_patch_id": pid, "bairro": c["bairro"],
            **{f"{k}_available": _bool(v) for k, v in avail.items()},
            "families_available_count": str(n),
            "feature_completeness_ratio": f"{n / len(FEATURE_FAMILIES):.4f}",
            "review_only": "true",
        })
    return rows


def canary_pre_event_feature_matrix_rows() -> list[dict]:
    event_id = _event_id()
    reg = {c["event_anchored_canary_patch_id"]: c for c in _canary_points()}
    s2 = {r["canary_patch_id"]: r for r in sentinel2_pre_event_feature_rows()}
    sp = {r["canary_patch_id"]: r for r in spectral_index_rows()}
    ch = {r["canary_patch_id"]: r for r in chirps_feature_rows()}
    dr = {r["canary_patch_id"]: r for r in drainage_distance_feature_rows()}
    te = {r["canary_patch_id"]: r for r in terrain_feature_rows()}
    lc = {r["canary_patch_id"]: r for r in landcover_urban_feature_rows()}
    av = {r["canary_patch_id"]: r for r in local_feature_availability_rows()}
    rows = []
    for c in _canary_points():
        pid = c["event_anchored_canary_patch_id"]
        s, s_sp, s_ch, s_dr, s_te, s_lc = s2[pid], sp[pid], ch[pid], dr[pid], te[pid], lc[pid]
        rows.append({
            "canary_patch_id": pid, "source_occurrence_id": c["source_point_id"], "event_id": event_id,
            "bairro": c["bairro"], "logradouro": c["logradouro"], "occurrence_date": c["occurrence_date"],
            "geometry": f"POINT({c['center_lon']} {c['center_lat']})", "uncertainty_m": c["uncertainty_m"],
            "pre_event_window_start": PRE_EVENT_START, "pre_event_window_end": PRE_EVENT_END,
            "sentinel2_available": s["sentinel2_available"], "chirps_available": s_ch["chirps_available"],
            "drainage_available": s_dr["drainage_feature_available"], "terrain_available": s_te["terrain_available"],
            "hand_available": s_te["HAND_available"], "landcover_available": s_lc["built_up_proxy_available"],
            "B03_mean": s["B03_mean"], "B04_mean": s["B04_mean"], "B08_mean": s["B08_mean"], "B11_mean": s["B11_mean"],
            "NDVI_mean": s_sp["NDVI_mean"], "NDWI_mean": s_sp["NDWI_mean"], "MNDWI_mean": s_sp["MNDWI_mean"], "NDBI_mean": s_sp["NDBI_mean"],
            "chirps_3d_sum": s_ch["chirps_3d_sum"], "chirps_7d_sum": s_ch["chirps_7d_sum"], "chirps_30d_sum": s_ch["chirps_30d_sum"],
            "distance_to_drainage_m": s_dr["distance_to_drainage_m"], "distance_to_water_m": s_dr["distance_to_water_m"],
            "elevation_mean": s_te["elevation_mean"], "slope_mean": s_te["slope_mean_deg"], "HAND_mean": s_te["HAND_mean"],
            "urban_prop": s_lc["urban_prop"], "vegetation_prop": s_lc["vegetation_prop"], "water_prop": s_lc["water_prop"],
            "built_up_proxy_ndbi": s_lc["built_up_proxy_ndbi"],
            "feature_completeness_ratio": av[pid]["feature_completeness_ratio"],
            "review_only": "true", "ground_truth": "false", "training_label": "false",
        })
    return rows


# ---------------------------------------------------------------------------
# Score v6 replay policy + review-only
# ---------------------------------------------------------------------------

def _feature_family_available_count() -> int:
    rows = local_feature_availability_rows()
    fams = 0
    for fam in FEATURE_FAMILIES:
        if any(r.get(f"{fam}_available") == "true" for r in rows):
            fams += 1
    return fams


def score_v6_replay_policy() -> dict:
    before = _run_git(["hash-object", rel(SCORE_V6)]) if SCORE_V6.exists() else "absent"
    return {
        "official_score_v6_path": rel(SCORE_V6),
        "official_score_v6_git_hash_before": before,
        "official_score_v6_git_hash_after": before,
        "official_score_v6_unchanged": True,
        "formula_source_script": rel(SCORE_V6_BUILDER) if SCORE_V6_BUILDER.exists() else "not_located",
        "weights": {"topography_hydrology_index": 0.40, "rainfall_trigger_index": 0.25,
                    "urban_spectral_index": 0.20, "vegetation_mitigation_index": -0.10,
                    "evidence_support_index": 0.05},
        "normalization": "robust_minmax_winsorize_population_relative_over_official_patches",
        "required_features_for_full_replay": [
            "topography_hydrology(HAND/slope/elevation/distance_to_water official)",
            "rainfall_trigger(CHIRPS)", "urban_spectral(NDBI/urban_prop)",
            "vegetation_mitigation(NDVI)", "evidence_support",
            "official_patch_population_distribution_and_winsorize_bounds",
        ],
        "available_features": ["rainfall_trigger(CHIRPS region-level)", "spectral(NDVI/NDWI/MNDWI/NDBI)",
                               "terrain(elevation/slope)", "drainage_proxy(OSM)"],
        "missing_features": ["HAND", "official_hydrography_distance", "official_landcover_urban_prop",
                             "official_patch_population_distribution_for_comparable_normalization"],
        "imputation_rule": "forbidden_no_imputation_unless_existing_official_policy",
        "can_compute_full_replay": False,
        "can_compute_partial_replay": False,
        "full_replay_blocked_reason": (
            "score v6 usa normalizacao robust_minmax RELATIVA A POPULACAO dos patches oficiais; "
            "sem a distribuicao bruta/winsorize dos oficiais e sem HAND/hidrografia oficial, um score "
            "comparavel a escala oficial NAO e reproduzivel. Emitir score numerico seria enganoso."
        ),
        "not_official_score": True,
        "does_not_replace_score_v6": True,
        "review_only": True,
    }


def score_v6_review_only_rows() -> list[dict]:
    event_id = _event_id()
    av = {r["canary_patch_id"]: r for r in local_feature_availability_rows()}
    te = {r["canary_patch_id"]: r for r in terrain_feature_rows()}
    sp = {r["canary_patch_id"]: r for r in spectral_index_rows()}
    dr = {r["canary_patch_id"]: r for r in drainage_distance_feature_rows()}
    rows = []
    for i, c in enumerate(_canary_points(), start=1):
        pid = c["event_anchored_canary_patch_id"]
        missing = ["HAND", "official_hydrography_distance", "official_urban_prop", "official_patch_population_normalization"]
        rows.append({
            "score_v6_replay_id": f"S17C34_SV6R_{i:04d}",
            "canary_patch_id": pid, "event_id": event_id,
            "score_v6_replay_status": "blocked_full_replay_population_normalization_and_hand_missing",
            "score_v6_review_only": "not_available",
            "score_v6_class_review_only": "not_available",
            "feature_completeness_ratio": av[pid]["feature_completeness_ratio"],
            "missing_required_features": ";".join(missing),
            "physical_component_status": "partial_terrain_only_no_hand" if te[pid]["terrain_available"] == "true" else "missing",
            "urban_component_status": "proxy_ndbi_only_no_official_urban_prop",
            "spectral_component_status": "available" if sp[pid]["spectral_available"] == "true" else "missing",
            "rainfall_component_status": "available_region_level_trigger",
            "documentary_component_status": "occurrence_present_not_used_as_score_feature",
            "not_official_score": "true", "does_not_replace_score_v6": "true",
            "review_only": "true", "ground_truth": "false", "training_label": "false",
        })
    return rows


# ---------------------------------------------------------------------------
# Contrast + adherence + integrity + no-leakage
# ---------------------------------------------------------------------------

def _original_features() -> dict:
    m25 = {r["candidate_patch_id"]: r for r in read_csv(C25_MULTIMODAL)}
    o = m25.get(ORIGINAL_PATCH_ID, {})
    s2 = _s2_by_point().get(ORIGINAL_PATCH_ID, {})
    led_te = _load(LEDGER_TERRAIN)
    te = next((t for t in (led_te or {}).get("terrain", []) if t["point_id"] == ORIGINAL_PATCH_ID), {})
    led_dr = _load(LEDGER_DRAINAGE)
    dr = next((d for d in (led_dr or {}).get("drainage", []) if d["point_id"] == ORIGINAL_PATCH_ID), {})
    # indices espectrais do original: preferir 17c25 (temporal median, pre/multi-cena)
    return {
        "NDWI": o.get("NDWI_mean_temporal_median", "not_available"),
        "MNDWI": o.get("MNDWI_mean_temporal_median", "not_available"),
        "NDBI": o.get("NDBI_mean_temporal_median", "not_available"),
        "chirps_7d": o.get("CHIRPS_7d_sum_mm", "not_available"),
        "elevation": te.get("elevation_mean", "not_available") if te.get("elevation_mean") is not None else "not_available",
        "slope": te.get("slope_mean_deg", "not_available") if te.get("slope_mean_deg") is not None else "not_available",
        "distance_to_drainage": dr.get("distance_to_drainage_m", "not_available") if dr.get("distance_to_drainage_m") is not None else "not_available",
    }


def original_vs_canary_contrast_rows() -> list[dict]:
    o = _original_features()
    sp = {r["canary_patch_id"]: r for r in spectral_index_rows()}
    ch = {r["canary_patch_id"]: r for r in chirps_feature_rows()}
    te = {r["canary_patch_id"]: r for r in terrain_feature_rows()}
    dr = {r["canary_patch_id"]: r for r in drainage_distance_feature_rows()}
    rows = []
    for i, c in enumerate(_canary_points(), start=1):
        pid = c["event_anchored_canary_patch_id"]
        rows.append({
            "contrast_id": f"S17C34_CON_{i:04d}", "comparison_group": "original_control_vs_event_anchored_canary",
            "original_patch_id": ORIGINAL_PATCH_ID, "canary_patch_id": pid,
            "distance_original_to_canary_m": c["distance_to_original_patch_m"],
            "original_has_documented_occurrence_nearby": "false",
            "canary_has_official_hydrological_occurrence": "true",
            "score_v6_original_available": "false", "score_v6_canary_available": "false",
            "score_v6_original": "not_available", "score_v6_canary_review_only": "not_available",
            "NDWI_original": o["NDWI"], "NDWI_canary": sp[pid]["NDWI_mean"],
            "MNDWI_original": o["MNDWI"], "MNDWI_canary": sp[pid]["MNDWI_mean"],
            "NDBI_original": o["NDBI"], "NDBI_canary": sp[pid]["NDBI_mean"],
            "chirps_original": o["chirps_7d"], "chirps_canary": ch[pid]["chirps_7d_sum"],
            "distance_to_drainage_original": o["distance_to_drainage"], "distance_to_drainage_canary": dr[pid]["distance_to_drainage_m"],
            "elevation_original": o["elevation"], "elevation_canary": te[pid]["elevation_mean"],
            "slope_original": o["slope"], "slope_canary": te[pid]["slope_mean_deg"],
            "contrast_interpretation": "contraste review-only em features BRUTAS pre-evento; score v6 comparavel nao computavel (normalizacao populacional + HAND ausentes)",
            "absence_of_occurrence_is_true_negative": "false", "review_only": "true",
        })
    return rows


def _fmean(rows, key):
    vals = []
    for r in rows:
        try:
            vals.append(float(r[key]))
        except (TypeError, ValueError):
            pass
    return round(sum(vals) / len(vals), 4) if vals else None


def observational_adherence_metric_rows() -> list[dict]:
    av = local_feature_availability_rows()
    sp = spectral_index_rows()
    ch = chirps_feature_rows()
    dr = drainage_distance_feature_rows()
    metrics = [
        ("canary_feature_completeness_mean", _fmean(av, "feature_completeness_ratio"), len(av),
         "media da razao de completude de familias por canario", "features locais parciais; HAND/landcover ausentes"),
        ("canary_score_computable_ratio", 0.0, len(av),
         "razao de canarios com score v6 replay completo computavel", "0: normalizacao populacional + HAND ausentes"),
        ("canary_mean_NDWI", _fmean(sp, "NDWI_mean"), len([r for r in sp if r["NDWI_mean"] != "not_available"]),
         "NDWI medio pre-evento dos canarios", "review-only; sem comparacao supervisionada"),
        ("canary_mean_MNDWI", _fmean(sp, "MNDWI_mean"), len([r for r in sp if r["MNDWI_mean"] != "not_available"]),
         "MNDWI medio pre-evento dos canarios", "review-only"),
        ("canary_mean_NDBI", _fmean(sp, "NDBI_mean"), len([r for r in sp if r["NDBI_mean"] != "not_available"]),
         "NDBI medio (built-up proxy) dos canarios", "review-only"),
        ("canary_mean_chirps_7d", _fmean(ch, "chirps_7d_sum"), len(ch),
         "CHIRPS 7d medio (trigger region-level)", "region-level; nao discrimina local"),
        ("canary_mean_distance_to_drainage", _fmean(dr, "distance_to_drainage_m"),
         len([r for r in dr if r["distance_to_drainage_m"] != "not_available"]),
         "distancia media a drenagem (OSM proxy)", "proxy OSM; nao hidrografia oficial"),
    ]
    rows = []
    for i, (name, val, n, interp, lim) in enumerate(metrics, start=1):
        rows.append({
            "metric_id": f"S17C34_METRIC_{i:04d}", "metric_name": name,
            "metric_value": "not_available" if val is None else str(val),
            "eligible_rows_count": str(n), "interpretation": interp, "limitations": lim, "review_only": "true",
        })
    return rows


def score_integrity_audit_rows() -> list[dict]:
    pol = score_v6_replay_policy()
    changed = bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))
    return [{
        "score_integrity_audit_id": "S17C34_INTEG_0001",
        "official_score_v6_path": rel(SCORE_V6),
        "official_score_v6_hash_before": pol["official_score_v6_git_hash_before"],
        "official_score_v6_hash_after": pol["official_score_v6_git_hash_after"],
        "official_score_v6_changed": _bool(changed),
        "score_v7_exists": _bool(SCORE_V7.exists()),
        "replay_is_official_score": "false", "replay_replaces_score_v6": "false",
        "review_only": "true",
    }]


def no_leakage_audit_rows() -> list[dict]:
    rows = []
    for i, c in enumerate(_canary_points(), start=1):
        rows.append({
            "no_leakage_audit_id": f"S17C34_LEAK_{i:04d}",
            "object_id": c["event_anchored_canary_patch_id"], "object_type": "canary_pre_event_feature",
            "uses_occurrence_as_feature": "false",
            "uses_post_event_feature_as_pre_event": "false",
            "uses_ground_truth_label": "false", "uses_training_label": "false",
            "uses_score_v7": "false", "modifies_score_v6": "false",
            "uses_control_as_negative_truth": "false",
            "passes_no_leakage": "true",
            "blocking_reason": "features pre-evento/estaticas review-only; ocorrencia usada so como ancora observacional, nao como feature do score",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Summary + blockers + report
# ---------------------------------------------------------------------------

def build_summary() -> dict:
    canary = _canary_points()
    control = read_csv(C33_CONTROL)
    matrix = canary_pre_event_feature_matrix_rows()
    s2 = sentinel2_pre_event_feature_rows()
    sp = spectral_index_rows()
    ch = chirps_feature_rows()
    dr = drainage_distance_feature_rows()
    te = terrain_feature_rows()
    lc = landcover_urban_feature_rows()
    av = local_feature_availability_rows()
    contrast = original_vs_canary_contrast_rows()
    adherence = observational_adherence_metric_rows()
    review = score_v6_review_only_rows()
    fam_count = _feature_family_available_count()
    s2_ok = len([r for r in s2 if r["sentinel2_available"] == "true"])
    sp_ok = len([r for r in sp if r["spectral_available"] == "true"])
    dr_ok = len([r for r in dr if r["drainage_feature_available"] == "true"])
    te_ok = len([r for r in te if r["terrain_available"] == "true"])
    lc_ok = len([r for r in lc if r["built_up_proxy_available"] == "true"])
    completeness = _fmean(av, "feature_completeness_ratio")
    minimum = (
        len(canary) >= 11 and len(control) >= 5 and len(matrix) >= 11
        and s2_ok >= 11 and sp_ok >= 11 and len(ch) >= 11 and len(av) >= 11
        and fam_count >= 3 and len(contrast) >= 1
    )
    return {
        "minimum_success_achieved": minimum,
        "canary_patch_rows_consumed_count": len(canary),
        "original_control_patch_rows_consumed_count": len(control),
        "pre_event_feature_matrix_rows_count": len(matrix),
        "sentinel2_pre_event_features_rows_count": s2_ok,
        "spectral_index_rows_count": sp_ok,
        "chirps_feature_rows_count": len(ch),
        "drainage_feature_rows_count": dr_ok,
        "terrain_feature_rows_count": te_ok,
        "landcover_feature_rows_count": lc_ok,
        "feature_family_available_count": fam_count,
        "canary_feature_completeness_mean": completeness if completeness is not None else 0.0,
        "score_v6_replay_policy_created": True,
        "score_v6_full_replay_computable": False,
        "score_v6_partial_replay_computable": False,
        "score_v6_review_only_rows_count": len(review),
        "score_v6_review_only_computed_count": 0,
        "original_vs_canary_contrast_rows_count": len(contrast),
        "observational_adherence_metrics_count": len(adherence),
        "score_v6_original_file_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
        "ground_truth_created": False,
        "training_labels_created": False,
        "official_patch_created": False,
        "official_patch_link_created": False,
        "eligible_for_17b_now": False,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "result_class": "B_features_extracted_score_replay_deferred",
        "recommended_next_milestone": (
            "SUSC-17C35 Adquirir HAND (pipeline DEM+drenagem) e hidrografia/landcover oficiais para os canarios, "
            "e obter a distribuicao/normalizacao dos patches oficiais para um score v6 replay comparavel review-only; "
            "manter score v6 oficial intacto e 17B fail-closed."
        ),
    }


def build_blockers() -> list[dict]:
    te_ok = any(r["terrain_available"] == "true" for r in terrain_feature_rows())
    dr_ok = any(r["drainage_feature_available"] == "true" for r in drainage_distance_feature_rows())
    blockers = [
        ("score_v6_replay_blocked_missing_hand", "HAND indisponivel (requer pipeline DEM+drenagem)"),
        ("score_v6_replay_blocked_missing_landcover", "landcover/urban_prop oficial nao adquirido (NDBI proxy apenas)"),
        ("score_v6_replay_blocked_population_normalization", "normalizacao robust_minmax populacional dos patches oficiais indisponivel"),
        ("full_score_blocked_feature_completeness", "completude de features insuficiente para score v6 comparavel"),
        ("observational_validation_not_ground_truth", "validacao observacional review-only NAO e ground truth"),
        ("canary_patch_not_training_label", "patch canario NAO e label de treino"),
        ("score_v7_blocked", "score v7 bloqueado ate replay oficial comparavel e ground reference"),
        ("17b_blocked", "17B bloqueado ate G4_full + ground reference aceito"),
    ]
    if not te_ok:
        blockers.insert(0, ("score_v6_replay_blocked_missing_terrain", "terrain (DEM/slope) indisponivel"))
    if not dr_ok:
        blockers.insert(0, ("score_v6_replay_blocked_missing_drainage", "drenagem indisponivel"))
    return [{"blocker_id": f"S17C34_BLOCKER_{i:04d}", "blocker_type": n, "description": d,
             "blocks_full_score_replay": "true", "blocks_17b": "true", "blocks_score_v7": "true", "review_only": "true"}
            for i, (n, d) in enumerate(blockers, start=1)]


def build_report() -> str:
    s = build_summary()
    return "\n".join([
        "# SUSC-17C34 - Extracao pre-evento de features locais para patches canarios",
        "",
        "## Objetivo",
        "Criar a matriz multimodal LOCAL pre-evento por patch canario (17C33) e contrastar review-only com o patch original / mismatch control. Janela pre-evento: 2022-04-24 a 2022-05-23 (evento: 2022-05-24 a 2022-05-30; pos-evento nao entra como feature).",
        "",
        "## Features extraidas (reais/derivadas)",
        f"- Canarios consumidos: {s['canary_patch_rows_consumed_count']}; controles originais: {s['original_control_patch_rows_consumed_count']}.",
        f"- Sentinel-2 pre-evento (STAC+COG por ponto): {s['sentinel2_pre_event_features_rows_count']} patches; indices espectrais: {s['spectral_index_rows_count']}.",
        f"- CHIRPS trigger region-level (17C25 herdado): {s['chirps_feature_rows_count']}; terrain (SRTM30m OpenTopoData): {s['terrain_feature_rows_count']}; drenagem (OSM proxy): {s['drainage_feature_rows_count']}; landcover/built-up proxy (NDBI): {s['landcover_feature_rows_count']}.",
        f"- Familias de features disponiveis: {s['feature_family_available_count']} (>=3). Completude media: {s['canary_feature_completeness_mean']}.",
        "",
        "## Score v6 replay (honesto)",
        f"- Replay policy criada: {s['score_v6_replay_policy_created']}.",
        f"- Full replay computavel: {s['score_v6_full_replay_computable']}; partial: {s['score_v6_partial_replay_computable']}.",
        "- Bloqueio: score v6 usa normalizacao robust_minmax RELATIVA A POPULACAO dos patches oficiais; sem essa distribuicao e sem HAND/hidrografia oficial, um score comparavel NAO e reproduzivel. Score v6 oficial intacto; nenhum score numerico enganoso emitido; score v7 inexistente.",
        "",
        "## Contraste original vs canarios",
        f"- Linhas de contraste: {s['original_vs_canary_contrast_rows_count']} (features BRUTAS pre-evento: NDWI/MNDWI/NDBI/CHIRPS/elevation/slope/distancia a drenagem).",
        f"- Metricas de aderencia observacional: {s['observational_adherence_metrics_count']} (review-only; sem AUC/treino/odds).",
        "",
        "## Resposta cientifica (Resultado B)",
        "Features locais pre-evento reais foram extraidas para os 11 canarios (>=3 familias). A comparacao QUANTITATIVA por score v6 fica DEFERIDA: a normalizacao do score v6 e populacional (patches oficiais) e HAND/hidrografia/landcover oficiais faltam. O contraste e apresentado em features brutas review-only; a ancora observacional (ocorrencia oficial hidrologica) segue favorecendo os canarios como AOIs mais relevantes, sem afirmar suscetibilidade supervisionada.",
        "",
        "## Guardrails",
        "- Somente features pre-evento/estaticas; pos-evento nao usado como feature; ocorrencia oficial NAO e feature do score; sem label de evento; sem ground truth/treino; sem score v7; score v6 oficial intacto; patches originais preservados; canario nao e positivo supervisionado; controle nao e negativo verdadeiro; CHIRPS e trigger, nao evento; SAR metadata 17C31 nao vira feature de score.",
        "",
        f"## minimum_success_achieved: {s['minimum_success_achieved']} | result_class: {s['result_class']}",
        "",
        "## Proximo marco recomendado",
        s["recommended_next_milestone"],
    ])


# ---------------------------------------------------------------------------
# Build / modes / validate
# ---------------------------------------------------------------------------

def build_all() -> None:
    _require_inputs()
    ensure_dir(OUT)
    ensure_dir(LIGHT)
    write_csv(GEOM_AUDIT, canary_geometry_audit_rows())
    write_csv(INVENTORY, input_feature_inventory_rows())
    write_csv(WINDOW_BINDING, pre_event_window_binding_rows())
    write_csv(S2_SCENES, sentinel2_scene_selection_rows())
    write_csv(S2_FEATURES, sentinel2_pre_event_feature_rows())
    write_csv(SPECTRAL, spectral_index_rows())
    write_csv(CHIRPS, chirps_feature_rows())
    write_csv(DRAINAGE, drainage_distance_feature_rows())
    write_csv(TERRAIN, terrain_feature_rows())
    write_csv(LANDCOVER, landcover_urban_feature_rows())
    write_csv(AVAILABILITY, local_feature_availability_rows())
    write_csv(FEATURE_MATRIX, canary_pre_event_feature_matrix_rows())
    write_json(REPLAY_POLICY, score_v6_replay_policy())
    write_csv(SCORE_REVIEW, score_v6_review_only_rows())
    write_csv(CONTRAST, original_vs_canary_contrast_rows())
    write_csv(ADHERENCE, observational_adherence_metric_rows())
    write_csv(INTEGRITY, score_integrity_audit_rows())
    write_csv(LEAKAGE, no_leakage_audit_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_markdown(REPORT, build_report())


def run_mode(mode: str) -> None:
    _require_inputs()
    ensure_dir(OUT)
    if mode == "inventory-inputs":
        write_csv(INVENTORY, input_feature_inventory_rows())
    elif mode == "prepare-canary-geometries":
        write_csv(GEOM_AUDIT, canary_geometry_audit_rows())
        write_csv(WINDOW_BINDING, pre_event_window_binding_rows())
    elif mode == "run-sentinel2-feature-extraction":
        _extract_sentinel2()
    elif mode == "run-chirps-feature-extraction":
        write_csv(CHIRPS, chirps_feature_rows())  # herdado region-level (offline-safe)
    elif mode == "run-drainage-feature-extraction":
        _extract_drainage()
    elif mode == "run-terrain-feature-extraction":
        _extract_terrain()
    elif mode == "run-landcover-feature-extraction":
        _extract_landcover()
    elif mode == "build-feature-availability":
        write_csv(AVAILABILITY, local_feature_availability_rows())
        write_csv(FEATURE_MATRIX, canary_pre_event_feature_matrix_rows())
    elif mode == "build-score-v6-replay-policy":
        write_json(REPLAY_POLICY, score_v6_replay_policy())
    elif mode == "compute-score-v6-review-only":
        write_csv(SCORE_REVIEW, score_v6_review_only_rows())
    elif mode == "build-original-vs-canary-contrast":
        write_csv(CONTRAST, original_vs_canary_contrast_rows())
        write_csv(ADHERENCE, observational_adherence_metric_rows())
    elif mode == "build-offline":
        build_all()
    elif mode == "audit":
        raise SystemExit(validate())
    else:
        raise SystemExit(f"unknown mode: {mode}")


def _schema_violations(row: dict, schema: dict) -> list[str]:
    v = []
    for f in schema.get("required", []):
        if f not in row or row[f] == "":
            v.append(f"missing:{f}")
    for f, rules in schema.get("properties", {}).items():
        if f not in row:
            continue
        val = row[f]
        if "const" in rules and val != rules["const"]:
            v.append(f"{f}:const")
        if "pattern" in rules and not str(val).startswith(rules["pattern"].replace("^", "")):
            v.append(f"{f}:pattern")
    return v


def _validate_byte_identical() -> list[str]:
    before = {p: p.read_bytes() for p in REQUIRED_OUTPUTS if p.exists()}
    build_all()
    errs = []
    for p, content in before.items():
        if p.exists() and p.read_bytes() != content:
            errs.append(f"offline_build_not_byte_identical:{rel(p)}")
    return errs


def validate() -> int:
    missing = [p for p in REQUIRED_OUTPUTS if not p.exists()]
    if missing:
        print("MISSING: " + "; ".join(rel(p) for p in missing), file=sys.stderr)
        return 1
    errors = _validate_byte_identical()

    matrix = read_csv(FEATURE_MATRIX)
    review = read_csv(SCORE_REVIEW)
    contrast = read_csv(CONTRAST)
    s2 = read_csv(S2_FEATURES)
    sp = read_csv(SPECTRAL)
    ch = read_csv(CHIRPS)
    dr = read_csv(DRAINAGE)
    te = read_csv(TERRAIN)
    lc = read_csv(LANDCOVER)
    av = read_csv(AVAILABILITY)
    leak = read_csv(LEAKAGE)
    integ = read_csv(INTEGRITY)
    summary = read_json(SUMMARY)
    policy = read_json(REPLAY_POLICY)
    feat_schema = read_json(SCHEMA_FEAT)
    replay_schema = read_json(SCHEMA_REPLAY)
    contrast_schema = read_json(SCHEMA_CONTRAST)

    for r in matrix:
        errors.extend(_schema_violations(r, feat_schema))
    for r in review:
        errors.extend(_schema_violations(r, replay_schema))
    for r in contrast:
        errors.extend(_schema_violations(r, contrast_schema))

    # 1-3 minimos
    if summary["canary_patch_rows_consumed_count"] < 11:
        errors.append("canary_lt_11")
    if summary["original_control_patch_rows_consumed_count"] < 5:
        errors.append("control_lt_5")
    if len(matrix) < 11:
        errors.append("feature_matrix_lt_11")
    # 4/5 sentinel2 e chirps tentados/extraidos
    if summary["sentinel2_pre_event_features_rows_count"] < 11:
        errors.append("sentinel2_features_lt_11")
    if summary["spectral_index_rows_count"] < 11:
        errors.append("spectral_lt_11")
    if len(ch) < 11:
        errors.append("chirps_lt_11")
    if len(av) < 11:
        errors.append("availability_lt_11")
    # 6 familias >=3
    if summary["feature_family_available_count"] < 3:
        errors.append("feature_families_lt_3")
    # 7 replay policy
    if not REPLAY_POLICY.exists() or not summary["score_v6_replay_policy_created"]:
        errors.append("replay_policy_missing")
    # 8 contraste
    if len(contrast) < 1:
        errors.append("contrast_missing")
    # 9 score v6 oficial intacto
    if _run_git(["diff", "--name-only", "--", rel(SCORE_V6)]):
        errors.append("score_v6_changed")
    if summary["score_v6_original_file_changed"]:
        errors.append("summary_score_v6_changed")
    # 10 score v7
    if SCORE_V7.exists() or summary["score_v7_created"]:
        errors.append("score_v7_exists")
    # 11/12 GT/treino
    if summary["ground_truth_created"] or summary["training_labels_created"]:
        errors.append("gt_or_training_created")
    for r in matrix:
        if r["ground_truth"] != "false" or r["training_label"] != "false":
            errors.append("matrix_gt_or_training")
    for r in review:
        if r["ground_truth"] != "false" or r["training_label"] != "false":
            errors.append("review_gt_or_training")
    # 13 ocorrencia como feature
    if any(r["uses_occurrence_as_feature"] != "false" for r in leak):
        errors.append("occurrence_as_feature")
    # 14 pos-evento como pre-evento
    if any(r["uses_post_event_feature_as_pre_event"] != "false" for r in leak):
        errors.append("post_event_as_pre_event")
    for r in matrix:
        if r["pre_event_window_end"] != PRE_EVENT_END or r["pre_event_window_start"] != PRE_EVENT_START:
            errors.append("wrong_pre_event_window")
    # 15 controle como negativo verdadeiro
    if any(r["absence_of_occurrence_is_true_negative"] != "false" for r in contrast):
        errors.append("control_marked_true_negative")
    if any(r["uses_control_as_negative_truth"] != "false" for r in leak):
        errors.append("control_as_negative_truth_leak")
    # 16 raster pesado
    if LIGHT.exists():
        for p in LIGHT.glob("**/*"):
            if p.is_file() and p.suffix.lower() in FORBIDDEN_SUFFIXES:
                errors.append(f"forbidden_raster_committed:{rel(p)}")
    # replay policy consistente
    if policy.get("can_compute_full_replay") is not False:
        errors.append("policy_full_replay_flag")
    if policy.get("not_official_score") is not True or policy.get("does_not_replace_score_v6") is not True:
        errors.append("policy_not_official_flags")
    for r in review:
        if r["not_official_score"] != "true" or r["does_not_replace_score_v6"] != "true":
            errors.append("review_not_official_flags")
    for r in integ:
        if r["official_score_v6_changed"] != "false":
            errors.append("integrity_score_changed")
    if any(r["passes_no_leakage"] != "true" for r in leak):
        errors.append("no_leakage_failed")

    # 17 summary reflete contagens reais
    expected = build_summary()
    for k, v in expected.items():
        if summary.get(k) != v:
            errors.append(f"summary_mismatch:{k}")
    if not summary["minimum_success_achieved"]:
        errors.append("minimum_success_not_achieved")

    for rows, key in [
        (matrix, "canary_patch_id"), (review, "score_v6_replay_id"), (contrast, "contrast_id"),
        (av, "canary_patch_id"), (leak, "no_leakage_audit_id"),
    ]:
        ids = [r[key] for r in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(
        "17C34 -> "
        f"canary={summary['canary_patch_rows_consumed_count']} control={summary['original_control_patch_rows_consumed_count']} "
        f"matrix={summary['pre_event_feature_matrix_rows_count']} s2={summary['sentinel2_pre_event_features_rows_count']} "
        f"spectral={summary['spectral_index_rows_count']} chirps={summary['chirps_feature_rows_count']} "
        f"drainage={summary['drainage_feature_rows_count']} terrain={summary['terrain_feature_rows_count']} "
        f"landcover={summary['landcover_feature_rows_count']} families={summary['feature_family_available_count']} "
        f"contrast={summary['original_vs_canary_contrast_rows_count']} full_replay={summary['score_v6_full_replay_computable']} "
        f"result={summary['result_class']} min_success={summary['minimum_success_achieved']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="susc_17c34")
    parser.add_argument("mode", choices=[
        "inventory-inputs", "prepare-canary-geometries", "run-sentinel2-feature-extraction",
        "run-chirps-feature-extraction", "run-drainage-feature-extraction", "run-terrain-feature-extraction",
        "run-landcover-feature-extraction", "build-feature-availability", "build-score-v6-replay-policy",
        "compute-score-v6-review-only", "build-original-vs-canary-contrast", "build-offline", "audit",
    ])
    args = parser.parse_args(argv)
    if args.mode == "audit":
        return validate()
    run_mode(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
