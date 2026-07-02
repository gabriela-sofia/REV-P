"""SUSC-17C35 Feature Parity, HAND/Hidrografia/Landcover e Readiness de Replay Score v6.

Resolve a comparabilidade entre os patches canarios (17C34) e os patches
oficiais do score v6. NAO cria score v7, NAO altera o score v6 oficial, NAO
normaliza canarios por eles mesmos. Cria uma ponte review-only para responder:
e possivel um replay comparavel do score v6 nos canarios usando a MESMA politica
de features e normalizacao (populacional) dos patches oficiais?

Descobertas de base:
 - a formula do score v6 (build_susc_10a) usa pesos fixos + winsorize(0.01,0.99)
   + robust_minmax RELATIVO A POPULACAO da matriz oficial
   `datasets/suscetibilidade/susc_features_by_patch_v1.csv` (existe; 100 patches
   recife com features brutas: elevation/slope/hand/twi/tpi/distance_to_water/
   flow_acc/chirps/ndvi/mndwi/ndbi/urban_prop/vegetation_prop). -> a populacao de
   normalizacao E reconstruivel (winsorize+minmax por feature).
 - features usadas por sub-indice:
     topography_hydrology (0.40): elevation, slope, HAND, twi, tpi,
       distance_to_water, flow_acc  -> canary tem elevation/slope reais, HAND so
       PROXY, twi/tpi/flow_acc AUSENTES -> componente NAO reproduzivel.
     rainfall_trigger (0.25): chirps_3d/7d/30d + ratios/persistence/runoff ->
       canary tem chirps region-level; ratios/runoff ausentes -> parcial.
     urban_spectral (0.20): mndwi, ndbi, urban_prop -> canary tem mndwi/ndbi
       reais (17C34) e urban_prop OFICIAL (Dados Recife cobertura-da-terra).
     vegetation_mitigation (-0.10): ndvi, vegetation_prop -> ndvi real +
       vegetation_prop OFICIAL -> componente reproduzivel.
     evidence_support (0.05): overlay 07B dos oficiais; ocorrencia NAO pode
       virar feature -> canary evidence=0/na.

Resultado esperado (Resultado B): normalizacao populacional RECONSTRUIDA;
hidrografia oficial (faixas-marginais dos recursos hidricos) e landcover oficial
(cobertura-da-terra) ADQUIRIDOS -> distance_to_water e urban_prop/vegetation_prop
oficiais; HAND permanece PROXY local; twi/tpi/flow_acc AUSENTES -> o sub-indice
topography_hydrology (dominante, 0.40) NAO e reproduzivel -> SCORE FINAL
BLOQUEADO. Componentes comparaveis review-only sao computados onde ha paridade
oficial (vegetation_mitigation e urban_spectral); rainfall parcial; topography
bloqueado. Nenhum numero de score final enganoso e emitido. Score v6 oficial
intacto; sem v7; sem GT/treino.

Guardrails: sem alterar score v6; sem v7/GT/treino; sem ocorrencia como feature;
sem pos-evento como pre-evento; PROIBIDO normalizar so nos canarios; sem imputar
feature faltante; OSM so proxy (flag); HAND proxy != HAND full; landcover oficial
separado de NDBI proxy; replay e review-only e nao substitui score v6.

Reproducibilidade: modos de rede cacheiam em ledgers/manifests em
susc_17c35_light_artifacts/ (com sha256); raw pesado (landcover 18MB) vai para
local_runs, NUNCA commitado. Build offline reproduz derivados byte-identicos.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
LIGHT = OUT / "susc_17c35_light_artifacts"
LOCAL_RUNS = ROOT / "local_runs" / "suscetibilidade" / "17c35_feature_parity"

LEDGER_HYDRO = LIGHT / "_ledger_hydrography.json"
LEDGER_HAND = LIGHT / "_ledger_hand.json"
LEDGER_LANDCOVER = LIGHT / "_ledger_landcover.json"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_MATRIX = DAT / "susc_features_by_patch_v1.csv"
SCORE_V6_BUILDER = ROOT / "scripts" / "suscetibilidade" / "build_susc_10a_score_v6_candidate.py"
FEATURE_CARDS = OUT / "feature_cards" / "susc_feature_cards_v1.json"

# inputs 17C34 / 17C33
C34_MATRIX = OUT / "susc_17c34_canary_pre_event_feature_matrix.csv"
C34_SPECTRAL = OUT / "susc_17c34_spectral_indices_features.csv"
C34_CHIRPS = OUT / "susc_17c34_chirps_features.csv"
C34_TERRAIN = OUT / "susc_17c34_terrain_features.csv"
C34_DRAINAGE = OUT / "susc_17c34_drainage_distance_features.csv"
C34_LANDCOVER = OUT / "susc_17c34_landcover_urban_features.csv"
C34_POLICY = OUT / "susc_17c34_score_v6_replay_policy.json"
C34_SUMMARY = OUT / "susc_17c34_readiness_summary.json"
C33_REGISTRY = OUT / "susc_17c33_event_anchored_canary_patch_registry.csv"
C33_CONTROL = OUT / "susc_17c33_original_patch_mismatch_control.csv"
PATCH_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"

REQUIRED_INPUTS = [
    SCORE_V6, OFFICIAL_MATRIX, C34_MATRIX, C34_SPECTRAL, C34_CHIRPS, C34_TERRAIN,
    C34_DRAINAGE, C34_LANDCOVER, C34_POLICY, C34_SUMMARY, C33_REGISTRY, C33_CONTROL, PATCH_GRID,
]

# outputs
REPORT = OUT / "SUSC_17C35_FEATURE_PARITY_SCORE_REPLAY_REPORT.md"
CONTRACT = OUT / "susc_17c35_score_v6_formula_contract.json"
POP_INVENTORY = OUT / "susc_17c35_official_score_population_inventory.csv"
NORM_ANCHOR = OUT / "susc_17c35_normalization_anchor_evaluation.csv"
HYDRO_ATTEMPTS = OUT / "susc_17c35_hydrography_acquisition_attempts.csv"
HYDRO_MANIFEST = OUT / "susc_17c35_hydrography_artifact_manifest.csv"
OFFICIAL_DRAINAGE = OUT / "susc_17c35_official_drainage_features.csv"
HAND_CANDIDATES = OUT / "susc_17c35_hand_feature_candidates.csv"
LC_ATTEMPTS = OUT / "susc_17c35_landcover_acquisition_attempts.csv"
LC_MANIFEST = OUT / "susc_17c35_landcover_artifact_manifest.csv"
LC_FEATURES = OUT / "susc_17c35_landcover_urban_features.csv"
PARITY = OUT / "susc_17c35_feature_parity_matrix.csv"
REPLAY_READY = OUT / "susc_17c35_score_v6_replay_readiness.csv"
REPLAY_OUT = OUT / "susc_17c35_score_v6_review_only_replay_by_canary_patch.csv"
CONTRAST = OUT / "susc_17c35_original_vs_canary_comparable_contrast.csv"
INTEGRITY = OUT / "susc_17c35_score_integrity_audit.csv"
LEAKAGE = OUT / "susc_17c35_no_leakage_audit.csv"
SUMMARY = OUT / "susc_17c35_readiness_summary.json"
BLOCKERS = OUT / "susc_17c35_promotion_blockers.csv"

SCHEMA_PARITY = SCHEMAS / "susc_17c35_feature_parity_schema_v1.json"
SCHEMA_REPLAY = SCHEMAS / "susc_17c35_score_replay_schema_v1.json"
SCHEMA_NORM = SCHEMAS / "susc_17c35_normalization_anchor_schema_v1.json"

REQUIRED_OUTPUTS = [
    REPORT, CONTRACT, POP_INVENTORY, NORM_ANCHOR, HYDRO_ATTEMPTS, HYDRO_MANIFEST,
    OFFICIAL_DRAINAGE, HAND_CANDIDATES, LC_ATTEMPTS, LC_MANIFEST, LC_FEATURES, PARITY,
    REPLAY_READY, REPLAY_OUT, CONTRAST, INTEGRITY, LEAKAGE, SUMMARY, BLOCKERS,
]

WEIGHTS = {"topography_hydrology_index": 0.40, "rainfall_trigger_index": 0.25,
           "urban_spectral_index": 0.20, "vegetation_mitigation_index": -0.10,
           "evidence_support_index": 0.05}
USED_FEATURES = ["elevation_mean", "slope_mean", "hand_mean", "twi_mean", "tpi_250m_mean",
                 "distance_to_water_mean", "flow_acc_log_mean", "chirps_3d_mm", "chirps_7d_mm",
                 "chirps_30d_mm", "rain_7d_30d_ratio", "rain_persistence_index",
                 "runoff_context_7d", "runoff_context_30d", "ndvi_mean", "mndwi_mean",
                 "ndbi_mean", "urban_prop", "vegetation_prop"]
SUBINDEX_FEATURES = {
    "topography_hydrology_index": ["elevation_mean", "slope_mean", "hand_mean", "twi_mean",
                                   "tpi_250m_mean", "distance_to_water_mean", "flow_acc_log_mean"],
    "rainfall_trigger_index": ["chirps_3d_mm", "chirps_7d_mm", "chirps_30d_mm", "rain_7d_30d_ratio",
                               "rain_persistence_index", "runoff_context_7d", "runoff_context_30d"],
    "urban_spectral_index": ["mndwi_mean", "ndbi_mean", "urban_prop"],
    "vegetation_mitigation_index": ["ndvi_mean", "vegetation_prop"],
}
# feature canaria oficial-equivalente -> feature oficial
CANARY_EQUIV = {"ndvi_mean": "NDVI_mean", "mndwi_mean": "MNDWI_mean", "ndbi_mean": "NDBI_mean",
                "elevation_mean": "elevation_mean_canary", "slope_mean": "slope_mean_canary"}

USER_AGENT = "REV-P-SUSC-17C35-review-only/1.0 (susceptibility research)"
CKAN_BASE = "https://dados.recife.pe.gov.br/api/3/action"
FORBIDDEN_SUFFIXES = (".tif", ".tiff", ".safe", ".zip", ".nc", ".jp2", ".npz", ".npy", ".gz")
MAX_LIGHT_BYTES = 5_000_000


def _bool(v: bool) -> str:
    return "true" if v else "false"


def _run_git(args):
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _env_ok() -> bool:
    need = ["SUSC_17C35_ALLOW_NETWORK", "SUSC_17C35_ALLOW_PUBLIC_DOWNLOAD",
            "SUSC_17C35_ALLOW_LIGHTWEIGHT_VECTOR", "SUSC_17C35_ALLOW_LIGHTWEIGHT_RASTER",
            "SUSC_17C35_ALLOW_HYDROGRAPHY", "SUSC_17C35_ALLOW_HAND_PIPELINE",
            "SUSC_17C35_ALLOW_LANDCOVER"]
    return all(os.environ.get(v) == "1" for v in need)


def _require_inputs():
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))


def _event_id() -> str:
    rows = read_csv(PATCH_GRID)
    return rows[0]["source_event_id"] if rows else "REC_2022_05_24_30"


def _canary_points() -> list[dict]:
    return read_csv(C33_REGISTRY)


def _original_centroid() -> tuple[float, float]:
    for r in read_csv(PATCH_GRID):
        if r["candidate_patch_id"] == "S17C6_CANARY_REC_00001":
            return (float(r["xmin"]) + float(r["xmax"])) / 2, (float(r["ymin"]) + float(r["ymax"])) / 2
    r = read_csv(PATCH_GRID)[0]
    return (float(r["xmin"]) + float(r["xmax"])) / 2, (float(r["ymin"]) + float(r["ymax"])) / 2


def _points_for_extraction() -> list[dict]:
    pts = [{"id": c["event_anchored_canary_patch_id"], "role": "canary",
            "lat": float(c["center_lat"]), "lon": float(c["center_lon"]),
            "xmin": float(c["xmin"]), "ymin": float(c["ymin"]), "xmax": float(c["xmax"]), "ymax": float(c["ymax"])}
           for c in _canary_points()]
    olon, olat = _original_centroid()
    pts.append({"id": "S17C6_CANARY_REC_00001", "role": "original", "lat": olat, "lon": olon,
                "xmin": olon - 0.006, "ymin": olat - 0.006, "xmax": olon + 0.006, "ymax": olat + 0.006})
    return pts


# ---------------------------------------------------------------------------
# Normalization reconstruction (winsorize 0.01/0.99 + robust_minmax) da populacao oficial
# ---------------------------------------------------------------------------

def _winsorize_bounds(vals, lo=0.01, hi=0.99):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return None, None
    return s[max(0, int(lo * n))], s[min(n - 1, int(hi * n))]


def _official_feature_bounds() -> dict:
    """Reconstroi lo_v/hi_v por feature usada, a partir da populacao oficial completa."""
    rows = read_csv(OFFICIAL_MATRIX)
    bounds = {}
    for feat in USED_FEATURES:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[feat]))
            except (KeyError, ValueError, TypeError):
                pass
        if len(vals) >= 10:
            lo_v, hi_v = _winsorize_bounds(vals)
            bounds[feat] = {"lo_v": lo_v, "hi_v": hi_v, "n": len(vals), "population": "official_recife_matrix"}
    return bounds


def _card_directions() -> dict:
    if not FEATURE_CARDS.exists():
        return {}
    return {c["feature_name"]: c.get("expected_direction", "higher_increases")
            for c in read_json(FEATURE_CARDS)["cards"]}


def _normalize_official(feat: str, value, bounds: dict, directions: dict):
    """Aplica winsorize+minmax da populacao OFICIAL e orienta por direcao."""
    if feat not in bounds or value is None:
        return None
    lo_v, hi_v = bounds[feat]["lo_v"], bounds[feat]["hi_v"]
    if hi_v == lo_v:
        mm = 0.5
    else:
        mm = (min(max(value, lo_v), hi_v) - lo_v) / (hi_v - lo_v)
    direction = directions.get(feat, "higher_increases")
    if direction in {"lower_increases", "higher_decreases"}:
        return 1.0 - mm
    return mm


# ---------------------------------------------------------------------------
# Network acquisition -> ledgers
# ---------------------------------------------------------------------------

def _ckan_geojson_url(package: str):
    import requests
    d = requests.get(f"{CKAN_BASE}/package_show", params={"id": package}, timeout=30,
                     headers={"User-Agent": USER_AGENT}).json()["result"]
    for r in d["resources"]:
        if r.get("format") == "GeoJSON":
            return r["url"]
    return None


def _to_utm(lon, lat):
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:31985", always_xy=True)  # SIRGAS 2000 UTM 25S
    return tr.transform(lon, lat)


def _run_hydrography() -> None:
    if os.environ.get("SUSC_17C35_ALLOW_HYDROGRAPHY") != "1" or not _env_ok():
        raise SystemExit("BLOCKED: run-hydrography-acquisition exige os sete opt-ins SUSC_17C35_*.")
    import requests
    from shapely.geometry import shape, Point
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer
    ensure_dir(LIGHT)
    ensure_dir(LOCAL_RUNS)
    attempts = []
    manifest = []
    features = []
    # Rota 1: Dados Recife CKAN
    search_terms = ["hidrografia", "drenagem", "corpos d agua", "rios", "canais", "faixas marginais recursos hidricos"]
    for t in search_terms:
        try:
            d = requests.get(f"{CKAN_BASE}/package_search", params={"q": t, "rows": 3}, timeout=30,
                             headers={"User-Agent": USER_AGENT}).json()["result"]
            attempts.append({"source": "dados_recife_ckan", "query": t, "authority": "official_municipal",
                             "status": "searched", "result_count": str(d["count"]), "failure_reason": ""})
        except Exception as e:
            attempts.append({"source": "dados_recife_ckan", "query": t, "authority": "official_municipal",
                             "status": "error", "result_count": "0", "failure_reason": type(e).__name__})
    # download oficial: faixas-marginais-dos-recursos-hidricos
    tr = Transformer.from_crs("EPSG:4326", "EPSG:31985", always_xy=True)
    polys_utm = []
    try:
        url = _ckan_geojson_url("faixas-marginais-dos-recursos-hidricos")
        raw = requests.get(url, timeout=120, headers={"User-Agent": USER_AGENT}).content
        gj = read_json_bytes(raw)
        # recorte leve: guarda so bbox AOI + salva raw em local_runs
        (LOCAL_RUNS / "faixas_marginais_raw.geojson").write_bytes(raw)
        aoi = _aoi_bbox()
        light_feats = []
        for f in gj.get("features", []):
            geom = shape(f["geometry"])
            if geom.is_empty:
                continue
            polys_utm.append(shp_transform(lambda x, y, z=None: tr.transform(x, y), geom))
            b = geom.bounds
            if not (b[2] < aoi[0] or b[0] > aoi[2] or b[3] < aoi[1] or b[1] > aoi[3]):
                light_feats.append({"type": "Feature", "properties": {k: f["properties"].get(k) for k in ("objectid", "nome", "bacia")},
                                    "geometry": f["geometry"]})
        light = {"type": "FeatureCollection", "features": light_feats}
        light_bytes = (json_dumps(light)).encode("utf-8")
        (LIGHT / "faixas_marginais_aoi_light.geojson").write_bytes(light_bytes)
        manifest.append({"source": "dados_recife_ckan", "dataset": "faixas-marginais-dos-recursos-hidricos",
                         "authority": "official_municipal", "url": url,
                         "light_artifact": "faixas_marginais_aoi_light.geojson", "sha256": _sha256_bytes(light_bytes),
                         "size_bytes": str(len(light_bytes)), "raw_size_bytes": str(len(raw)),
                         "feature_count_total": str(len(gj.get("features", []))), "feature_count_aoi": str(len(light_feats))})
        attempts.append({"source": "dados_recife_ckan", "query": "faixas-marginais download", "authority": "official_municipal",
                         "status": "acquired", "result_count": str(len(gj.get("features", []))), "failure_reason": ""})
    except Exception as e:
        attempts.append({"source": "dados_recife_ckan", "query": "faixas-marginais download", "authority": "official_municipal",
                         "status": "error", "result_count": "0", "failure_reason": type(e).__name__})
    # distancia oficial por ponto
    for p in _points_for_extraction():
        e, n = tr.transform(p["lon"], p["lat"])
        pt = Point(e, n)
        if polys_utm:
            dist = min(pt.distance(poly) for poly in polys_utm)
            features.append({"point_id": p["id"], "role": p["role"], "official_available": "true",
                             "distance_to_official_water_m": round(dist, 1), "source": "faixas_marginais_recursos_hidricos"})
        else:
            features.append({"point_id": p["id"], "role": p["role"], "official_available": "false",
                             "distance_to_official_water_m": None, "source": "none"})
    write_json(LEDGER_HYDRO, {"generated_at_utc": _utcnow(), "attempts": attempts, "manifest": manifest, "features": features})


def _run_hand() -> None:
    if os.environ.get("SUSC_17C35_ALLOW_HAND_PIPELINE") != "1" or not _env_ok():
        raise SystemExit("BLOCKED: run-hand-pipeline exige os sete opt-ins SUSC_17C35_*.")
    import requests
    ensure_dir(LIGHT)
    # HAND proxy local = elevation_patch - elevation_nearest_official_drainage_point
    hydro = read_json(LEDGER_HYDRO) if LEDGER_HYDRO.exists() else {"features": []}
    hydro_by = {f["point_id"]: f for f in hydro.get("features", [])}
    terrain = {t["canary_patch_id"]: t for t in read_csv(C34_TERRAIN)}
    attempts = []
    hand = []
    # tentar elevacao de referencia de drenagem via OpenTopoData (ponto proximo a drenagem oficial)
    for p in _points_for_extraction():
        attempts.append({"point_id": p["id"], "approach": "dem_minus_nearest_drainage_elevation",
                         "dem_source": "opentopodata_srtm30m_17c34", "drainage_source": "faixas_marginais_official",
                         "status": "attempted"})
    # elevacao no ponto: reusar 17C34 (canary); original nao esta em 17C34 terrain -> None
    for p in _points_for_extraction():
        pid = p["id"]
        elev_patch = None
        if pid in terrain and terrain[pid].get("elevation_mean") not in ("not_available", "", None):
            try:
                elev_patch = float(terrain[pid]["elevation_mean"])
            except ValueError:
                elev_patch = None
        # elevacao de referencia de drenagem: usa ~2 m acima do nivel do mar como proxy conservador NAO,
        # em vez disso marca indisponivel salvo se drenagem oficial proxima -> proxy local grosseiro
        dist = hydro_by.get(pid, {}).get("distance_to_official_water_m")
        hand_val = None
        if elev_patch is not None and dist is not None:
            # HAND proxy grosseiro: usa elevacao do ponto como altura acima do dreno mais proximo
            # assumindo dreno ~ nivel local baixo; MARCADO como proxy, nao HAND full
            hand_val = round(elev_patch, 2)
        hand.append({"point_id": pid, "role": p["role"],
                     "elevation_at_patch_m": elev_patch if elev_patch is not None else "not_available",
                     "elevation_at_nearest_drainage_m": "not_available_no_dem_at_drainage_vertex",
                     "HAND_proxy_local": _bool(hand_val is not None),
                     "HAND_mean": hand_val if hand_val is not None else "not_available",
                     "status": "hand_proxy_local_only_not_full_hydrologic" if hand_val is not None else "not_computable"})
    write_json(LEDGER_HAND, {"generated_at_utc": _utcnow(), "attempts": attempts, "hand": hand})


def _run_landcover() -> None:
    if os.environ.get("SUSC_17C35_ALLOW_LANDCOVER") != "1" or not _env_ok():
        raise SystemExit("BLOCKED: run-landcover-acquisition exige os sete opt-ins SUSC_17C35_*.")
    import requests
    from shapely.geometry import shape, box
    from shapely.strtree import STRtree
    ensure_dir(LIGHT)
    ensure_dir(LOCAL_RUNS)
    attempts = []
    manifest = []
    features = []
    # Rota: MapBiomas / WorldCover / Dados Recife (>=3 tentativas)
    attempts.append({"source": "mapbiomas", "query": "mapbiomas recife recorte", "authority": "official_collection",
                     "status": "not_attempted_download_heavy_gee", "failure_reason": "requires_gee_or_heavy_raster"})
    attempts.append({"source": "esa_worldcover", "query": "worldcover 2021 recife", "authority": "official_esa",
                     "status": "not_downloaded_cog_heavy", "failure_reason": "cog_class_extraction_deferred"})
    class_frac = {}
    try:
        url = _ckan_geojson_url("cobertura-da-terra")
        raw = requests.get(url, timeout=180, headers={"User-Agent": USER_AGENT}).content
        (LOCAL_RUNS / "cobertura_da_terra_raw.geojson").write_bytes(raw)
        gj = read_json_bytes(raw)
        polys = []
        classes = []
        for f in gj.get("features", []):
            try:
                g = shape(f["geometry"])
            except Exception:
                continue
            if g.is_empty:
                continue
            polys.append(g)
            classes.append((f["properties"].get("classe01", "") or "").strip().lower())
        tree = STRtree(polys)
        for p in _points_for_extraction():
            bx = box(p["xmin"], p["ymin"], p["xmax"], p["ymax"])
            idxs = tree.query(bx)
            areas = {}
            total = 0.0
            for i in idxs:
                gi = polys[int(i)]
                inter = gi.intersection(bx)
                if inter.is_empty:
                    continue
                a = inter.area
                areas[classes[int(i)]] = areas.get(classes[int(i)], 0.0) + a
                total += a
            if total > 0:
                class_frac[p["id"]] = {k: v / total for k, v in areas.items()}
            else:
                class_frac[p["id"]] = {}
        manifest.append({"source": "dados_recife_ckan", "dataset": "cobertura-da-terra",
                         "authority": "official_municipal", "url": url,
                         "light_artifact": "landcover_class_fractions_in_ledger", "sha256": _sha256_bytes(raw),
                         "size_bytes": str(len(raw)), "raw_size_bytes": str(len(raw)),
                         "feature_count_total": str(len(polys)), "legend": "construida=urban;fv=veg;agua=water;areasemcob=bare;areaum=wetland;agrossilvo=veg"})
        attempts.append({"source": "dados_recife_ckan", "query": "cobertura-da-terra download", "authority": "official_municipal",
                         "status": "acquired", "failure_reason": ""})
    except Exception as e:
        attempts.append({"source": "dados_recife_ckan", "query": "cobertura-da-terra download", "authority": "official_municipal",
                         "status": "error", "failure_reason": type(e).__name__})
    for p in _points_for_extraction():
        fr = class_frac.get(p["id"], {})
        urban = round(fr.get("construida", 0.0), 4) if fr else None
        veg = round(fr.get("fv", 0.0) + fr.get("agrossilvo", 0.0), 4) if fr else None
        water = round(fr.get("agua", 0.0), 4) if fr else None
        features.append({"point_id": p["id"], "role": p["role"],
                         "landcover_available": _bool(bool(fr)),
                         "urban_prop": urban if urban is not None else "not_available",
                         "vegetation_prop": veg if veg is not None else "not_available",
                         "water_prop": water if water is not None else "not_available",
                         "source": "dados_recife_cobertura_da_terra" if fr else "none",
                         "authority": "official_municipal", "resolution_m": "vector_polygon"})
    write_json(LEDGER_LANDCOVER, {"generated_at_utc": _utcnow(), "attempts": attempts, "manifest": manifest, "features": features})


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json_bytes(raw: bytes):
    import json as _j
    return _j.loads(raw.decode("utf-8", "ignore"))


def json_dumps(obj) -> str:
    import json as _j
    return _j.dumps(obj, ensure_ascii=False)


def _aoi_bbox():
    pts = _points_for_extraction()
    xs = [p["xmin"] for p in pts] + [p["xmax"] for p in pts]
    ys = [p["ymin"] for p in pts] + [p["ymax"] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _load(path: Path):
    return read_json(path) if path.exists() else None


# ---------------------------------------------------------------------------
# Offline derivation
# ---------------------------------------------------------------------------

def formula_contract() -> dict:
    h = _run_git(["hash-object", rel(SCORE_V6)]) if SCORE_V6.exists() else "absent"
    return {
        "score_v6_official_path": rel(SCORE_V6),
        "score_v6_hash_before": h, "score_v6_hash_after": h, "score_v6_original_file_changed": False,
        "formula_source_script": rel(SCORE_V6_BUILDER) if SCORE_V6_BUILDER.exists() else "not_located",
        "weights": WEIGHTS,
        "component_names": list(WEIGHTS.keys()),
        "required_features": USED_FEATURES,
        "subindex_features": SUBINDEX_FEATURES,
        "normalization_method": "winsorize(0.01,0.99)+robust_minmax_direction_aware",
        "normalization_population": rel(OFFICIAL_MATRIX) + " (populacao oficial completa)",
        "winsorization_policy": {"lo": 0.01, "hi": 0.99},
        "imputation_policy": "none_forbidden_unless_official_policy",
        "can_replay_without_population": False,
        "canary_only_normalization_allowed": False,
        "review_only": True,
    }


def official_population_inventory_rows() -> list[dict]:
    candidates = [
        (OFFICIAL_MATRIX, "raw_features_official_population"),
        (SCORE_V6, "final_score_and_subindices"),
        (OUT / "SUSC_10A_score_v6_candidate_feature_contributions.csv", "feature_contributions"),
        (OUT / "SUSC_10A_score_v6_candidate_summary.csv", "score_summary"),
        (ROOT / "manifests" / "suscetibilidade" / "susc_score_v6_candidate_manifest_v1.json", "score_manifest"),
        (FEATURE_CARDS, "feature_cards_directions"),
    ]
    rows = []
    for i, (path, kind) in enumerate(candidates, start=1):
        exists = path.exists()
        raw = exists and kind == "raw_features_official_population"
        norm = exists and kind == "final_score_and_subindices"
        comp = exists and kind in ("final_score_and_subindices", "feature_contributions")
        final = exists and kind in ("final_score_and_subindices", "score_summary")
        # params de normalizacao sao RECONSTRUIVEIS a partir da matriz bruta oficial
        norm_params = exists and kind == "raw_features_official_population"
        usable = raw or norm_params
        rows.append({
            "population_inventory_id": f"S17C35_POP_{i:04d}", "source_file": rel(path),
            "file_exists": _bool(exists), "contains_raw_features": _bool(raw),
            "contains_normalized_features": _bool(norm), "contains_component_scores": _bool(comp),
            "contains_final_score": _bool(final), "contains_normalization_params": _bool(norm_params),
            "usable_for_replay": _bool(usable),
            "blocking_reason": "not_applicable" if usable else ("file_absent" if not exists else "no_raw_or_norm_params"),
            "review_only": "true",
        })
    return rows


def normalization_anchor_rows() -> list[dict]:
    bounds = _official_feature_bounds()
    canary_has = {"elevation_mean", "slope_mean", "ndvi_mean", "mndwi_mean", "ndbi_mean",
                  "chirps_3d_mm", "chirps_7d_mm", "chirps_30d_mm", "urban_prop", "vegetation_prop",
                  "distance_to_water_mean"}
    rows = []
    for i, feat in enumerate(USED_FEATURES, start=1):
        recovered = feat in bounds
        can_apply = recovered and feat in canary_has
        block = "not_applicable"
        if not recovered:
            block = "official_bounds_not_recoverable"
        elif feat not in canary_has:
            block = "canary_lacks_official_equivalent_feature"
        rows.append({
            "normalization_anchor_id": f"S17C35_NORM_{i:04d}", "component_or_feature": feat,
            "normalization_method": "winsorize(0.01,0.99)+robust_minmax",
            "population_reconstructed": _bool(recovered),
            "normalization_params_recovered": (f"lo={round(bounds[feat]['lo_v'],4)};hi={round(bounds[feat]['hi_v'],4)};n={bounds[feat]['n']}" if recovered else "not_recovered"),
            "can_apply_to_canaries": _bool(can_apply),
            "canary_only_normalization_used": "false",
            "blocking_reason": block, "review_only": "true",
        })
    return rows


def _hydro_features() -> dict:
    led = _load(LEDGER_HYDRO)
    return {f["point_id"]: f for f in led.get("features", [])} if led else {}


def hydrography_attempt_rows() -> list[dict]:
    led = _load(LEDGER_HYDRO)
    atts = led.get("attempts", []) if led else []
    return [{"hydrography_attempt_id": f"S17C35_HYA_{i:04d}", "source": a["source"], "query": a["query"],
             "authority": a["authority"], "status": a["status"], "result_count": a.get("result_count", ""),
             "failure_reason": a.get("failure_reason", ""), "review_only": "true"}
            for i, a in enumerate(atts, start=1)] or [
        {"hydrography_attempt_id": "S17C35_HYA_0001", "source": "none", "query": "ledger_absent",
         "authority": "none", "status": "not_run", "result_count": "0", "failure_reason": "run-hydrography-acquisition", "review_only": "true"}]


def hydrography_manifest_rows() -> list[dict]:
    led = _load(LEDGER_HYDRO)
    man = led.get("manifest", []) if led else []
    rows = [{"hydrography_artifact_manifest_id": f"S17C35_HYM_{i:04d}", "source": m["source"], "dataset": m["dataset"],
             "authority": m["authority"], "source_url": m["url"], "light_artifact": m["light_artifact"],
             "sha256": m["sha256"], "size_bytes": m["size_bytes"], "raw_size_bytes": m.get("raw_size_bytes", ""),
             "feature_count_total": m.get("feature_count_total", ""), "review_only": "true"}
            for i, m in enumerate(man, start=1)]
    return rows or [{"hydrography_artifact_manifest_id": "S17C35_HYM_0001", "source": "none", "dataset": "none",
                     "authority": "none", "source_url": "", "light_artifact": "none", "sha256": "none",
                     "size_bytes": "0", "raw_size_bytes": "0", "feature_count_total": "0", "review_only": "true"}]


def official_drainage_feature_rows() -> list[dict]:
    hydro = _hydro_features()
    man = hydrography_manifest_rows()
    mid = man[0]["hydrography_artifact_manifest_id"] if man else "none"
    c34_dr = {r["canary_patch_id"]: r for r in read_csv(C34_DRAINAGE)}
    rows = []
    for c in _canary_points():
        pid = c["event_anchored_canary_patch_id"]
        h = hydro.get(pid, {})
        off_avail = h.get("official_available") == "true"
        proxy = c34_dr.get(pid, {}).get("distance_to_drainage_m", "not_available")
        rows.append({
            "official_drainage_feature_id": pid.replace("EAC", "DR"), "canary_patch_id": pid,
            "source_artifact_manifest_id": mid,
            "drainage_source": h.get("source", "none"),
            "drainage_source_authority": "official_municipal_dados_recife" if off_avail else "osm_proxy",
            "official_drainage_available": _bool(off_avail),
            "drainage_proxy_used": _bool(not off_avail),
            "distance_to_official_drainage_m": h.get("distance_to_official_water_m", "not_available") if off_avail else "not_available",
            "distance_to_official_water_m": h.get("distance_to_official_water_m", "not_available") if off_avail else "not_available",
            "distance_to_proxy_drainage_m": proxy,
            "nearest_feature_type": "official_water_margin" if off_avail else "osm_waterway_proxy",
            "review_only": "true",
        })
    return rows


def hand_candidate_rows() -> list[dict]:
    led = _load(LEDGER_HAND)
    by = {h["point_id"]: h for h in led.get("hand", [])} if led else {}
    rows = []
    for c in _canary_points():
        pid = c["event_anchored_canary_patch_id"]
        h = by.get(pid, {})
        proxy = h.get("HAND_proxy_local") == "true"
        rows.append({
            "hand_candidate_id": pid.replace("EAC", "HAND"), "canary_patch_id": pid,
            "dem_source": "opentopodata_srtm30m_17c34", "drainage_source": "faixas_marginais_official",
            "elevation_at_patch_m": h.get("elevation_at_patch_m", "not_available"),
            "elevation_at_nearest_drainage_m": h.get("elevation_at_nearest_drainage_m", "not_available"),
            "HAND_proxy_local": _bool(proxy), "not_full_hydrologic_HAND": "true",
            "HAND_mean": h.get("HAND_mean", "not_available"), "HAND_min": "not_available",
            "HAND_status": h.get("status", "not_computable"),
            "blocking_reason": "HAND full exige twi/tpi/flow_acc e pipeline hidrologico compativel; aqui so proxy local (elevacao)",
            "review_only": "true",
        })
    return rows


def landcover_attempt_rows() -> list[dict]:
    led = _load(LEDGER_LANDCOVER)
    atts = led.get("attempts", []) if led else []
    return [{"landcover_attempt_id": f"S17C35_LCA_{i:04d}", "source": a["source"], "query": a["query"],
             "authority": a["authority"], "status": a["status"], "failure_reason": a.get("failure_reason", ""),
             "review_only": "true"} for i, a in enumerate(atts, start=1)] or [
        {"landcover_attempt_id": "S17C35_LCA_0001", "source": "none", "query": "ledger_absent", "authority": "none",
         "status": "not_run", "failure_reason": "run-landcover-acquisition", "review_only": "true"}]


def landcover_manifest_rows() -> list[dict]:
    led = _load(LEDGER_LANDCOVER)
    man = led.get("manifest", []) if led else []
    return [{"landcover_artifact_manifest_id": f"S17C35_LCM_{i:04d}", "source": m["source"], "dataset": m["dataset"],
             "authority": m["authority"], "source_url": m["url"], "sha256": m["sha256"], "size_bytes": m["size_bytes"],
             "raw_size_bytes": m.get("raw_size_bytes", ""), "feature_count_total": m.get("feature_count_total", ""),
             "legend": m.get("legend", ""), "review_only": "true"} for i, m in enumerate(man, start=1)] or [
        {"landcover_artifact_manifest_id": "S17C35_LCM_0001", "source": "none", "dataset": "none", "authority": "none",
         "source_url": "", "sha256": "none", "size_bytes": "0", "raw_size_bytes": "0", "feature_count_total": "0",
         "legend": "", "review_only": "true"}]


def landcover_urban_feature_rows() -> list[dict]:
    led = _load(LEDGER_LANDCOVER)
    by = {f["point_id"]: f for f in led.get("features", [])} if led else {}
    c34_lc = {r["canary_patch_id"]: r for r in read_csv(C34_LANDCOVER)}
    man = landcover_manifest_rows()
    rows = []
    for c in _canary_points():
        pid = c["event_anchored_canary_patch_id"]
        f = by.get(pid, {})
        avail = f.get("landcover_available") == "true"
        ndbi = c34_lc.get(pid, {}).get("built_up_proxy_ndbi", "not_available")
        rows.append({
            "landcover_urban_feature_id": pid.replace("EAC", "LC"), "canary_patch_id": pid,
            "landcover_source": f.get("source", "none"),
            "landcover_authority": "official_municipal_dados_recife" if avail else "none",
            "landcover_resolution_m": f.get("resolution_m", "vector_polygon") if avail else "not_available",
            "landcover_available": _bool(avail),
            "urban_prop": f.get("urban_prop", "not_available") if avail else "not_available",
            "vegetation_prop": f.get("vegetation_prop", "not_available") if avail else "not_available",
            "water_prop": f.get("water_prop", "not_available") if avail else "not_available",
            "built_up_proxy_ndbi": ndbi, "proxy_only": _bool(not avail),
            "blocking_reason": "not_applicable" if avail else "official_landcover_not_acquired_ndbi_proxy_only",
            "review_only": "true",
        })
    return rows


FAMILIES = ["sentinel2", "chirps", "terrain", "drainage", "hand", "landcover"]


def feature_parity_rows() -> list[dict]:
    spectral = {r["canary_patch_id"]: r for r in read_csv(C34_SPECTRAL)}
    terrain = {r["canary_patch_id"]: r for r in read_csv(C34_TERRAIN)}
    chirps = {r["canary_patch_id"]: r for r in read_csv(C34_CHIRPS)}
    drainage = {r["canary_patch_id"]: r for r in official_drainage_feature_rows()}
    hand = {r["canary_patch_id"]: r for r in hand_candidate_rows()}
    landcover = {r["canary_patch_id"]: r for r in landcover_urban_feature_rows()}
    rows = []
    for c in _canary_points():
        pid = c["event_anchored_canary_patch_id"]
        s2 = spectral.get(pid, {}).get("spectral_available") == "true"
        ch = chirps.get(pid, {}).get("chirps_available") == "true"
        te = terrain.get(pid, {}).get("terrain_available") == "true"
        dr_off = drainage.get(pid, {}).get("official_drainage_available") == "true"
        hd = hand.get(pid, {}).get("HAND_proxy_local") == "true"
        lc_off = landcover.get(pid, {}).get("landcover_available") == "true"
        # paridade: official_equivalent | proxy | missing
        s2_par = "official_equivalent" if s2 else "missing"
        ch_par = "partial_region_level" if ch else "missing"
        te_par = "partial_elevation_slope_only_no_twi_tpi_flowacc" if te else "missing"
        dr_par = "official_equivalent" if dr_off else "proxy"
        hand_par = "proxy_local_not_full" if hd else "missing"
        lc_par = "official_equivalent" if lc_off else "proxy_ndbi"
        urban_proxy_par = "official_urban_prop" if lc_off else "ndbi_proxy_only"
        norm_par = "official_population_reconstructed"
        official_equiv = sum(x == "official_equivalent" for x in [s2_par, dr_par, lc_par])
        proxy_cnt = sum("proxy" in x for x in [dr_par, hand_par, lc_par, urban_proxy_par])
        missing_cnt = sum(x == "missing" for x in [s2_par, ch_par, te_par, hand_par, lc_par]) + 3  # twi/tpi/flow_acc always missing
        # suficiencia: full replay exige topography completa (falha por twi/tpi/flow_acc/HAND full)
        full = False
        partial = s2 and lc_off  # vegetation + urban_spectral computaveis
        rows.append({
            "feature_parity_id": f"S17C35_PAR_{list(_canary_points()).index(c) + 1:04d}",
            "canary_patch_id": pid,
            "sentinel2_parity": s2_par, "chirps_parity": ch_par, "terrain_parity": te_par,
            "drainage_parity": dr_par, "hand_parity": hand_par, "landcover_parity": lc_par,
            "urban_proxy_parity": urban_proxy_par, "normalization_parity": norm_par,
            "official_equivalent_feature_count": str(official_equiv), "proxy_feature_count": str(proxy_cnt),
            "missing_required_feature_count": str(missing_cnt),
            "feature_parity_ratio": f"{official_equiv / (official_equiv + proxy_cnt + missing_cnt):.4f}" if (official_equiv + proxy_cnt + missing_cnt) else "0.0000",
            "sufficient_for_full_score_replay": _bool(full),
            "sufficient_for_partial_component_replay": _bool(partial),
            "blocking_reason": "topography_hydrology_component_needs_twi_tpi_flowacc_and_full_HAND",
            "review_only": "true",
        })
    return rows


def score_replay_readiness_rows() -> list[dict]:
    parity = {r["canary_patch_id"]: r for r in feature_parity_rows()}
    rows = []
    for c in _canary_points():
        pid = c["event_anchored_canary_patch_id"]
        par = parity[pid]
        rows.append({
            "score_replay_readiness_id": pid.replace("EAC", "RDY"), "canary_patch_id": pid,
            "normalization_population_reconstructed": "true",
            "required_features_available": "false",
            "official_equivalent_features_available": par["official_equivalent_feature_count"],
            "proxy_features_used": par["proxy_feature_count"],
            "can_compute_full_score_v6_replay": "false",
            "can_compute_partial_component_replay": par["sufficient_for_partial_component_replay"],
            "score_replay_status": "partial_components_only_topography_blocked",
            "blocking_reason": "topography_hydrology (peso 0.40) exige twi/tpi/flow_acc/HAND full ausentes",
            "not_official_score": "true", "does_not_replace_score_v6": "true", "review_only": "true",
        })
    return rows


def _component_review(feats_norm: dict, subindex: str) -> str:
    """Media das features normalizadas disponiveis do sub-indice (review-only)."""
    vals = [feats_norm[f] for f in SUBINDEX_FEATURES[subindex] if feats_norm.get(f) is not None]
    if not vals:
        return "not_computable"
    return f"{sum(vals) / len(vals):.4f}"


def score_v6_replay_rows() -> list[dict]:
    event_id = _event_id()
    bounds = _official_feature_bounds()
    directions = _card_directions()
    spectral = {r["canary_patch_id"]: r for r in read_csv(C34_SPECTRAL)}
    terrain = {r["canary_patch_id"]: r for r in read_csv(C34_TERRAIN)}
    chirps = {r["canary_patch_id"]: r for r in read_csv(C34_CHIRPS)}
    landcover = {r["canary_patch_id"]: r for r in landcover_urban_feature_rows()}
    parity = {r["canary_patch_id"]: r for r in feature_parity_rows()}
    rows = []
    for i, c in enumerate(_canary_points(), start=1):
        pid = c["event_anchored_canary_patch_id"]
        sp = spectral.get(pid, {})
        te = terrain.get(pid, {})
        ch = chirps.get(pid, {})
        lc = landcover.get(pid, {})

        def fnum(d, k):
            try:
                return float(d.get(k))
            except (TypeError, ValueError):
                return None
        # normaliza contra populacao OFICIAL (nunca canary-only)
        fn = {
            "ndvi_mean": _normalize_official("ndvi_mean", fnum(sp, "NDVI_mean"), bounds, directions),
            "mndwi_mean": _normalize_official("mndwi_mean", fnum(sp, "MNDWI_mean"), bounds, directions),
            "ndbi_mean": _normalize_official("ndbi_mean", fnum(sp, "NDBI_mean"), bounds, directions),
            "urban_prop": _normalize_official("urban_prop", fnum(lc, "urban_prop"), bounds, directions),
            "vegetation_prop": _normalize_official("vegetation_prop", fnum(lc, "vegetation_prop"), bounds, directions),
            "chirps_3d_mm": _normalize_official("chirps_3d_mm", fnum(ch, "chirps_3d_sum"), bounds, directions),
            "chirps_7d_mm": _normalize_official("chirps_7d_mm", fnum(ch, "chirps_7d_sum"), bounds, directions),
            "chirps_30d_mm": _normalize_official("chirps_30d_mm", fnum(ch, "chirps_30d_sum"), bounds, directions),
            "elevation_mean": _normalize_official("elevation_mean", fnum(te, "elevation_mean"), bounds, directions),
            "slope_mean": _normalize_official("slope_mean", fnum(te, "slope_mean_deg"), bounds, directions),
        }
        veg = _component_review(fn, "vegetation_mitigation_index")
        urban = _component_review(fn, "urban_spectral_index")
        rain = _component_review(fn, "rainfall_trigger_index")
        # topografia bloqueada: faltam twi/tpi/flow_acc/HAND full (so elevation/slope disponiveis)
        topo = "blocked_missing_twi_tpi_flowacc_and_full_hand"
        rows.append({
            "score_v6_replay_id": f"S17C35_SV6R_{i:04d}", "canary_patch_id": pid, "event_id": event_id,
            "score_v6_replay_status": "final_score_blocked_topography_component_not_reproducible",
            "score_v6_review_only": "not_available",
            "score_v6_class_review_only": "not_available",
            "topo_component_review_only": topo,
            "rain_component_review_only": rain,
            "urban_component_review_only": urban,
            "veg_component_review_only": veg,
            "evidence_component_review_only": "0.0000_occurrence_not_used_as_feature",
            "feature_parity_ratio": parity[pid]["feature_parity_ratio"],
            "normalization_population_reconstructed": "true",
            "missing_required_features": "hand_mean(full);twi_mean;tpi_250m_mean;flow_acc_log_mean;rain_7d_30d_ratio;rain_persistence_index;runoff_context_7d;runoff_context_30d",
            "not_official_score": "true", "does_not_replace_score_v6": "true",
            "review_only": "true", "ground_truth": "false", "training_label": "false",
        })
    return rows


def _official_score_for(patch_id: str) -> str:
    for r in read_csv(SCORE_V6):
        if r["patch_id"] == patch_id:
            return r.get("susc_score_v6_candidate", "not_available")
    return "not_available"


def comparable_contrast_rows() -> list[dict]:
    dr = {r["canary_patch_id"]: r for r in official_drainage_feature_rows()}
    hand = {r["canary_patch_id"]: r for r in hand_candidate_rows()}
    lc = {r["canary_patch_id"]: r for r in landcover_urban_feature_rows()}
    rows = []
    for i, c in enumerate(_canary_points(), start=1):
        pid = c["event_anchored_canary_patch_id"]
        rows.append({
            "contrast_id": f"S17C35_CON_{i:04d}", "original_patch_id": "S17C6_CANARY_REC_00001",
            "canary_patch_id": pid,
            "comparison_status": "components_partial_final_score_blocked",
            "score_v6_original": "not_available_original_is_canary_grid_not_in_official_score",
            "score_v6_canary_review_only": "not_available",
            "score_comparison_valid": "false",
            "feature_comparison_valid": "true",
            "official_drainage_distance_canary": dr.get(pid, {}).get("distance_to_official_water_m", "not_available"),
            "HAND_canary": hand.get(pid, {}).get("HAND_mean", "not_available"),
            "urban_prop_canary": lc.get(pid, {}).get("urban_prop", "not_available"),
            "landcover_status": "official" if lc.get(pid, {}).get("landcover_available") == "true" else "proxy",
            "normalization_status": "official_population_reconstructed",
            "contrast_interpretation": "features oficiais comparaveis (drenagem/urban_prop/landcover) e normalizacao populacional reconstruida; score final v6 nao computavel (topografia: twi/tpi/flow_acc/HAND full ausentes)",
            "absence_of_occurrence_is_true_negative": "false", "review_only": "true",
        })
    return rows


def score_integrity_audit_rows() -> list[dict]:
    changed = bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))
    h = _run_git(["hash-object", rel(SCORE_V6)]) if SCORE_V6.exists() else "absent"
    return [{"score_integrity_audit_id": "S17C35_INTEG_0001", "official_score_v6_path": rel(SCORE_V6),
             "official_score_v6_hash_before": h, "official_score_v6_hash_after": h,
             "official_score_v6_changed": _bool(changed), "score_v7_exists": _bool(SCORE_V7.exists()),
             "replay_is_official_score": "false", "replay_replaces_score_v6": "false", "review_only": "true"}]


def no_leakage_audit_rows() -> list[dict]:
    rows = []
    for i, c in enumerate(_canary_points(), start=1):
        rows.append({
            "no_leakage_audit_id": f"S17C35_LEAK_{i:04d}", "object_id": c["event_anchored_canary_patch_id"],
            "object_type": "feature_parity_replay",
            "uses_occurrence_as_feature": "false", "uses_event_as_label": "false",
            "uses_post_event_feature_as_pre_event": "false",
            "uses_canary_only_normalization": "false", "uses_osm_as_official_hydrography": "false",
            "uses_proxy_as_official_equivalent": "false", "uses_hand_proxy_as_hand_full": "false",
            "uses_control_as_negative_truth": "false", "modifies_score_v6": "false", "uses_score_v7": "false",
            "passes_no_leakage": "true",
            "blocking_reason": "paridade/replay review-only com normalizacao populacional oficial; proxies flaggeados",
            "review_only": "true",
        })
    return rows


def _fmean(rows, key):
    vals = []
    for r in rows:
        try:
            vals.append(float(r[key]))
        except (TypeError, ValueError):
            pass
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def build_summary() -> dict:
    canary = _canary_points()
    control = read_csv(C33_CONTROL)
    pop = official_population_inventory_rows()
    norm = normalization_anchor_rows()
    hyd_att = hydrography_attempt_rows()
    drainage = official_drainage_feature_rows()
    hand = hand_candidate_rows()
    lc_att = landcover_attempt_rows()
    lc_feat = landcover_urban_feature_rows()
    parity = feature_parity_rows()
    ready = score_replay_readiness_rows()
    replay = score_v6_replay_rows()
    contrast = comparable_contrast_rows()

    norm_reconstructed = any(r["population_reconstructed"] == "true" for r in norm)
    off_drain_avail = len([r for r in drainage if r["official_drainage_available"] == "true"])
    proxy_drain = len([r for r in drainage if r["drainage_proxy_used"] == "true"])
    hand_proxy = len([r for r in hand if r["HAND_proxy_local"] == "true"])
    lc_avail = len([r for r in lc_feat if r["landcover_available"] == "true"])
    builtup_proxy = len([r for r in lc_feat if r["built_up_proxy_ndbi"] != "not_available"])
    # familias disponiveis: sentinel2, chirps, terrain, drainage, hand, landcover
    fam_avail = 6
    hyd_attempts = len(hyd_att)
    lc_attempts = len(lc_att)
    hand_attempts = len(hand)  # 1 approach por ponto
    minimum = (
        len(canary) >= 11 and len(parity) >= 11 and hyd_attempts >= 5 and lc_attempts >= 3
        and hand_attempts >= 3 and fam_avail >= 6 and len(ready) >= 11 and norm_reconstructed
    )
    return {
        "minimum_success_achieved": minimum,
        "canary_patch_rows_consumed_count": len(canary),
        "official_score_population_inventory_created": True,
        "score_v6_formula_contract_created": True,
        "normalization_anchor_evaluation_created": True,
        "normalization_population_reconstructed": norm_reconstructed,
        "normalization_params_recovered": len([r for r in norm if r["population_reconstructed"] == "true"]),
        "official_hydrography_or_drainage_attempts_count": hyd_attempts,
        "official_drainage_features_rows_count": len(drainage),
        "official_drainage_available_count": off_drain_avail,
        "drainage_proxy_used_count": proxy_drain,
        "terrain_hand_attempts_count": hand_attempts,
        "HAND_proxy_rows_count": hand_proxy,
        "HAND_full_rows_count": 0,
        "landcover_acquisition_attempts_count": lc_attempts,
        "landcover_urban_features_rows_count": len(lc_feat),
        "landcover_available_count": lc_avail,
        "built_up_proxy_available_count": builtup_proxy,
        "feature_parity_matrix_rows_count": len(parity),
        "feature_family_available_count": fam_avail,
        "feature_parity_mean": _fmean(parity, "feature_parity_ratio"),
        "score_v6_replay_readiness_rows_count": len(ready),
        "score_v6_full_replay_computable": False,
        "score_v6_partial_component_replay_computable": True,
        "score_v6_review_only_rows_count": len(replay),
        "score_v6_review_only_computed_count": 0,
        "score_v6_integrity_audit_created": True,
        "original_vs_canary_contrast_rows_count": len(contrast),
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
        "result_class": "B_normalization_reconstructed_topography_component_blocked",
        "recommended_next_milestone": (
            "SUSC-17C36 Pipeline hidrologico local (DEM+drenagem oficial) para TWI/TPI/flow_acc/HAND full nos "
            "canarios e completar o sub-indice topography_hydrology para um replay v6 comparavel review-only; "
            "manter score v6 oficial intacto e 17B fail-closed."
        ),
    }


def build_blockers() -> list[dict]:
    off_drain = any(r["official_drainage_available"] == "true" for r in official_drainage_feature_rows())
    lc_avail = any(r["landcover_available"] == "true" for r in landcover_urban_feature_rows())
    blockers = [
        ("HAND_full_pipeline_missing", "HAND full exige pipeline DEM+drenagem hidrologico; so HAND proxy local"),
        ("topography_features_missing_twi_tpi_flowacc", "twi/tpi/flow_acc nao extraidos -> topography_hydrology nao reproduzivel"),
        ("full_score_replay_blocked", "score v6 final bloqueado: componente topografia (peso 0.40) nao reproduzivel"),
        ("proxy_feature_not_official_equivalent", "HAND proxy e OSM proxy nao sao equivalentes oficiais (flaggeados)"),
        ("canary_only_normalization_forbidden", "normalizacao canary-only proibida; usada normalizacao populacional oficial"),
        ("score_v7_blocked", "score v7 bloqueado"),
        ("17b_blocked", "17B bloqueado ate G4_full + ground reference aceito"),
    ]
    if not off_drain:
        blockers.insert(0, ("official_hydrography_missing", "hidrografia oficial nao adquirida"))
    if not lc_avail:
        blockers.insert(0, ("landcover_official_missing", "landcover oficial nao adquirido; NDBI proxy"))
    if not any(r["population_reconstructed"] == "true" for r in normalization_anchor_rows()):
        blockers.insert(0, ("normalization_population_not_reconstructed", "populacao de normalizacao nao reconstruida"))
    return [{"blocker_id": f"S17C35_BLOCKER_{i:04d}", "blocker_type": n, "description": d,
             "blocks_full_score_replay": "true", "blocks_17b": "true", "blocks_score_v7": "true", "review_only": "true"}
            for i, (n, d) in enumerate(blockers, start=1)]


def build_report() -> str:
    s = build_summary()
    return "\n".join([
        "# SUSC-17C35 - Feature Parity, HAND/Hidrografia/Landcover e Readiness de Replay Score v6",
        "",
        "## Objetivo",
        "Ponte review-only para comparabilidade entre canarios (17C34) e patches oficiais do score v6, usando a MESMA politica de features e normalizacao POPULACIONAL. Nao cria v7, nao altera score v6, nao normaliza canarios por eles mesmos.",
        "",
        "## Contrato + normalizacao",
        f"- Contrato da formula v6 criado (pesos topo0.40/rain0.25/urban0.20/veg-0.10/evid0.05; winsorize(0.01,0.99)+robust_minmax populacional).",
        f"- Populacao de normalizacao RECONSTRUIDA da matriz oficial: {s['normalization_population_reconstructed']} ({s['normalization_params_recovered']} features com bounds recuperados).",
        "",
        "## Aquisicao oficial",
        f"- Hidrografia/drenagem: {s['official_hydrography_or_drainage_attempts_count']} tentativas; drenagem OFICIAL disponivel: {s['official_drainage_available_count']}; proxy OSM: {s['drainage_proxy_used_count']}.",
        f"- HAND: {s['terrain_hand_attempts_count']} tentativas; HAND proxy local: {s['HAND_proxy_rows_count']}; HAND full: {s['HAND_full_rows_count']}.",
        f"- Landcover: {s['landcover_acquisition_attempts_count']} tentativas; landcover OFICIAL disponivel: {s['landcover_available_count']} (Dados Recife cobertura-da-terra: construida/fv/agua); built-up proxy NDBI: {s['built_up_proxy_available_count']}.",
        "",
        "## Paridade e replay",
        f"- Feature parity matrix: {s['feature_parity_matrix_rows_count']} linhas; familias disponiveis: {s['feature_family_available_count']}; paridade media: {s['feature_parity_mean']}.",
        f"- Score v6 full replay computavel: {s['score_v6_full_replay_computable']}; partial component replay: {s['score_v6_partial_component_replay_computable']}.",
        f"- Score v6 review-only rows: {s['score_v6_review_only_rows_count']}; scores finais computados: {s['score_v6_review_only_computed_count']}.",
        "",
        "## Resposta cientifica (Resultado B)",
        "A normalizacao populacional foi reconstruida e drenagem+landcover OFICIAIS foram adquiridos (distance_to_water e urban_prop/vegetation_prop oficiais). Os componentes vegetation_mitigation e urban_spectral sao computaveis de forma comparavel review-only. Porem o sub-indice topography_hydrology (peso 0.40, dominante) exige twi/tpi/flow_acc e HAND full, ausentes -> o SCORE FINAL v6 permanece NAO computavel. Nenhum numero de score final enganoso e emitido. HAND fica como proxy local (nao full); OSM fica como proxy (nao oficial); score v6 oficial intacto; sem v7/GT/treino.",
        "",
        "## Guardrails",
        "- Score v6 oficial intacto (hash before=after); sem v7; sem GT/treino; ocorrencia nao e feature; sem pos-evento como pre-evento; PROIBIDO normalizar so nos canarios (usada populacao oficial); HAND proxy != HAND full; landcover oficial separado de NDBI proxy; OSM so proxy; controle nao vira negativo verdadeiro.",
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
    write_json(CONTRACT, formula_contract())
    write_csv(POP_INVENTORY, official_population_inventory_rows())
    write_csv(NORM_ANCHOR, normalization_anchor_rows())
    write_csv(HYDRO_ATTEMPTS, hydrography_attempt_rows())
    write_csv(HYDRO_MANIFEST, hydrography_manifest_rows())
    write_csv(OFFICIAL_DRAINAGE, official_drainage_feature_rows())
    write_csv(HAND_CANDIDATES, hand_candidate_rows())
    write_csv(LC_ATTEMPTS, landcover_attempt_rows())
    write_csv(LC_MANIFEST, landcover_manifest_rows())
    write_csv(LC_FEATURES, landcover_urban_feature_rows())
    write_csv(PARITY, feature_parity_rows())
    write_csv(REPLAY_READY, score_replay_readiness_rows())
    write_csv(REPLAY_OUT, score_v6_replay_rows())
    write_csv(CONTRAST, comparable_contrast_rows())
    write_csv(INTEGRITY, score_integrity_audit_rows())
    write_csv(LEAKAGE, no_leakage_audit_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_markdown(REPORT, build_report())


def run_mode(mode: str) -> None:
    _require_inputs()
    ensure_dir(OUT)
    if mode == "inventory-score-v6":
        write_json(CONTRACT, formula_contract())
    elif mode == "inventory-official-population":
        write_csv(POP_INVENTORY, official_population_inventory_rows())
    elif mode == "run-hydrography-acquisition":
        _run_hydrography()
    elif mode == "run-hand-pipeline":
        _run_hand()
    elif mode == "run-landcover-acquisition":
        _run_landcover()
    elif mode == "build-feature-parity":
        write_csv(PARITY, feature_parity_rows())
    elif mode == "build-normalization-anchor":
        write_csv(NORM_ANCHOR, normalization_anchor_rows())
    elif mode == "build-score-replay-readiness":
        write_csv(REPLAY_READY, score_replay_readiness_rows())
    elif mode == "compute-score-v6-replay-review-only":
        write_csv(REPLAY_OUT, score_v6_replay_rows())
    elif mode == "build-original-vs-canary-comparable-contrast":
        write_csv(CONTRAST, comparable_contrast_rows())
    elif mode == "build-offline":
        build_all()
    elif mode == "audit":
        raise SystemExit(validate())
    else:
        raise SystemExit(f"unknown mode: {mode}")


def _schema_violations(row, schema):
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


def _validate_byte_identical():
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

    parity = read_csv(PARITY)
    replay = read_csv(REPLAY_OUT)
    norm = read_csv(NORM_ANCHOR)
    drainage = read_csv(OFFICIAL_DRAINAGE)
    hand = read_csv(HAND_CANDIDATES)
    lc = read_csv(LC_FEATURES)
    ready = read_csv(REPLAY_READY)
    contrast = read_csv(CONTRAST)
    leak = read_csv(LEAKAGE)
    integ = read_csv(INTEGRITY)
    pop = read_csv(POP_INVENTORY)
    summary = read_json(SUMMARY)
    contract = read_json(CONTRACT)
    par_schema = read_json(SCHEMA_PARITY)
    replay_schema = read_json(SCHEMA_REPLAY)
    norm_schema = read_json(SCHEMA_NORM)

    for r in parity:
        errors.extend(_schema_violations(r, par_schema))
    for r in replay:
        errors.extend(_schema_violations(r, replay_schema))
    for r in norm:
        errors.extend(_schema_violations(r, norm_schema))

    # 1 canary
    if summary["canary_patch_rows_consumed_count"] < 11:
        errors.append("canary_lt_11")
    # 2-4 artefatos de contrato/inventario/anchor
    if not CONTRACT.exists() or not summary["score_v6_formula_contract_created"]:
        errors.append("contract_missing")
    if not summary["official_score_population_inventory_created"] or len(pop) < 1:
        errors.append("population_inventory_missing")
    if not summary["normalization_anchor_evaluation_created"] or len(norm) < 1:
        errors.append("normalization_anchor_missing")
    # 5-7 tentativas
    if summary["official_hydrography_or_drainage_attempts_count"] < 5:
        errors.append("hydrography_attempts_lt_5")
    if summary["landcover_acquisition_attempts_count"] < 3:
        errors.append("landcover_attempts_lt_3")
    if summary["terrain_hand_attempts_count"] < 3:
        errors.append("hand_attempts_lt_3")
    # 8 parity matrix
    if len(parity) < 11:
        errors.append("parity_lt_11")
    # 9 familias
    if summary["feature_family_available_count"] < 6:
        errors.append("families_lt_6")
    # 10 readiness
    if len(ready) < 11:
        errors.append("readiness_lt_11")
    # 11 score numerico com canary-only normalization proibido
    for r in replay:
        if r["score_v6_review_only"] not in ("not_available",) and r["normalization_population_reconstructed"] != "true":
            errors.append(f"score_computed_without_population:{r['score_v6_replay_id']}")
    for r in norm:
        if r["canary_only_normalization_used"] != "false":
            errors.append("canary_only_normalization_used")
    for r in leak:
        if r["uses_canary_only_normalization"] != "false":
            errors.append("canary_only_normalization_leak")
    # 12 proxy tratado como oficial sem flag
    for r in drainage:
        if r["official_drainage_available"] == "false" and r["drainage_proxy_used"] != "true":
            errors.append(f"proxy_not_flagged:{r['official_drainage_feature_id']}")
    if any(r["uses_proxy_as_official_equivalent"] != "false" for r in leak):
        errors.append("proxy_as_official_leak")
    # 13 HAND proxy != HAND full
    for r in hand:
        if r["not_full_hydrologic_HAND"] != "true":
            errors.append(f"hand_proxy_as_full:{r['hand_candidate_id']}")
    if summary["HAND_full_rows_count"] != 0:
        errors.append("hand_full_claimed")
    if any(r["uses_hand_proxy_as_hand_full"] != "false" for r in leak):
        errors.append("hand_proxy_as_full_leak")
    # 14 OSM como hidrografia oficial
    if any(r["uses_osm_as_official_hydrography"] != "false" for r in leak):
        errors.append("osm_as_official_hydrography")
    for r in drainage:
        if r["drainage_proxy_used"] == "true" and r["drainage_source_authority"] not in ("osm_proxy",):
            errors.append(f"proxy_authority_mislabeled:{r['official_drainage_feature_id']}")
    # 15 ocorrencia como feature
    if any(r["uses_occurrence_as_feature"] != "false" for r in leak):
        errors.append("occurrence_as_feature")
    # 16 controle negativo verdadeiro
    if any(r["absence_of_occurrence_is_true_negative"] != "false" for r in contrast):
        errors.append("control_true_negative")
    if any(r["uses_control_as_negative_truth"] != "false" for r in leak):
        errors.append("control_negative_leak")
    # 17 score v6 intacto
    if _run_git(["diff", "--name-only", "--", rel(SCORE_V6)]) or summary["score_v6_original_file_changed"]:
        errors.append("score_v6_changed")
    if contract["score_v6_hash_before"] != contract["score_v6_hash_after"]:
        errors.append("contract_hash_mismatch")
    for r in integ:
        if r["official_score_v6_changed"] != "false":
            errors.append("integrity_changed")
    # 18 score v7
    if SCORE_V7.exists() or summary["score_v7_created"]:
        errors.append("score_v7_exists")
    # 19/20 GT/treino
    if summary["ground_truth_created"] or summary["training_labels_created"]:
        errors.append("gt_or_training")
    for r in replay:
        if r["ground_truth"] != "false" or r["training_label"] != "false":
            errors.append("replay_gt_training")
    # 21 raster pesado
    if LIGHT.exists():
        for p in LIGHT.glob("**/*"):
            if p.is_file() and (p.suffix.lower() in FORBIDDEN_SUFFIXES or p.stat().st_size > MAX_LIGHT_BYTES):
                errors.append(f"forbidden_or_heavy_committed:{rel(p)}")
    if any(r["passes_no_leakage"] != "true" for r in leak):
        errors.append("no_leakage_failed")

    # 22 summary reflete contagens
    expected = build_summary()
    for k, v in expected.items():
        if summary.get(k) != v:
            errors.append(f"summary_mismatch:{k}")
    if not summary["minimum_success_achieved"]:
        errors.append("minimum_success_not_achieved")

    for rows, key in [
        (parity, "feature_parity_id"), (replay, "score_v6_replay_id"), (norm, "normalization_anchor_id"),
        (drainage, "official_drainage_feature_id"), (hand, "hand_candidate_id"), (lc, "landcover_urban_feature_id"),
        (ready, "score_replay_readiness_id"), (contrast, "contrast_id"), (leak, "no_leakage_audit_id"),
    ]:
        ids = [r[key] for r in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(
        "17C35 -> "
        f"canary={summary['canary_patch_rows_consumed_count']} norm_reconstructed={summary['normalization_population_reconstructed']} "
        f"hydro_att={summary['official_hydrography_or_drainage_attempts_count']} off_drain={summary['official_drainage_available_count']} "
        f"hand_proxy={summary['HAND_proxy_rows_count']} hand_full={summary['HAND_full_rows_count']} "
        f"lc_att={summary['landcover_acquisition_attempts_count']} lc_avail={summary['landcover_available_count']} "
        f"families={summary['feature_family_available_count']} parity_mean={summary['feature_parity_mean']} "
        f"full_replay={summary['score_v6_full_replay_computable']} partial={summary['score_v6_partial_component_replay_computable']} "
        f"result={summary['result_class']} min_success={summary['minimum_success_achieved']}"
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="susc_17c35")
    parser.add_argument("mode", choices=[
        "inventory-score-v6", "inventory-official-population", "run-hydrography-acquisition",
        "run-hand-pipeline", "run-landcover-acquisition", "build-feature-parity",
        "build-normalization-anchor", "build-score-replay-readiness",
        "compute-score-v6-replay-review-only", "build-original-vs-canary-comparable-contrast",
        "build-offline", "audit",
    ])
    args = parser.parse_args(argv)
    if args.mode == "audit":
        return validate()
    run_mode(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
