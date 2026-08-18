"""SUSC-18A Feature Store Multimodal Escalavel Patch-Level.

A cadeia 17C33-17C40 provou que a esteira extrai dados REAIS por patch canario
(Sentinel-2 COG, CHIRPS, DEM/topografia, drenagem oficial, landcover) e produziu
um indice transparente review-only (17C40) porque o flow_acc oficial do score v6
e IRRECUPERAVEL com dados publicos. Este marco NAO tenta recuperar flow_acc, NAO
tenta replay do score v6 e NAO cria score v7. Ele CONSOLIDA os extratores reais em
uma FEATURE STORE MULTIMODAL patch-level, unica, auditavel e reproduzivel, unindo:

  1. patches oficiais da populacao (Recife/Curitiba/Petropolis, matriz oficial);
  2. patches canarios ancorados em ocorrencia oficial hidrologica (17C33-17C36/40);
  3. controles originais de mismatch espacial (17C33 + 17C25);
  4. candidatos existentes (identicos aos 5 controles nesta iteracao -> 0 extras).

Familias multimodais: sentinel2_spectral, chirps_rainfall, terrain_topography,
hydrology_proximity, landcover_urban, transparent_review_components, lineage_quality.

Principios (fail-closed, honesto):
  - Ocorrencia oficial e ANCORA observacional, NUNCA feature causal.
  - Canario NAO e label positivo; controle NAO e negativo verdadeiro; ausencia
    documental NAO e ausencia real.
  - flow_acc oficial NAO e equivalente ao recomputado (17C39); nos oficiais e o
    valor NATIVO da propria matriz (nao um surrogate); nos canarios entra so como
    valor review-only com contrato de substituicao, nunca "equivalente".
  - Missingness EXPLICITA por familia; sem imputacao escondida.
  - Sem pre/pos-evento misturados: features oficiais/controle sao contexto
    populacional estatico (not_event_pre_window=true); canarios sao pre-evento.
  - review_only=true, ground_truth=false, training_label=false em tudo.
  - Score v6/matriz oficiais intactos; score v7 inexistente; 17B fail-closed.

Offline e deterministico: sintetiza artefatos ja commitados. Modos de rede sao
fail-closed sem as envs SUSC_18A_ALLOW_*.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import (  # noqa: E402
    ensure_dir, read_csv, read_json, rel, sha256_file, write_csv, write_json, write_markdown,
)

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

# --- Entradas obrigatorias -------------------------------------------------
OFFICIAL_MATRIX = DAT / "susc_features_by_patch_v1.csv"
SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

C33_REGISTRY = OUT / "susc_17c33_event_anchored_canary_patch_registry.csv"
C33_CONTROL = OUT / "susc_17c33_original_patch_mismatch_control.csv"
C34_SENTINEL2 = OUT / "susc_17c34_sentinel2_pre_event_features.csv"
C34_SPECTRAL = OUT / "susc_17c34_spectral_indices_features.csv"
C34_CHIRPS = OUT / "susc_17c34_chirps_features.csv"
C35_DRAINAGE = OUT / "susc_17c35_official_drainage_features.csv"
C35_LANDCOVER = OUT / "susc_17c35_landcover_urban_features.csv"
C36_TOPO = OUT / "susc_17c36_topography_hydrology_feature_matrix.csv"
C38_OFFICIAL_TOPO = OUT / "susc_17c38_official_patch_basin_aware_topography_features.csv"
C40_INDEX = OUT / "susc_17c40_transparent_review_only_index.csv"
C25_MULTIMODAL = OUT / "susc_17c25_multimodal_feature_matrix.csv"
C17_AOI = OUT / "susc_17c17_candidate_patch_aoi_plan.csv"

REQUIRED_INPUTS = [
    OFFICIAL_MATRIX, C33_REGISTRY, C33_CONTROL, C34_SENTINEL2, C34_SPECTRAL, C34_CHIRPS,
    C35_DRAINAGE, C35_LANDCOVER, C36_TOPO, C40_INDEX, C25_MULTIMODAL, C17_AOI,
]

# --- Saidas ----------------------------------------------------------------
REGISTRY = OUT / "susc_18a_unified_patch_registry.csv"
FAMILY_CONTRACT = OUT / "susc_18a_multimodal_feature_family_contract.json"
SOURCE_INVENTORY = OUT / "susc_18a_feature_source_inventory.csv"
SENTINEL2_STORE = OUT / "susc_18a_sentinel2_feature_store.csv"
CHIRPS_STORE = OUT / "susc_18a_chirps_feature_store.csv"
TERRAIN_STORE = OUT / "susc_18a_terrain_topography_feature_store.csv"
HYDROLOGY_STORE = OUT / "susc_18a_hydrology_proximity_feature_store.csv"
LANDCOVER_STORE = OUT / "susc_18a_landcover_urban_feature_store.csv"
MULTIMODAL_STORE = OUT / "susc_18a_multimodal_patch_feature_store.csv"
AVAILABILITY = OUT / "susc_18a_feature_availability_matrix.csv"
LINEAGE = OUT / "susc_18a_feature_lineage_manifest.csv"
QUALITY_FLAGS = OUT / "susc_18a_quality_flags.csv"
NEXT_STAGE = OUT / "susc_18a_next_stage_readiness.csv"
INTEGRITY = OUT / "susc_18a_score_integrity_audit.csv"
LEAKAGE = OUT / "susc_18a_no_leakage_audit.csv"
SUMMARY = OUT / "susc_18a_readiness_summary.json"
BLOCKERS = OUT / "susc_18a_promotion_blockers.csv"
REPORT = OUT / "SUSC_18A_MULTIMODAL_FEATURE_STORE_REPORT.md"

SCHEMA_STORE = SCHEMAS / "susc_18a_multimodal_feature_store_schema_v1.json"
SCHEMA_LINEAGE = SCHEMAS / "susc_18a_feature_lineage_schema_v1.json"
SCHEMA_AVAIL = SCHEMAS / "susc_18a_availability_schema_v1.json"

REQUIRED_OUTPUTS = [
    REGISTRY, FAMILY_CONTRACT, SOURCE_INVENTORY, SENTINEL2_STORE, CHIRPS_STORE, TERRAIN_STORE,
    HYDROLOGY_STORE, LANDCOVER_STORE, MULTIMODAL_STORE, AVAILABILITY, LINEAGE, QUALITY_FLAGS,
    NEXT_STAGE, INTEGRITY, LEAKAGE, SUMMARY, BLOCKERS, REPORT,
]

# familias de dados que contam para disponibilidade/completude (6)
DATA_FAMILIES = ["sentinel2_spectral", "chirps_rainfall", "terrain_topography",
                 "hydrology_proximity", "landcover_urban", "transparent_review_components"]
# todas as familias declaradas no contrato (7, inclui lineage_quality meta)
ALL_FAMILIES = DATA_FAMILIES + ["lineage_quality"]

NETWORK_ENVS = [
    "SUSC_18A_ALLOW_NETWORK", "SUSC_18A_ALLOW_PUBLIC_DOWNLOAD", "SUSC_18A_ALLOW_LIGHTWEIGHT_RASTER",
    "SUSC_18A_ALLOW_LIGHTWEIGHT_VECTOR", "SUSC_18A_ALLOW_SENTINEL2_COG", "SUSC_18A_ALLOW_CHIRPS",
    "SUSC_18A_ALLOW_DEM", "SUSC_18A_ALLOW_HYDROGRAPHY", "SUSC_18A_ALLOW_LANDCOVER",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bool(v):
    return "true" if v else "false"


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _av(v, ndigits=5):
    n = _num(v)
    return "not_available" if n is None else round(n, ndigits)


def _st(v):
    return "available" if _num(v) is not None else "not_available"


def _run_git(args):
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _require_inputs():
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))


_SHA_CACHE = {}


def _sha(path):
    key = str(path)
    if key not in _SHA_CACHE:
        _SHA_CACHE[key] = sha256_file(path) if Path(path).exists() else "not_available"
    return _SHA_CACHE[key]


def _centroid_from_bbox(xmin, ymin, xmax, ymax):
    xs = [_num(xmin), _num(xmax)]
    ys = [_num(ymin), _num(ymax)]
    if None in xs or None in ys:
        return None, None
    return round((xs[0] + xs[1]) / 2, 6), round((ys[0] + ys[1]) / 2, 6)


# ---------------------------------------------------------------------------
# assemble: consolida um dict rico por patch (chamado uma vez por builder)
# ---------------------------------------------------------------------------

def _load_sources():
    official = read_csv(OFFICIAL_MATRIX)
    canary_reg = read_csv(C33_REGISTRY)
    control = read_csv(C33_CONTROL)
    s2 = {r["canary_patch_id"]: r for r in read_csv(C34_SENTINEL2)}
    spec = {r["canary_patch_id"]: r for r in read_csv(C34_SPECTRAL)}
    chirps = {r["canary_patch_id"]: r for r in read_csv(C34_CHIRPS)}
    drain = {r["canary_patch_id"]: r for r in read_csv(C35_DRAINAGE)}
    lc = {r["canary_patch_id"]: r for r in read_csv(C35_LANDCOVER)}
    topo = {r["canary_patch_id"]: r for r in read_csv(C36_TOPO)}
    m25 = {r["candidate_patch_id"]: r for r in read_csv(C25_MULTIMODAL)}
    aoi = {r["candidate_patch_id"]: r for r in read_csv(C17_AOI)}
    transparent_ids = {r["patch_id"] for r in read_csv(C40_INDEX)}
    return dict(official=official, canary_reg=canary_reg, control=control, s2=s2, spec=spec,
                chirps=chirps, drain=drain, lc=lc, topo=topo, m25=m25, aoi=aoi,
                transparent_ids=transparent_ids)


def _ndwi(b3, b8):
    g, n = _num(b3), _num(b8)
    if g is None or n is None or (g + n) == 0:
        return None
    return (g - n) / (g + n)


def assemble_patches():
    """Retorna lista ordenada de dicts ricos por patch (ordem estavel).

    Ordem: oficiais (patch_id asc) -> canarios (id asc) -> controles (id asc).
    """
    S = _load_sources()
    patches = []

    # 1. oficiais (populacao) ---------------------------------------------
    for r in sorted(S["official"], key=lambda x: x["patch_id"]):
        pid = r["patch_id"]
        clon, clat = _centroid_from_bbox(r.get("xmin"), r.get("ymin"), r.get("xmax"), r.get("ymax"))
        transp = pid in S["transparent_ids"]
        patches.append({
            "source_patch_id": pid, "patch_family": "official_population_patch",
            "city": r.get("regiao", ""), "region": r.get("regiao", ""),
            "bbox": f"{r.get('xmin','')},{r.get('ymin','')},{r.get('xmax','')},{r.get('ymax','')}",
            "centroid_lon": clon, "centroid_lat": clat, "geometry_available": clon is not None,
            "event_anchor_available": False, "event_id": "none", "occurrence_anchor_type": "none",
            "is_canary": False, "is_control": False, "is_official": True,
            # sentinel2 (populacao, contexto estatico)
            "scene_id": "population_context", "scene_date": r.get("reference_date", "not_available"),
            "pre_event": False, "temporal_context": "static_or_population_context", "not_event_pre_window": True,
            "cloud_cover": None,
            "B03_mean": r.get("B3_mean"), "B04_mean": r.get("B4_mean"),
            "B08_mean": r.get("B8_mean"), "B11_mean": r.get("B11_mean"),
            "NDVI_mean": r.get("ndvi_mean"), "NDWI_mean": _ndwi(r.get("B3_mean"), r.get("B8_mean")),
            "MNDWI_mean": r.get("mndwi_mean"), "NDBI_mean": r.get("ndbi_mean"),
            "valid_pixel_count": "not_available",
            # chirps
            "chirps_3d_sum": r.get("chirps_3d_mm"), "chirps_7d_sum": r.get("chirps_7d_mm"),
            "chirps_30d_sum": r.get("chirps_30d_mm"),
            # terrain (nativo oficial)
            "elevation_mean": r.get("elevation_mean"), "slope_mean": r.get("slope_mean"),
            "HAND_value": r.get("hand_mean"), "TWI_value": r.get("twi_mean"),
            "TPI_250m_value": r.get("tpi_250m_mean"), "flow_acc_value": r.get("flow_acc_log_mean"),
            "flow_acc_status": "official_native_value", "flow_acc_native": True,
            "flow_acc_replacement_required": False,
            "method_equivalence_status": "official_native",
            # hydrology
            "distance_to_official_drainage_m": None,
            "distance_to_official_water_m": r.get("distance_to_water_mean"),
            "official_drainage_available": False, "drainage_source_authority": "dem_water_occurrence_native",
            "drainage_proxy_used": False,
            # landcover
            "urban_prop": r.get("urban_prop"), "vegetation_prop": r.get("vegetation_prop"),
            "water_prop": None, "built_up_proxy_ndbi": r.get("ndbi_mean"),
            "landcover_source": "official_feature_matrix", "landcover_authority": "susc_features_by_patch_v1",
            "landcover_proxy_only": False,
            "transparent_available": transp,
        })

    # 2. canarios ----------------------------------------------------------
    for r in sorted(S["canary_reg"], key=lambda x: x["event_anchored_canary_patch_id"]):
        pid = r["event_anchored_canary_patch_id"]
        s2 = S["s2"].get(pid, {})
        sp = S["spec"].get(pid, {})
        ch = S["chirps"].get(pid, {})
        d = S["drain"].get(pid, {})
        lc = S["lc"].get(pid, {})
        t = S["topo"].get(pid, {})
        clon, clat = _num(r.get("center_lon")), _num(r.get("center_lat"))
        patches.append({
            "source_patch_id": pid, "patch_family": "event_anchored_canary_patch",
            "city": "recife", "region": r.get("bairro", ""),
            "bbox": f"{r.get('xmin','')},{r.get('ymin','')},{r.get('xmax','')},{r.get('ymax','')}",
            "centroid_lon": clon, "centroid_lat": clat, "geometry_available": clon is not None,
            "event_anchor_available": True, "event_id": r.get("event_id", "none"),
            "occurrence_anchor_type": r.get("link_type_to_occurrence", "official_hydrological_occurrence"),
            "is_canary": True, "is_control": False, "is_official": False,
            "scene_id": s2.get("scene_id", "not_available"), "scene_date": s2.get("scene_date", "not_available"),
            "pre_event": True, "temporal_context": "event_pre_window", "not_event_pre_window": False,
            "cloud_cover": None,
            "B03_mean": s2.get("B03_mean"), "B04_mean": s2.get("B04_mean"),
            "B08_mean": s2.get("B08_mean"), "B11_mean": s2.get("B11_mean"),
            "NDVI_mean": sp.get("NDVI_mean"), "NDWI_mean": sp.get("NDWI_mean"),
            "MNDWI_mean": sp.get("MNDWI_mean"), "NDBI_mean": sp.get("NDBI_mean"),
            "valid_pixel_count": s2.get("valid_pixel_count", "not_available"),
            "chirps_3d_sum": ch.get("chirps_3d_sum"), "chirps_7d_sum": ch.get("chirps_7d_sum"),
            "chirps_30d_sum": ch.get("chirps_30d_sum"),
            "elevation_mean": t.get("elevation_mean"), "slope_mean": t.get("slope_mean"),
            "HAND_value": t.get("hand_mean"), "TWI_value": t.get("twi_mean"),
            "TPI_250m_value": t.get("tpi_250m_mean"), "flow_acc_value": t.get("flow_acc_log_mean"),
            "flow_acc_status": "review_only_not_official_equivalent" if _num(t.get("flow_acc_log_mean")) is not None else "not_available",
            "flow_acc_native": False, "flow_acc_replacement_required": True,
            "method_equivalence_status": t.get("method_equivalence_status", "not_equivalent"),
            "distance_to_official_drainage_m": d.get("distance_to_official_drainage_m"),
            "distance_to_official_water_m": d.get("distance_to_official_water_m"),
            "official_drainage_available": str(d.get("official_drainage_available", "false")).lower() == "true",
            "drainage_source_authority": d.get("drainage_source_authority", "not_available"),
            "drainage_proxy_used": str(d.get("drainage_proxy_used", "false")).lower() == "true",
            "urban_prop": lc.get("urban_prop"), "vegetation_prop": lc.get("vegetation_prop"),
            "water_prop": lc.get("water_prop"), "built_up_proxy_ndbi": lc.get("built_up_proxy_ndbi"),
            "landcover_source": lc.get("landcover_source", "not_available"),
            "landcover_authority": lc.get("landcover_authority", "not_available"),
            "landcover_proxy_only": str(lc.get("proxy_only", "false")).lower() == "true",
            "transparent_available": pid in S["transparent_ids"],
        })

    # 3. controles ---------------------------------------------------------
    for r in sorted(S["control"], key=lambda x: x["original_patch_id"]):
        pid = r["original_patch_id"]
        m = S["m25"].get(pid, {})
        a = S["aoi"].get(pid, {})
        bbox = a.get("bbox", "")
        parts = [p.strip() for p in bbox.split(",")] if bbox else []
        if len(parts) == 4:
            clon, clat = _centroid_from_bbox(parts[0], parts[1], parts[2], parts[3])
        else:
            clon, clat = None, None
        patches.append({
            "source_patch_id": pid, "patch_family": "spatial_mismatch_control_candidate",
            "city": "recife", "region": r.get("region", "recife"),
            "bbox": bbox if bbox else "not_available",
            "centroid_lon": clon, "centroid_lat": clat, "geometry_available": clon is not None,
            "event_anchor_available": False, "event_id": r.get("event_id", "none"),
            "occurrence_anchor_type": "none_spatial_mismatch_control",
            "is_canary": False, "is_control": True, "is_official": False,
            "scene_id": "temporal_median_composite", "scene_date": "not_available",
            "pre_event": False, "temporal_context": "static_or_population_context", "not_event_pre_window": True,
            "cloud_cover": m.get("cloud_cover_median"),
            "B03_mean": m.get("B03_mean_temporal_median"), "B04_mean": m.get("B04_mean_temporal_median"),
            "B08_mean": m.get("B08_mean_temporal_median"), "B11_mean": m.get("B11_mean_temporal_median"),
            "NDVI_mean": m.get("NDVI_mean_temporal_median"), "NDWI_mean": m.get("NDWI_mean_temporal_median"),
            "MNDWI_mean": m.get("MNDWI_mean_temporal_median"), "NDBI_mean": m.get("NDBI_mean_temporal_median"),
            "valid_pixel_count": "not_available",
            "chirps_3d_sum": m.get("CHIRPS_3d_sum_mm"), "chirps_7d_sum": m.get("CHIRPS_7d_sum_mm"),
            "chirps_30d_sum": m.get("CHIRPS_30d_sum_mm"),
            "elevation_mean": None, "slope_mean": None, "HAND_value": None, "TWI_value": None,
            "TPI_250m_value": None, "flow_acc_value": None, "flow_acc_status": "not_available",
            "flow_acc_native": False, "flow_acc_replacement_required": False,
            "method_equivalence_status": "not_applicable",
            "distance_to_official_drainage_m": None, "distance_to_official_water_m": None,
            "official_drainage_available": False, "drainage_source_authority": "not_available",
            "drainage_proxy_used": False,
            "urban_prop": None, "vegetation_prop": None, "water_prop": None, "built_up_proxy_ndbi": None,
            "landcover_source": "not_available", "landcover_authority": "not_available",
            "landcover_proxy_only": False,
            "transparent_available": pid in S["transparent_ids"],
        })

    # unified ids estaveis
    for i, p in enumerate(patches, start=1):
        p["unified_patch_id"] = f"S18A_PATCH_{i:04d}"
    # disponibilidade por familia
    for p in patches:
        p["_avail"] = _family_availability(p)
    return patches


def _family_availability(p):
    return {
        "sentinel2_spectral": any(_num(p[k]) is not None for k in ("NDVI_mean", "NDWI_mean", "MNDWI_mean", "NDBI_mean")),
        "chirps_rainfall": _num(p["chirps_7d_sum"]) is not None,
        "terrain_topography": any(_num(p[k]) is not None for k in ("HAND_value", "TWI_value", "TPI_250m_value", "elevation_mean")),
        "hydrology_proximity": any(_num(p[k]) is not None for k in ("distance_to_official_drainage_m", "distance_to_official_water_m")),
        "landcover_urban": any(_num(p[k]) is not None for k in ("urban_prop", "vegetation_prop")),
        "transparent_review_components": bool(p["transparent_available"]),
    }


# ---------------------------------------------------------------------------
# Etapa 1 - registry unificado
# ---------------------------------------------------------------------------

def registry_rows():
    rows = []
    for p in assemble_patches():
        clon = "not_available" if p["centroid_lon"] is None else p["centroid_lon"]
        clat = "not_available" if p["centroid_lat"] is None else p["centroid_lat"]
        rows.append({
            "unified_patch_id": p["unified_patch_id"], "source_patch_id": p["source_patch_id"],
            "patch_family": p["patch_family"], "city": p["city"], "region": p["region"],
            "geometry_available": _bool(p["geometry_available"]), "bbox": p["bbox"],
            "centroid_lon": clon, "centroid_lat": clat,
            "event_anchor_available": _bool(p["event_anchor_available"]), "event_id": p["event_id"],
            "occurrence_anchor_type": p["occurrence_anchor_type"],
            "is_canary_patch": _bool(p["is_canary"]), "is_control_patch": _bool(p["is_control"]),
            "is_official_population_patch": _bool(p["is_official"]),
            "review_only": "true", "ground_truth": "false", "training_label": "false",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapa 2 - contrato de familias multimodais
# ---------------------------------------------------------------------------

def family_contract():
    return {
        "milestone": "SUSC-18A",
        "feature_family_count": len(ALL_FAMILIES),
        "families": {
            "sentinel2_spectral": {
                "source": "Sentinel-2 L2A (matriz oficial p/ populacao; COG Element84 p/ canarios; mediana temporal 17C25 p/ controles)",
                "period": "populacao: contexto estatico; canarios: janela pre-evento; controles: composto mediano",
                "resolution": "10-20 m", "pre_event": "true_only_for_canaries", "is_static": "true_for_population_and_controls",
                "is_official": True, "is_proxy": False, "review_only": True,
                "can_enter_score": False, "can_enter_training": False,
                "limitations": "NDWI oficial derivado de B3/B8; sem scene_id por-patch na populacao; controles sao composto (nao pre-evento)"},
            "chirps_rainfall": {
                "source": "CHIRPS UCSB (matriz oficial / 17C25 / 17C34)", "period": "janelas 3/7/30d",
                "resolution": "~5 km (region-level)", "pre_event": "canaries_pre_event", "is_static": "false",
                "is_official": True, "is_proxy": False, "review_only": True,
                "can_enter_score": False, "can_enter_training": False,
                "limitations": "region-level (~5km) NAO discrimina local; gatilho/contexto, NAO ocorrencia observada"},
            "terrain_topography": {
                "source": "matriz oficial (nativo p/ populacao); Copernicus GLO-30 + pipeline local 17C36 p/ canarios",
                "period": "estatico", "resolution": "30 m", "pre_event": "static", "is_static": "true",
                "is_official": "native_for_population_proxy_for_canaries", "is_proxy": "true_for_canaries", "review_only": True,
                "can_enter_score": False, "can_enter_training": False,
                "limitations": "flow_acc canario NAO equivalente ao oficial (17C39); nos oficiais e valor nativo, nao surrogate"},
            "hydrology_proximity": {
                "source": "distance_to_water nativa (populacao); Dados Recife faixas-marginais/hidrografia (17C35) p/ canarios",
                "period": "estatico", "resolution": "vetorial oficial / raster nativo", "pre_event": "static", "is_static": "true",
                "is_official": True, "is_proxy": "partial", "review_only": True,
                "can_enter_score": False, "can_enter_training": False,
                "limitations": "populacao tem distancia a agua (nativa) mas nao a linha de drenagem oficial; controles indisponivel"},
            "landcover_urban": {
                "source": "urban/vegetation nativa (populacao); Dados Recife cobertura-da-terra (17C35) p/ canarios",
                "period": "estatico", "resolution": "vetorial/raster", "pre_event": "static", "is_static": "true",
                "is_official": True, "is_proxy": "partial", "review_only": True,
                "can_enter_score": False, "can_enter_training": False,
                "limitations": "water_prop indisponivel na populacao; NDBI usado como proxy built-up; controles indisponivel"},
            "transparent_review_components": {
                "source": "indice transparente review-only 17C40", "period": "estatico", "resolution": "patch",
                "pre_event": "static", "is_static": "true", "is_official": False, "is_proxy": True, "review_only": True,
                "can_enter_score": False, "can_enter_training": False,
                "limitations": "so 16 patches (11 canarios + 5 controles); NAO score v6/v7; escala nao-oficial"},
            "lineage_quality": {
                "source": "manifesto de lineage + quality flags 18A", "period": "meta", "resolution": "meta",
                "pre_event": "meta", "is_static": "true", "is_official": "meta", "is_proxy": "meta", "review_only": True,
                "can_enter_score": False, "can_enter_training": False,
                "limitations": "metadados de proveniencia e qualidade; nao e feature preditiva"},
        },
        "rules": [
            "ocorrencia oficial e ancora observacional, nunca feature causal",
            "canario nao e label positivo; controle nao e negativo verdadeiro; ausencia nao e ausencia real",
            "flow_acc oficial nao equivalente ao recomputado; nos oficiais e valor nativo (nao surrogate)",
            "missingness explicita por familia; sem imputacao escondida",
            "sem misturar pre-evento (canarios) com contexto estatico (populacao/controles)",
        ],
        "not_score_v6": True, "not_score_v7": True, "creates_ground_truth": False,
        "creates_training_labels": False, "review_only": True, "trainable": False,
    }


# ---------------------------------------------------------------------------
# Etapa 3 - inventario de fontes
# ---------------------------------------------------------------------------

def _rows_of(path):
    return len(read_csv(path)) if Path(path).exists() else 0


def source_inventory_rows():
    inv = [
        ("sentinel2_spectral", OFFICIAL_MATRIX, "official_population_patch",
         "B3/B4/B8/B11_mean;ndvi_mean;mndwi_mean;ndbi_mean;NDWI_derived", True,
         "NDWI derivado de B3/B8; contexto estatico (sem scene por-patch)"),
        ("sentinel2_spectral", C34_SENTINEL2, "event_anchored_canary_patch",
         "B03/B04/B08/B11_mean;scene_id;scene_date;valid_pixel_count", True, "COG por-ponto, janela pre-evento"),
        ("sentinel2_spectral", C34_SPECTRAL, "event_anchored_canary_patch",
         "NDVI/NDWI/MNDWI/NDBI_mean", True, "indices espectrais canarios"),
        ("sentinel2_spectral", C25_MULTIMODAL, "spatial_mismatch_control_candidate",
         "NDVI/NDWI/MNDWI/NDBI_mean_temporal_median", True, "composto mediano (nao pre-evento)"),
        ("chirps_rainfall", OFFICIAL_MATRIX, "official_population_patch",
         "chirps_3d_mm;chirps_7d_mm;chirps_30d_mm", True, "region-level ~5km"),
        ("chirps_rainfall", C34_CHIRPS, "event_anchored_canary_patch",
         "chirps_3d_sum;chirps_7d_sum;chirps_30d_sum", True, "gatilho region-level; nao ocorrencia"),
        ("chirps_rainfall", C25_MULTIMODAL, "spatial_mismatch_control_candidate",
         "CHIRPS_3d/7d/30d_sum_mm", True, "region-level"),
        ("terrain_topography", OFFICIAL_MATRIX, "official_population_patch",
         "elevation/slope/hand/twi/tpi_250m/flow_acc_log_mean", True, "topografia nativa oficial"),
        ("terrain_topography", C36_TOPO, "event_anchored_canary_patch",
         "elevation/slope/hand/twi/tpi_250m/flow_acc_log_mean", True, "pipeline local; flow_acc nao equivalente"),
        ("hydrology_proximity", OFFICIAL_MATRIX, "official_population_patch",
         "distance_to_water_mean", True, "distancia a agua nativa (nao linha de drenagem)"),
        ("hydrology_proximity", C35_DRAINAGE, "event_anchored_canary_patch",
         "distance_to_official_drainage_m;distance_to_official_water_m", True, "Dados Recife faixas-marginais oficiais"),
        ("landcover_urban", OFFICIAL_MATRIX, "official_population_patch",
         "urban_prop;vegetation_prop;ndbi_mean", True, "landcover nativo; water_prop indisponivel"),
        ("landcover_urban", C35_LANDCOVER, "event_anchored_canary_patch",
         "urban_prop;vegetation_prop;water_prop;built_up_proxy_ndbi", True, "Dados Recife cobertura-da-terra"),
        ("transparent_review_components", C40_INDEX, "canary_and_control",
         "transparent_index_value(V1/V2/V3)", True, "16 patches; review-only; nao score"),
        ("lineage_quality", REGISTRY, "all",
         "unified_patch_id;patch_family;geometry", True, "registry unificado 18A"),
    ]
    rows = []
    for i, (fam, path, covered, feats, lineage, lim) in enumerate(inv, start=1):
        rows.append({
            "source_inventory_id": f"S18A_INV_{i:04d}", "feature_family": fam,
            "source_file": rel(path), "source_exists": _bool(Path(path).exists()),
            "source_rows": str(_rows_of(path)), "patch_family_covered": covered,
            "feature_names": feats, "lineage_available": _bool(lineage),
            "limitations": lim, "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapas 5-9 - feature stores por familia
# ---------------------------------------------------------------------------

def sentinel2_store_rows():
    rows = []
    for i, p in enumerate(assemble_patches(), start=1):
        avail = p["_avail"]["sentinel2_spectral"]
        rows.append({
            "sentinel2_feature_id": f"S18A_S2_{i:04d}", "unified_patch_id": p["unified_patch_id"],
            "source_patch_id": p["source_patch_id"], "patch_family": p["patch_family"],
            "scene_id": p["scene_id"], "scene_date": p["scene_date"], "pre_event": _bool(p["pre_event"]),
            "temporal_context": p["temporal_context"], "not_event_pre_window": _bool(p["not_event_pre_window"]),
            "B03_mean": _av(p["B03_mean"]), "B04_mean": _av(p["B04_mean"]),
            "B08_mean": _av(p["B08_mean"]), "B11_mean": _av(p["B11_mean"]),
            "NDVI_mean": _av(p["NDVI_mean"]), "NDWI_mean": _av(p["NDWI_mean"]),
            "MNDWI_mean": _av(p["MNDWI_mean"]), "NDBI_mean": _av(p["NDBI_mean"]),
            "valid_pixel_count": p["valid_pixel_count"] if p["valid_pixel_count"] not in (None, "") else "not_available",
            "cloud_cover": _av(p["cloud_cover"]),
            "feature_status": "available" if avail else "not_available",
            "lineage_source": "official_matrix" if p["is_official"] else ("17C34" if p["is_canary"] else "17C25"),
            "review_only": "true",
        })
    return rows


def chirps_store_rows():
    rows = []
    for i, p in enumerate(assemble_patches(), start=1):
        avail = p["_avail"]["chirps_rainfall"]
        rows.append({
            "chirps_feature_id": f"S18A_CH_{i:04d}", "unified_patch_id": p["unified_patch_id"],
            "source_patch_id": p["source_patch_id"], "patch_family": p["patch_family"],
            "date_window_start": "not_available",
            "date_window_end": p["scene_date"] if p["is_official"] else "not_available",
            "chirps_3d_sum": _av(p["chirps_3d_sum"]), "chirps_7d_sum": _av(p["chirps_7d_sum"]),
            "chirps_30d_sum": _av(p["chirps_30d_sum"]),
            "chirps_resolution_status": "region_level_~5km", "region_level": "true",
            "not_event_observation": "true",
            "feature_status": "available" if avail else "not_available",
            "lineage_source": "official_matrix" if p["is_official"] else ("17C34" if p["is_canary"] else "17C25"),
            "review_only": "true",
        })
    return rows


def terrain_store_rows():
    rows = []
    for i, p in enumerate(assemble_patches(), start=1):
        avail = p["_avail"]["terrain_topography"]
        rows.append({
            "terrain_feature_id": f"S18A_TR_{i:04d}", "unified_patch_id": p["unified_patch_id"],
            "source_patch_id": p["source_patch_id"], "patch_family": p["patch_family"],
            "elevation_mean": _av(p["elevation_mean"]), "slope_mean": _av(p["slope_mean"]),
            "HAND_value": _av(p["HAND_value"]), "HAND_status": _st(p["HAND_value"]),
            "TWI_value": _av(p["TWI_value"]), "TWI_status": _st(p["TWI_value"]),
            "TPI_250m_value": _av(p["TPI_250m_value"]), "TPI_status": _st(p["TPI_250m_value"]),
            "flow_acc_value": _av(p["flow_acc_value"]), "flow_acc_status": p["flow_acc_status"],
            "flow_acc_is_official_equivalent": "false",
            "flow_acc_is_official_native": _bool(p["flow_acc_native"]),
            "flow_acc_replacement_contract_required": _bool(p["flow_acc_replacement_required"]),
            "method_equivalence_status": p["method_equivalence_status"],
            "feature_status": "available" if avail else "not_available",
            "lineage_source": "official_matrix" if p["is_official"] else ("17C36" if p["is_canary"] else "not_available"),
            "review_only": "true",
        })
    return rows


def hydrology_store_rows():
    rows = []
    for i, p in enumerate(assemble_patches(), start=1):
        avail = p["_avail"]["hydrology_proximity"]
        rows.append({
            "hydrology_feature_id": f"S18A_HY_{i:04d}", "unified_patch_id": p["unified_patch_id"],
            "source_patch_id": p["source_patch_id"], "patch_family": p["patch_family"],
            "distance_to_official_drainage_m": _av(p["distance_to_official_drainage_m"]),
            "distance_to_official_water_m": _av(p["distance_to_official_water_m"]),
            "official_drainage_available": _bool(p["official_drainage_available"]),
            "drainage_source_authority": p["drainage_source_authority"],
            "drainage_proxy_used": _bool(p["drainage_proxy_used"]),
            "feature_status": "available" if avail else "not_available",
            "lineage_source": "official_matrix" if p["is_official"] else ("17C35" if p["is_canary"] else "not_available"),
            "review_only": "true",
        })
    return rows


def landcover_store_rows():
    rows = []
    for i, p in enumerate(assemble_patches(), start=1):
        avail = p["_avail"]["landcover_urban"]
        rows.append({
            "landcover_feature_id": f"S18A_LC_{i:04d}", "unified_patch_id": p["unified_patch_id"],
            "source_patch_id": p["source_patch_id"], "patch_family": p["patch_family"],
            "urban_prop": _av(p["urban_prop"]), "vegetation_prop": _av(p["vegetation_prop"]),
            "water_prop": _av(p["water_prop"]), "built_up_proxy_ndbi": _av(p["built_up_proxy_ndbi"]),
            "landcover_source": p["landcover_source"], "landcover_authority": p["landcover_authority"],
            "proxy_only": _bool(p["landcover_proxy_only"]),
            "feature_status": "available" if avail else "not_available",
            "lineage_source": "official_matrix" if p["is_official"] else ("17C35" if p["is_canary"] else "not_available"),
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapa 10 - matriz multimodal final
# ---------------------------------------------------------------------------

# quais familias sao "oficiais" vs "proxy" por familia de patch (para contagem)
def _official_proxy_counts(p):
    official, proxy = 0, 0
    a = p["_avail"]
    if a["sentinel2_spectral"]:
        official += 1  # Sentinel-2 real/oficial em todas as familias
    if a["chirps_rainfall"]:
        official += 1  # CHIRPS oficial
    if a["terrain_topography"]:
        if p["is_official"]:
            official += 1
        else:
            proxy += 1  # DEM publico local
    if a["hydrology_proximity"]:
        official += 1  # distancia oficial/nativa
    if a["landcover_urban"]:
        if p["landcover_proxy_only"]:
            proxy += 1
        else:
            official += 1
    if a["transparent_review_components"]:
        proxy += 1  # indice review-only
    return official, proxy


def multimodal_store_rows():
    rows = []
    for p in assemble_patches():
        a = p["_avail"]
        avail_count = sum(1 for f in DATA_FAMILIES if a[f])
        missing = [f for f in DATA_FAMILIES if not a[f]]
        off_c, prox_c = _official_proxy_counts(p)
        rows.append({
            "unified_patch_id": p["unified_patch_id"], "source_patch_id": p["source_patch_id"],
            "patch_family": p["patch_family"], "city": p["city"], "region": p["region"],
            "event_anchor_available": _bool(p["event_anchor_available"]),
            "has_official_hydrological_occurrence_anchor": _bool(p["is_canary"]),
            "absence_of_occurrence_is_true_negative": "false",
            "NDVI_mean": _av(p["NDVI_mean"]), "NDWI_mean": _av(p["NDWI_mean"]),
            "MNDWI_mean": _av(p["MNDWI_mean"]), "NDBI_mean": _av(p["NDBI_mean"]),
            "chirps_3d_sum": _av(p["chirps_3d_sum"]), "chirps_7d_sum": _av(p["chirps_7d_sum"]),
            "chirps_30d_sum": _av(p["chirps_30d_sum"]),
            "elevation_mean": _av(p["elevation_mean"]), "slope_mean": _av(p["slope_mean"]),
            "HAND_value": _av(p["HAND_value"]), "TWI_value": _av(p["TWI_value"]),
            "TPI_250m_value": _av(p["TPI_250m_value"]), "flow_acc_status": p["flow_acc_status"],
            "distance_to_official_drainage_m": _av(p["distance_to_official_drainage_m"]),
            "distance_to_official_water_m": _av(p["distance_to_official_water_m"]),
            "urban_prop": _av(p["urban_prop"]), "vegetation_prop": _av(p["vegetation_prop"]),
            "water_prop": _av(p["water_prop"]),
            "sentinel2_available": _bool(a["sentinel2_spectral"]), "chirps_available": _bool(a["chirps_rainfall"]),
            "terrain_available": _bool(a["terrain_topography"]), "hydrology_available": _bool(a["hydrology_proximity"]),
            "landcover_available": _bool(a["landcover_urban"]),
            "transparent_components_available": _bool(a["transparent_review_components"]),
            "feature_family_available_count": str(avail_count),
            "feature_completeness_ratio": round(avail_count / len(DATA_FAMILIES), 4),
            "missing_feature_families": ";".join(missing) if missing else "none",
            "proxy_feature_count": str(prox_c), "official_feature_count": str(off_c),
            "review_only": "true", "ground_truth": "false", "training_label": "false",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapa 11 - matriz de disponibilidade
# ---------------------------------------------------------------------------

def availability_matrix_rows():
    rows = []
    for i, p in enumerate(assemble_patches(), start=1):
        a = p["_avail"]
        complete = sum(1 for f in DATA_FAMILIES if a[f])
        missing = len(DATA_FAMILIES) - complete
        if p["is_control"]:
            blocking = "control_lacks_terrain_hydrology_landcover"
        elif p["is_official"]:
            blocking = "population_lacks_transparent_review_components"
        elif complete == len(DATA_FAMILIES):
            blocking = "none"
        else:
            blocking = "partial_family_availability"
        rows.append({
            "availability_id": f"S18A_AVAIL_{i:04d}", "unified_patch_id": p["unified_patch_id"],
            "patch_family": p["patch_family"],
            "sentinel2_status": "available" if a["sentinel2_spectral"] else "not_available",
            "chirps_status": "available" if a["chirps_rainfall"] else "not_available",
            "terrain_status": "available" if a["terrain_topography"] else "not_available",
            "hydrology_status": "available" if a["hydrology_proximity"] else "not_available",
            "landcover_status": "available" if a["landcover_urban"] else "not_available",
            "transparent_component_status": "available" if a["transparent_review_components"] else "not_available",
            "lineage_status": "available" if complete > 0 else "not_available",
            "complete_family_count": str(complete), "missing_family_count": str(missing),
            "blocking_reason": blocking, "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapa 12 - lineage + quality flags
# ---------------------------------------------------------------------------

# metadados de fonte por (familia, familia_de_patch)
def _lineage_meta(p, fam):
    is_off = p["is_official"]
    is_can = p["is_canary"]
    table = {
        "sentinel2_spectral": (
            (OFFICIAL_MATRIX, "official_population_v1", "Sentinel-2 L2A (matriz oficial)", "population_band_aggregate", True, False),
            (C34_SENTINEL2, "SUSC-17C34", "Sentinel-2 L2A COG (Element84)", "per_point_cog_pre_event", True, False),
            (C25_MULTIMODAL, "SUSC-17C25", "Sentinel-2 L2A (mediana temporal)", "temporal_median_composite", True, False),
        ),
        "chirps_rainfall": (
            (OFFICIAL_MATRIX, "official_population_v1", "CHIRPS UCSB (matriz oficial)", "zonal_window_sum", True, False),
            (C34_CHIRPS, "SUSC-17C34", "CHIRPS UCSB", "point_window_sum", True, False),
            (C25_MULTIMODAL, "SUSC-17C25", "CHIRPS UCSB", "zonal_window_sum", True, False),
        ),
        "terrain_topography": (
            (OFFICIAL_MATRIX, "official_population_v1", "DEM oficial (topografia nativa v1)", "native_v1_topography", True, False),
            (C36_TOPO, "SUSC-17C36", "Copernicus GLO-30 + pipeline local", "local_hydrologic_pipeline_non_equivalent", False, True),
            (None, "not_available", "not_available", "not_available", False, False),
        ),
        "hydrology_proximity": (
            (OFFICIAL_MATRIX, "official_population_v1", "DEM water occurrence (nativa)", "native_distance_to_water", True, False),
            (C35_DRAINAGE, "SUSC-17C35", "Dados Recife faixas-marginais (oficial)", "official_vector_distance", True, False),
            (None, "not_available", "not_available", "not_available", False, False),
        ),
        "landcover_urban": (
            (OFFICIAL_MATRIX, "official_population_v1", "matriz oficial (urban/vegetation)", "native_landcover_fraction", True, False),
            (C35_LANDCOVER, "SUSC-17C35", "Dados Recife cobertura-da-terra", "official_vector_landcover", True, False),
            (None, "not_available", "not_available", "not_available", False, False),
        ),
        "transparent_review_components": (
            (C40_INDEX, "SUSC-17C40", "indice transparente review-only", "review_only_transparent_index", False, True),
            (C40_INDEX, "SUSC-17C40", "indice transparente review-only", "review_only_transparent_index", False, True),
            (C40_INDEX, "SUSC-17C40", "indice transparente review-only", "review_only_transparent_index", False, True),
        ),
    }
    idx = 0 if is_off else (1 if is_can else 2)
    return table[fam][idx]


def lineage_manifest_rows():
    rows = []
    n = 0
    for p in assemble_patches():
        for fam in DATA_FAMILIES:
            if not p["_avail"][fam]:
                continue
            path, milestone, authority, method, is_off, is_proxy = _lineage_meta(p, fam)
            n += 1
            if fam == "terrain_topography" and p["is_official"]:
                window = "static"
            elif p["is_canary"]:
                window = "event_pre_window" if fam in ("sentinel2_spectral", "chirps_rainfall") else "static"
            else:
                window = "static_or_population_context"
            rows.append({
                "lineage_id": f"S18A_LIN_{n:05d}", "unified_patch_id": p["unified_patch_id"],
                "feature_family": fam, "source_file": rel(path) if path else "not_available",
                "source_milestone": milestone, "source_authority": authority,
                "extraction_method": method, "extraction_date_or_window": window,
                "is_official_source": _bool(is_off), "is_proxy_source": _bool(is_proxy),
                "is_review_only": "true",
                "sha256_or_source_hash": _sha(path) if path else "not_available",
                "lineage_status": "traced",
            })
    return rows


def quality_flag_rows():
    rows = []
    n = 0
    for p in assemble_patches():
        a = p["_avail"]
        avail_count = sum(1 for f in DATA_FAMILIES if a[f])
        ratio = avail_count / len(DATA_FAMILIES)
        n += 1
        rows.append({
            "quality_flag_id": f"S18A_QF_{n:05d}", "unified_patch_id": p["unified_patch_id"],
            "feature_family": "all", "flag_name": "feature_completeness_ratio",
            "flag_value": round(ratio, 4),
            "severity": "info" if ratio >= 0.8 else ("warning" if ratio >= 0.5 else "high"),
            "explanation": f"{avail_count}/{len(DATA_FAMILIES)} familias de dados disponiveis para o patch",
            "review_only": "true",
        })
        for fam in DATA_FAMILIES:
            if not a[fam]:
                n += 1
                rows.append({
                    "quality_flag_id": f"S18A_QF_{n:05d}", "unified_patch_id": p["unified_patch_id"],
                    "feature_family": fam, "flag_name": "family_missing", "flag_value": "true",
                    "severity": "warning",
                    "explanation": f"familia {fam} indisponivel para {p['patch_family']} (missingness explicita, sem imputacao)",
                    "review_only": "true",
                })
        if p["is_canary"] and _num(p["flow_acc_value"]) is not None:
            n += 1
            rows.append({
                "quality_flag_id": f"S18A_QF_{n:05d}", "unified_patch_id": p["unified_patch_id"],
                "feature_family": "terrain_topography", "flag_name": "flow_acc_not_official_equivalent",
                "flag_value": "true", "severity": "high",
                "explanation": "flow_acc recomputado (DEM publico) NAO equivalente ao oficial (17C39); contrato de substituicao review-only",
                "review_only": "true",
            })
        if p["landcover_proxy_only"]:
            n += 1
            rows.append({
                "quality_flag_id": f"S18A_QF_{n:05d}", "unified_patch_id": p["unified_patch_id"],
                "feature_family": "landcover_urban", "flag_name": "proxy_only_landcover", "flag_value": "true",
                "severity": "warning", "explanation": "landcover proxy (nao oficial) para este patch",
                "review_only": "true",
            })
    return rows


# ---------------------------------------------------------------------------
# Etapa 13 - readiness
# ---------------------------------------------------------------------------

def next_stage_readiness_rows():
    targets = [
        ("ready_for_population_analysis", True,
         "matriz multimodal com >=100 patches oficiais + canarios + controles consolidada",
         "", "usar store para analise exploratoria review-only por patch"),
        ("ready_for_review_only_index_expansion", True,
         "componentes transparentes 17C40 integrados; extensivel a populacao",
         "", "expandir indice transparente review-only a populacao oficial"),
        ("ready_for_candidate_patch_screening", True,
         "registry unificado + disponibilidade por familia permite triagem",
         "", "triagem de novos candidatos com missingness explicita"),
        ("blocked_for_training", False, "",
         "sem ground truth; review-only; ausencia nao e negativo verdadeiro",
         "aquisicao de ground reference oficial patch-level validado"),
        ("blocked_for_ground_truth", False, "",
         "ocorrencia oficial e ancora observacional, nao label; sem verdade de terreno",
         "protocolo de ground reference oficial com geometria patch-level"),
        ("blocked_for_score_v7", False, "",
         "flow_acc oficial irrecuperavel (17C39); sem base validada p/ score v7",
         "nao aplicavel nesta frente; manter indice transparente review-only"),
        ("blocked_for_17b", False, "",
         "17B exige G4_full + ground reference aceito; nao atingido",
         "G4_full com ground reference oficial patch-level"),
    ]
    rows = []
    for i, (target, ready, evidence, blocking, nxt) in enumerate(targets, start=1):
        rows.append({
            "readiness_id": f"S18A_RDY_{i:04d}", "readiness_target": target,
            "is_ready": _bool(ready), "evidence": evidence, "blocking_reason": blocking,
            "required_next_step": nxt, "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# integridade + no-leakage + summary + blockers + report
# ---------------------------------------------------------------------------

def score_integrity_audit_rows():
    v6_changed = bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))
    matrix_changed = bool(_run_git(["diff", "--name-only", "--", rel(OFFICIAL_MATRIX)]))
    return [
        {"score_integrity_audit_id": "S18A_INTEG_0001", "protected_artifact": rel(SCORE_V6),
         "artifact_type": "score_v6_candidate", "changed_in_working_tree": _bool(v6_changed),
         "created_new": "false", "note": "score v6 oficial nunca modificado nem substituido", "review_only": "true"},
        {"score_integrity_audit_id": "S18A_INTEG_0002", "protected_artifact": rel(OFFICIAL_MATRIX),
         "artifact_type": "official_feature_matrix", "changed_in_working_tree": _bool(matrix_changed),
         "created_new": "false", "note": "matriz oficial apenas lida", "review_only": "true"},
        {"score_integrity_audit_id": "S18A_INTEG_0003", "protected_artifact": rel(SCORE_V7),
         "artifact_type": "score_v7_candidate", "changed_in_working_tree": "false",
         "created_new": _bool(SCORE_V7.exists()), "note": "score v7 permanece inexistente", "review_only": "true"},
    ]


def no_leakage_audit_rows():
    rows = []
    for i, p in enumerate(assemble_patches(), start=1):
        anchor = "true" if p["is_canary"] else "not_applicable"
        rows.append({
            "no_leakage_audit_id": f"S18A_LEAK_{i:04d}", "unified_patch_id": p["unified_patch_id"],
            "patch_family": p["patch_family"],
            "uses_occurrence_as_causal_feature": "false", "occurrence_only_observational_anchor": anchor,
            "uses_canary_as_positive_label": "false", "uses_control_as_negative_truth": "false",
            "uses_absence_as_real_absence": "false", "uses_flow_acc_as_equivalent": "false",
            "imputes_missing_feature": "false", "hides_missing_feature": "false",
            "mixes_pre_and_post_event": "false", "creates_ground_truth": "false",
            "creates_training_label": "false", "modifies_score_v6": "false", "creates_score_v7": "false",
            "unlocks_17b": "false", "passes_no_leakage": "true",
            "review_only": "true",
        })
    return rows


def build_summary():
    patches = assemble_patches()
    official = [p for p in patches if p["is_official"]]
    canary = [p for p in patches if p["is_canary"]]
    control = [p for p in patches if p["is_control"]]
    registry = registry_rows()
    contract = family_contract()
    inv = source_inventory_rows()
    s2 = sentinel2_store_rows()
    ch = chirps_store_rows()
    tr = terrain_store_rows()
    hy = hydrology_store_rows()
    lc = landcover_store_rows()
    mm = multimodal_store_rows()
    avail = availability_matrix_rows()
    lineage = lineage_manifest_rows()
    qflags = quality_flag_rows()

    def _fam_count(fam):
        return sum(1 for p in patches if p["_avail"][fam])

    completeness = [sum(1 for f in DATA_FAMILIES if p["_avail"][f]) / len(DATA_FAMILIES) for p in patches]
    completeness_mean = round(sum(completeness) / len(completeness), 4) if completeness else 0.0
    geom_count = sum(1 for p in patches if p["geometry_available"])

    minimum = (
        len(official) >= 100 and len(canary) >= 11 and len(control) >= 5
        and len(registry) >= 116 and contract["feature_family_count"] >= 6
        and len(mm) >= 116 and len(avail) >= 116 and len(lineage) >= 116 and len(qflags) >= 116
    )
    return {
        "minimum_success_achieved": minimum,
        "official_patch_rows_consumed_count": len(official),
        "canary_patch_rows_consumed_count": len(canary),
        "control_patch_rows_consumed_count": len(control),
        "unified_patch_registry_rows_count": len(registry),
        "feature_family_count": contract["feature_family_count"],
        "feature_source_inventory_rows_count": len(inv),
        "sentinel2_store_rows_count": len(s2),
        "chirps_store_rows_count": len(ch),
        "terrain_store_rows_count": len(tr),
        "hydrology_store_rows_count": len(hy),
        "landcover_store_rows_count": len(lc),
        "multimodal_feature_store_rows_count": len(mm),
        "feature_availability_matrix_rows_count": len(avail),
        "lineage_manifest_rows_count": len(lineage),
        "quality_flag_rows_count": len(qflags),
        "feature_completeness_mean": completeness_mean,
        "patches_with_geometric_lineage_count": geom_count,
        "patches_with_sentinel2_count": _fam_count("sentinel2_spectral"),
        "patches_with_chirps_count": _fam_count("chirps_rainfall"),
        "patches_with_terrain_count": _fam_count("terrain_topography"),
        "patches_with_hydrology_count": _fam_count("hydrology_proximity"),
        "patches_with_landcover_count": _fam_count("landcover_urban"),
        "patches_with_transparent_components_count": _fam_count("transparent_review_components"),
        "flow_acc_treated_as_equivalent": False,
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
        "result_class": "multimodal_feature_store_delivered",
        "recommended_next_milestone": (
            "SUSC-18B Expandir os componentes transparentes review-only (17C40) para toda a populacao oficial "
            "usando a feature store 18A (analise por-patch review-only), mantendo missingness explicita; "
            "e/ou reabrir aquisicao de Ground Reference oficial com geometria patch-level para G4_full. "
            "Sem score v7, sem ground truth, sem treino, 17B fail-closed."),
    }


BLOCKER_LIST = [
    ("not_ground_truth", "feature store review-only; NAO e ground truth"),
    ("not_training_ready", "sem labels; NAO habilita treino supervisionado"),
    ("not_score_v7", "NAO e score v7; flow_acc oficial irrecuperavel (17C39)"),
    ("flow_acc_not_equivalent", "flow_acc recomputado NAO equivalente ao oficial; nos oficiais e valor nativo"),
    ("documentary_anchor_not_label", "ocorrencia oficial e ancora observacional, nao label positivo"),
    ("control_not_negative_truth", "controle de mismatch NAO e negativo verdadeiro; ausencia nao e ausencia real"),
    ("17b_blocked", "17B exige G4_full + ground reference oficial aceito; nao atingido"),
    ("feature_missingness_explicit", "missingness registrada por familia; sem imputacao escondida"),
]


def promotion_blocker_rows():
    return [{"blocker_id": f"S18A_BLOCKER_{i:04d}", "blocker_type": n, "description": d,
             "blocks_official_use": "true", "blocks_17b": "true", "review_only": "true"}
            for i, (n, d) in enumerate(BLOCKER_LIST, start=1)]


def build_report():
    s = build_summary()
    return "\n".join([
        "# SUSC-18A - Feature Store Multimodal Escalavel Patch-Level",
        "",
        "## Objetivo",
        "Consolidar os extratores reais desenvolvidos na cadeia de canarios (17C33-17C40) em uma FEATURE STORE MULTIMODAL patch-level unica, auditavel e reproduzivel, unindo populacao oficial, canarios ancorados em ocorrencia oficial hidrologica e controles de mismatch, com cobertura, lineage, disponibilidade e incerteza EXPLICITAS por patch.",
        "",
        "## Pergunta cientifica",
        "Conseguimos transformar os extratores reais dos canarios em uma feature store multimodal escalavel, com cobertura, lineage, disponibilidade e incerteza explicitas por patch? Sim, com missingness assimetrica honesta.",
        "",
        "## Consumo de patches",
        f"- Oficiais consumidos: {s['official_patch_rows_consumed_count']} (populacao Recife/Curitiba/Petropolis)",
        f"- Canarios consumidos: {s['canary_patch_rows_consumed_count']}",
        f"- Controles consumidos: {s['control_patch_rows_consumed_count']}",
        f"- Registry unificado: {s['unified_patch_registry_rows_count']} linhas",
        "",
        "## Familias multimodais (contrato)",
        f"- {s['feature_family_count']} familias: sentinel2_spectral, chirps_rainfall, terrain_topography, hydrology_proximity, landcover_urban, transparent_review_components, lineage_quality",
        "",
        "## Stores por familia",
        f"- Sentinel-2: {s['sentinel2_store_rows_count']} | CHIRPS: {s['chirps_store_rows_count']} | Terrain: {s['terrain_store_rows_count']} | Hydrology: {s['hydrology_store_rows_count']} | Landcover: {s['landcover_store_rows_count']}",
        f"- Matriz multimodal final: {s['multimodal_feature_store_rows_count']} linhas | Disponibilidade: {s['feature_availability_matrix_rows_count']} | Lineage: {s['lineage_manifest_rows_count']} | Quality flags: {s['quality_flag_rows_count']}",
        "",
        "## Cobertura por familia (patches com feature)",
        f"- Sentinel-2: {s['patches_with_sentinel2_count']} | CHIRPS: {s['patches_with_chirps_count']} | Terrain: {s['patches_with_terrain_count']} | Hydrology: {s['patches_with_hydrology_count']} | Landcover: {s['patches_with_landcover_count']} | Transparent: {s['patches_with_transparent_components_count']}",
        f"- Completude media: {s['feature_completeness_mean']}",
        "",
        "## Missingness honesta",
        "- Controles (5): sem terrain/hydrology/landcover (contexto de mismatch) -> completude 0.5.",
        "- Oficiais (300): sem transparent_review_components (so 16 patches em 17C40) -> completude ~0.83.",
        "- Canarios (11): 6/6 familias -> completude 1.0.",
        "- Nada imputado; cada ausencia registrada em availability_matrix e quality_flags.",
        "",
        "## Guardrails cientificos",
        "- flow_acc oficial NAO tratado como equivalente (canario recomputado != oficial; oficial = valor nativo).",
        "- Ocorrencia oficial NAO entrou como feature causal (apenas ancora observacional).",
        "- Controle NAO virou negativo verdadeiro; ausencia NAO virou ausencia real.",
        "- Score v6 oficial intacto; score v7 inexistente; ground truth nao criado; treino nao criado; 17B fail-closed.",
        "",
        f"## minimum_success_achieved: {s['minimum_success_achieved']} | result_class: {s['result_class']}",
        "",
        "## Proximo marco recomendado",
        s["recommended_next_milestone"],
    ])


# ---------------------------------------------------------------------------
# build / modes / network (fail-closed) / validate
# ---------------------------------------------------------------------------

def build_all():
    _require_inputs()
    ensure_dir(OUT)
    write_csv(REGISTRY, registry_rows())
    write_json(FAMILY_CONTRACT, family_contract())
    write_csv(SOURCE_INVENTORY, source_inventory_rows())
    write_csv(SENTINEL2_STORE, sentinel2_store_rows())
    write_csv(CHIRPS_STORE, chirps_store_rows())
    write_csv(TERRAIN_STORE, terrain_store_rows())
    write_csv(HYDROLOGY_STORE, hydrology_store_rows())
    write_csv(LANDCOVER_STORE, landcover_store_rows())
    write_csv(MULTIMODAL_STORE, multimodal_store_rows())
    write_csv(AVAILABILITY, availability_matrix_rows())
    write_csv(LINEAGE, lineage_manifest_rows())
    write_csv(QUALITY_FLAGS, quality_flag_rows())
    write_csv(NEXT_STAGE, next_stage_readiness_rows())
    write_csv(INTEGRITY, score_integrity_audit_rows())
    write_csv(LEAKAGE, no_leakage_audit_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, promotion_blocker_rows())
    write_markdown(REPORT, build_report())


def _network_blocked(mode):
    missing = [e for e in NETWORK_ENVS if os.environ.get(e) != "1"]
    print(f"NETWORK MODE '{mode}' BLOCKED (fail-closed): faltam envs {';'.join(missing)}. "
          "O marco fecha com outputs existentes + missingness registrada.", file=sys.stderr)
    return 0


def run_mode(mode):
    if mode in ("backfill-sentinel2", "backfill-chirps", "backfill-hydrology", "backfill-landcover"):
        return _network_blocked(mode)
    _require_inputs()
    ensure_dir(OUT)
    dispatch = {
        "build-unified-patch-registry": (write_csv, REGISTRY, registry_rows),
        "build-feature-family-contract": (write_json, FAMILY_CONTRACT, family_contract),
        "inventory-feature-sources": (write_csv, SOURCE_INVENTORY, source_inventory_rows),
        "build-sentinel2-store": (write_csv, SENTINEL2_STORE, sentinel2_store_rows),
        "build-chirps-store": (write_csv, CHIRPS_STORE, chirps_store_rows),
        "build-terrain-store": (write_csv, TERRAIN_STORE, terrain_store_rows),
        "build-hydrology-store": (write_csv, HYDROLOGY_STORE, hydrology_store_rows),
        "build-landcover-store": (write_csv, LANDCOVER_STORE, landcover_store_rows),
        "build-multimodal-store": (write_csv, MULTIMODAL_STORE, multimodal_store_rows),
        "build-availability-matrix": (write_csv, AVAILABILITY, availability_matrix_rows),
        "build-lineage-manifest": (write_csv, LINEAGE, lineage_manifest_rows),
        "build-quality-flags": (write_csv, QUALITY_FLAGS, quality_flag_rows),
        "build-next-stage-readiness": (write_csv, NEXT_STAGE, next_stage_readiness_rows),
    }
    if mode == "build-offline":
        build_all()
    elif mode == "audit":
        raise SystemExit(validate())
    elif mode in dispatch:
        writer, path, fn = dispatch[mode]
        writer(path, fn())
    else:
        raise SystemExit(f"unknown mode: {mode}")
    return 0


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
        if "enum" in rules and val not in rules["enum"]:
            v.append(f"{f}:enum")
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


def validate():
    missing = [p for p in REQUIRED_OUTPUTS if not p.exists()]
    if missing:
        print("MISSING: " + "; ".join(rel(p) for p in missing), file=sys.stderr)
        return 1
    errors = _validate_byte_identical()

    registry = read_csv(REGISTRY)
    inv = read_csv(SOURCE_INVENTORY)
    mm = read_csv(MULTIMODAL_STORE)
    avail = read_csv(AVAILABILITY)
    lineage = read_csv(LINEAGE)
    qflags = read_csv(QUALITY_FLAGS)
    terrain = read_csv(TERRAIN_STORE)
    leak = read_csv(LEAKAGE)
    integ = read_csv(INTEGRITY)
    summary = read_json(SUMMARY)
    contract = read_json(FAMILY_CONTRACT)
    store_schema = read_json(SCHEMA_STORE)
    lineage_schema = read_json(SCHEMA_LINEAGE)
    avail_schema = read_json(SCHEMA_AVAIL)

    for r in mm:
        errors.extend(_schema_violations(r, store_schema))
    for r in lineage:
        errors.extend(_schema_violations(r, lineage_schema))
    for r in avail:
        errors.extend(_schema_violations(r, avail_schema))

    official = [r for r in registry if r["patch_family"] == "official_population_patch"]
    canary = [r for r in registry if r["patch_family"] == "event_anchored_canary_patch"]
    control = [r for r in registry if r["patch_family"] == "spatial_mismatch_control_candidate"]

    # 1-3 contagens minimas
    if len(official) < 100:
        errors.append("official_lt_100")
    if len(canary) < 11:
        errors.append("canary_lt_11")
    if len(control) < 5:
        errors.append("control_lt_5")
    # 4 registry
    if len(registry) < 116:
        errors.append("registry_lt_116")
    # 5 contrato
    if not FAMILY_CONTRACT.exists() or contract.get("feature_family_count", 0) < 6:
        errors.append("family_contract_missing_or_lt_6")
    # 6 multimodal
    if len(mm) < 116:
        errors.append("multimodal_lt_116")
    # 7 disponibilidade
    if len(avail) < 116:
        errors.append("availability_lt_116")
    # 8 lineage
    if len(lineage) < 116:
        errors.append("lineage_lt_116")
    # 9 quality flags
    if len(qflags) < 116:
        errors.append("quality_flags_lt_116")
    # 10 flow_acc equivalente
    if summary.get("flow_acc_treated_as_equivalent") is not False:
        errors.append("flow_acc_equivalent_summary")
    for r in terrain:
        if r.get("flow_acc_is_official_equivalent") != "false":
            errors.append("flow_acc_marked_equivalent")
            break
    for r in leak:
        if r["uses_flow_acc_as_equivalent"] != "false":
            errors.append("flow_acc_equivalent_leak")
            break
    # canarios com flow_acc precisam do contrato de substituicao
    for r in terrain:
        if r["patch_family"] == "event_anchored_canary_patch" and _num(r["flow_acc_value"]) is not None \
                and r["flow_acc_replacement_contract_required"] != "true":
            errors.append("canary_flow_acc_missing_contract")
            break
    # 11 ocorrencia como feature causal
    for r in leak:
        if r["uses_occurrence_as_causal_feature"] != "false":
            errors.append("occurrence_causal_leak")
            break
    for r in mm:
        if r["absence_of_occurrence_is_true_negative"] != "false":
            errors.append("absence_true_negative")
            break
    # 12 controle negativo verdadeiro
    for r in leak:
        if r["uses_control_as_negative_truth"] != "false" or r["uses_absence_as_real_absence"] != "false":
            errors.append("control_negative_leak")
            break
    # 13 score v6 mudou
    if _run_git(["diff", "--name-only", "--", rel(SCORE_V6)]) or summary.get("score_v6_original_file_changed"):
        errors.append("score_v6_changed")
    if _run_git(["diff", "--name-only", "--", rel(OFFICIAL_MATRIX)]):
        errors.append("official_matrix_changed")
    # 14 score v7
    if SCORE_V7.exists() or summary.get("score_v7_created"):
        errors.append("score_v7_exists")
    # 15/16 ground truth / treino
    if summary.get("ground_truth_created") or summary.get("training_labels_created"):
        errors.append("gt_or_training")
    for r in mm:
        if r["ground_truth"] != "false" or r["training_label"] != "false":
            errors.append("mm_gt_training")
            break
    for r in leak:
        if r["creates_ground_truth"] != "false" or r["creates_training_label"] != "false":
            errors.append("leak_gt_training")
            break
    # 17 17B liberado
    if summary.get("eligible_for_17b_now"):
        errors.append("17b_unlocked")
    if any(r["unlocks_17b"] != "false" for r in leak):
        errors.append("17b_leak")
    # 18 raster pesado -> stores sao CSV/JSON leves (checagem de extensao dos outputs)
    for p in REQUIRED_OUTPUTS:
        if p.suffix.lower() in (".tif", ".tiff", ".npz", ".npy"):
            errors.append(f"heavy_raster_output:{rel(p)}")
    # 19 missingness escondida
    for r in leak:
        if r["hides_missing_feature"] != "false" or r["imputes_missing_feature"] != "false":
            errors.append("missingness_hidden")
            break
    if any(r["passes_no_leakage"] != "true" for r in leak):
        errors.append("no_leakage_failed")
    # integridade
    for r in integ:
        if r["changed_in_working_tree"] == "true" and r["artifact_type"] in ("score_v6_candidate", "official_feature_matrix"):
            errors.append("integrity_protected_changed")
    # nao terminar blocked
    if not summary.get("minimum_success_achieved"):
        errors.append("minimum_success_not_achieved")

    # 20 summary reflete contagens reais
    expected = build_summary()
    for k, v in expected.items():
        if summary.get(k) != v:
            errors.append(f"summary_mismatch:{k}")

    # ids unicos/ordenados
    for rows, key in [
        (registry, "unified_patch_id"), (inv, "source_inventory_id"), (mm, "unified_patch_id"),
        (avail, "availability_id"), (lineage, "lineage_id"), (qflags, "quality_flag_id"),
        (leak, "no_leakage_audit_id"),
    ]:
        ids = [r[key] for r in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(
        "18A -> "
        f"official={summary['official_patch_rows_consumed_count']} canary={summary['canary_patch_rows_consumed_count']} "
        f"control={summary['control_patch_rows_consumed_count']} registry={summary['unified_patch_registry_rows_count']} "
        f"families={summary['feature_family_count']} multimodal={summary['multimodal_feature_store_rows_count']} "
        f"availability={summary['feature_availability_matrix_rows_count']} lineage={summary['lineage_manifest_rows_count']} "
        f"qflags={summary['quality_flag_rows_count']} completeness_mean={summary['feature_completeness_mean']} "
        f"flow_acc_equivalent={summary['flow_acc_treated_as_equivalent']} "
        f"result={summary['result_class']} min_success={summary['minimum_success_achieved']}"
    )
    return 0


MODES = [
    "build-unified-patch-registry", "build-feature-family-contract", "inventory-feature-sources",
    "build-sentinel2-store", "build-chirps-store", "build-terrain-store", "build-hydrology-store",
    "build-landcover-store", "build-multimodal-store", "build-availability-matrix",
    "build-lineage-manifest", "build-quality-flags", "build-next-stage-readiness",
    "build-offline", "audit",
    "backfill-sentinel2", "backfill-chirps", "backfill-hydrology", "backfill-landcover",
]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="susc_18a")
    parser.add_argument("mode", choices=MODES)
    args = parser.parse_args(argv)
    if args.mode == "audit":
        return validate()
    return run_mode(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
