"""SUSC-17C33 Event-Anchored Canary Patch Set e Validacao Observacional Review-Only.

Cria uma NOVA familia de patches canarios ancorados em ocorrencias oficiais
hidrologicas (geocodificadas no 17C32), SEM substituir o patch original. O
objetivo NAO e mover o patch antigo para "dar certo"; e materializar AOIs
ancorados no evento observado e contrastar, review-only, com o patch original
(que falhou no vinculo espacial no 17C32).

Fluxo:
  ocorrencia oficial hidrologica (17C31/17C32)
    -> localizacao oficial/controlada (ponto geocodificado 17C32, street-level)
    -> patch canario novo (namespace event_anchored_canary_patch)
    -> features pre-evento DISPONIVEIS (CHIRPS region-level herdado do 17C25;
       Sentinel-2/HAND/slope/drenagem/urban_prop = nao disponiveis sem extracao)
    -> score v6 (NAO recomputado: features locais indisponiveis -> not_computable)
    -> comparacao observacional com patch original (spatial_mismatch_control)
    -> validacao observacional review-only.

Achado honesto esperado: o gatilho de chuva (CHIRPS) foi region-wide (mesmo para
patch original e canarios); o que discrimina alagamento e a topografia/drenagem/
urbanizacao LOCAL, cujas features NAO estao disponiveis offline. Logo o score v6
nao e computavel nos pontos de ocorrencia neste marco. O unico discriminador
local disponivel e a ANCORA OBSERVACIONAL: os patches canarios tem ocorrencia
oficial hidrologica documentada; o patch original NAO tem ocorrencia proxima
(ausencia != negativo verdadeiro). O conjunto canario fica materializado para
extracao de features futura.

Guardrails (fail-closed):
  - patch original permanece e vira spatial_mismatch_control_candidate;
  - patches novos usam namespace event_anchored_canary_patch;
  - nenhum patch novo vira ground truth / treino / substitui SUSC original;
  - nenhum score v7; score v6 intacto;
  - ocorrencia sem hidrologico nao entra como canario positivo;
  - centroide de bairro nao usado sem incerteza (usa ponto geocodificado + incerteza_m);
  - ausencia de ocorrencia NAO vira negativo verdadeiro.

Tudo offline e deterministico: reusa artefatos ja commitados (17C32 pontos
geocodificados; 17C25 CHIRPS; 17C6 grid). Sem rede.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

# inputs (reuso; nao re-adquirir nem recomputar)
C32_POINTS = OUT / "susc_17c32_geocoded_point_registry.csv"
C32_DIST = OUT / "susc_17c32_patch_buffer_distance_evaluation.csv"
C32_TARGETS = OUT / "susc_17c32_hydrological_occurrence_targets.csv"
C32_SUMMARY = OUT / "susc_17c32_readiness_summary.json"
C31_G5D = OUT / "susc_17c31_g5d_hydrological_separation_evaluation.csv"
C31_PHSEP = OUT / "susc_17c31_phenomenon_separation_candidate_registry.csv"
C25_MULTIMODAL = OUT / "susc_17c25_multimodal_feature_matrix.csv"
PATCH_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"

REQUIRED_INPUTS = [
    SCORE_V6, C32_POINTS, C32_DIST, C32_TARGETS, C32_SUMMARY, C31_G5D, C31_PHSEP,
    C25_MULTIMODAL, PATCH_GRID,
]

# outputs
REPORT = OUT / "SUSC_17C33_EVENT_ANCHORED_CANARY_PATCH_REPORT.md"
REGISTRY = OUT / "susc_17c33_event_anchored_canary_patch_registry.csv"
GEOJSON = OUT / "susc_17c33_event_anchored_patch_geojson.geojson"
MISMATCH_CONTROL = OUT / "susc_17c33_original_patch_mismatch_control.csv"
FEATURE_BINDING = OUT / "susc_17c33_pre_event_feature_binding.csv"
SCORE_ON_CANARY = OUT / "susc_17c33_score_v6_on_canary_patches.csv"
CONTRAST = OUT / "susc_17c33_observational_contrast_matrix.csv"
SUMMARY = OUT / "susc_17c33_review_only_validation_summary.json"

SCHEMA_EAC = SCHEMAS / "susc_17c33_event_anchored_canary_patch_schema_v1.json"

REQUIRED_OUTPUTS = [
    REPORT, REGISTRY, GEOJSON, MISMATCH_CONTROL, FEATURE_BINDING, SCORE_ON_CANARY,
    CONTRAST, SUMMARY,
]

GOV = {"review_only": "true", "is_training": "false", "is_ground_truth": "false"}
NAMESPACE = "event_anchored_canary_patch"
# incerteza street-level herdada do 17C32; meia-aresta do patch canario = incerteza.
DEG_LAT_M = 110540.0
OCCURRENCE_PROXIMITY_BUFFER_M = 1500.0  # mesmo buffer do 17C32 para o controle


def _bool(v: bool) -> str:
    return "true" if v else "false"


def _run_git(args: list[str]) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _require_inputs() -> None:
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))


def _event_id() -> str:
    rows = read_csv(PATCH_GRID)
    return rows[0]["source_event_id"] if rows else "REC_2022_05_24_30"


def _patches() -> list[dict]:
    return read_csv(PATCH_GRID)


def _patch_centers() -> list[tuple[str, float, float]]:
    out = []
    for r in _patches():
        cx = (float(r["xmin"]) + float(r["xmax"])) / 2
        cy = (float(r["ymin"]) + float(r["ymax"])) / 2
        out.append((r["candidate_patch_id"], cx, cy))
    return out


def _primary_patch_id() -> str:
    c = _patch_centers()
    return c[0][0] if c else "S17C6_CANARY_REC_00001"


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _nearest_patch(lon: float, lat: float) -> tuple[str, float]:
    best_id, best_d = "", float("inf")
    for pid, cx, cy in _patch_centers():
        d = _haversine_m(lon, lat, cx, cy)
        if d < best_d:
            best_id, best_d = pid, d
    return best_id, best_d


def _deg_lon_m(lat: float) -> float:
    return 111320.0 * math.cos(math.radians(lat))


def _chirps_region_level() -> dict:
    """CHIRPS pre-evento region-level herdado do 17C25 (patch original).
    CHIRPS ~5km: valido a nivel metropolitano para o evento; NAO discrimina local."""
    rows = read_csv(C25_MULTIMODAL)
    primary = _primary_patch_id()
    row = next((r for r in rows if r["candidate_patch_id"] == primary), rows[0] if rows else {})
    return {
        "chirps_3d_sum_mm": row.get("CHIRPS_3d_sum_mm", "not_available"),
        "chirps_7d_sum_mm": row.get("CHIRPS_7d_sum_mm", "not_available"),
        "chirps_30d_sum_mm": row.get("CHIRPS_30d_sum_mm", "not_available"),
    }


# ---------------------------------------------------------------------------
# Event-anchored canary patch registry (a partir dos pontos hidrologicos 17C32)
# ---------------------------------------------------------------------------

def _hydrological_points() -> list[dict]:
    """Pontos geocodificados do 17C32 (todos hidrologicos por construcao do 17C32)."""
    points = read_csv(C32_POINTS)
    # garante sinal hidrologico: o 17C32 so geocodificou alvos hidrologicos
    targets = {t["hydrological_occurrence_target_id"]: t for t in read_csv(C32_TARGETS)}
    out = []
    for p in points:
        tgt = targets.get(p["hydrological_occurrence_target_id"], {})
        if tgt.get("hydrological_signal", "true") != "true":
            continue
        out.append({**p, "mass_movement_signal": tgt.get("mass_movement_signal", "false"),
                    "occurrence_date": tgt.get("occurrence_date", ""),
                    "occurrence_type": tgt.get("occurrence_type", ""),
                    "priority_rank": tgt.get("priority_rank", "99")})
    return out


def event_anchored_canary_patch_rows() -> list[dict]:
    event_id = _event_id()
    original = _primary_patch_id()
    rows = []
    for i, p in enumerate(sorted(_hydrological_points(),
                                 key=lambda r: (int(r.get("priority_rank", 99)), r["bairro"], r["logradouro"])), start=1):
        lat, lon = float(p["resolved_lat"]), float(p["resolved_lon"])
        unc = float(p["uncertainty_m"])
        half = unc  # meia-aresta = incerteza street-level (honesto: patch cobre a incerteza)
        dlat = half / DEG_LAT_M
        dlon = half / _deg_lon_m(lat)
        nearest_id, dist = _nearest_patch(lon, lat)
        rows.append({
            "event_anchored_canary_patch_id": f"S17C33_EAC_{i:04d}",
            "namespace": NAMESPACE,
            "candidate_patch_id_original": original,
            "event_id": event_id,
            "bairro": p["bairro"], "logradouro": p["logradouro"],
            "occurrence_date": p["occurrence_date"], "occurrence_type": p["occurrence_type"],
            "hydrological_signal": "true", "mass_movement_signal": p.get("mass_movement_signal", "false"),
            "center_lat": f"{lat:.7f}", "center_lon": f"{lon:.7f}",
            "xmin": f"{lon - dlon:.7f}", "ymin": f"{lat - dlat:.7f}",
            "xmax": f"{lon + dlon:.7f}", "ymax": f"{lat + dlat:.7f}", "crs": "EPSG:4326",
            "uncertainty_m": str(int(unc)), "patch_half_size_m": str(int(half)),
            "distance_to_original_patch_m": f"{dist:.1f}", "nearest_original_patch_id": nearest_id,
            "link_type_to_occurrence": "official_hydrological_occurrence_geocoded_street_level",
            "geocoder_is_official": "false",
            "source_point_id": p["geocoded_point_id"],
            "replaces_original_patch": "false", "is_ground_truth": "false", "is_training": "false",
            "review_only": "true",
        })
    return rows


def build_geojson() -> dict:
    feats = []
    for r in event_anchored_canary_patch_rows():
        lon, lat = float(r["center_lon"]), float(r["center_lat"])
        xmin, ymin, xmax, ymax = float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"])
        feats.append({
            "type": "Feature",
            "properties": {
                "event_anchored_canary_patch_id": r["event_anchored_canary_patch_id"],
                "namespace": NAMESPACE, "bairro": r["bairro"], "logradouro": r["logradouro"],
                "occurrence_date": r["occurrence_date"], "uncertainty_m": r["uncertainty_m"],
                "distance_to_original_patch_m": r["distance_to_original_patch_m"],
                "is_ground_truth": "false", "review_only": "true",
            },
            "geometry": {"type": "Polygon", "coordinates": [[
                [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]]},
        })
        feats.append({
            "type": "Feature",
            "properties": {"event_anchored_canary_patch_id": r["event_anchored_canary_patch_id"],
                           "role": "center_point", "review_only": "true"},
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
        })
    return {"type": "FeatureCollection", "name": "susc_17c33_event_anchored_patches",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": feats}


# ---------------------------------------------------------------------------
# Original patch mismatch control (patch original permanece, vira controle)
# ---------------------------------------------------------------------------

def original_patch_mismatch_control_rows() -> list[dict]:
    event_id = _event_id()
    points = _hydrological_points()
    rows = []
    for r in _patches():
        cx = (float(r["xmin"]) + float(r["xmax"])) / 2
        cy = (float(r["ymin"]) + float(r["ymax"])) / 2
        # menor distancia a uma ocorrencia hidrologica oficial
        best_b, best_d = "none", float("inf")
        for p in points:
            d = _haversine_m(cx, cy, float(p["resolved_lon"]), float(p["resolved_lat"]))
            if d < best_d:
                best_d, best_b = d, p["bairro"]
        has_nearby = best_d <= OCCURRENCE_PROXIMITY_BUFFER_M
        rows.append({
            "original_patch_id": r["candidate_patch_id"],
            "event_id": event_id, "region": r.get("region", "Recife"),
            "control_role": "spatial_mismatch_control_candidate",
            "distance_to_nearest_official_occurrence_m": f"{best_d:.1f}",
            "nearest_occurrence_bairro": best_b,
            "proximity_buffer_m": str(int(OCCURRENCE_PROXIMITY_BUFFER_M)),
            "has_nearby_official_hydrological_occurrence": _bool(has_nearby),
            "absence_of_occurrence_is_true_negative": "false",
            "note": "patch original preservado; ausencia de ocorrencia proxima e mismatch espacial, NAO negativo verdadeiro; nao substituido, nao ground truth, nao treino",
            "replaces_by_canary": "false", "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Pre-event feature binding (honesto: CHIRPS herdado; resto nao disponivel)
# ---------------------------------------------------------------------------

def pre_event_feature_binding_rows() -> list[dict]:
    chirps = _chirps_region_level()
    rows = []
    for r in event_anchored_canary_patch_rows():
        rows.append({
            "feature_binding_id": r["event_anchored_canary_patch_id"].replace("EAC", "FB"),
            "event_anchored_canary_patch_id": r["event_anchored_canary_patch_id"],
            "bairro": r["bairro"], "logradouro": r["logradouro"],
            # CHIRPS region-level herdado (5km): valido metropolitano, NAO discrimina local
            "chirps_3d_sum_mm_preevent": chirps["chirps_3d_sum_mm"],
            "chirps_7d_sum_mm_preevent": chirps["chirps_7d_sum_mm"],
            "chirps_30d_sum_mm_preevent": chirps["chirps_30d_sum_mm"],
            "chirps_source": "inherited_region_level_17c25_same_event",
            "chirps_discriminates_local": "false",
            # features locais (10-30m) NAO extraiveis offline neste marco
            "ndvi_preevent": "not_available_requires_extraction",
            "ndwi_preevent": "not_available_requires_extraction",
            "mndwi_preevent": "not_available_requires_extraction",
            "ndbi_preevent": "not_available_requires_extraction",
            "hand_m": "not_available_requires_extraction",
            "elevation_m": "not_available_requires_extraction",
            "slope_deg": "not_available_requires_extraction",
            "distance_to_drainage_m": "not_available_requires_extraction",
            "urban_prop": "not_available_requires_extraction",
            "occurrence_anchor_present": "true",
            "feature_completeness": "partial_chirps_region_level_only",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Score v6 on canary patches (NAO recomputado: features locais indisponiveis)
# ---------------------------------------------------------------------------

def score_v6_on_canary_patch_rows() -> list[dict]:
    rows = []
    for r in event_anchored_canary_patch_rows():
        rows.append({
            "score_v6_on_canary_id": r["event_anchored_canary_patch_id"].replace("EAC", "SV6"),
            "event_anchored_canary_patch_id": r["event_anchored_canary_patch_id"],
            "bairro": r["bairro"], "logradouro": r["logradouro"],
            "score_v6_status": "not_computable_pre_event_features_unavailable",
            "score_v6_value": "not_available",
            "score_v6_class": "not_available",
            "score_v6_recomputed": "false",
            "score_v6_source_dataset": rel(SCORE_V6),
            "score_v6_source_dataset_altered": "false",
            "score_v7_created": "false",
            "rainfall_trigger_note": "CHIRPS region-level identico ao patch original (gatilho de chuva foi metropolitano); nao discrimina local",
            "blocking_reason": "features locais discriminantes (HAND/slope/drenagem/Sentinel-2/urban_prop) indisponiveis offline; score v6 nao computavel no ponto de ocorrencia sem extracao",
            "is_ground_truth": "false", "is_training": "false", "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Observational contrast matrix (original control vs canary anchors)
# ---------------------------------------------------------------------------

def observational_contrast_matrix_rows() -> list[dict]:
    chirps = _chirps_region_level()
    control = original_patch_mismatch_control_rows()
    canary = event_anchored_canary_patch_rows()
    rows = []
    idx = 0
    # linha do patch original (controle)
    primary_ctrl = next((c for c in control if c["original_patch_id"] == _primary_patch_id()), control[0] if control else {})
    idx += 1
    rows.append({
        "contrast_id": f"S17C33_CONTRAST_{idx:04d}",
        "entity_id": primary_ctrl.get("original_patch_id", _primary_patch_id()),
        "entity_type": "spatial_mismatch_control_candidate",
        "bairro": "Recife Antigo/Santo Amaro (AOI Charter758)",
        "has_official_hydrological_occurrence_within_buffer": primary_ctrl.get("has_nearby_official_hydrological_occurrence", "false"),
        "distance_to_nearest_official_occurrence_m": primary_ctrl.get("distance_to_nearest_official_occurrence_m", "unknown"),
        "distance_to_original_patch_m": "0.0",
        "chirps_7d_sum_mm_region_level": chirps["chirps_7d_sum_mm"],
        "score_v6_available": "false",
        "observational_relevance_to_event": "low_no_nearby_official_occurrence",
        "absence_is_true_negative": "false",
        "interpretation": "patch original nao tem ocorrencia hidrologica oficial proxima; mismatch espacial (NAO negativo verdadeiro)",
        "review_only": "true",
    })
    # linhas dos patches canarios
    for c in canary:
        idx += 1
        rows.append({
            "contrast_id": f"S17C33_CONTRAST_{idx:04d}",
            "entity_id": c["event_anchored_canary_patch_id"],
            "entity_type": "event_anchored_canary_patch",
            "bairro": c["bairro"],
            "has_official_hydrological_occurrence_within_buffer": "true",
            "distance_to_nearest_official_occurrence_m": "0.0",
            "distance_to_original_patch_m": c["distance_to_original_patch_m"],
            "chirps_7d_sum_mm_region_level": chirps["chirps_7d_sum_mm"],
            "score_v6_available": "false",
            "observational_relevance_to_event": "high_official_hydrological_occurrence_anchor",
            "absence_is_true_negative": "not_applicable",
            "interpretation": "patch canario ancorado em ocorrencia oficial hidrologica documentada (alagamento) na janela do evento; AOI observacionalmente mais relevante, porem score v6 nao computavel sem extracao",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Summary + report
# ---------------------------------------------------------------------------

def build_summary() -> dict:
    canary = event_anchored_canary_patch_rows()
    control = original_patch_mismatch_control_rows()
    binding = pre_event_feature_binding_rows()
    score_rows = score_v6_on_canary_patch_rows()
    contrast = observational_contrast_matrix_rows()
    c32 = read_json(C32_SUMMARY)
    dists = [float(r["distance_to_original_patch_m"]) for r in canary] if canary else []
    controls_with_occurrence = [c for c in control if c["has_nearby_official_hydrological_occurrence"] == "true"]
    score_computable = [s for s in score_rows if s["score_v6_status"] != "not_computable_pre_event_features_unavailable"]
    minimum = (
        len(canary) >= 3 and len(control) >= 1 and len(binding) >= 3
        and len(score_rows) >= 3 and len(contrast) >= 4
    )
    return {
        "minimum_success_achieved": minimum,
        "event_anchored_canary_patches_count": len(canary),
        "priority_bairros_covered": sorted({c["bairro"] for c in canary}),
        "original_patches_as_mismatch_control_count": len(control),
        "original_controls_with_nearby_official_occurrence_count": len(controls_with_occurrence),
        "pre_event_feature_binding_rows_count": len(binding),
        "canary_patches_with_chirps_region_level_count": len(binding),
        "canary_patches_with_local_features_count": 0,
        "score_v6_on_canary_rows_count": len(score_rows),
        "score_v6_computable_on_canary_count": len(score_computable),
        "observational_contrast_rows_count": len(contrast),
        "nearest_canary_to_original_m": round(min(dists), 1) if dists else -1,
        "farthest_canary_to_original_m": round(max(dists), 1) if dists else -1,
        "chirps_trigger_region_level_same_for_control_and_canary": True,
        "original_patch_overwritten": False,
        "canary_is_ground_truth": False,
        "canary_is_training": False,
        "score_v6_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
        "ground_truth_created": False,
        "training_labels_created": False,
        "review_only": True,
        # herdado do 17C32/17C31 (nao recalculado)
        "g4_full_true_count_inherited": int(c32.get("G4_full_true_count", 0)),
        "eligible_for_17b_now": False,
        "observational_answer": (
            "As areas com ocorrencia oficial hidrologica sao observacionalmente MAIS relevantes ao evento "
            "(ancora de ocorrencia documentada) do que o patch original, que nao tem ocorrencia proxima "
            "(mismatch espacial, NAO negativo). Porem a comparacao QUANTITATIVA de score v6 fica DEFERIDA: "
            "o gatilho de chuva (CHIRPS) foi region-wide (identico); as features discriminantes locais "
            "(HAND/slope/drenagem/Sentinel-2/urban_prop) nao estao disponiveis offline. Conjunto canario "
            "materializado para extracao futura."
        ),
        "result_class": "A_observational_anchor_B_score_deferred",
        "recommended_next_milestone": (
            "SUSC-17C34 Extracao pre-evento de features locais (HAND/slope/distance_to_drainage/Sentinel-2/urban_prop) "
            "para os patches canarios ancorados e calculo de score v6 review-only por patch canario, para comparar "
            "quantitativamente com o patch original; manter score v6 original intacto e 17B fail-closed."
        ),
    }


def build_report() -> str:
    s = build_summary()
    canary = event_anchored_canary_patch_rows()
    return "\n".join([
        "# SUSC-17C33 - Event-Anchored Canary Patch Set e Validacao Observacional Review-Only",
        "",
        "## Objetivo",
        "Criar uma familia NOVA de patches canarios ancorados em ocorrencias oficiais hidrologicas (geocodificadas no 17C32), SEM substituir o patch original. Nao e mover o patch antigo para dar certo: e ancorar AOIs no evento observado e contrastar review-only com o patch original que falhou no vinculo espacial.",
        "",
        "## Patches canarios ancorados",
        f"- Patches canarios (namespace {NAMESPACE}): {s['event_anchored_canary_patches_count']}.",
        f"- Bairros cobertos: {', '.join(s['priority_bairros_covered'])}.",
        f"- Distancia ao patch original: {s['nearest_canary_to_original_m']} m (min) a {s['farthest_canary_to_original_m']} m (max).",
        "- Cada patch canario: bbox centrado no ponto geocodificado (street-level) +- incerteza; ocorrencia oficial hidrologica na janela 2022-05-24..30.",
        "",
        "## Patch original como controle",
        f"- Patches originais marcados spatial_mismatch_control_candidate: {s['original_patches_as_mismatch_control_count']} (preservados, nao substituidos).",
        f"- Controles com ocorrencia oficial proxima (buffer {int(OCCURRENCE_PROXIMITY_BUFFER_M)} m): {s['original_controls_with_nearby_official_occurrence_count']}.",
        "- Ausencia de ocorrencia proxima e mismatch espacial, NAO negativo verdadeiro.",
        "",
        "## Features pre-evento (honesto)",
        f"- CHIRPS region-level herdado do 17C25 (mesmo evento): {s['canary_patches_with_chirps_region_level_count']} patches. CHIRPS ~5km NAO discrimina local.",
        f"- Features locais (HAND/slope/drenagem/Sentinel-2/urban_prop) disponiveis: {s['canary_patches_with_local_features_count']} (indisponiveis offline; requerem extracao).",
        "",
        "## Score v6 nos patches canarios",
        f"- Score v6 computavel: {s['score_v6_computable_on_canary_count']} de {s['score_v6_on_canary_rows_count']}.",
        "- score_v6_status = not_computable_pre_event_features_unavailable. Score v6 NAO recomputado, NAO alterado; score v7 NAO criado.",
        "",
        "## Resposta observacional",
        s["observational_answer"],
        "",
        "## Guardrails",
        "- Patch original preservado (vira spatial_mismatch_control_candidate); patches novos em namespace proprio; nenhum vira ground truth/treino/substitui SUSC original; score v6 intacto; score v7 inexistente; ocorrencia sem hidrologico nao entra; centroide de bairro nao usado sem incerteza (ponto geocodificado + incerteza_m); ausencia de ocorrencia NAO vira negativo verdadeiro.",
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
    write_csv(REGISTRY, event_anchored_canary_patch_rows())
    write_json(GEOJSON, build_geojson())
    write_csv(MISMATCH_CONTROL, original_patch_mismatch_control_rows())
    write_csv(FEATURE_BINDING, pre_event_feature_binding_rows())
    write_csv(SCORE_ON_CANARY, score_v6_on_canary_patch_rows())
    write_csv(CONTRAST, observational_contrast_matrix_rows())
    write_json(SUMMARY, build_summary())
    write_markdown(REPORT, build_report())


def run_mode(mode: str) -> None:
    _require_inputs()
    ensure_dir(OUT)
    if mode == "build-event-anchored-canary-patches":
        write_csv(REGISTRY, event_anchored_canary_patch_rows())
        write_json(GEOJSON, build_geojson())
    elif mode == "build-original-patch-mismatch-control":
        write_csv(MISMATCH_CONTROL, original_patch_mismatch_control_rows())
    elif mode == "bind-pre-event-features":
        write_csv(FEATURE_BINDING, pre_event_feature_binding_rows())
    elif mode == "associate-score-v6":
        write_csv(SCORE_ON_CANARY, score_v6_on_canary_patch_rows())
    elif mode == "build-observational-contrast":
        write_csv(CONTRAST, observational_contrast_matrix_rows())
    elif mode == "build-offline":
        build_all()
    elif mode == "audit":
        raise SystemExit(validate())
    else:
        raise SystemExit(f"unknown mode: {mode}")


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
            violations.append(f"{field}:const")
        if "pattern" in rules and not value.startswith(rules["pattern"].replace("^", "")):
            violations.append(f"{field}:pattern")
    return violations


def _validate_byte_identical() -> list[str]:
    before = {p: p.read_bytes() for p in REQUIRED_OUTPUTS if p.exists()}
    build_all()
    errors = []
    for p, content in before.items():
        if p.exists() and p.read_bytes() != content:
            errors.append(f"offline_build_not_byte_identical:{rel(p)}")
    return errors


def validate() -> int:
    missing = [p for p in REQUIRED_OUTPUTS if not p.exists()]
    if missing:
        print("MISSING: " + "; ".join(rel(p) for p in missing), file=sys.stderr)
        return 1
    errors = _validate_byte_identical()

    registry = read_csv(REGISTRY)
    control = read_csv(MISMATCH_CONTROL)
    binding = read_csv(FEATURE_BINDING)
    score_rows = read_csv(SCORE_ON_CANARY)
    contrast = read_csv(CONTRAST)
    summary = read_json(SUMMARY)
    geojson = read_json(GEOJSON)
    eac_schema = read_json(SCHEMA_EAC)

    for row in registry:
        errors.extend(_schema_violations(row, eac_schema))

    # minimos
    if len(registry) < 3:
        errors.append("canary_patches_lt_3")
    if len(control) < 1:
        errors.append("no_mismatch_control")
    if len(binding) < 3:
        errors.append("feature_binding_lt_3")
    if len(score_rows) < 3:
        errors.append("score_rows_lt_3")
    if len(contrast) < 4:
        errors.append("contrast_lt_4")

    # namespace proprio; nao substitui original; nao GT/treino
    original_ids = {r["candidate_patch_id"] for r in read_csv(PATCH_GRID)}
    for r in registry:
        if r["namespace"] != NAMESPACE:
            errors.append(f"wrong_namespace:{r['event_anchored_canary_patch_id']}")
        if r["event_anchored_canary_patch_id"] in original_ids:
            errors.append(f"canary_overwrites_original:{r['event_anchored_canary_patch_id']}")
        if r["replaces_original_patch"] != "false":
            errors.append("canary_replaces_original")
        if r["is_ground_truth"] != "false" or r["is_training"] != "false":
            errors.append(f"canary_gt_or_training:{r['event_anchored_canary_patch_id']}")
        if r["hydrological_signal"] != "true":
            errors.append(f"non_hydrological_canary:{r['event_anchored_canary_patch_id']}")
        if int(r["uncertainty_m"]) < 300:
            errors.append(f"canary_without_uncertainty:{r['event_anchored_canary_patch_id']}")

    # patch original preservado como controle; ausencia != negativo
    for r in control:
        if r["control_role"] != "spatial_mismatch_control_candidate":
            errors.append("original_not_marked_control")
        if r["absence_of_occurrence_is_true_negative"] != "false":
            errors.append("absence_marked_true_negative")
        if r["replaces_by_canary"] != "false":
            errors.append("original_replaced_by_canary")
    control_ids = {r["original_patch_id"] for r in control}
    if control_ids != original_ids:
        errors.append("control_does_not_cover_all_original_patches")

    # score v6 nao computado/alterado; score v7 nao criado
    for r in score_rows:
        if r["score_v6_recomputed"] != "false":
            errors.append("score_v6_recomputed")
        if r["score_v6_source_dataset_altered"] != "false":
            errors.append("score_v6_source_altered")
        if r["score_v7_created"] != "false":
            errors.append("score_v7_created_flag")
        if r["is_ground_truth"] != "false" or r["is_training"] != "false":
            errors.append("score_row_gt_or_training")
    if SCORE_V7.exists():
        errors.append("score_v7_exists")
    if _run_git(["diff", "--name-only", "--", rel(SCORE_V6)]):
        errors.append("score_v6_dataset_changed")

    # ocorrencia sem hidrologico nao entra como canario positivo (checado acima)
    # centroide de bairro nao usado sem incerteza: cada canario tem uncertainty_m e ponto geocodificado
    for r in registry:
        if r["link_type_to_occurrence"] != "official_hydrological_occurrence_geocoded_street_level":
            errors.append(f"canary_not_geocoded_point:{r['event_anchored_canary_patch_id']}")
        if r["geocoder_is_official"] != "false":
            errors.append("geocoder_marked_official")

    # geojson consistente
    if geojson.get("type") != "FeatureCollection" or not geojson.get("features"):
        errors.append("geojson_invalid")

    # summary guardrails
    if summary["original_patch_overwritten"] or summary["canary_is_ground_truth"] or summary["canary_is_training"]:
        errors.append("summary_guardrail_violated")
    if summary["score_v6_changed"] or summary["score_v7_created"]:
        errors.append("summary_score_guardrail_violated")
    if summary["eligible_for_17b_now"]:
        errors.append("17b_eligible")

    # summary reflete contagens reais
    expected = build_summary()
    for k, v in expected.items():
        if summary.get(k) != v:
            errors.append(f"summary_mismatch:{k}")
    if not summary["minimum_success_achieved"]:
        errors.append("minimum_success_not_achieved")

    # ids unicos/ordenados
    for rows, key in [
        (registry, "event_anchored_canary_patch_id"), (control, "original_patch_id"),
        (binding, "feature_binding_id"), (score_rows, "score_v6_on_canary_id"),
        (contrast, "contrast_id"),
    ]:
        ids = [row[key] for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(
        "17C33 -> "
        f"canary={summary['event_anchored_canary_patches_count']} control={summary['original_patches_as_mismatch_control_count']} "
        f"binding={summary['pre_event_feature_binding_rows_count']} score_computable={summary['score_v6_computable_on_canary_count']} "
        f"contrast={summary['observational_contrast_rows_count']} nearest_m={summary['nearest_canary_to_original_m']} "
        f"result={summary['result_class']} min_success={summary['minimum_success_achieved']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="susc_17c33")
    parser.add_argument("mode", choices=[
        "build-event-anchored-canary-patches", "build-original-patch-mismatch-control",
        "bind-pre-event-features", "associate-score-v6", "build-observational-contrast",
        "build-offline", "audit",
    ])
    args = parser.parse_args(argv)
    if args.mode == "audit":
        return validate()
    run_mode(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
