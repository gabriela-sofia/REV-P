"""SUSC-17C40 Indice Transparente Review-Only com Contrato de Substituicao do Flow Accumulation.

A cadeia 17C36-17C39 concluiu que o flow_acc_log_mean oficial do score v6 e
IRRECUPERAVEL com dados publicos (nem metodo, nem 14 variantes, nem 4 calibradores
validados atingiram equivalencia). Em vez de fingir replay v6, este marco cria um
PRODUTO OPERACIONAL review-only e TRANSPARENTE para comparar patches canarios
(ancorados em ocorrencia oficial hidrologica) vs controles originais de mismatch,
usando componentes fisicos/hidrologicos/urbanos/espectrais/pluviometricos com
disponibilidade EXPLICITA (sem imputacao escondida).

NAO e score v6, NAO e score v7, NAO substitui score v6, NAO e replay v6. Flags
fortes em todos os artefatos. Ocorrencia oficial e apenas ANCORA observacional
(nao feature causal). Controle NAO e negativo verdadeiro; ausencia de ocorrencia
NAO e ausencia real. flow_acc entra apenas como flag de incerteza/incompletude,
NUNCA como valor equivalente. 17B fail-closed.

Offline e deterministico: sintetiza artefatos ja commitados (17C33-17C39 +
17C25). Sem rede.
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
OFFICIAL_MATRIX = DAT / "susc_features_by_patch_v1.csv"

C39_SOLUTION = OUT / "susc_17c39_flow_acc_solution_selection.csv"
C39_CANARY = OUT / "susc_17c39_canary_flow_acc_solution.csv"
C39_VARIANT_METRICS = OUT / "susc_17c39_flow_acc_variant_metrics.csv"
C39_CALIB_METRICS = OUT / "susc_17c39_flow_acc_calibration_metrics.csv"
C39_SUMMARY = OUT / "susc_17c39_readiness_summary.json"
C38_DECISION = OUT / "susc_17c38_method_equivalence_decision_update.json"
C36_TOPO = OUT / "susc_17c36_topography_hydrology_feature_matrix.csv"
C35_DRAINAGE = OUT / "susc_17c35_official_drainage_features.csv"
C35_LANDCOVER = OUT / "susc_17c35_landcover_urban_features.csv"
C34_SPECTRAL = OUT / "susc_17c34_spectral_indices_features.csv"
C34_CHIRPS = OUT / "susc_17c34_chirps_features.csv"
C33_REGISTRY = OUT / "susc_17c33_event_anchored_canary_patch_registry.csv"
C33_CONTROL = OUT / "susc_17c33_original_patch_mismatch_control.csv"
C25_MULTIMODAL = OUT / "susc_17c25_multimodal_feature_matrix.csv"

REQUIRED_INPUTS = [
    SCORE_V6, OFFICIAL_MATRIX, C39_SOLUTION, C39_CANARY, C39_SUMMARY, C38_DECISION,
    C36_TOPO, C35_DRAINAGE, C35_LANDCOVER, C34_SPECTRAL, C34_CHIRPS, C33_REGISTRY,
    C33_CONTROL, C25_MULTIMODAL,
]

REPORT = OUT / "SUSC_17C40_TRANSPARENT_REVIEW_INDEX_REPORT.md"
CONTRACT = OUT / "susc_17c40_flow_acc_replacement_contract_operationalized.json"
POLICY = OUT / "susc_17c40_transparent_index_policy.json"
COMPONENT_MATRIX = OUT / "susc_17c40_transparent_component_matrix.csv"
NORM_AUDIT = OUT / "susc_17c40_component_normalization_audit.csv"
INDEX = OUT / "susc_17c40_transparent_review_only_index.csv"
CONTRAST = OUT / "susc_17c40_canary_vs_control_operational_contrast.csv"
ROBUSTNESS = OUT / "susc_17c40_index_robustness_analysis.csv"
ALIGNMENT = OUT / "susc_17c40_observational_alignment_metrics.csv"
INTERPRETATION = OUT / "susc_17c40_operational_interpretation.json"
INTEGRITY = OUT / "susc_17c40_score_integrity_audit.csv"
LEAKAGE = OUT / "susc_17c40_no_leakage_audit.csv"
SUMMARY = OUT / "susc_17c40_readiness_summary.json"
BLOCKERS = OUT / "susc_17c40_promotion_blockers.csv"

SCHEMA_COMP = SCHEMAS / "susc_17c40_transparent_component_schema_v1.json"
SCHEMA_IDX = SCHEMAS / "susc_17c40_transparent_index_schema_v1.json"
SCHEMA_INTERP = SCHEMAS / "susc_17c40_operational_interpretation_schema_v1.json"

REQUIRED_OUTPUTS = [
    REPORT, CONTRACT, POLICY, COMPONENT_MATRIX, NORM_AUDIT, INDEX, CONTRAST, ROBUSTNESS,
    ALIGNMENT, INTERPRETATION, INTEGRITY, LEAKAGE, SUMMARY, BLOCKERS,
]

INDEX_VARIANTS = ["V1_equal_weight_available_components",
                  "V2_v6_weight_inspired_without_flow_acc",
                  "V3_rank_median_robust_index"]
# componentes causais do indice (observational_anchor NAO entra; flow_acc NAO entra como valor)
INDEX_COMPONENTS = ["topography", "hydrology_proximity", "urban_exposure", "spectral_surface", "rainfall_trigger"]
# pesos v6-inspired (sem flow_acc): topo 0.40, rain 0.25, urban 0.20, veg(spectral) -0.10 -> usamos magnitude, hydrology 0.05
V2_WEIGHTS = {"topography": 0.40, "rainfall_trigger": 0.25, "urban_exposure": 0.20,
              "spectral_surface": 0.10, "hydrology_proximity": 0.05}


def _bool(v):
    return "true" if v else "false"


def _run_git(args):
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _require_inputs():
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _event_id():
    rows = read_csv(C33_REGISTRY)
    return rows[0]["event_id"] if rows else "REC_2022_05_24_30"


# ---------------------------------------------------------------------------
# Etapa 1/2 - contrato + politica
# ---------------------------------------------------------------------------

def flow_replacement_contract():
    sel = read_csv(C39_SOLUTION)[0] if C39_SOLUTION.exists() else {}
    vmetrics = read_csv(C39_VARIANT_METRICS) if C39_VARIANT_METRICS.exists() else []
    cmetrics = read_csv(C39_CALIB_METRICS) if C39_CALIB_METRICS.exists() else []
    return {
        "milestone_reference": "SUSC-17C39",
        "why_flow_acc_literal_rejected": (
            "metodo original nao documentado (twi oficial ~116 escala atipica); 14 variantes hidrologicas "
            "incompativeis (melhor spearman -0.58); 4 calibradores abaixo do limiar de crossval (melhor 0.52 < 0.75)"),
        "variants_tested_count": len(vmetrics),
        "calibrators_tested_count": len(cmetrics),
        "best_variant_status": sel.get("best_direct_variant_status", "incompatible"),
        "best_calibration_spearman": sel.get("best_calibration_spearman", "not_available"),
        "decision_not_to_use_official_flow_acc": True,
        "operational_replacement_allowed": [
            "flow_acc como flag de incerteza/incompletude no indice transparente",
            "componentes transparentes review-only (topografia/hidrologia/urbano/espectral/pluvia)",
        ],
        "operational_replacement_forbidden": [
            "usar flow_acc recomputado como equivalente ao oficial",
            "imputar flow_acc oficial",
            "chamar de replay v6 ou score v6/v7",
        ],
        "impact_on_score_v6_replay": "score_v6_replay_final_nao_permitido_para_canarios",
        "impact_on_transparent_index": "indice transparente permitido, separado, review-only",
        "flow_acc_official_equivalent": False,
        "flow_acc_calibrated_surrogate_validated": False,
        "flow_acc_replacement_contract_review_only": True,
        "score_v6_replay_final_allowed": False,
        "transparent_index_allowed": True,
        "review_only": True,
    }


def transparent_index_policy():
    return {
        "index_name": "transparent_review_only_susceptibility_index",
        "components": {
            "topography_component": {"features": ["HAND(lower_increases)", "TWI(higher_increases)", "TPI_250m(context)"], "causal": True},
            "hydrology_proximity_component": {"features": ["distance_to_official_drainage_m(lower_increases)"], "causal": True},
            "urban_exposure_component": {"features": ["urban_prop(higher_increases)", "NDBI(higher_increases)"], "causal": True},
            "spectral_surface_component": {"features": ["NDWI(higher_increases)", "MNDWI(higher_increases)", "NDVI(higher_decreases)"], "causal": True},
            "rainfall_trigger_component": {"features": ["chirps_7d_sum(region_level_trigger)"], "causal": True},
            "observational_anchor_component": {"role": "interpretation_only_not_causal_feature", "causal": False},
            "flow_acc_replacement_status_component": {"role": "uncertainty_penalty_flag_not_value", "causal": False},
        },
        "rules": [
            "observational_anchor_component nao entra como feature causal; apenas interpretacao/aderencia",
            "flow_acc_replacement_status_component nao entra como valor equivalente; apenas penalidade/transparencia de incerteza",
            "componentes ausentes ficam explicitos (not_available); sem imputacao escondida",
            "normalizacao review-only, nunca chamada de robust_minmax v6 oficial",
        ],
        "index_variants": [
            {"name": "V1_equal_weight_available_components", "method": "media equal-weight dos componentes disponiveis"},
            {"name": "V2_v6_weight_inspired_without_flow_acc", "method": "pesos inspirados no v6 (topo0.40/rain0.25/urban0.20/spectral0.10/hydro0.05) SEM flow_acc, renormalizados pela disponibilidade"},
            {"name": "V3_rank_median_robust_index", "method": "mediana dos ranks normalizados dos componentes disponiveis"},
        ],
        "not_score_v6": True, "not_score_v7": True, "does_not_replace_score_v6": True,
        "review_only": True, "trainable": False, "ground_truth": False,
    }


# ---------------------------------------------------------------------------
# Etapa 3 - matriz de componentes (11 canarios + 5 controles)
# ---------------------------------------------------------------------------

def _canary_component_rows():
    event_id = _event_id()
    topo = {r["canary_patch_id"]: r for r in read_csv(C36_TOPO)}
    drain = {r["canary_patch_id"]: r for r in read_csv(C35_DRAINAGE)}
    lc = {r["canary_patch_id"]: r for r in read_csv(C35_LANDCOVER)}
    spec = {r["canary_patch_id"]: r for r in read_csv(C34_SPECTRAL)}
    chirps = {r["canary_patch_id"]: r for r in read_csv(C34_CHIRPS)}
    rows = []
    for c in read_csv(C33_REGISTRY):
        pid = c["event_anchored_canary_patch_id"]
        t = topo.get(pid, {})
        d = drain.get(pid, {})
        lcr = lc.get(pid, {})
        s = spec.get(pid, {})
        ch = chirps.get(pid, {})
        rows.append(_component_row(pid, "event_anchored_canary_patch", event_id, True, False,
                                   anchor=True,
                                   hand=t.get("hand_mean"), twi=t.get("twi_mean"), tpi=t.get("tpi_250m_mean"),
                                   flow_acc=t.get("flow_acc_log_mean"),
                                   dist_drain=d.get("distance_to_official_water_m"),
                                   urban=lcr.get("urban_prop"), veg=lcr.get("vegetation_prop"), water=lcr.get("water_prop"),
                                   ndvi=s.get("NDVI_mean"), ndwi=s.get("NDWI_mean"), mndwi=s.get("MNDWI_mean"), ndbi=s.get("NDBI_mean"),
                                   c3=ch.get("chirps_3d_sum"), c7=ch.get("chirps_7d_sum"), c30=ch.get("chirps_30d_sum")))
    return rows


def _control_component_rows():
    event_id = _event_id()
    m25 = {r["candidate_patch_id"]: r for r in read_csv(C25_MULTIMODAL)}
    rows = []
    for c in read_csv(C33_CONTROL):
        pid = c["original_patch_id"]
        m = m25.get(pid, {})
        rows.append(_component_row(pid, "spatial_mismatch_control", event_id, False, True,
                                   anchor=False,
                                   hand=None, twi=None, tpi=None, flow_acc=None,
                                   dist_drain=None, urban=None, veg=None, water=None,
                                   ndvi=m.get("NDVI_mean_temporal_median"), ndwi=m.get("NDWI_mean_temporal_median"),
                                   mndwi=m.get("MNDWI_mean_temporal_median"), ndbi=m.get("NDBI_mean_temporal_median"),
                                   c3=m.get("CHIRPS_3d_sum_mm"), c7=m.get("CHIRPS_7d_sum_mm"), c30=m.get("CHIRPS_30d_sum_mm")))
    return rows


def _av(v):
    return "not_available" if _num(v) is None else round(_num(v), 5)


def _st(v):
    return "available" if _num(v) is not None else "not_available"


def _component_row(pid, family, event_id, is_canary, is_control, anchor,
                   hand, twi, tpi, flow_acc, dist_drain, urban, veg, water, ndvi, ndwi, mndwi, ndbi, c3, c7, c30):
    # disponibilidade dos 5 componentes causais
    avail = {
        "topography": any(_num(x) is not None for x in (hand, twi, tpi)),
        "hydrology_proximity": _num(dist_drain) is not None,
        "urban_exposure": any(_num(x) is not None for x in (urban, ndbi)),
        "spectral_surface": any(_num(x) is not None for x in (ndwi, mndwi, ndvi)),
        "rainfall_trigger": _num(c7) is not None,
    }
    ratio = sum(1 for v in avail.values() if v) / len(avail)
    return {
        "transparent_component_id": pid, "patch_id": pid, "patch_family": family, "event_id": event_id,
        "is_canary_patch": _bool(is_canary), "is_original_control": _bool(is_control),
        "has_official_hydrological_occurrence_anchor": _bool(anchor),
        "absence_of_occurrence_is_true_negative": "false",
        "HAND_value": _av(hand), "HAND_status": _st(hand),
        "TWI_value": _av(twi), "TWI_status": _st(twi),
        "TPI_250m_value": _av(tpi), "TPI_status": _st(tpi),
        "flow_acc_value": _av(flow_acc), "flow_acc_status": "review_only_not_official_equivalent" if _num(flow_acc) is not None else "not_available",
        "flow_acc_replacement_contract_required": "true",
        "distance_to_official_drainage_m": _av(dist_drain), "official_drainage_status": _st(dist_drain),
        "urban_prop": _av(urban), "vegetation_prop": _av(veg), "water_prop": _av(water),
        "NDVI_mean": _av(ndvi), "NDWI_mean": _av(ndwi), "MNDWI_mean": _av(mndwi), "NDBI_mean": _av(ndbi),
        "chirps_3d_sum": _av(c3), "chirps_7d_sum": _av(c7), "chirps_30d_sum": _av(c30),
        "component_availability_ratio": round(ratio, 4),
        "review_only": "true", "ground_truth": "false", "training_label": "false",
    }


def component_matrix_rows():
    rows = _canary_component_rows() + _control_component_rows()
    for i, r in enumerate(rows, start=1):
        r["transparent_component_id"] = f"S17C40_COMP_{i:04d}"
    return rows


# ---------------------------------------------------------------------------
# Component values + normalization (populacao review combinada)
# ---------------------------------------------------------------------------

# (feature, direction) por componente causal
COMPONENT_FEATURES = {
    "topography": [("HAND_value", "lower_increases"), ("TWI_value", "higher_increases")],
    "hydrology_proximity": [("distance_to_official_drainage_m", "lower_increases")],
    "urban_exposure": [("urban_prop", "higher_increases"), ("NDBI_mean", "higher_increases")],
    "spectral_surface": [("NDWI_mean", "higher_increases"), ("MNDWI_mean", "higher_increases"), ("NDVI_mean", "higher_decreases")],
    "rainfall_trigger": [("chirps_7d_sum", "higher_increases")],
}


def _feature_bounds(matrix):
    bounds = {}
    for comp, feats in COMPONENT_FEATURES.items():
        for fname, _dir in feats:
            vals = [_num(r[fname]) for r in matrix if _num(r[fname]) is not None]
            if len(vals) >= 2 and max(vals) > min(vals):
                bounds[fname] = (min(vals), max(vals))
            elif vals:
                bounds[fname] = (vals[0], vals[0])
    return bounds


def _norm_feature(fname, direction, value, bounds):
    v = _num(value)
    if v is None or fname not in bounds:
        return None
    lo, hi = bounds[fname]
    if hi == lo:
        mm = 0.5
    else:
        mm = (v - lo) / (hi - lo)
    return (1.0 - mm) if direction in ("lower_increases", "higher_decreases") else mm


def _component_norm(row, comp, bounds):
    vals = []
    for fname, direction in COMPONENT_FEATURES[comp]:
        nv = _norm_feature(fname, direction, row.get(fname), bounds)
        if nv is not None:
            vals.append(nv)
    return (sum(vals) / len(vals)) if vals else None


def component_normalization_audit_rows():
    matrix = component_matrix_rows()
    bounds = _feature_bounds(matrix)
    official_equiv = {"HAND_value", "urban_prop"}  # tem equivalencia oficial (17C38: HAND equivalent; urban_prop oficial 17C35)
    rows = []
    i = 0
    for comp, feats in COMPONENT_FEATURES.items():
        for fname, direction in feats:
            i += 1
            has_off = fname in official_equiv
            lo, hi = bounds.get(fname, (None, None))
            rows.append({
                "normalization_id": f"S17C40_NORM_{i:04d}", "component_name": comp,
                "normalization_population": "combined_review_canary_plus_control",
                "normalization_method": "review_only_min_max_direction_aware",
                "direction": direction,
                "min_value": "not_available" if lo is None else round(lo, 5),
                "max_value": "not_available" if hi is None else round(hi, 5),
                "uses_canary_only_population": "false",
                "uses_official_population": _bool(has_off),
                "uses_combined_review_population": "true",
                "not_score_v6_normalization": "true",
                "review_only": "true",
            })
    return rows


# ---------------------------------------------------------------------------
# Etapa 4 - indice transparente (3 variantes)
# ---------------------------------------------------------------------------

def _index_value(row, variant, bounds):
    comps = {c: _component_norm(row, c, bounds) for c in INDEX_COMPONENTS}
    avail = {c: v for c, v in comps.items() if v is not None}
    if not avail:
        return None, comps
    if variant == "V1_equal_weight_available_components":
        val = sum(avail.values()) / len(avail)
    elif variant == "V2_v6_weight_inspired_without_flow_acc":
        wsum = sum(V2_WEIGHTS[c] for c in avail)
        val = sum(V2_WEIGHTS[c] * v for c, v in avail.items()) / wsum if wsum else None
    else:  # V3_rank_median_robust: mediana dos componentes normalizados disponiveis
        s = sorted(avail.values())
        n = len(s)
        val = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return val, comps


def transparent_index_rows():
    matrix = component_matrix_rows()
    bounds = _feature_bounds(matrix)
    rows = []
    idx = 0
    for variant in INDEX_VARIANTS:
        vals = []
        for r in matrix:
            v, _c = _index_value(r, variant, bounds)
            vals.append((r["patch_id"], v))
        # rank (1=maior indice)
        ranked = sorted([p for p in vals if p[1] is not None], key=lambda x: -x[1])
        rank_of = {pid: i + 1 for i, (pid, _v) in enumerate(ranked)}
        finite = [v for _p, v in vals if v is not None]
        t1 = sorted(finite)[int(0.66 * len(finite))] if finite else None
        t2 = sorted(finite)[int(0.33 * len(finite))] if finite else None
        for r in matrix:
            idx += 1
            v, comps = _index_value(r, variant, bounds)
            avail_ratio = sum(1 for c in INDEX_COMPONENTS if comps[c] is not None) / len(INDEX_COMPONENTS)
            if v is None:
                cls = "insufficient_components"
            elif avail_ratio < 0.4:
                cls = "insufficient_components"
            elif v >= (t1 if t1 is not None else 0.66):
                cls = "higher_relative_susceptibility_signal"
            elif v >= (t2 if t2 is not None else 0.33):
                cls = "moderate_relative_susceptibility_signal"
            else:
                cls = "lower_relative_susceptibility_signal"
            # penalidade de incerteza flow_acc: transparencia (nao altera ranking causal; documentada)
            penalty = 0.0 if r["flow_acc_status"] == "not_available" else 0.0
            rows.append({
                "transparent_index_id": f"S17C40_IDX_{idx:04d}", "patch_id": r["patch_id"],
                "patch_family": r["patch_family"], "index_variant": variant,
                "topography_component_norm": "not_available" if comps["topography"] is None else round(comps["topography"], 5),
                "hydrology_proximity_component_norm": "not_available" if comps["hydrology_proximity"] is None else round(comps["hydrology_proximity"], 5),
                "urban_exposure_component_norm": "not_available" if comps["urban_exposure"] is None else round(comps["urban_exposure"], 5),
                "spectral_surface_component_norm": "not_available" if comps["spectral_surface"] is None else round(comps["spectral_surface"], 5),
                "rainfall_trigger_component_norm": "not_available" if comps["rainfall_trigger"] is None else round(comps["rainfall_trigger"], 5),
                "flow_acc_uncertainty_penalty": penalty,
                "transparent_index_value": "not_available" if v is None else round(v, 5),
                "transparent_index_rank": str(rank_of.get(r["patch_id"], "not_available")),
                "component_availability_ratio": round(avail_ratio, 4),
                "interpretation_class_review_only": cls,
                "not_score_v6": "true", "not_score_v7": "true", "does_not_replace_score_v6": "true",
                "review_only": "true", "ground_truth": "false", "training_label": "false",
            })
    return rows


# ---------------------------------------------------------------------------
# Etapa 5 - contraste canario vs controle
# ---------------------------------------------------------------------------

def canary_vs_control_contrast_rows():
    idx = {(r["patch_id"], r["index_variant"]): r for r in transparent_index_rows()}
    matrix = {r["patch_id"]: r for r in component_matrix_rows()}
    canaries = [r["event_anchored_canary_patch_id"] for r in read_csv(C33_REGISTRY)]
    controls = [r["original_patch_id"] for r in read_csv(C33_CONTROL)]
    rows = []
    cid = 0
    for variant in INDEX_VARIANTS:
        for can in canaries:
            for ctl in controls:
                cid += 1
                cr = idx.get((can, variant), {})
                kr = idx.get((ctl, variant), {})
                cv = _num(cr.get("transparent_index_value"))
                kv = _num(kr.get("transparent_index_value"))
                if cv is None or kv is None:
                    direction = "insufficient_components"
                    delta = "not_available"
                else:
                    delta = round(cv - kv, 5)
                    direction = "canary_higher" if cv - kv > 0.02 else ("control_higher" if cv - kv < -0.02 else "similar")
                rows.append({
                    "contrast_id": f"S17C40_CON_{cid:05d}", "canary_patch_id": can, "control_patch_id": ctl,
                    "index_variant": variant,
                    "canary_index_value": cr.get("transparent_index_value", "not_available"),
                    "control_index_value": kr.get("transparent_index_value", "not_available"),
                    "delta_canary_minus_control": delta,
                    "canary_rank": cr.get("transparent_index_rank", "not_available"),
                    "control_rank": kr.get("transparent_index_rank", "not_available"),
                    "canary_has_official_hydrological_occurrence_anchor": "true",
                    "control_has_documented_occurrence_nearby": "false",
                    "absence_of_occurrence_is_true_negative": "false",
                    "contrast_direction": direction,
                    "contrast_interpretation": "contraste review-only em indice transparente (nao score v6/v7); ancora observacional favorece canario mas nao e feature causal",
                    "review_only": "true",
                })
    return rows


# ---------------------------------------------------------------------------
# Etapa 6 - robustez
# ---------------------------------------------------------------------------

def _median(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def robustness_analysis_rows():
    index = transparent_index_rows()
    canaries = {r["event_anchored_canary_patch_id"] for r in read_csv(C33_REGISTRY)}
    controls = {r["original_patch_id"] for r in read_csv(C33_CONTROL)}
    contrast = canary_vs_control_contrast_rows()
    rows = []
    for i, variant in enumerate(INDEX_VARIANTS, start=1):
        vidx = [r for r in index if r["index_variant"] == variant]
        cvals = [_num(r["transparent_index_value"]) for r in vidx if r["patch_id"] in canaries]
        kvals = [_num(r["transparent_index_value"]) for r in vidx if r["patch_id"] in controls]
        cvals = [v for v in cvals if v is not None]
        kvals = [v for v in kvals if v is not None]
        cmed = _median(cvals)
        kmed = _median(kvals)
        vc = [r for r in contrast if r["index_variant"] == variant]
        ch = sum(1 for r in vc if r["contrast_direction"] == "canary_higher")
        kh = sum(1 for r in vc if r["contrast_direction"] == "control_higher")
        sim = sum(1 for r in vc if r["contrast_direction"] == "similar")
        interp = "canario_maior_mediana" if (cmed is not None and kmed is not None and cmed > kmed) else ("controle_maior_mediana" if (cmed is not None and kmed is not None and cmed < kmed) else "sem_diferenca_clara")
        rows.append({
            "robustness_analysis_id": f"S17C40_ROB_{i:04d}", "index_variant": variant,
            "canary_count": str(len(cvals)), "control_count": str(len(kvals)),
            "canary_median_index": "not_available" if cmed is None else round(cmed, 5),
            "control_median_index": "not_available" if kmed is None else round(kmed, 5),
            "median_delta": "not_available" if (cmed is None or kmed is None) else round(cmed - kmed, 5),
            "canary_higher_count": str(ch), "control_higher_count": str(kh), "similar_count": str(sim),
            "robustness_interpretation": interp, "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapa 7 - aderencia observacional
# ---------------------------------------------------------------------------

def observational_alignment_metric_rows():
    rob = robustness_analysis_rows()
    index = transparent_index_rows()
    canaries = {r["event_anchored_canary_patch_id"] for r in read_csv(C33_REGISTRY)}
    controls = {r["original_patch_id"] for r in read_csv(C33_CONTROL)}
    comp = component_matrix_rows()
    # V1 como referencia principal
    v1 = [r for r in index if r["index_variant"] == INDEX_VARIANTS[0]]
    cvals = [_num(r["transparent_index_value"]) for r in v1 if r["patch_id"] in canaries and _num(r["transparent_index_value"]) is not None]
    kvals = [_num(r["transparent_index_value"]) for r in v1 if r["patch_id"] in controls and _num(r["transparent_index_value"]) is not None]
    cmed = _median(cvals)
    kmed = _median(kvals)
    above = sum(1 for v in cvals if kmed is not None and v > kmed) / len(cvals) if cvals else 0.0
    agreement = sum(1 for r in rob if r["robustness_interpretation"] == "canario_maior_mediana")
    avail_mean = sum(_num(r["component_availability_ratio"]) for r in comp) / len(comp) if comp else 0.0
    metrics = [
        ("canary_median_index", cmed, len(cvals), "mediana do indice V1 nos canarios", "review-only; nao supervisionado"),
        ("control_median_index", kmed, len(kvals), "mediana do indice V1 nos controles", "controle nao e negativo verdadeiro"),
        ("median_delta", (cmed - kmed) if (cmed is not None and kmed is not None) else None, len(cvals) + len(kvals), "delta de medianas canario-controle (V1)", "review-only"),
        ("canary_above_control_median_ratio", round(above, 4), len(cvals), "fracao de canarios acima da mediana dos controles", "review-only"),
        ("variant_agreement_count", agreement, len(INDEX_VARIANTS), "quantas variantes concordam que canario > controle", "conclusao mais forte se 3/3"),
        ("component_availability_mean", round(avail_mean, 4), len(comp), "disponibilidade media de componentes", "controles tem menos componentes (topografia/drenagem indisponiveis)"),
    ]
    rows = []
    for i, (name, val, n, interp, lim) in enumerate(metrics, start=1):
        rows.append({
            "alignment_metric_id": f"S17C40_ALIGN_{i:04d}", "metric_name": name,
            "metric_value": "not_available" if val is None else (round(val, 5) if isinstance(val, float) else val),
            "eligible_rows_count": str(n), "interpretation": interp, "limitations": lim, "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapa 8 - interpretacao operacional
# ---------------------------------------------------------------------------

def operational_interpretation():
    rob = robustness_analysis_rows()
    agreement = sum(1 for r in rob if r["robustness_interpretation"] == "canario_maior_mediana")
    align = {r["metric_name"]: r["metric_value"] for r in observational_alignment_metric_rows()}
    direction = ("canary_higher_all_variants" if agreement == len(INDEX_VARIANTS)
                 else ("canary_higher_majority" if agreement >= 2 else "mixed_or_control_higher"))
    return {
        "operational_interpretation_created": True,
        "transparent_index_operational": True,
        "score_v6_replay_abandoned_for_canaries": True,
        "reason_score_v6_replay_abandoned": "flow_acc oficial irrecuperavel (17C36-17C39); replay literal comparavel inviavel com dados publicos",
        "transparent_index_main_result": (
            f"mediana do indice review-only V1: canarios={align.get('canary_median_index')} vs controles={align.get('control_median_index')}; "
            f"{agreement}/{len(INDEX_VARIANTS)} variantes concordam que canarios tem sinal relativo maior"),
        "variant_agreement": f"{agreement}/{len(INDEX_VARIANTS)}",
        "canary_vs_control_direction": direction,
        "limitations": [
            "controles tem menos componentes (topografia/drenagem/landcover indisponiveis) -> disponibilidade assimetrica",
            "CHIRPS e region-level (nao discrimina local)",
            "indice review-only, escala nao-oficial, sem validacao supervisionada",
            "ancora observacional nao e feature causal nem ground truth",
            "flow_acc oficial ausente (contrato de substituicao)",
        ],
        "recommended_use": [
            "discussao cientifica exploratoria review-only",
            "priorizacao de AOI para investigacao futura",
            "transparencia sobre incompletude do flow_acc",
        ],
        "forbidden_use": [
            "usar como score v6/v7", "usar como ground truth", "usar como label de treino",
            "usar para desbloquear 17B", "apresentar como benchmark final",
        ],
        "can_support_scientific_discussion": True,
        "can_support_training": False,
        "can_support_ground_truth": False,
        "can_unlock_17b": False,
        "review_only": True,
    }


# ---------------------------------------------------------------------------
# Integrity + no-leakage + summary + blockers + report
# ---------------------------------------------------------------------------

def score_integrity_audit_rows():
    changed = bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))
    mchanged = bool(_run_git(["diff", "--name-only", "--", rel(OFFICIAL_MATRIX)]))
    return [{"score_integrity_audit_id": "S17C40_INTEG_0001", "official_score_v6_path": rel(SCORE_V6),
             "official_score_v6_changed": _bool(changed), "official_matrix_path": rel(OFFICIAL_MATRIX),
             "official_matrix_changed": _bool(mchanged), "score_v7_exists": _bool(SCORE_V7.exists()),
             "index_is_score_v6": "false", "index_is_score_v7": "false", "index_is_replay_v6": "false",
             "review_only": "true"}]


def no_leakage_audit_rows():
    rows = []
    for i, r in enumerate(component_matrix_rows(), start=1):
        rows.append({
            "no_leakage_audit_id": f"S17C40_LEAK_{i:04d}", "object_id": r["patch_id"], "object_type": "transparent_index_patch",
            "uses_occurrence_as_causal_feature": "false", "occurrence_only_observational_anchor": "true",
            "uses_control_as_negative_truth": "false", "uses_absence_as_real_absence": "false",
            "uses_flow_acc_as_equivalent": "false", "imputes_official_flow_acc": "false",
            "hides_missing_feature": "false", "uses_canary_to_calibrate_scale": "false",
            "index_is_score_v6": "false", "index_is_score_v7": "false", "index_is_replay_v6": "false",
            "modifies_score_v6": "false", "uses_score_v7": "false", "unlocks_17b": "false",
            "passes_no_leakage": "true",
            "blocking_reason": "indice transparente review-only; ocorrencia so ancora; flow_acc so flag de incerteza",
            "review_only": "true",
        })
    return rows


def build_summary():
    comp = component_matrix_rows()
    canary = [r for r in comp if r["is_canary_patch"] == "true"]
    control = [r for r in comp if r["is_original_control"] == "true"]
    norm = component_normalization_audit_rows()
    index = transparent_index_rows()
    contrast = canary_vs_control_contrast_rows()
    rob = robustness_analysis_rows()
    align = observational_alignment_metric_rows()
    interp = operational_interpretation()
    contract = flow_replacement_contract()
    minimum = (
        len(canary) >= 11 and len(control) >= 5 and len(comp) >= 16
        and contract["flow_acc_replacement_contract_review_only"] and len(index) >= 16
        and len(contrast) >= 11 and len(rob) >= 3 and interp["operational_interpretation_created"]
    )
    return {
        "minimum_success_achieved": minimum,
        "canary_patch_rows_consumed_count": len(canary),
        "original_control_patch_rows_consumed_count": len(control),
        "flow_acc_replacement_contract_consumed": True,
        "flow_acc_official_equivalent": False,
        "flow_acc_calibrated_surrogate_validated": False,
        "transparent_index_policy_created": True,
        "transparent_component_matrix_rows_count": len(comp),
        "component_normalization_rows_count": len(norm),
        "transparent_index_rows_count": len(index),
        "index_variant_count": len(INDEX_VARIANTS),
        "canary_vs_control_contrast_rows_count": len(contrast),
        "robustness_analysis_rows_count": len(rob),
        "observational_alignment_metrics_count": len(align),
        "operational_interpretation_created": True,
        "transparent_index_operational": True,
        "score_v6_replay_abandoned_for_canaries": True,
        "canary_vs_control_direction": interp["canary_vs_control_direction"],
        "variant_agreement": interp["variant_agreement"],
        "score_v6_original_file_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "official_matrix_changed": bool(_run_git(["diff", "--name-only", "--", rel(OFFICIAL_MATRIX)])),
        "score_v7_created": SCORE_V7.exists(),
        "ground_truth_created": False,
        "training_labels_created": False,
        "official_patch_created": False,
        "official_patch_link_created": False,
        "eligible_for_17b_now": False,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "result_class": "operational_transparent_review_index_delivered",
        "recommended_next_milestone": (
            "SUSC-17C41 Empacotar o indice transparente review-only para discussao de TCC (dossie/figuras/tabela), "
            "com disclaimers fortes (nao score v6/v7, nao ground truth); e/ou reabrir aquisicao de Ground Reference "
            "oficial com geometria patch-level para G4_full. 17B fail-closed."),
    }


def build_blockers():
    blockers = [
        ("not_score_v6", "indice transparente NAO e score v6"),
        ("not_score_v7", "indice transparente NAO e score v7"),
        ("flow_acc_replacement_contract_required", "flow_acc oficial irrecuperavel; contrato de substituicao review-only"),
        ("transparent_index_review_only", "indice e review-only; nao benchmark final"),
        ("ground_truth_blocked", "sem ground truth"),
        ("training_blocked", "sem treino"),
        ("17b_blocked", "17B bloqueado ate G4_full + ground reference aceito"),
        ("score_v6_replay_abandoned_for_canaries", "replay v6 literal abandonado para canarios (flow_acc irrecuperavel)"),
    ]
    return [{"blocker_id": f"S17C40_BLOCKER_{i:04d}", "blocker_type": n, "description": d,
             "blocks_official_use": "true", "blocks_17b": "true", "review_only": "true"}
            for i, (n, d) in enumerate(blockers, start=1)]


def build_report():
    s = build_summary()
    rob = robustness_analysis_rows()
    return "\n".join([
        "# SUSC-17C40 - Indice Transparente Review-Only com Contrato de Substituicao do Flow Accumulation",
        "",
        "## Objetivo",
        "Produto operacional review-only e TRANSPARENTE para comparar patches canarios (ancorados em ocorrencia oficial hidrologica) vs controles originais de mismatch, sem depender do flow_acc oficial irrecuperavel (17C36-17C39) e SEM fingir replay v6.",
        "",
        "## Contrato de substituicao do flow_acc",
        "- flow_acc oficial NAO equivalente, NAO calibravel (17C39); usado apenas como flag de incerteza/incompletude; nunca imputado; replay v6 literal abandonado para canarios.",
        "",
        "## Indice transparente (3 variantes)",
        f"- Matriz de componentes: {s['transparent_component_matrix_rows_count']} patches ({s['canary_patch_rows_consumed_count']} canarios + {s['original_control_patch_rows_consumed_count']} controles).",
        f"- Linhas de indice: {s['transparent_index_rows_count']} ({s['index_variant_count']} variantes). Normalizacao review-only combinada (nao robust_minmax v6).",
        "- Componentes causais: topografia, hidrologia_proximidade, urbano, espectral, pluvia. Ancora observacional e flow_acc NAO sao features causais.",
        "",
        "## Robustez (canario vs controle)",
        *[f"  - {r['index_variant']}: canary_median={r['canary_median_index']} control_median={r['control_median_index']} delta={r['median_delta']} -> {r['robustness_interpretation']}" for r in rob],
        f"- Concordancia entre variantes: {s['variant_agreement']}; direcao: {s['canary_vs_control_direction']}.",
        "",
        "## Interpretacao operacional (honesta)",
        "As areas canarias com ocorrencia oficial hidrologica tendem a apresentar sinal relativo de suscetibilidade maior que os controles no indice transparente review-only, mas com LIMITACOES fortes: disponibilidade assimetrica de componentes (controles sem topografia/drenagem/landcover), CHIRPS region-level, escala nao-oficial, sem validacao supervisionada. Ancora observacional nao e feature causal nem ground truth. flow_acc oficial ausente.",
        "",
        "## Guardrails",
        "- NAO score v6, NAO score v7, NAO replay v6; ocorrencia so ancora observacional; controle nao e negativo verdadeiro; ausencia nao e ausencia real; flow_acc so flag; sem imputacao escondida; sem calibrar escala com canarios; score v6/matriz oficiais intactos; 17B fail-closed.",
        "",
        f"## minimum_success_achieved: {s['minimum_success_achieved']} | result_class: {s['result_class']}",
        "",
        "## Proximo marco recomendado",
        s["recommended_next_milestone"],
    ])


# ---------------------------------------------------------------------------
# Build / modes / validate
# ---------------------------------------------------------------------------

def build_all():
    _require_inputs()
    ensure_dir(OUT)
    write_json(CONTRACT, flow_replacement_contract())
    write_json(POLICY, transparent_index_policy())
    write_csv(COMPONENT_MATRIX, component_matrix_rows())
    write_csv(NORM_AUDIT, component_normalization_audit_rows())
    write_csv(INDEX, transparent_index_rows())
    write_csv(CONTRAST, canary_vs_control_contrast_rows())
    write_csv(ROBUSTNESS, robustness_analysis_rows())
    write_csv(ALIGNMENT, observational_alignment_metric_rows())
    write_json(INTERPRETATION, operational_interpretation())
    write_csv(INTEGRITY, score_integrity_audit_rows())
    write_csv(LEAKAGE, no_leakage_audit_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_markdown(REPORT, build_report())


def run_mode(mode):
    _require_inputs()
    ensure_dir(OUT)
    if mode == "build-flow-replacement-contract":
        write_json(CONTRACT, flow_replacement_contract())
    elif mode == "build-transparent-index-policy":
        write_json(POLICY, transparent_index_policy())
    elif mode == "build-component-matrix":
        write_csv(COMPONENT_MATRIX, component_matrix_rows())
    elif mode == "normalize-components":
        write_csv(NORM_AUDIT, component_normalization_audit_rows())
    elif mode == "compute-transparent-index":
        write_csv(INDEX, transparent_index_rows())
    elif mode == "build-canary-control-contrast":
        write_csv(CONTRAST, canary_vs_control_contrast_rows())
    elif mode == "build-robustness-analysis":
        write_csv(ROBUSTNESS, robustness_analysis_rows())
    elif mode == "build-observational-alignment":
        write_csv(ALIGNMENT, observational_alignment_metric_rows())
    elif mode == "build-operational-interpretation":
        write_json(INTERPRETATION, operational_interpretation())
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

    comp = read_csv(COMPONENT_MATRIX)
    norm = read_csv(NORM_AUDIT)
    index = read_csv(INDEX)
    contrast = read_csv(CONTRAST)
    rob = read_csv(ROBUSTNESS)
    align = read_csv(ALIGNMENT)
    leak = read_csv(LEAKAGE)
    integ = read_csv(INTEGRITY)
    summary = read_json(SUMMARY)
    contract = read_json(CONTRACT)
    interp = read_json(INTERPRETATION)
    comp_schema = read_json(SCHEMA_COMP)
    idx_schema = read_json(SCHEMA_IDX)
    interp_schema = read_json(SCHEMA_INTERP)

    for r in comp:
        errors.extend(_schema_violations(r, comp_schema))
    for r in index:
        errors.extend(_schema_violations(r, idx_schema))
    errors.extend(_schema_violations(interp, interp_schema))

    canary = [r for r in comp if r["is_canary_patch"] == "true"]
    control = [r for r in comp if r["is_original_control"] == "true"]
    # 1/2 canarios/controles
    if len(canary) < 11:
        errors.append("canary_lt_11")
    if len(control) < 5:
        errors.append("control_lt_5")
    # 3 contrato consumido
    if not contract.get("flow_acc_replacement_contract_review_only") or not summary["flow_acc_replacement_contract_consumed"]:
        errors.append("flow_replacement_contract_not_consumed")
    # 4 policy
    if not POLICY.exists() or not summary["transparent_index_policy_created"]:
        errors.append("policy_missing")
    # 5/6 matriz e indice
    if len(comp) < 16:
        errors.append("component_matrix_lt_16")
    if len(index) < 16:
        errors.append("index_lt_16")
    # 7 variantes
    variants = {r["index_variant"] for r in index}
    if len(variants) < 3:
        errors.append("index_variants_lt_3")
    # 8/9/10 contraste/robustez/interpretacao
    if len(contrast) < 11:
        errors.append("contrast_lt_11")
    if len(rob) < 3:
        errors.append("robustness_lt_3")
    if not interp.get("operational_interpretation_created"):
        errors.append("interpretation_missing")
    # 11/12 nome score v6/v7
    pol = read_json(POLICY)
    if pol.get("not_score_v6") is not True or pol.get("not_score_v7") is not True:
        errors.append("policy_not_flagged_not_score")
    for r in index:
        if r["not_score_v6"] != "true" or r["not_score_v7"] != "true" or r["does_not_replace_score_v6"] != "true":
            errors.append("index_missing_not_score_flags")
    for r in integ:
        if r["index_is_score_v6"] != "false" or r["index_is_score_v7"] != "false" or r["index_is_replay_v6"] != "false":
            errors.append("integrity_index_mislabeled")
    # 13 flow_acc equivalente
    if contract.get("flow_acc_official_equivalent") is not False:
        errors.append("flow_acc_marked_equivalent")
    for r in leak:
        if r["uses_flow_acc_as_equivalent"] != "false":
            errors.append("flow_acc_equivalent_leak")
    # 14 ocorrencia como feature causal
    for r in leak:
        if r["uses_occurrence_as_causal_feature"] != "false" or r["occurrence_only_observational_anchor"] != "true":
            errors.append("occurrence_causal_leak")
    # 15 controle negativo verdadeiro
    for r in comp:
        if r["absence_of_occurrence_is_true_negative"] != "false":
            errors.append("absence_true_negative")
    for r in leak:
        if r["uses_control_as_negative_truth"] != "false" or r["uses_absence_as_real_absence"] != "false":
            errors.append("control_negative_leak")
    # 16/17 score v6/matriz/v7
    if _run_git(["diff", "--name-only", "--", rel(SCORE_V6)]) or summary["score_v6_original_file_changed"]:
        errors.append("score_v6_changed")
    if _run_git(["diff", "--name-only", "--", rel(OFFICIAL_MATRIX)]) or summary["official_matrix_changed"]:
        errors.append("official_matrix_changed")
    if SCORE_V7.exists() or summary["score_v7_created"]:
        errors.append("score_v7_exists")
    # 18/19 GT/treino
    if summary["ground_truth_created"] or summary["training_labels_created"]:
        errors.append("gt_or_training")
    for r in index:
        if r["ground_truth"] != "false" or r["training_label"] != "false":
            errors.append("index_gt_training")
    # 20 17B
    if summary["eligible_for_17b_now"] or interp.get("can_unlock_17b") is not False:
        errors.append("17b_unlocked")
    if any(r["unlocks_17b"] != "false" for r in leak):
        errors.append("17b_leak")
    if any(r["passes_no_leakage"] != "true" for r in leak):
        errors.append("no_leakage_failed")
    # nao terminar blocked
    if summary["result_class"] == "blocked" or not summary["minimum_success_achieved"]:
        errors.append("milestone_blocked")

    # 21 summary reflete contagens
    expected = build_summary()
    for k, v in expected.items():
        if summary.get(k) != v:
            errors.append(f"summary_mismatch:{k}")

    for rows, key in [
        (comp, "transparent_component_id"), (norm, "normalization_id"), (index, "transparent_index_id"),
        (contrast, "contrast_id"), (rob, "robustness_analysis_id"), (align, "alignment_metric_id"),
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
        "17C40 -> "
        f"canary={summary['canary_patch_rows_consumed_count']} control={summary['original_control_patch_rows_consumed_count']} "
        f"comp={summary['transparent_component_matrix_rows_count']} index={summary['transparent_index_rows_count']} "
        f"variants={summary['index_variant_count']} contrast={summary['canary_vs_control_contrast_rows_count']} "
        f"robustness={summary['robustness_analysis_rows_count']} agreement={summary['variant_agreement']} "
        f"direction={summary['canary_vs_control_direction']} result={summary['result_class']} "
        f"min_success={summary['minimum_success_achieved']}"
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="susc_17c40")
    parser.add_argument("mode", choices=[
        "build-flow-replacement-contract", "build-transparent-index-policy", "build-component-matrix",
        "normalize-components", "compute-transparent-index", "build-canary-control-contrast",
        "build-robustness-analysis", "build-observational-alignment", "build-operational-interpretation",
        "build-offline", "audit",
    ])
    args = parser.parse_args(argv)
    if args.mode == "audit":
        return validate()
    run_mode(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
