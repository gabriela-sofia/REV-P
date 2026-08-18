"""SUSC-16D review-only calibration candidate design.

This module turns the SUSC-16C divergence diagnostic into a controlled,
review-only set of calibration candidates. It NEVER:

* changes the official score v6,
* creates score v7,
* creates training data, supervised labels, models, ground truth, or true
  negatives,
* treats an observed footprint as automatic ground truth,
* treats documentary absence as a true negative.

It separates two distinct concepts that SUSC-16C kept entangled inside the
score:

* ``susceptibility_signal`` -> physical / hydrological / urban / spectral /
  rainfall features that describe how flood-prone a patch is;
* ``evidence_confidence`` -> documentary / source / footprint / uncertainty
  signals that describe how well-documented a case is.

The ``documentary_component`` that collapsed 65/65 SUSC-16C cases is here
treated as documentary *confidence*, not as proof of low physical
susceptibility.

It also lays the minimal, empty foundation for the SUSC-17A Reference Evidence
Protocol (schema + class policy + headers-only registry stub) without inventing
any record.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_common import find_model_artifacts, matrix_sha256_ok  # noqa: E402
from susc_io import read_csv, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

AUDIT = OUT / "SUSC_WORKTREE_AUDIT_BEFORE_16D.md"
SCHEMA = SCHEMAS / "susc_16d_calibration_candidate_schema_v1.json"
PROTOCOL_SCHEMA = SCHEMAS / "susc_17a_reference_evidence_protocol_schema_v1.json"

# --- SUSC-16C inputs consumed (read-only) ---------------------------------- #
UNIFIED = DAT / "susc_16c_unified_observational_analysis_table_v1.csv"
UNIFIED_SUMMARY = OUT / "SUSC_16C_unified_observational_analysis_summary.csv"
DECOMP = OUT / "SUSC_16C_score_v6_component_decomposition.csv"
DECOMP_SUMMARY = OUT / "SUSC_16C_score_v6_component_decomposition_summary.json"
FAILURES = OUT / "SUSC_16C_proxy_failure_modes.csv"
FAILURE_SUMMARY = OUT / "SUSC_16C_proxy_failure_modes_summary.json"
WEIGHT_SUMMARY = OUT / "SUSC_16C_weight_sensitivity_summary.json"
FEATURE_STABILITY = OUT / "SUSC_16C_feature_direction_stability.csv"

# --- score v6 official (read-only, never written) -------------------------- #
SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

# --- SUSC-16D outputs ------------------------------------------------------ #
MATRIX = OUT / "susc_16d_calibration_candidate_matrix.csv"
HYPOTHESES = OUT / "susc_16d_calibration_hypotheses.csv"
FAILURE_TO_WEIGHT = OUT / "susc_16d_failure_mode_to_weight_hypothesis.csv"
FEATURE_POLICY = OUT / "susc_16d_feature_direction_policy.json"
SENSITIVITY_SUMMARY = OUT / "susc_16d_sensitivity_summary.json"
REPORT = OUT / "SUSC_16D_CALIBRATION_CANDIDATE_REVIEW_ONLY_REPORT.md"

# --- SUSC-17A reference evidence protocol foundation ----------------------- #
PROTOCOL_REGISTRY_STUB = OUT / "susc_17a_reference_evidence_protocol_registry_stub.csv"
PROTOCOL_CLASS_POLICY = OUT / "susc_17a_reference_evidence_protocol_class_policy.json"

REQUIRED_INPUTS = [
    AUDIT, SCHEMA, PROTOCOL_SCHEMA, UNIFIED, UNIFIED_SUMMARY, DECOMP,
    DECOMP_SUMMARY, FAILURES, FAILURE_SUMMARY, WEIGHT_SUMMARY, SCORE_V6,
]

MANDATORY = (
    "O SUSC-16D desenha calibracao candidata review-only a partir do "
    "diagnostico SUSC-16C. Ele separa susceptibility_signal de "
    "evidence_confidence, trata o documentary_component como confianca "
    "documental e nao como prova de baixa suscetibilidade fisica, e nao altera "
    "o score v6 oficial, nao cria score v7, nao cria treino, modelo, ground "
    "truth ou negativo verdadeiro."
)

ADJUSTMENT_FAMILIES = (
    "decouple_documentary_component_from_susceptibility",
    "increase_urban_flash_flood_weight",
    "increase_rainfall_trigger_weight",
    "increase_spectral_water_weight",
    "guard_low_documentary_high_physical_signal",
)

# Concept separation: which signals describe physical susceptibility vs which
# describe how well-documented a case is. documentary_component is confidence,
# NOT susceptibility.
SUSCEPTIBILITY_SIGNAL_FEATURES = {
    "hand_mean": "lower_in_event",
    "slope_mean": "lower_in_event",
    "elevation_mean": "lower_in_event",
    "distance_to_water_mean": "lower_in_event",
    "twi_mean": "higher_in_event",
    "flow_accumulation_mean": "higher_in_event",
    "urban_prop": "higher_in_event",
    "vegetation_prop": "lower_in_event",
    "water_prop": "higher_in_event",
    "ndvi_mean": "lower_in_event",
    "mndwi_mean": "higher_in_event",
    "ndbi_mean": "higher_in_event",
    "chirps_3d_mm": "higher_in_event",
    "chirps_7d_mm": "higher_in_event",
    "chirps_30d_mm": "higher_in_event",
    "runoff_context_7d": "higher_in_event",
}

EVIDENCE_CONFIDENCE_FEATURES = {
    "documentary_component": "confidence_not_susceptibility",
    "footprint_quality_tier": "confidence_not_susceptibility",
    "evidence_strength": "confidence_not_susceptibility",
    "overlap_ratio": "confidence_not_susceptibility",
    "distance_m": "confidence_not_susceptibility",
}

REFERENCE_CLASSES = (
    ("official_observed_event_polygon", True),
    ("official_observed_event_point", True),
    ("technical_remote_sensing_flood_footprint", True),
    ("official_address_resolved", False),
    ("street_neighborhood_context_only", False),
    ("risk_area_not_event", False),
    ("alert_only", False),
    ("rejected_non_event", False),
)

PROTOCOL_FIELDS = [
    "source_id", "event_id", "geometry_id", "patch_link_id", "event_date",
    "pre_event_window", "post_event_window", "geometry_type",
    "source_authority", "source_confidence", "annotation_method",
    "uncertainty_m", "eligible_for_evaluation", "eligible_for_calibration",
    "not_ground_truth_reason",
]

MATRIX_FIELDS = [
    "patch_id", "event_id", "footprint_id", "case_id", "region", "score_v6",
    "score_v6_class", "has_observed_footprint", "documentary_component",
    "urban_prop", "divergent_features_available", "failure_mode",
    "candidate_adjustment_family", "secondary_adjustment_families",
    "signal_domain", "review_only", "trainable", "ground_truth",
    "score_v6_unchanged", "score_v7_created", "eligible_for_score_v7_future",
    "not_ground_truth_reason",
]

NOT_GT_REASON = (
    "observed_footprint_is_candidate_evidence_not_ground_truth; "
    "control_is_no_documented_footprint_not_true_negative"
)


def _require_inputs() -> None:
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(str(p.relative_to(ROOT)) for p in missing))


def _index(rows: list[dict], key: str) -> dict[str, dict]:
    return {r.get(key, ""): r for r in rows if r.get(key)}


def _primary_family(failure_mode: str) -> str:
    if failure_mode == "urban_flash_flood_underrepresented":
        return "increase_urban_flash_flood_weight"
    if failure_mode == "rainfall_trigger_underweighted":
        return "increase_rainfall_trigger_weight"
    return "guard_low_documentary_high_physical_signal"


def _secondary_families(divergent: str, component_lowest: str, primary: str) -> list[str]:
    families: list[str] = []
    # documentary_component collapsed every SUSC-16C case -> structural decoupling
    # candidate for all rows where it is the lowest driver.
    if component_lowest == "documentary_component":
        families.append("decouple_documentary_component_from_susceptibility")
        if divergent and primary != "guard_low_documentary_high_physical_signal":
            families.append("guard_low_documentary_high_physical_signal")
    if "wet_spectral_signal" in divergent and "increase_spectral_water_weight" not in (primary,):
        families.append("increase_spectral_water_weight")
    return [fam for fam in families if fam != primary]


def build_calibration_candidate_matrix() -> int:
    _require_inputs()
    failures = read_csv(FAILURES)
    decomp_by_case = _index(read_csv(DECOMP), "case_id")
    unified_by_case = _index(
        [r for r in read_csv(UNIFIED) if r.get("is_event_patch") == "true"], "case_id"
    )
    rows = []
    for src in failures:
        case_id = src.get("case_id", "")
        decomp = decomp_by_case.get(case_id, {})
        unified = unified_by_case.get(case_id, {})
        divergent = src.get("supporting_features", "")
        failure_mode = src.get("failure_mode_primary", "")
        component_lowest = decomp.get("component_lowest", "")
        primary = _primary_family(failure_mode)
        secondary = _secondary_families(divergent, component_lowest, primary)
        signal_domain = "mixed" if divergent else "evidence_confidence"
        rows.append({
            "patch_id": src.get("patch_id", ""),
            "event_id": "",
            "footprint_id": src.get("footprint_id", ""),
            "case_id": case_id,
            "region": unified.get("region", ""),
            "score_v6": src.get("score_v6", ""),
            "score_v6_class": src.get("score_v6_class", ""),
            "has_observed_footprint": "true",
            "documentary_component": decomp.get("documentary_component", ""),
            "urban_prop": unified.get("urban_prop", ""),
            "divergent_features_available": divergent,
            "failure_mode": failure_mode,
            "candidate_adjustment_family": primary,
            "secondary_adjustment_families": "|".join(secondary),
            "signal_domain": signal_domain,
            "review_only": "true",
            "trainable": "false",
            "ground_truth": "false",
            "score_v6_unchanged": "true",
            "score_v7_created": "false",
            "eligible_for_score_v7_future": "false",
            "not_ground_truth_reason": NOT_GT_REASON,
        })
    write_csv(MATRIX, rows, MATRIX_FIELDS)
    print(f"candidate matrix -> {MATRIX.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


def _hypotheses() -> list[dict]:
    base = {
        "review_only": "true",
        "trainable": "false",
        "ground_truth": "false",
        "score_v6_unchanged": "true",
        "score_v7_created": "false",
        "eligible_for_score_v7_future": "false",
        "status": "review_only_design_candidate",
    }
    rows = [
        {**base,
         "hypothesis_id": "decouple_documentary_component_from_susceptibility",
         "signal_domain": "evidence_confidence",
         "justification": (
             "O documentary_component foi o componente mais baixo em 65/65 casos "
             "SUSC-16C. Usa-lo dentro do score como se fosse suscetibilidade fisica "
             "faz a ausencia de documentacao derrubar a suscetibilidade. A hipotese "
             "separa confianca documental do sinal fisico."),
         "features_involved": "documentary_component;evidence_support_index",
         "failure_modes_covered": "documentary_lowest_component",
         "methodological_risk": (
             "medium - redefinir o que o score mede; risco de descartar exposicao "
             "real se a separacao for mal calibrada")},
        {**base,
         "hypothesis_id": "increase_urban_flash_flood_weight",
         "signal_domain": "susceptibility_signal",
         "justification": (
             "60/65 casos cairam em urban_flash_flood_underrepresented. Patches "
             "urbanos com footprint observacional ficaram low/medium apesar de "
             "exposicao urbana e drenagem."),
         "features_involved": "urban_prop;ndbi_mean;twi_mean;flow_accumulation_mean",
         "failure_modes_covered": "urban_flash_flood_underrepresented",
         "methodological_risk": (
             "high - peso urbano excessivo pode inflar areas urbanas sem evento; "
             "exige validacao regional")},
        {**base,
         "hypothesis_id": "increase_rainfall_trigger_weight",
         "signal_domain": "susceptibility_signal",
         "justification": (
             "5/65 casos cairam em rainfall_trigger_underweighted. O gatilho de "
             "chuva/runoff aparece subponderado em casos com contexto pluviometrico."),
         "features_involved": "chirps_3d_mm;chirps_7d_mm;chirps_30d_mm;runoff_context_7d",
         "failure_modes_covered": "rainfall_trigger_underweighted",
         "methodological_risk": (
             "high - chuva e gatilho temporal, nao suscetibilidade estatica; risco "
             "de misturar trigger com condicionante")},
        {**base,
         "hypothesis_id": "increase_spectral_water_weight",
         "signal_domain": "susceptibility_signal",
         "justification": (
             "O cenario increase_spectral_water_weight teve a melhor sensibilidade "
             "review-only no SUSC-16C (event_hit_rate_top_30_simulated=0.366667). "
             "Sinal espectral umido sustenta suscetibilidade em casos low/medium."),
         "features_involved": "mndwi_mean;water_prop;ndvi_mean",
         "failure_modes_covered": "spectral_best_sensitivity_scenario",
         "methodological_risk": (
             "medium - sinal espectral pode refletir agua permanente, nao evento; "
             "exige controle temporal pre/pos")},
        {**base,
         "hypothesis_id": "guard_low_documentary_high_physical_signal",
         "signal_domain": "mixed",
         "justification": (
             "Guarda de seguranca: quando o documentary_component e baixo mas o "
             "sinal fisico (hidrologico/urbano/espectral/chuva) e alto, o score nao "
             "deve ser colapsado pela ausencia documental."),
         "features_involved": (
             "documentary_component;urban_prop;twi_mean;mndwi_mean;chirps_7d_mm"),
         "failure_modes_covered": "low_documentary_high_physical_signal",
         "methodological_risk": (
             "medium - definir 'sinal fisico alto' exige limiar auditavel; risco de "
             "guarda permissiva demais")},
    ]
    return rows


def build_hypotheses() -> int:
    fields = [
        "hypothesis_id", "signal_domain", "justification", "features_involved",
        "failure_modes_covered", "methodological_risk", "status", "review_only",
        "trainable", "ground_truth", "score_v6_unchanged", "score_v7_created",
        "eligible_for_score_v7_future",
    ]
    write_csv(HYPOTHESES, _hypotheses(), fields)
    print(f"hypotheses -> {HYPOTHESES.relative_to(ROOT)}")
    return 0


def build_failure_mode_to_weight() -> int:
    failure_dist = json.loads(FAILURE_SUMMARY.read_text(encoding="utf-8")).get(
        "primary_failure_mode_distribution", {})
    decomp = json.loads(DECOMP_SUMMARY.read_text(encoding="utf-8"))
    weight = json.loads(WEIGHT_SUMMARY.read_text(encoding="utf-8"))
    documentary_count = decomp.get("component_lowest_distribution", {}).get(
        "documentary_component", 0)
    rows = [
        {"failure_mode": "documentary_lowest_component",
         "observed_count_or_metric": str(documentary_count),
         "candidate_adjustment_family": "decouple_documentary_component_from_susceptibility",
         "hypothesis_id": "decouple_documentary_component_from_susceptibility",
         "concept_separation": "evidence_confidence_not_susceptibility_signal",
         "justification": "documentary_component foi o componente mais baixo em todos os casos",
         "methodological_risk": "medium"},
        {"failure_mode": "urban_flash_flood_underrepresented",
         "observed_count_or_metric": str(failure_dist.get("urban_flash_flood_underrepresented", 0)),
         "candidate_adjustment_family": "increase_urban_flash_flood_weight",
         "hypothesis_id": "increase_urban_flash_flood_weight",
         "concept_separation": "susceptibility_signal",
         "justification": "exposicao urbana subrepresentada em casos low/medium",
         "methodological_risk": "high"},
        {"failure_mode": "rainfall_trigger_underweighted",
         "observed_count_or_metric": str(failure_dist.get("rainfall_trigger_underweighted", 0)),
         "candidate_adjustment_family": "increase_rainfall_trigger_weight",
         "hypothesis_id": "increase_rainfall_trigger_weight",
         "concept_separation": "susceptibility_signal_trigger",
         "justification": "gatilho de chuva/runoff subponderado",
         "methodological_risk": "high"},
        {"failure_mode": "spectral_best_sensitivity_scenario",
         "observed_count_or_metric": str(weight.get("best_hit_rate_top_30_simulated", "")),
         "candidate_adjustment_family": "increase_spectral_water_weight",
         "hypothesis_id": "increase_spectral_water_weight",
         "concept_separation": "susceptibility_signal",
         "justification": "melhor sensibilidade review-only no SUSC-16C",
         "methodological_risk": "medium"},
        {"failure_mode": "low_documentary_high_physical_signal",
         "observed_count_or_metric": "guard_rule",
         "candidate_adjustment_family": "guard_low_documentary_high_physical_signal",
         "hypothesis_id": "guard_low_documentary_high_physical_signal",
         "concept_separation": "mixed_guard",
         "justification": "evitar colapso do score por ausencia documental quando o sinal fisico e alto",
         "methodological_risk": "medium"},
    ]
    for row in rows:
        row.update({
            "review_only": "true", "trainable": "false", "ground_truth": "false",
            "score_v6_unchanged": "true", "score_v7_created": "false",
            "eligible_for_score_v7_future": "false",
        })
    fields = [
        "failure_mode", "observed_count_or_metric", "candidate_adjustment_family",
        "hypothesis_id", "concept_separation", "justification",
        "methodological_risk", "review_only", "trainable", "ground_truth",
        "score_v6_unchanged", "score_v7_created", "eligible_for_score_v7_future",
    ]
    write_csv(FAILURE_TO_WEIGHT, rows, fields)
    print(f"failure->weight -> {FAILURE_TO_WEIGHT.relative_to(ROOT)}")
    return 0


def build_feature_direction_policy() -> int:
    stability_by_feature = _index(read_csv(FEATURE_STABILITY), "feature_name") \
        if FEATURE_STABILITY.exists() else {}
    features = {}
    for name, direction in SUSCEPTIBILITY_SIGNAL_FEATURES.items():
        stab = stability_by_feature.get(name, {})
        stability_class = stab.get("stability_class", "not_evaluated")
        features[name] = {
            "signal_domain": "susceptibility_signal",
            "expected_direction": direction,
            "observed_stability_class_16c": stability_class,
            "eligible_for_future_weight_discussion": stability_class in {
                "stable_support", "regional_conflict", "quality_dependent",
                "partial_support"},
            "can_change_official_score_v6": False,
        }
    for name, note in EVIDENCE_CONFIDENCE_FEATURES.items():
        features[name] = {
            "signal_domain": "evidence_confidence",
            "expected_direction": note,
            "observed_stability_class_16c": "not_a_susceptibility_signal",
            "eligible_for_future_weight_discussion": False,
            "can_change_official_score_v6": False,
        }
    policy = {
        "concept": (
            "susceptibility_signal descreve o quao propenso a inundacao um patch e; "
            "evidence_confidence descreve o quao bem documentado o caso e. O "
            "documentary_component pertence a evidence_confidence e nunca deve ser "
            "lido como prova de baixa suscetibilidade fisica."),
        "susceptibility_signal_count": len(SUSCEPTIBILITY_SIGNAL_FEATURES),
        "evidence_confidence_count": len(EVIDENCE_CONFIDENCE_FEATURES),
        "documentary_component_is_susceptibility_signal": False,
        "documentary_absence_is_true_negative": False,
        "footprint_is_ground_truth": False,
        "official_score_v6_changed": False,
        "score_v7_created": False,
        "review_only": True,
        "features": features,
    }
    write_json(FEATURE_POLICY, policy)
    print(f"feature policy -> {FEATURE_POLICY.relative_to(ROOT)}")
    return 0


def build_sensitivity_summary() -> int:
    unified_summary = {r["metric"]: r["value"] for r in read_csv(UNIFIED_SUMMARY)}
    failure = json.loads(FAILURE_SUMMARY.read_text(encoding="utf-8"))
    decomp = json.loads(DECOMP_SUMMARY.read_text(encoding="utf-8"))
    weight = json.loads(WEIGHT_SUMMARY.read_text(encoding="utf-8"))
    matrix_rows = read_csv(MATRIX) if MATRIX.exists() else []
    failure_dist = failure.get("primary_failure_mode_distribution", {})
    summary = {
        "calibration_candidate_created": True,
        "official_score_changed": False,
        "score_v7_created": False,
        "trainable": False,
        "ground_truth": False,
        "eligible_for_score_v7_future": 0,
        "readiness_status": "BLOCKED_REVIEW_ONLY",
        "candidate_rows": len(matrix_rows),
        "footprint_patch_links": int(unified_summary.get("event_link_rows", 0) or 0),
        "unique_event_patches": int(unified_summary.get("unique_event_patches", 0) or 0),
        "low_or_medium_event_links": int(unified_summary.get("low_or_medium_event_rows", 0) or 0),
        "documentary_lowest_component_count": decomp.get(
            "component_lowest_distribution", {}).get("documentary_component", 0),
        "urban_flash_flood_underrepresented": failure_dist.get(
            "urban_flash_flood_underrepresented", 0),
        "rainfall_trigger_underweighted": failure_dist.get(
            "rainfall_trigger_underweighted", 0),
        "best_sensitivity_scenario_16c": weight.get("top_scenario_by_hit30", ""),
        "best_hit_rate_top_30_simulated_16c": weight.get(
            "best_hit_rate_top_30_simulated", ""),
        "candidate_adjustment_family_distribution": dict(
            Counter(r["candidate_adjustment_family"] for r in matrix_rows)),
        "hypotheses_count": len(_hypotheses()),
        "footprint_treated_as_ground_truth": False,
        "control_treated_as_negative": False,
        "documentary_absence_treated_as_negative": False,
        "review_only": True,
    }
    write_json(SENSITIVITY_SUMMARY, summary)
    print(f"sensitivity summary -> {SENSITIVITY_SUMMARY.relative_to(ROOT)}")
    return 0


def build_reference_evidence_protocol_foundation() -> int:
    """Minimal SUSC-17A foundation: headers-only registry stub + class policy.

    No record is invented. Only the first three reference classes may ever
    support strong evaluation; none becomes ground truth or a true negative.
    """
    write_csv(PROTOCOL_REGISTRY_STUB, [], PROTOCOL_FIELDS)
    policy = {
        "protocol": "susc_17a_reference_evidence_protocol",
        "status": "stub_foundation_no_records",
        "schema": str(PROTOCOL_SCHEMA.relative_to(ROOT)).replace("\\", "/"),
        "record_count": 0,
        "note": (
            "Stub fail-closed. Nenhum registro foi preenchido. Footprints e "
            "referencias permanecem candidatos; nenhuma classe vira ground truth "
            "ou negativo verdadeiro."),
        "classes": [
            {
                "geometry_type": name,
                "can_be_strong_evaluation": strong,
                "can_be_ground_truth": False,
                "can_be_true_negative": False,
                "eligible_for_calibration": strong,
            }
            for name, strong in REFERENCE_CLASSES
        ],
        "strong_evaluation_classes": [n for n, s in REFERENCE_CLASSES if s],
        "review_only": True,
    }
    write_json(PROTOCOL_CLASS_POLICY, policy)
    print(f"protocol foundation -> {PROTOCOL_REGISTRY_STUB.relative_to(ROOT)}")
    return 0


def build_report() -> int:
    unified_summary = {r["metric"]: r["value"] for r in read_csv(UNIFIED_SUMMARY)}
    summary = json.loads(SENSITIVITY_SUMMARY.read_text(encoding="utf-8")) \
        if SENSITIVITY_SUMMARY.exists() else {}
    matrix_rows = read_csv(MATRIX) if MATRIX.exists() else []
    family_dist = summary.get("candidate_adjustment_family_distribution", {})
    report = f"""# SUSC-16D - desenho controlado de calibracao candidata review-only

Status: review-only. `trainable=false`; `ground_truth=false`; `official_score_changed=false`; `score_v7_created=false`; `eligible_for_score_v7_future=0`; `readiness_status=BLOCKED_REVIEW_ONLY`.

{MANDATORY}

## 1. Entrada herdada do SUSC-16C

O SUSC-16D consome o diagnostico do SUSC-16C sem reabrir o escopo. Numeros reproduzidos do 16C:

- Links footprint-patch: {unified_summary.get('event_link_rows')} (esperado 65).
- Patches observacionais unicos: {unified_summary.get('unique_event_patches')} (esperado 62).
- Casos low/medium com footprint: {unified_summary.get('low_or_medium_event_rows')} (esperado 57).
- `documentary_component` foi o componente mais baixo em {summary.get('documentary_lowest_component_count')}/65 casos.
- `urban_flash_flood_underrepresented`: {summary.get('urban_flash_flood_underrepresented')}.
- `rainfall_trigger_underweighted`: {summary.get('rainfall_trigger_underweighted')}.
- Melhor sensibilidade review-only: `{summary.get('best_sensitivity_scenario_16c')}` com `event_hit_rate_top_30_simulated`={summary.get('best_hit_rate_top_30_simulated_16c')}.

## 2. Separacao conceitual obrigatoria

O 16D separa dois conceitos que o score v6 mantinha misturados:

- `susceptibility_signal`: fisico, hidrologico, urbano, espectral e de chuva ({summary.get('candidate_rows') and len(SUSCEPTIBILITY_SIGNAL_FEATURES)} features).
- `evidence_confidence`: documental, fonte, footprint e incerteza ({len(EVIDENCE_CONFIDENCE_FEATURES)} sinais).

O `documentary_component` e tratado como confianca documental. A ausencia de documentacao NAO e prova de baixa suscetibilidade fisica e NAO e negativo verdadeiro.

## 3. Matriz de candidatos por link/patch

A matriz `susc_16d_calibration_candidate_matrix.csv` tem {len(matrix_rows)} linhas (uma por link footprint-patch elegivel do 16C). Cada linha registra `score_v6`, `score_v6_class`, `documentary_component`, `urban_prop`, features divergentes, `failure_mode` e a familia de ajuste candidata. Todas as linhas: `has_observed_footprint=true`, `ground_truth=false`, `trainable=false`, `score_v6_unchanged=true`, `score_v7_created=false`, `eligible_for_score_v7_future=false`, com `not_ground_truth_reason` preenchido.

Distribuicao por familia de ajuste primaria: {json.dumps(family_dist, ensure_ascii=False, sort_keys=True)}.

## 4. Hipoteses de calibracao candidatas

As hipoteses estao em `susc_16d_calibration_hypotheses.csv`, com justificativa, features, modos de falha cobertos, risco metodologico e status review-only:

1. `decouple_documentary_component_from_susceptibility` - separar confianca documental do sinal fisico.
2. `increase_urban_flash_flood_weight` - exposicao urbana subrepresentada (60/65).
3. `increase_rainfall_trigger_weight` - gatilho de chuva/runoff subponderado (5/65).
4. `increase_spectral_water_weight` - melhor sensibilidade review-only do 16C.
5. `guard_low_documentary_high_physical_signal` - guarda contra colapso do score por ausencia documental.

Nenhuma hipotese e aplicada ao score oficial. `eligible_for_score_v7_future=false` para todas.

## 5. Mapeamento modo de falha -> hipotese de peso

`susc_16d_failure_mode_to_weight_hypothesis.csv` liga cada modo de falha do 16C a uma familia de ajuste candidata e marca a separacao de conceito (susceptibility_signal vs evidence_confidence).

## 6. Politica de direcao de features

`susc_16d_feature_direction_policy.json` classifica cada feature como `susceptibility_signal` ou `evidence_confidence`, registra a direcao esperada e a estabilidade observada no 16C, e marca `can_change_official_score_v6=false` para todas.

## 7. Fundacao minima para SUSC-17A (Reference Evidence Protocol)

Foi criada a base minima do Reference Evidence Protocol como stub fail-closed:

- Schema: `schemas/suscetibilidade/susc_17a_reference_evidence_protocol_schema_v1.json`.
- Registro stub headers-only (0 registros): `susc_17a_reference_evidence_protocol_registry_stub.csv`.
- Politica de classes: `susc_17a_reference_evidence_protocol_class_policy.json`.

Somente `official_observed_event_polygon`, `official_observed_event_point` e `technical_remote_sensing_flood_footprint` podem sustentar avaliacao forte. Nenhuma classe vira ground truth ou negativo verdadeiro. Nenhum registro foi inventado.

## 8. O que nao pode ser afirmado

Nao se pode afirmar ground truth, negativo verdadeiro, prontidao para treino, score v7 pronto ou causalidade. Footprints permanecem evidencia observacional candidata; controles permanecem 'no documented footprint', nao negativos verdadeiros.

## 9. Proximo marco recomendado

`SUSC-17A Reference Evidence Protocol`: popular o protocolo de referencia com evidencia observacional auditada (sem ground truth automatico), ou `SUSC-17B Event-Based Mini-Benchmark` se houver referencias fortes suficientes para um mini-benchmark review-only.
"""
    write_markdown(REPORT, report)
    print(f"report -> {REPORT.relative_to(ROOT)}")
    return 0


def run_all() -> int:
    build_calibration_candidate_matrix()
    build_hypotheses()
    build_failure_mode_to_weight()
    build_feature_direction_policy()
    build_sensitivity_summary()
    build_reference_evidence_protocol_foundation()
    build_report()
    return 0


# --------------------------------------------------------------------------- #
# Validation (fail-closed)
# --------------------------------------------------------------------------- #

def matrix_row_violations(row: dict) -> list[str]:
    """Fail-closed guard for a single calibration candidate row.

    Returns the list of governance violations. An empty list means the row is
    clean. Exposed for tests so the validator's failure conditions are testable.
    """
    out = []
    if row.get("review_only") != "true":
        out.append("review_only not true")
    if row.get("trainable") != "false":
        out.append("trainable true")
    if row.get("ground_truth") != "false":
        out.append("ground_truth true")
    if row.get("score_v6_unchanged") != "true":
        out.append("score_v6_unchanged not true")
    if row.get("score_v7_created") != "false":
        out.append("score_v7_created true")
    if row.get("eligible_for_score_v7_future") not in ("false", "0", ""):
        out.append("eligible_for_score_v7_future positive")
    if not row.get("not_ground_truth_reason"):
        out.append("missing not_ground_truth_reason")
    if row.get("candidate_adjustment_family") not in ADJUSTMENT_FAMILIES:
        out.append("unknown candidate_adjustment_family")
    # observed footprint must never be promoted to ground truth.
    if row.get("has_observed_footprint") == "true" and row.get("ground_truth") != "false":
        out.append("footprint treated as ground truth")
    return out


def validate() -> int:
    errors: list[str] = []

    required = [
        AUDIT, SCHEMA, PROTOCOL_SCHEMA, MATRIX, HYPOTHESES, FAILURE_TO_WEIGHT,
        FEATURE_POLICY, SENSITIVITY_SUMMARY, REPORT, PROTOCOL_REGISTRY_STUB,
        PROTOCOL_CLASS_POLICY,
    ]
    for path in required:
        if not path.exists():
            errors.append(f"missing: {path.relative_to(ROOT)}")
    if errors:
        print("FAILED: SUSC-16D validation")
        for error in errors:
            print("  ERROR:", error)
        return 1

    # Per-row governance guards on the candidate matrix.
    matrix_rows = read_csv(MATRIX)
    for line_no, row in enumerate(matrix_rows, start=2):
        for violation in matrix_row_violations(row):
            errors.append(f"{MATRIX.name}:{line_no}: {violation}")

    # Hypotheses and failure->weight tables must stay review-only and blocked.
    for path in (HYPOTHESES, FAILURE_TO_WEIGHT):
        for line_no, row in enumerate(read_csv(path), start=2):
            if row.get("review_only") != "true":
                errors.append(f"{path.name}:{line_no}: review_only not true")
            if row.get("trainable") != "false":
                errors.append(f"{path.name}:{line_no}: trainable true")
            if row.get("ground_truth") != "false":
                errors.append(f"{path.name}:{line_no}: ground_truth true")
            if row.get("score_v7_created") != "false":
                errors.append(f"{path.name}:{line_no}: score_v7_created true")
            if row.get("eligible_for_score_v7_future") not in ("false", "0", ""):
                errors.append(f"{path.name}:{line_no}: eligible_for_score_v7_future positive")

    # Sensitivity summary flags.
    summary = json.loads(SENSITIVITY_SUMMARY.read_text(encoding="utf-8"))
    if summary.get("calibration_candidate_created") is not True:
        errors.append("summary: calibration_candidate_created not true")
    if summary.get("official_score_changed") is not False:
        errors.append("summary: official_score_changed true")
    if summary.get("score_v7_created") is not False:
        errors.append("summary: score_v7_created true")
    if summary.get("trainable") is not False:
        errors.append("summary: trainable true")
    if summary.get("ground_truth") is not False:
        errors.append("summary: ground_truth true")
    if summary.get("eligible_for_score_v7_future") not in (0, "0"):
        errors.append("summary: eligible_for_score_v7_future > 0")
    if summary.get("readiness_status") != "BLOCKED_REVIEW_ONLY":
        errors.append("summary: readiness_status not BLOCKED_REVIEW_ONLY")
    if summary.get("footprint_treated_as_ground_truth") is not False:
        errors.append("summary: footprint treated as ground truth")
    if summary.get("control_treated_as_negative") is not False:
        errors.append("summary: control treated as negative")

    # Reproduce the SUSC-16C diagnostic exactly (fail if drift).
    unified_summary = {r["metric"]: r["value"] for r in read_csv(UNIFIED_SUMMARY)}
    expected = {
        "event_link_rows": "65",
        "unique_event_patches": "62",
        "low_or_medium_event_rows": "57",
    }
    for metric, value in expected.items():
        if unified_summary.get(metric) != value:
            errors.append(f"16C drift: {metric}={unified_summary.get(metric)} expected {value}")
    decomp = json.loads(DECOMP_SUMMARY.read_text(encoding="utf-8"))
    if decomp.get("component_lowest_distribution", {}).get("documentary_component") != 65:
        errors.append("16C drift: documentary_component lowest != 65")
    failure_dist = json.loads(FAILURE_SUMMARY.read_text(encoding="utf-8")).get(
        "primary_failure_mode_distribution", {})
    if failure_dist.get("urban_flash_flood_underrepresented") != 60:
        errors.append("16C drift: urban_flash_flood_underrepresented != 60")
    if failure_dist.get("rainfall_trigger_underweighted") != 5:
        errors.append("16C drift: rainfall_trigger_underweighted != 5")
    weight = json.loads(WEIGHT_SUMMARY.read_text(encoding="utf-8"))
    if weight.get("top_scenario_by_hit30") != "increase_spectral_water_weight":
        errors.append("16C drift: best sensitivity scenario changed")
    if str(weight.get("best_hit_rate_top_30_simulated")) != "0.366667":
        errors.append("16C drift: best hit30 != 0.366667")
    if len(matrix_rows) != 65:
        errors.append(f"candidate matrix rows {len(matrix_rows)} expected 65")

    # Feature direction policy: documentary stays evidence_confidence.
    policy = json.loads(FEATURE_POLICY.read_text(encoding="utf-8"))
    if policy.get("documentary_component_is_susceptibility_signal") is not False:
        errors.append("policy: documentary marked as susceptibility signal")
    if policy.get("documentary_absence_is_true_negative") is not False:
        errors.append("policy: documentary absence marked true negative")
    doc = policy.get("features", {}).get("documentary_component", {})
    if doc.get("signal_domain") != "evidence_confidence":
        errors.append("policy: documentary_component not evidence_confidence")

    # Reference evidence protocol foundation: stub stays empty, fail-closed.
    stub_rows = read_csv(PROTOCOL_REGISTRY_STUB)
    if stub_rows:
        errors.append("protocol registry stub has invented records")
    class_policy = json.loads(PROTOCOL_CLASS_POLICY.read_text(encoding="utf-8"))
    strong = {c["geometry_type"] for c in class_policy.get("classes", [])
              if c.get("can_be_strong_evaluation")}
    if strong != {
        "official_observed_event_polygon",
        "official_observed_event_point",
        "technical_remote_sensing_flood_footprint",
    }:
        errors.append("protocol: strong-evaluation classes are not exactly the three allowed")
    if any(c.get("can_be_ground_truth") for c in class_policy.get("classes", [])):
        errors.append("protocol: a reference class is marked ground truth")
    if any(c.get("can_be_true_negative") for c in class_policy.get("classes", [])):
        errors.append("protocol: a reference class is marked true negative")

    # Cross guards shared with the SUSC chain.
    if SCORE_V7.exists():
        errors.append("score v7 artifact exists")
    if not SCORE_V6.exists():
        errors.append("score v6 official source missing")
    if not matrix_sha256_ok():
        errors.append("SUSC-03 matrix sha256 mismatch (official source guard)")
    if find_model_artifacts():
        errors.append("model artifact found under SUSC scope")
    if MANDATORY not in REPORT.read_text(encoding="utf-8"):
        errors.append("mandatory report phrase missing")

    if errors:
        print("FAILED: SUSC-16D validation")
        for error in errors:
            print("  ERROR:", error)
        return 1
    print("PASSED: SUSC-16D validations passed")
    return 0
