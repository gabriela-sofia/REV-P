"""
SUSC-06B Proxy Baseline Diagnostics (review-only)

Formal diagnostic layer over the SUSC-06A proxy baseline. Prevents misreading the
results: documents target circularity, reframes high R2 as heuristic
recoverability (NOT predictive power), audits partial-effect direction
divergences, checks regional stability, builds a per-feature action matrix and an
event-overlay readiness matrix for SUSC-07.

Reads:
  - datasets/suscetibilidade/susc_06_proxy_baseline_dataset_v1.csv
  - manifests/suscetibilidade/susc_06_proxy_baseline_dataset_manifest_v1.json
  - outputs_public/suscetibilidade/SUSC_06_proxy_baseline_target_policy.json
  - outputs_public/suscetibilidade/SUSC_06_proxy_baseline_metrics.csv
  - outputs_public/suscetibilidade/SUSC_06_proxy_baseline_region_metrics.csv
  - outputs_public/suscetibilidade/SUSC_06_proxy_baseline_partial_effects.csv
  - outputs_public/suscetibilidade/feature_cards/susc_feature_cards_v1.json
  - outputs_public/suscetibilidade/SUSC_04_score_v6_feature_decision_matrix.csv

Governance: review_only, proxy_based, not_event_validation. No model is fitted or
persisted, no ground truth is created, no training is unlocked.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())

BASE_DS = ROOT / "datasets" / "suscetibilidade" / "susc_06_proxy_baseline_dataset_v1.csv"
BASE_MANIFEST = (
    ROOT / "manifests" / "suscetibilidade" / "susc_06_proxy_baseline_dataset_manifest_v1.json"
)
MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
TARGET_POLICY = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_06_proxy_baseline_target_policy.json"
)
METRICS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_06_proxy_baseline_metrics.csv"
REGION_METRICS = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_06_proxy_baseline_region_metrics.csv"
)
PARTIAL = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_06_proxy_baseline_partial_effects.csv"
)
CARDS_JSON = (
    ROOT / "outputs_public" / "suscetibilidade" / "feature_cards" / "susc_feature_cards_v1.json"
)
DECISION = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_04_score_v6_feature_decision_matrix.csv"
)
SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_06b_baseline_diagnostics_schema_v1.json"

OUT = ROOT / "outputs_public" / "suscetibilidade"
CIRC_CSV = OUT / "SUSC_06B_baseline_circularity_report.csv"
DIR_AUDIT_CSV = OUT / "SUSC_06B_partial_effect_direction_audit.csv"
REGION_DIAG_CSV = OUT / "SUSC_06B_region_stability_diagnostics.csv"
ACTION_CSV = OUT / "SUSC_06B_feature_action_matrix.csv"
OVERLAY_CSV = OUT / "SUSC_06B_event_overlay_readiness_matrix.csv"
SUMMARY_JSON = OUT / "SUSC_06B_diagnostic_summary.json"

NEAR_ZERO = 0.005
REGION_SPREAD = 0.05
AMBIGUOUS_DIRS = {"ambiguous", "non_monotonic", "not_scored"}


def load_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _num(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def main() -> int:
    print("=" * 60)
    print("SUSC-06B Proxy Baseline Diagnostics (review-only)")
    print("=" * 60)

    for p in (BASE_DS, BASE_MANIFEST, TARGET_POLICY, METRICS, REGION_METRICS,
              PARTIAL, CARDS_JSON, DECISION, SCHEMA, MATRIX):
        if not p.exists():
            print(f"STOP: required input not found: {p}")
            return 1

    manifest = load_json(BASE_MANIFEST)
    policy = load_json(TARGET_POLICY)
    metrics = load_csv(METRICS)
    region_metrics = load_csv(REGION_METRICS)
    partial = load_csv(PARTIAL)
    cards = {c["feature_name"]: c for c in load_json(CARDS_JSON)["cards"]}
    decision = {r["feature_name"]: r for r in load_csv(DECISION)}

    features_used = manifest.get("features_used", [])
    target = policy.get("chosen_target")
    target_card = cards.get(target, {})

    # ---------- 1) Circularity ----------
    # The target is a heuristic composite score (group 'score') computed FROM the
    # physical conditionals; the model regresses those same conditionals onto it.
    target_is_score = target_card.get("feature_group") == "score"
    global_r2 = _num(metrics[0].get("r2_cv_mean")) if metrics else None
    circularity_risk = "high" if target_is_score else "medium"
    r2_interpretation = "heuristic_recoverability"
    circ_rows = [{
        "diagnostic": "target_circularity",
        "target": target,
        "target_group": target_card.get("feature_group", ""),
        "target_is_proxy": "true",
        "target_is_ground_truth": "false",
        "target_derived_from_model_features": "true" if target_is_score else "unknown",
        "n_model_features": len(features_used),
        "global_r2_cv": global_r2 if global_r2 is not None else "",
        "r2_interpretation": r2_interpretation,
        "predictive_power_claim": "false",
        "circularity_risk": circularity_risk,
        "explanation": (
            "O target e um score heuristico composto v5 derivado dos proprios "
            "condicionantes fisicos usados como features. R2 alto mede "
            "RECUPERABILIDADE do heuristico (consistencia interna), NAO poder "
            "preditivo contra evento real."
        ),
        "review_only": "true",
    }]

    # ---------- 2) Partial effect direction audit ----------
    dir_rows = []
    divergent = []
    for pe in partial:
        feat = pe["feature"]
        exp = pe.get("expected_direction", "not_scored")
        effect = _num(pe.get("partial_effect_low_to_high")) or 0.0
        agrees = pe.get("agrees_with_expected", "")
        if exp in AMBIGUOUS_DIRS:
            cls = "ambiguous_expected_direction"
            action = "acceptable_due_to_non_monotonicity"
        elif abs(effect) < NEAR_ZERO:
            cls = "near_zero_effect"
            action = "investigate_before_score_v6"
        elif agrees == "true":
            cls = "direction_agrees"
            action = ""
        else:
            cls = "direction_diverges"
            action = "investigate_before_score_v6"
        if cls in {"direction_diverges", "near_zero_effect"}:
            divergent.append(feat)
        dir_rows.append({
            "feature": feat,
            "feature_group": cards.get(feat, {}).get("feature_group", ""),
            "expected_direction": exp,
            "observed_effect": round(effect, 6),
            "observed_effect_sign": pe.get("effect_sign", ""),
            "agrees_with_expected": agrees,
            "partial_effect_class": cls,
            "divergence_action": action,
            "review_only": "true",
        })

    # ---------- 3) Regional stability ----------
    r2_vals = [v for v in (_num(r.get("r2_cv_mean")) for r in region_metrics) if v is not None]
    spread = (max(r2_vals) - min(r2_vals)) if r2_vals else None
    if spread is None:
        stability = "unstable"
    elif spread < REGION_SPREAD:
        stability = "stable"
    elif spread < 2 * REGION_SPREAD:
        stability = "moderately_variable"
    else:
        stability = "unstable"
    region_rows = []
    for r in region_metrics:
        region_rows.append({
            "region": r["scope"],
            "n": r.get("n", ""),
            "r2_cv_mean": r.get("r2_cv_mean", ""),
            "mae": r.get("mae", ""),
            "rmse": r.get("rmse", ""),
            "status": r.get("status", ""),
            "metric_basis": "proxy_not_event",
        })
    region_rows.append({
        "region": "ALL_SPREAD",
        "n": "",
        "r2_cv_mean": round(spread, 6) if spread is not None else "",
        "mae": "", "rmse": "",
        "status": f"regional_stability_status={stability}",
        "metric_basis": "proxy_not_event",
    })

    # ---------- 4) Feature action matrix ----------
    dir_by_feat = {d["feature"]: d for d in dir_rows}
    action_rows = []
    caution = manifest.get("optional_features_caution", [])
    excluded = manifest.get("optional_features_excluded", [])
    for feat in features_used:
        card = cards.get(feat, {})
        dd = dir_by_feat.get(feat, {})
        cls = dd.get("partial_effect_class", "")
        if cls in {"direction_diverges", "near_zero_effect"}:
            action = "manual_review_required"
        else:
            action = "advance_to_score_v6_candidate"
        action_rows.append({
            "feature": feat,
            "feature_group": card.get("feature_group", ""),
            "card_status": card.get("status", ""),
            "partial_effect_class": cls,
            "feature_action": action,
            "advance_to_score_v6_candidate": str(action == "advance_to_score_v6_candidate").lower(),
            "advance_to_spgam_candidate": str(action == "advance_to_score_v6_candidate").lower(),
            "advance_to_dino_comparison": "true",
            "weight_class": decision.get(feat, {}).get("weight_class", ""),
            "review_only": "true",
        })
    for feat in caution:
        card = cards.get(feat, {})
        action_rows.append({
            "feature": feat,
            "feature_group": card.get("feature_group", ""),
            "card_status": card.get("status", ""),
            "partial_effect_class": "ambiguous_expected_direction",
            "feature_action": "advance_to_dino_comparison",
            "advance_to_score_v6_candidate": "false",
            "advance_to_spgam_candidate": "false",
            "advance_to_dino_comparison": "true",
            "weight_class": decision.get(feat, {}).get("weight_class", ""),
            "review_only": "true",
        })
    # heuristic / blocked features summarised
    for feat in excluded:
        card = cards.get(feat, {})
        st = card.get("status", "")
        if st == "blocked_until_recomputed":
            act = "hold_until_recomputed"
        elif st == "unresolved":
            act = "hold_until_recomputed"
        else:
            act = "proxy_only_do_not_score"
        action_rows.append({
            "feature": feat,
            "feature_group": card.get("feature_group", ""),
            "card_status": st,
            "partial_effect_class": "",
            "feature_action": act,
            "advance_to_score_v6_candidate": "false",
            "advance_to_spgam_candidate": "false",
            "advance_to_dino_comparison": "false",
            "weight_class": decision.get(feat, {}).get("weight_class", ""),
            "review_only": "true",
        })

    # ---------- 5) Event overlay readiness (SUSC-07) ----------
    with MATRIX.open("r", encoding="utf-8", newline="") as f:
        matrix_header = next(csv.reader(f))
    matrix_cols = set(matrix_header)
    base_cols = set(load_csv(BASE_DS)[0].keys()) if BASE_DS.exists() else set()

    bbox_cols = ["xmin", "ymin", "xmax", "ymax"]
    has_bbox_matrix = all(c in matrix_cols for c in bbox_cols)

    overlay_rows = [
        {"required_field": "patch_id",
         "present_in_baseline": str("patch_id" in base_cols).lower(),
         "present_in_susc03_matrix": str("patch_id" in matrix_cols).lower(),
         "status": "available", "gap": "none",
         "note": "Chave primaria do patch."},
        {"required_field": "regiao",
         "present_in_baseline": str("regiao" in base_cols).lower(),
         "present_in_susc03_matrix": str("regiao" in matrix_cols).lower(),
         "status": "available", "gap": "none",
         "note": "Estratificacao regional."},
        {"required_field": "geometry_or_bbox",
         "present_in_baseline": str(all(c in base_cols for c in bbox_cols)).lower(),
         "present_in_susc03_matrix": str(has_bbox_matrix).lower(),
         "status": "available_in_matrix" if has_bbox_matrix else "missing",
         "gap": "not_in_baseline_dataset_join_from_susc03" if has_bbox_matrix else "no_bbox",
         "note": "bbox xmin/ymin/xmax/ymax existe na matriz SUSC-03; juntar por patch_id no overlay."},
        {"required_field": "score_or_proxy",
         "present_in_baseline": str(target in base_cols).lower(),
         "present_in_susc03_matrix": str(target in matrix_cols).lower(),
         "status": "available", "gap": "none",
         "note": "Proxy interno; NUNCA ground truth de evento."},
        {"required_field": "main_physical_features",
         "present_in_baseline": "true",
         "present_in_susc03_matrix": "true",
         "status": "available", "gap": "none",
         "note": f"{len(features_used)} features ready_for_methodology."},
        {"required_field": "future_documentary_evidence",
         "present_in_baseline": "false",
         "present_in_susc03_matrix": "false",
         "status": "missing", "gap": "external_acquisition_required",
         "note": "Evidencia documental/evento NAO existe ainda; e o objetivo do SUSC-07."},
        {"required_field": "date_or_period",
         "present_in_baseline": str("reference_date" in base_cols).lower(),
         "present_in_susc03_matrix": str("reference_date" in matrix_cols).lower(),
         "status": "weak", "gap": "reference_date_constant_2022-12-31",
         "note": "Data de referencia constante; periodo de evento real ainda ausente."},
    ]
    overlay_gaps = [r["required_field"] for r in overlay_rows if r["status"] in {"missing", "weak"}]

    # ---------- write outputs ----------
    OUT.mkdir(parents=True, exist_ok=True)
    _write_csv(CIRC_CSV, circ_rows)
    _write_csv(DIR_AUDIT_CSV, dir_rows)
    _write_csv(REGION_DIAG_CSV, region_rows)
    _write_csv(ACTION_CSV, action_rows)
    _write_csv(OVERLAY_CSV, overlay_rows)

    n_advance = sum(1 for r in action_rows if r["feature_action"] == "advance_to_score_v6_candidate")
    n_manual = sum(1 for r in action_rows if r["feature_action"] == "manual_review_required")

    summary = {
        "diagnostic_name": "SUSC-06B baseline proxy diagnostics",
        "input_baseline": "SUSC-06A",
        "diagnostic_status": "review_only_proxy_diagnostic",
        "result_tags": ["review_only", "proxy_based", "not_event_validation"],
        "target": target,
        "circularity_risk": circularity_risk,
        "r2_interpretation": r2_interpretation,
        "predictive_power_claim": False,
        "global_r2_cv": global_r2,
        "n_features_audited": len(features_used),
        "divergent_features": divergent,
        "n_divergent": len(divergent),
        "regional_stability_status": stability,
        "regional_r2_spread": round(spread, 6) if spread is not None else None,
        "n_advance_to_score_v6_candidate": n_advance,
        "n_manual_review_required": n_manual,
        "overlay_gaps_for_susc07": overlay_gaps,
        "allowed_for_training": False,
        "can_be_used_as_ground_truth": False,
        "review_only": True,
        "model_persisted": False,
        "scientific_status": "review_only_proxy_diagnostic_not_event_validation",
        "disclaimer": (
            "O SUSC-06B nao valida ocorrencia real de enchente. Diagnostica a "
            "consistencia interna de um baseline ajustado contra proxy heuristico."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\ncircularity_risk: {circularity_risk} | r2_interpretation: {r2_interpretation}")
    print(f"divergent features ({len(divergent)}): {divergent}")
    print(f"regional_stability_status: {stability} (spread={summary['regional_r2_spread']})")
    print(f"advance_to_score_v6_candidate: {n_advance} | manual_review_required: {n_manual}")
    print(f"overlay gaps for SUSC-07: {overlay_gaps}")
    print("\nreview_only, proxy_based, not_event_validation. No model. No ground truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
