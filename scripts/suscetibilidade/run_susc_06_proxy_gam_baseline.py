"""
SUSC-06A Proxy GAM/SPGAM-style Baseline (interpretable, review-only)

Fits an interpretable baseline AGAINST an internal proxy target (heuristic v5
score / label), purely as a methodological diagnostic. This is NOT a flood
predictor. It does NOT validate real occurrence, does NOT create ground truth,
and does NOT unlock supervised training. No model is persisted to disk.

Target selection priority:
  1. final susceptibility score  (score_evento_enchente_potencial_v5_core)
  2. composite v5 score          (any other score_* column)
  3. heuristic label fallback    (label_* column)

Modeling backend priority: pygam -> statsmodels GAM -> sklearn spline fallback
(SplineTransformer + StandardScaler + Ridge/LogisticRegression). No new packages
are installed; whatever is available is used.

Outputs (outputs_public/suscetibilidade/):
  - SUSC_06_proxy_baseline_feature_table.csv
  - SUSC_06_proxy_baseline_metrics.csv
  - SUSC_06_proxy_baseline_region_metrics.csv
  - SUSC_06_proxy_baseline_partial_effects.csv
  - SUSC_06_proxy_baseline_target_policy.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())

DATASET = ROOT / "datasets" / "suscetibilidade" / "susc_06_proxy_baseline_dataset_v1.csv"
DS_MANIFEST = (
    ROOT / "manifests" / "suscetibilidade" / "susc_06_proxy_baseline_dataset_manifest_v1.json"
)
CARDS_JSON = (
    ROOT / "outputs_public" / "suscetibilidade" / "feature_cards" / "susc_feature_cards_v1.json"
)
OUT_DIR = ROOT / "outputs_public" / "suscetibilidade"

FEATURE_TABLE = OUT_DIR / "SUSC_06_proxy_baseline_feature_table.csv"
METRICS = OUT_DIR / "SUSC_06_proxy_baseline_metrics.csv"
REGION_METRICS = OUT_DIR / "SUSC_06_proxy_baseline_region_metrics.csv"
PARTIAL_EFFECTS = OUT_DIR / "SUSC_06_proxy_baseline_partial_effects.csv"
TARGET_POLICY = OUT_DIR / "SUSC_06_proxy_baseline_target_policy.json"

FINAL_SCORE = "score_evento_enchente_potencial_v5_core"
RANDOM_STATE = 42

EXPECTED_SIGN = {
    "higher_increases": 1.0,
    "lower_increases": -1.0,
    "higher_decreases": -1.0,
    "lower_decreases": 1.0,
}


def load_dataset():
    with DATASET.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or []
    return cols, rows


def choose_target(cols, cards_by_name):
    """Pick proxy target by priority; record reason."""
    score_cols = [c for c in cols if cards_by_name.get(c, {}).get("feature_group") == "score"]
    label_cols = [c for c in cols if cards_by_name.get(c, {}).get("feature_group") == "heuristic_label"]

    if FINAL_SCORE in cols:
        return FINAL_SCORE, "continuous", "final_susceptibility_score", score_cols, label_cols
    if score_cols:
        return score_cols[0], "continuous", "composite_v5_score", score_cols, label_cols
    if label_cols:
        return label_cols[0], "binary", "heuristic_label_fallback", score_cols, label_cols
    return None, None, None, score_cols, label_cols


def to_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan


def main() -> int:
    print("=" * 60)
    print("SUSC-06A Proxy GAM/SPGAM-style Baseline (review-only)")
    print("=" * 60)

    for p in (DATASET, DS_MANIFEST, CARDS_JSON):
        if not p.exists():
            print(f"STOP: required input not found: {p}")
            return 1

    with CARDS_JSON.open("r", encoding="utf-8") as f:
        cards = json.load(f)["cards"]
    cards_by_name = {c["feature_name"]: c for c in cards}

    with DS_MANIFEST.open("r", encoding="utf-8") as f:
        ds_manifest = json.load(f)
    features = [f for f in ds_manifest["features_used"]]  # ready_for_methodology

    cols, rows = load_dataset()

    target, target_type, target_reason, score_cols, label_cols = choose_target(cols, cards_by_name)
    if target is None:
        # Rule 13: no proxy target -> readiness report only, no model.
        policy = {
            "baseline_status": "proxy_baseline_review_only",
            "chosen_target": None,
            "reason": "NO_PROXY_TARGET_AVAILABLE",
            "model_fitted": False,
            "is_ground_truth": False,
        }
        TARGET_POLICY.write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("No proxy target available; readiness only, no model fitted.")
        return 0

    # backend detection (no install; whatever is available is used)
    import importlib.util as _ilu
    if _ilu.find_spec("pygam") is not None:
        backend = "pygam"
    elif _ilu.find_spec("statsmodels.gam.api") is not None:
        backend = "statsmodels_gam"
    else:
        backend = "sklearn_spline_fallback"

    print(f"\nTarget: {target} ({target_type}) via {target_reason}")
    print(f"Backend: {backend}")

    policy = {
        "baseline_name": "SUSC-06A proxy GAM/SPGAM-style baseline",
        "baseline_status": "proxy_baseline_review_only",
        "chosen_target": target,
        "target_type": target_type,
        "target_priority_matched": target_reason,
        "target_is_proxy": True,
        "target_is_ground_truth": False,
        "candidate_score_columns": score_cols,
        "candidate_label_columns": label_cols,
        "selection_reason": (
            f"Target '{target}' escolhido pela prioridade '{target_reason}'. "
            "E um proxy heuristico interno (score/label v5), NUNCA ocorrencia confirmada."
        ),
        "backend": backend,
        "model_persisted": False,
        "metric_basis": "metrics_against_internal_proxy_not_real_event",
        "allowed_for_training": False,
        "can_be_used_as_ground_truth": False,
        "review_only": True,
    }
    TARGET_POLICY.write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # build matrices
    feat_present = [f for f in features if f in cols]
    X = np.array([[to_float(r[f]) for f in feat_present] for r in rows], dtype=float)
    y = np.array([to_float(r[target]) for r in rows], dtype=float)
    regions = [r.get("regiao", "") for r in rows]

    # drop rows with nan (matrix has 0 missing, but be safe)
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[valid], y[valid]
    regions = [reg for reg, ok in zip(regions, valid) if ok]

    from sklearn.preprocessing import StandardScaler, SplineTransformer
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
    from sklearn.metrics import (
        r2_score, mean_absolute_error, mean_squared_error,
        accuracy_score, balanced_accuracy_score, roc_auc_score,
    )

    def make_pipeline(binary):
        steps = [
            ("scaler", StandardScaler()),
            ("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False)),
        ]
        if binary:
            steps.append(("model", LogisticRegression(max_iter=2000, C=1.0)))
        else:
            steps.append(("model", Ridge(alpha=1.0)))
        return Pipeline(steps)

    binary = target_type == "binary"

    def fit_eval(Xs, ys, scope):
        out = {"scope": scope, "target": target, "target_type": target_type,
               "n": int(len(ys)), "backend": backend,
               "metric_basis": "proxy_not_event"}
        if len(ys) < 20 or (binary and len(np.unique(ys)) < 2):
            out["status"] = "INSUFFICIENT_DATA"
            return out
        pipe = make_pipeline(binary)
        pipe.fit(Xs, ys)
        pred = pipe.predict(Xs)
        if binary:
            out["accuracy"] = round(float(accuracy_score(ys, pred)), 4)
            out["balanced_accuracy"] = round(float(balanced_accuracy_score(ys, pred)), 4)
            try:
                proba = pipe.predict_proba(Xs)[:, 1]
                out["roc_auc"] = round(float(roc_auc_score(ys, proba)), 4)
            except Exception:
                out["roc_auc"] = ""
            try:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                scores = cross_val_score(make_pipeline(True), Xs, ys, cv=cv, scoring="balanced_accuracy")
                out["balanced_accuracy_cv_mean"] = round(float(scores.mean()), 4)
                out["balanced_accuracy_cv_std"] = round(float(scores.std()), 4)
            except Exception:
                pass
        else:
            out["r2_in_sample"] = round(float(r2_score(ys, pred)), 4)
            out["mae"] = round(float(mean_absolute_error(ys, pred)), 4)
            out["rmse"] = round(float(np.sqrt(mean_squared_error(ys, pred))), 4)
            try:
                cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                scores = cross_val_score(make_pipeline(False), Xs, ys, cv=cv, scoring="r2")
                out["r2_cv_mean"] = round(float(scores.mean()), 4)
                out["r2_cv_std"] = round(float(scores.std()), 4)
            except Exception:
                pass
        out["status"] = "FITTED_PROXY_BASELINE_REVIEW_ONLY"
        return out

    # global
    global_metrics = fit_eval(X, y, "global")

    # per region
    region_rows = []
    uniq_regions = sorted(set(regions))
    for reg in uniq_regions:
        mask = np.array([rr == reg for rr in regions])
        if mask.sum() >= 20:
            region_rows.append(fit_eval(X[mask], y[mask], reg))
        else:
            region_rows.append({"scope": reg, "target": target, "n": int(mask.sum()),
                                "status": "INSUFFICIENT_DATA", "metric_basis": "proxy_not_event"})

    # feature table: standardized linear coef (interpretable sign) + expected direction
    from sklearn.linear_model import Ridge as RidgeLin
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    if binary:
        lin = LogisticRegression(max_iter=2000).fit(Xs, y)
        coefs = lin.coef_[0]
    else:
        lin = RidgeLin(alpha=1.0).fit(Xs, y)
        coefs = lin.coef_

    feature_table = []
    for j, feat in enumerate(feat_present):
        card = cards_by_name.get(feat, {})
        exp_dir = card.get("expected_direction", "not_scored")
        exp_sign = EXPECTED_SIGN.get(exp_dir)
        coef = float(coefs[j])
        obs_sign = "positive" if coef > 0 else ("negative" if coef < 0 else "zero")
        agrees = "" if exp_sign is None else str((coef > 0) == (exp_sign > 0)).lower()
        feature_table.append({
            "feature": feat,
            "feature_group": card.get("feature_group", ""),
            "expected_direction": exp_dir,
            "standardized_linear_coef": round(coef, 6),
            "observed_effect_sign": obs_sign,
            "agrees_with_expected": agrees,
            "weight_class": card.get("weight_class", ""),
        })

    # partial effects via spline pipeline: sweep each feature p10->p90, others at median
    partial = []
    if global_metrics.get("status", "").startswith("FITTED"):
        pipe = make_pipeline(binary).fit(X, y)
        med = np.median(X, axis=0)
        for j, feat in enumerate(feat_present):
            lo, hi = np.percentile(X[:, j], 10), np.percentile(X[:, j], 90)
            base = np.tile(med, (2, 1))
            base[0, j] = lo
            base[1, j] = hi
            if binary:
                try:
                    pv = pipe.predict_proba(base)[:, 1]
                except Exception:
                    pv = pipe.predict(base)
            else:
                pv = pipe.predict(base)
            effect = float(pv[1] - pv[0])
            card = cards_by_name.get(feat, {})
            exp_dir = card.get("expected_direction", "not_scored")
            exp_sign = EXPECTED_SIGN.get(exp_dir)
            eff_sign = "positive" if effect > 0 else ("negative" if effect < 0 else "zero")
            agrees = "" if exp_sign is None else str((effect > 0) == (exp_sign > 0)).lower()
            partial.append({
                "feature": feat,
                "x_low_p10": round(float(lo), 6),
                "x_high_p90": round(float(hi), 6),
                "pred_low": round(float(pv[0]), 6),
                "pred_high": round(float(pv[1]), 6),
                "partial_effect_low_to_high": round(effect, 6),
                "effect_sign": eff_sign,
                "expected_direction": exp_dir,
                "agrees_with_expected": agrees,
            })

    # write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(FEATURE_TABLE, feature_table)
    _write_csv(METRICS, [global_metrics])
    _write_csv(REGION_METRICS, region_rows)
    _write_csv(PARTIAL_EFFECTS, partial)

    print(f"\n[global] {global_metrics}")
    print(f"[regions] {len(region_rows)} region rows")
    agree_n = sum(1 for r in partial if r["agrees_with_expected"] == "true")
    scored = sum(1 for r in partial if r["agrees_with_expected"] in {"true", "false"})
    print(f"[partial effects] {agree_n}/{scored} agree with expected direction")
    print("\nMetrics are against an INTERNAL PROXY, not a real flood event.")
    print("No model persisted. No ground truth. No training unlocked.")
    return 0


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
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


if __name__ == "__main__":
    sys.exit(main())
