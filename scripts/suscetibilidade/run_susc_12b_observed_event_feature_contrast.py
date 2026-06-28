"""SUSC-12B observed event feature contrast runner."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, read_json, write_csv, write_json  # noqa: E402

MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
SCORE = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
DATASET = ROOT / "datasets" / "suscetibilidade" / "susc_12b_event_feature_contrast_dataset_v1.csv"
CARDS = ROOT / "outputs_public" / "suscetibilidade" / "feature_cards" / "susc_feature_cards_v1.json"
ACTION = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_06B_feature_action_matrix.csv"
SCORE_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_score_v6_candidate_manifest_v1.json"

OUT = ROOT / "outputs_public" / "suscetibilidade"
FEATURE_CONTRAST = OUT / "SUSC_12B_observed_event_feature_contrast.csv"
SCORE_CONTRAST = OUT / "SUSC_12B_observed_event_score_contrast.csv"
REGION_CONTRAST = OUT / "SUSC_12B_observed_event_region_contrast.csv"
VALIDITY = OUT / "SUSC_12B_observed_event_feature_validity_matrix.csv"
SUMMARY = OUT / "SUSC_12B_observed_event_contrast_summary.json"

EVENT_GROUPS = {"observed_event_patch", "moderate_observed_event_patch"}
CONTROL_GROUP = "no_documented_event_control"


def _f(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def _median(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return float("nan")
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _round(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(round(value, 6))


def _rankdata(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def _mann_whitney(event: list[float], control: list[float]) -> str:
    if not event or not control:
        return ""
    vals = event + control
    ranks = _rankdata(vals)
    r1 = sum(ranks[:len(event)])
    u1 = r1 - len(event) * (len(event) + 1) / 2
    return str(round(u1, 6))


def _ks_stat(event: list[float], control: list[float]) -> str:
    if not event or not control:
        return ""
    vals = sorted(set(event + control))
    best = 0.0
    for v in vals:
        e = sum(1 for x in event if x <= v) / len(event)
        c = sum(1 for x in control if x <= v) / len(control)
        best = max(best, abs(e - c))
    return str(round(best, 6))


def _direction_agreement(diff: float, expected: str) -> tuple[str, str]:
    if expected in {"lower_increases", "higher_decreases"}:
        return ("lower_in_event", "true" if diff < 0 else "false")
    if expected in {"higher_increases", "lower_decreases"}:
        return ("higher_in_event", "true" if diff > 0 else "false")
    return ("ambiguous", "")


def main() -> int:
    print("=" * 60)
    print("SUSC-12B Observed Event Feature Contrast")
    print("=" * 60)
    for path in (MATRIX, SCORE, DATASET, CARDS, ACTION, SCORE_MANIFEST):
        if not path.exists():
            print(f"STOP: required input not found: {path}")
            return 1

    matrix_by_patch = {r["patch_id"]: r for r in read_csv(MATRIX)}
    score_by_patch = {r["patch_id"]: r for r in read_csv(SCORE)}
    dataset = read_csv(DATASET)
    cards = {c["feature_name"]: c for c in read_json(CARDS)["cards"]}
    actions = {r["feature"]: r for r in read_csv(ACTION)}
    features = read_json(SCORE_MANIFEST)["features_used"]
    blocked = set(read_json(SCORE_MANIFEST).get("features_excluded_flagged", []))

    event_rows = [r for r in dataset if r["contrast_group"] in EVENT_GROUPS]
    control_rows = [r for r in dataset if r["contrast_group"] == CONTROL_GROUP]
    event_patches = [matrix_by_patch[r["patch_id"]] for r in event_rows if r["patch_id"] in matrix_by_patch]
    control_patches = [matrix_by_patch[r["patch_id"]] for r in control_rows if r["patch_id"] in matrix_by_patch]

    feature_rows = []
    for feat in features:
        ev = [_f(r.get(feat)) for r in event_patches]
        ct = [_f(r.get(feat)) for r in control_patches]
        ev = [v for v in ev if v is not None]
        ct = [v for v in ct if v is not None]
        em, cm = _mean(ev), _mean(ct)
        diff = em - cm if ev and ct else float("nan")
        pooled = math.sqrt((_std(ev) ** 2 + _std(ct) ** 2) / 2) if len(ev) > 1 and len(ct) > 1 else float("nan")
        std_diff = diff / pooled if pooled and not math.isnan(pooled) else float("nan")
        expected = cards.get(feat, {}).get("expected_direction", "ambiguous")
        observed_dir, agree = _direction_agreement(diff, expected)
        confidence = "low_sample_size" if min(len(ev), len(ct)) < 20 else ("direction_agrees" if agree == "true" else "direction_diverges")
        if abs(std_diff) < 0.1 if not math.isnan(std_diff) else False:
            confidence = "near_zero_effect"
        feature_rows.append({
            "feature": feat,
            "feature_group": cards.get(feat, {}).get("feature_group", ""),
            "mean_event": _round(em),
            "mean_control": _round(cm),
            "median_event": _round(_median(ev)),
            "median_control": _round(_median(ct)),
            "effect_direction_observed": observed_dir,
            "expected_direction": expected,
            "direction_agreement": agree,
            "difference": _round(diff),
            "standardized_difference": _round(std_diff),
            "mann_whitney_if_available": _mann_whitney(ev, ct),
            "ks_if_available": _ks_stat(ev, ct),
            "sample_size_event": len(ev),
            "sample_size_control": len(ct),
            "confidence_flag": confidence,
            "feature_action_06b": actions.get(feat, {}).get("feature_action", ""),
            "blocked_in_v6": "true" if feat in blocked else "false",
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
        })
    write_csv(FEATURE_CONTRAST, feature_rows)

    def score_vals(rows: list[dict]) -> list[float]:
        vals = [_f(score_by_patch.get(r["patch_id"], {}).get("susc_score_v6_candidate")) for r in rows]
        return [v for v in vals if v is not None]

    evs, cts = score_vals(event_rows), score_vals(control_rows)
    score_rows = [
        {"group": "event_patch", "n": len(evs), "mean_score_v6": _round(_mean(evs)), "median_score_v6": _round(_median(evs)), "min_score_v6": _round(min(evs) if evs else None), "max_score_v6": _round(max(evs) if evs else None), "can_be_ground_truth": "false", "allowed_for_training": "false", "review_only": "true"},
        {"group": "no_documented_event_control", "n": len(cts), "mean_score_v6": _round(_mean(cts)), "median_score_v6": _round(_median(cts)), "min_score_v6": _round(min(cts) if cts else None), "max_score_v6": _round(max(cts) if cts else None), "can_be_ground_truth": "false", "allowed_for_training": "false", "review_only": "true"},
        {"group": "event_minus_control", "n": min(len(evs), len(cts)), "mean_score_v6": _round(_mean(evs) - _mean(cts) if evs and cts else None), "median_score_v6": "", "min_score_v6": "", "max_score_v6": "", "can_be_ground_truth": "false", "allowed_for_training": "false", "review_only": "true"},
    ]
    write_csv(SCORE_CONTRAST, score_rows)

    region_rows = []
    for region in sorted({r["region"] for r in dataset}):
        er = [r for r in event_rows if r["region"] == region]
        cr = [r for r in control_rows if r["region"] == region]
        es, cs = score_vals(er), score_vals(cr)
        region_rows.append({
            "region": region,
            "event_patch_count": len(er),
            "control_patch_count": len(cr),
            "mean_score_event": _round(_mean(es)),
            "mean_score_control": _round(_mean(cs)),
            "score_difference_event_minus_control": _round(_mean(es) - _mean(cs) if es and cs else None),
            "review_only": "true",
        })
    write_csv(REGION_CONTRAST, region_rows)

    validity_rows = []
    for row in feature_rows:
        validity_rows.append({
            "feature": row["feature"],
            "feature_group": row["feature_group"],
            "expected_direction": row["expected_direction"],
            "direction_agreement": row["direction_agreement"],
            "confidence_flag": row["confidence_flag"],
            "feature_action_06b": row["feature_action_06b"],
            "valid_for_auto_calibration": "false",
            "reason": "low_sample_size_review_only" if row["confidence_flag"] == "low_sample_size" else "review_only_requires_manual_review",
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
        })
    write_csv(VALIDITY, validity_rows)

    agree = [r["feature"] for r in feature_rows if r["direction_agreement"] == "true"]
    diverge = [r["feature"] for r in feature_rows if r["direction_agreement"] == "false"]
    inconclusive = [r["feature"] for r in feature_rows if not r["direction_agreement"]]
    summary = {
        "artifact": "SUSC-12B observed event feature contrast",
        "event_patch_count": len(event_rows),
        "control_patch_count": len(control_rows),
        "weak_context_patch_count": sum(1 for r in dataset if r["contrast_group"] == "weak_context_patch"),
        "score_v6_event_mean": round(_mean(evs), 6) if evs else None,
        "score_v6_control_mean": round(_mean(cts), 6) if cts else None,
        "score_v6_event_minus_control": round(_mean(evs) - _mean(cts), 6) if evs and cts else None,
        "features_direction_agree": agree,
        "features_direction_diverge": diverge,
        "features_inconclusive": inconclusive,
        "feature_confidence_counts": dict(Counter(r["confidence_flag"] for r in feature_rows)),
        "region_contrast": region_rows,
        "low_sample_size": len(event_rows) < 20,
        "can_be_ground_truth": False,
        "allowed_for_training": False,
        "review_only": True,
    }
    write_json(SUMMARY, summary)

    print(f"event patches: {len(event_rows)} | controls: {len(control_rows)}")
    print("score event mean:", _round(_mean(evs)), "control mean:", _round(_mean(cts)))
    print("direction agree:", len(agree), "diverge:", len(diverge), "inconclusive:", len(inconclusive))
    print("review-only. No ground truth. No supervised training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
