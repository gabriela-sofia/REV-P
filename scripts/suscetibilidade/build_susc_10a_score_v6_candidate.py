"""
SUSC-10A Score v6 Candidate Builder (deterministic, explainable, review-only)

Builds a transparent, non-supervised candidate susceptibility score using ONLY
gate-approved features (SUSC-06B advance_to_score_v6 + cards ready_for_methodology).
No training, no ground truth, no proxy/label/score-v5 as feature. The four
features flagged in SUSC-06B are kept OUT of the main score (diagnostic only).

Deterministic pipeline per feature: winsorize 1/99 -> robust min-max -> direction
-aware orientation (so higher normalized = more susceptible). Subindices are
weighted into susc_score_v6_candidate in [0,1], with per-patch top contributions.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, write_csv, write_json, write_markdown, read_json, sha256_file, rel  # noqa: E402

MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
DECISION = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_06B_feature_action_matrix.csv"
CARDS = ROOT / "outputs_public" / "suscetibilidade" / "feature_cards" / "susc_feature_cards_v1.json"
OVERLAY_07B = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_refined_patch_event_overlay.csv"

OUT_SCORE = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
OUT_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_score_v6_candidate_manifest_v1.json"
OUT_SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_10A_score_v6_candidate_summary.csv"
OUT_BY_REGION = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_10A_score_v6_candidate_by_region.csv"
OUT_TOP = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_10A_score_v6_candidate_top_patches.csv"
OUT_CONTRIB = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_10A_score_v6_candidate_feature_contributions.csv"
OUT_REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_10A_score_v6_candidate_methodology_report.md"
OUT_LIMITS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_10A_score_v6_candidate_limitations.json"

FLAGGED_OUT = {"curvature_laplacian_mean", "rain_3d_7d_ratio", "flow_acc_log_p75", "water_occurrence_patch"}
FORBIDDEN_GROUPS = {"score", "heuristic_label", "proxy_v5"}

WEIGHTS = {
    "topography_hydrology_index": 0.40,
    "rainfall_trigger_index": 0.25,
    "urban_spectral_index": 0.20,
    "vegetation_mitigation_index": -0.10,
    "evidence_support_index": 0.05,
}
# group -> subindex bucket
SUBINDEX_OF_GROUP = {
    "topography": "topography_hydrology_index",
    "topography_hydrology": "topography_hydrology_index",
    "hydrology": "topography_hydrology_index",
    "precipitation": "rainfall_trigger_index",
    "land_use": None,       # split by feature below
    "spectral_index": None,  # split by feature below
}
MITIGATION_FEATURES = {"ndvi_mean", "vegetation_prop"}
URBAN_SPECTRAL_FEATURES = {"ndbi_mean", "urban_prop", "mndwi_mean"}


def _num(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def winsorize(vals, lo=0.01, hi=0.99):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return vals
    lo_v = s[max(0, int(lo * n))]
    hi_v = s[min(n - 1, int(hi * n))]
    return [min(max(v, lo_v), hi_v) for v in vals], lo_v, hi_v


def robust_minmax(vals):
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [0.5 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def subindex_for(feat, group):
    if feat in MITIGATION_FEATURES:
        return "vegetation_mitigation_index"
    if feat in URBAN_SPECTRAL_FEATURES:
        return "urban_spectral_index"
    return SUBINDEX_OF_GROUP.get(group)


def main() -> int:
    print("=" * 60)
    print("SUSC-10A Score v6 Candidate Builder (deterministic, review-only)")
    print("=" * 60)

    for p in (MATRIX, DECISION, CARDS):
        if not p.exists():
            print(f"STOP: required input not found: {p}")
            return 1

    rows = read_csv(MATRIX)
    decision = {r["feature"]: r for r in read_csv(DECISION)}
    cards = {c["feature_name"]: c for c in read_json(CARDS)["cards"]}

    # select gate-approved features for the MAIN score
    selected = []
    for feat, d in decision.items():
        if d.get("feature_action") != "advance_to_score_v6_candidate":
            continue
        if feat in FLAGGED_OUT:
            continue
        card = cards.get(feat, {})
        if card.get("status") != "ready_for_methodology":
            continue
        if card.get("feature_group") in FORBIDDEN_GROUPS:
            continue
        if feat not in rows[0]:
            continue
        selected.append(feat)
    selected = sorted(selected)

    # direction-aware normalization per feature
    feat_norm = {}     # feat -> list of contribution-in-[0,1]
    feat_meta = {}     # feat -> (group, subindex, direction)
    for feat in selected:
        raw = [_num(r.get(feat)) for r in rows]
        if any(v is None for v in raw):
            continue
        wins, _, _ = winsorize(raw)
        mm = robust_minmax(wins)
        direction = cards[feat].get("expected_direction", "higher_increases")
        # orient so higher = more susceptible contribution
        if direction in {"lower_increases", "higher_decreases"}:
            contrib = [1.0 - v for v in mm]
        else:
            contrib = mm
        group = cards[feat].get("feature_group", "")
        sidx = subindex_for(feat, group)
        if sidx is None:
            continue
        feat_norm[feat] = contrib
        feat_meta[feat] = (group, sidx, direction)

    used_features = sorted(feat_norm.keys())

    # evidence support per patch (low modulator) from 07B spatial relations
    support = {r["patch_id"]: 0.0 for r in rows}
    if OVERLAY_07B.exists():
        for o in read_csv(OVERLAY_07B):
            if o["spatial_relation"] in {"bbox_overlap", "near_patch_buffer_candidate"}:
                pid = o["patch_id"]
                if pid in support:
                    support[pid] = min(1.0, support[pid] + 0.5)

    # build subindices per patch
    n = len(rows)
    subindex_features = {k: [] for k in WEIGHTS if k != "evidence_support_index"}
    for feat in used_features:
        subindex_features[feat_meta[feat][1]].append(feat)

    raw_scores = []
    contrib_rows = []
    per_patch_subidx = []
    for i, r in enumerate(rows):
        sub = {}
        for sidx, feats in subindex_features.items():
            if feats:
                sub[sidx] = sum(feat_norm[f][i] for f in feats) / len(feats)
            else:
                sub[sidx] = 0.0
        sub["evidence_support_index"] = support.get(r["patch_id"], 0.0)
        per_patch_subidx.append(sub)
        raw = sum(WEIGHTS[k] * sub.get(k, 0.0) for k in WEIGHTS)
        raw_scores.append(raw)
        # per-feature weighted contribution for explainability
        for f in used_features:
            w = WEIGHTS[feat_meta[f][1]]
            k = len(subindex_features[feat_meta[f][1]]) or 1
            contrib_rows.append({
                "patch_id": r["patch_id"], "feature": f,
                "subindex": feat_meta[f][1],
                "normalized_contribution": round(feat_norm[f][i], 6),
                "weighted_contribution": round(w * feat_norm[f][i] / k, 6),
            })

    # final score: robust min-max of raw to [0,1]
    final = robust_minmax(raw_scores)
    # global tertiles
    s = sorted(final)
    t1, t2 = s[int(0.33 * n)], s[int(0.66 * n)]

    def cls(v):
        return "low" if v <= t1 else ("high" if v > t2 else "medium")

    # top contributions text per patch
    contrib_by_patch = {}
    for c in contrib_rows:
        contrib_by_patch.setdefault(c["patch_id"], []).append(c)

    score_rows = []
    for i, r in enumerate(rows):
        pid = r["patch_id"]
        sub = per_patch_subidx[i]
        tops = sorted(contrib_by_patch.get(pid, []), key=lambda c: c["weighted_contribution"], reverse=True)[:5]
        top_str = "; ".join(f"{c['feature']}={c['weighted_contribution']}" for c in tops)
        reason = (f"Score dominado por {tops[0]['feature']} e {tops[1]['feature']}" if len(tops) >= 2
                  else "Score baseado nas features fisicas/hidro aprovadas")
        score_rows.append({
            "patch_id": pid, "regiao": r.get("regiao", ""),
            "topography_hydrology_index": round(sub["topography_hydrology_index"], 6),
            "rainfall_trigger_index": round(sub["rainfall_trigger_index"], 6),
            "urban_spectral_index": round(sub["urban_spectral_index"], 6),
            "vegetation_mitigation_index": round(sub["vegetation_mitigation_index"], 6),
            "evidence_support_index": round(sub["evidence_support_index"], 6),
            "susc_score_v6_candidate": round(final[i], 6),
            "susc_class_v6_candidate": cls(final[i]),
            "top5_feature_contributions": top_str,
            "reason": reason + " (review-only; nao e ocorrencia confirmada).",
            "caution_flags": "diagnostic_features_excluded:curvature/rain_3d_7d_ratio/flow_acc_log_p75/water_occurrence",
            "can_be_used_as_ground_truth": "false", "allowed_for_training": "false", "review_only": "true",
        })

    write_csv(OUT_SCORE, score_rows)

    # diagnostic-only normalized values for the 4 flagged features (not in score)
    diag = {}
    for feat in FLAGGED_OUT:
        if feat in rows[0]:
            raw = [_num(r.get(feat)) for r in rows]
            if all(v is not None for v in raw):
                wins, _, _ = winsorize(raw)
                diag[feat] = robust_minmax(wins)

    # summary
    from collections import Counter
    cls_counts = dict(Counter(s["susc_class_v6_candidate"] for s in score_rows))
    vals = [s["susc_score_v6_candidate"] for s in score_rows]
    summary_rows = [
        {"metric": "n_patches", "value": n},
        {"metric": "score_min", "value": round(min(vals), 6)},
        {"metric": "score_max", "value": round(max(vals), 6)},
        {"metric": "score_mean", "value": round(sum(vals) / n, 6)},
        {"metric": "class_low", "value": cls_counts.get("low", 0)},
        {"metric": "class_medium", "value": cls_counts.get("medium", 0)},
        {"metric": "class_high", "value": cls_counts.get("high", 0)},
        {"metric": "n_features_used", "value": len(used_features)},
        {"metric": "n_features_excluded_flagged", "value": len(FLAGGED_OUT)},
    ]
    write_csv(OUT_SUMMARY, summary_rows, ["metric", "value"])

    # by region
    region_rows = []
    for reg in sorted({s["regiao"] for s in score_rows}):
        rv = [s["susc_score_v6_candidate"] for s in score_rows if s["regiao"] == reg]
        rc = Counter(s["susc_class_v6_candidate"] for s in score_rows if s["regiao"] == reg)
        region_rows.append({
            "regiao": reg, "n_patches": len(rv),
            "score_mean": round(sum(rv) / len(rv), 6), "score_min": round(min(rv), 6),
            "score_max": round(max(rv), 6),
            "class_low": rc.get("low", 0), "class_medium": rc.get("medium", 0),
            "class_high": rc.get("high", 0), "metric_basis": "deterministic_candidate_review_only",
        })
    write_csv(OUT_BY_REGION, region_rows)

    # top patches
    top = sorted(score_rows, key=lambda s: s["susc_score_v6_candidate"], reverse=True)[:20]
    write_csv(OUT_TOP, [{
        "patch_id": s["patch_id"], "regiao": s["regiao"],
        "susc_score_v6_candidate": s["susc_score_v6_candidate"],
        "susc_class_v6_candidate": s["susc_class_v6_candidate"],
        "top5_feature_contributions": s["top5_feature_contributions"],
    } for s in top], ["patch_id", "regiao", "susc_score_v6_candidate", "susc_class_v6_candidate", "top5_feature_contributions"])

    # feature contributions (aggregate mean per feature)
    agg = {}
    for c in contrib_rows:
        agg.setdefault(c["feature"], []).append(c["weighted_contribution"])
    write_csv(OUT_CONTRIB, [{
        "feature": f, "subindex": feat_meta[f][1], "expected_direction": feat_meta[f][2],
        "mean_weighted_contribution": round(sum(v) / len(v), 6),
        "weight_class": decision.get(f, {}).get("weight_class", ""),
        "review_only": "true",
    } for f, v in sorted(agg.items())],
        ["feature", "subindex", "expected_direction", "mean_weighted_contribution", "weight_class", "review_only"])

    digest = sha256_file(OUT_SCORE)
    write_json(OUT_MANIFEST, {
        "artifact": rel(OUT_SCORE), "source": rel(MATRIX),
        "rows": n, "sha256": digest, "n_features_used": len(used_features),
        "features_used": used_features,
        "features_excluded_flagged": sorted(FLAGGED_OUT),
        "weights": WEIGHTS, "class_method": "global_tertiles",
        "tertiles": [round(t1, 6), round(t2, 6)],
        "uses_label_as_feature": False, "uses_score_v5_as_feature": False,
        "uses_proxy_target": False, "trained_model": False, "model_persisted": False,
        "allowed_for_training": False, "can_be_used_as_ground_truth": False, "review_only": True,
        "scientific_status": "deterministic_candidate_score_not_supervised_not_ground_truth",
    })

    write_json(OUT_LIMITS, {
        "stage": "SUSC-10A score v6 candidate",
        "deterministic": True, "supervised": False, "trained": False,
        "ground_truth_used": False, "proxy_target_used": False,
        "features_excluded_from_main_score": sorted(FLAGGED_OUT),
        "diagnostic_features_normalized": sorted(diag.keys()),
        "evidence_support_weight": WEIGHTS["evidence_support_index"],
        "key_limitations": [
            "Pesos sao escolha metodologica documentada, nao aprendidos.",
            "Score e candidato review-only; nao valida ocorrencia real de enchente.",
            "evidence_support_index usa aderencia espacial review-only (07B), nao ground truth.",
            "As 4 features sinalizadas no SUSC-06B ficam fora do score principal (diagnostico).",
            "Classes por tercis globais; sensiveis a distribuicao da amostra (300 patches).",
        ],
        "can_be_used_as_ground_truth": False, "allowed_for_training": False, "review_only": True,
    })

    write_markdown(OUT_REPORT, "\n".join([
        "# SUSC-10A — Score v6 Candidato (determinístico, review-only)", "",
        "> O SUSC-10A cria um score v6 candidato determinístico e review-only. Ele não é modelo "
        "supervisionado, não usa ground truth, não valida ocorrência real de enchente e não autoriza "
        "afirmações de evento observado por patch.", "",
        "## Construção",
        f"- Features usadas (aprovadas pelos gates): **{len(used_features)}**.",
        f"- Excluídas do score principal (sinalizadas SUSC-06B, diagnóstico): {sorted(FLAGGED_OUT)}.",
        "- Normalização determinística: winsorize 1/99 → robust min-max → orientação por direção esperada.",
        "- Subíndices e pesos documentados: " + ", ".join(f"{k}={v}" for k, v in WEIGHTS.items()) + ".",
        "- Score final reescalado para [0,1]; classes por tercis globais (low/medium/high).", "",
        "## Resultados",
        f"- n=300; score médio={round(sum(vals)/n,4)}; classes: {cls_counts}.",
        f"- Por região: ver `SUSC_10A_score_v6_candidate_by_region.csv`.",
        f"- Top 20 patches: `SUSC_10A_score_v6_candidate_top_patches.csv`.",
        f"- Contribuições por feature: `SUSC_10A_score_v6_candidate_feature_contributions.csv`.", "",
        "## O que NÃO é",
        "- Não usa label heurístico, score v5 ou proxy como feature/target.",
        "- Não treina modelo; não persiste modelo; não cria ground truth.",
        "- Não é validação de ocorrência por patch.", "",
        "## Limitações",
        "Pesos são escolha metodológica (não aprendidos). `evidence_support_index` usa aderência "
        "espacial review-only (07B), não ground truth. As 4 features sinalizadas ficam em diagnóstico.", "",
        "> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.", "",
    ]))

    print(f"\nscore v6 candidate: {n} patches | features used: {len(used_features)} | classes: {cls_counts}")
    print(f"sha256: {digest}")
    print(f"top patch: {top[0]['patch_id']} ({top[0]['regiao']}) = {top[0]['susc_score_v6_candidate']}")
    print("\nDeterministic candidate. No training. No ground truth. review-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
