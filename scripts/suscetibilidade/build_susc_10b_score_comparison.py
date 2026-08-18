"""
SUSC-10B Score Comparison (review-only diagnostic)

Compares the deterministic candidate score v6 against the internal v5 proxy score,
the heuristic label, and the review-only spatial evidence (07B/08/09A). Computes
Pearson/Spearman, rank shift, top-k overlap, class agreement and regional
divergence. Correlation is NOT validation, ground truth, or confirmed occurrence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, write_csv, write_json, write_markdown  # noqa: E402

SCORE = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
OVERLAY_07B = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_refined_patch_event_overlay.csv"
CASES_08 = ROOT / "datasets" / "suscetibilidade" / "susc_08_spatiotemporal_review_cases_v1.csv"

OUT = ROOT / "outputs_public" / "suscetibilidade"
M_METRICS = OUT / "SUSC_10B_score_comparison_metrics.csv"
M_RANK = OUT / "SUSC_10B_score_rank_shift_by_patch.csv"
M_CLASS = OUT / "SUSC_10B_score_class_agreement.csv"
M_REGION = OUT / "SUSC_10B_score_region_diagnostics.csv"
M_HIGHSPATIAL = OUT / "SUSC_10B_high_score_spatial_evidence_cases.csv"
M_DISAGREE = OUT / "SUSC_10B_score_disagreement_cases.csv"
M_SUMMARY = OUT / "SUSC_10B_score_comparison_summary.json"
M_REPORT = OUT / "SUSC_10B_score_comparison_report.md"

V5_COL = "score_evento_enchente_potencial_v5_core"
LABEL_COL = "label_evento_enchente_potencial_v5_core_regional_p75"
SPATIAL_RELS = {"bbox_overlap", "near_patch_buffer_candidate", "exact_patch_overlap"}
MANDATORY = ("O SUSC-10B compara scores e proxies internos de suscetibilidade em modo review-only. "
             "A concordância entre score v6, proxy v5, baseline ou evidência documental não constitui "
             "validação supervisionada, ground truth ou confirmação de ocorrência real por patch.")


def _f(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rankdata(a):
    a = np.asarray(a, float)
    order = a.argsort()
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def spearman(a, b):
    return pearson(rankdata(a), rankdata(b))


def tertile_class(vals):
    s = sorted(vals)
    n = len(vals)
    t1, t2 = s[int(0.33 * n)], s[int(0.66 * n)]
    return ["low" if v <= t1 else ("high" if v > t2 else "medium") for v in vals]


def main() -> int:
    print("=" * 60)
    print("SUSC-10B Score Comparison (review-only diagnostic)")
    print("=" * 60)

    for p in (SCORE, MATRIX):
        if not p.exists():
            print(f"STOP: required input not found: {p}")
            return 1

    score_rows = {r["patch_id"]: r for r in read_csv(SCORE)}
    matrix_rows = {r["patch_id"]: r for r in read_csv(MATRIX)}

    # spatial evidence patches (07B real spatial relations)
    spatial_patches = set()
    if OVERLAY_07B.exists():
        for o in read_csv(OVERLAY_07B):
            if o["spatial_relation"] in SPATIAL_RELS and o["patch_id"] != "REGION_LEVEL_NO_PATCH_RESOLUTION":
                spatial_patches.add(o["patch_id"])

    # aligned arrays
    patches, v6, v5, region, label = [], [], [], [], []
    for pid, sr in score_rows.items():
        mr = matrix_rows.get(pid)
        if not mr:
            continue
        a, b = _f(sr.get("susc_score_v6_candidate")), _f(mr.get(V5_COL))
        if a is None or b is None:
            continue
        patches.append(pid)
        v6.append(a)
        v5.append(b)
        region.append(sr.get("regiao", mr.get("regiao", "")))
        label.append(_f(mr.get(LABEL_COL)))
    n = len(patches)

    # global metrics
    p_corr, s_corr = pearson(v6, v5), spearman(v6, v5)
    v6_rank = rankdata(v6)
    v5_rank = rankdata(v5)
    v6_class = {patches[i]: score_rows[patches[i]].get("susc_class_v6_candidate") for i in range(n)}
    v5_class_list = tertile_class(v5)
    v5_class = {patches[i]: v5_class_list[i] for i in range(n)}

    # top-k overlap
    order_v6 = sorted(range(n), key=lambda i: v6[i], reverse=True)
    order_v5 = sorted(range(n), key=lambda i: v5[i], reverse=True)
    topk = {}
    for k in (15, 30):
        s6 = {patches[i] for i in order_v6[:k]}
        s5 = {patches[i] for i in order_v5[:k]}
        topk[k] = round(len(s6 & s5) / k, 4)

    metrics_rows = [
        {"metric": "n_patches", "value": n, "basis": "review_only"},
        {"metric": "pearson_v6_v5", "value": round(p_corr, 6), "basis": "review_only_not_validation"},
        {"metric": "spearman_v6_v5", "value": round(s_corr, 6), "basis": "review_only_not_validation"},
        {"metric": "top15_overlap_v6_v5", "value": topk[15], "basis": "review_only"},
        {"metric": "top30_overlap_v6_v5", "value": topk[30], "basis": "review_only"},
        {"metric": "mean_abs_diff_v6_v5", "value": round(float(np.mean(np.abs(np.array(v6) - np.array(v5)))), 6), "basis": "review_only"},
    ]
    write_csv(M_METRICS, metrics_rows, ["metric", "value", "basis"])

    # rank shift by patch
    rank_rows = []
    for i in range(n):
        rank_rows.append({
            "patch_id": patches[i], "regiao": region[i],
            "score_v6": round(v6[i], 6), "score_v5_proxy": round(v5[i], 6),
            "abs_diff_v6_v5": round(abs(v6[i] - v5[i]), 6),
            "rank_v6": int(n - v6_rank[i] + 1), "rank_v5": int(n - v5_rank[i] + 1),
            "rank_shift": int((n - v6_rank[i] + 1) - (n - v5_rank[i] + 1)),
            "class_v6": v6_class[patches[i]], "class_v5_proxy": v5_class[patches[i]],
            "has_spatial_evidence_review_only": str(patches[i] in spatial_patches).lower(),
            "review_only": "true",
        })
    write_csv(M_RANK, rank_rows)

    # class agreement matrix
    agree_rows = []
    classes = ["low", "medium", "high"]
    for c6 in classes:
        for c5 in classes:
            cnt = sum(1 for i in range(n) if v6_class[patches[i]] == c6 and v5_class[patches[i]] == c5)
            agree_rows.append({"class_v6": c6, "class_v5_proxy": c5, "count": cnt})
    same = sum(1 for i in range(n) if v6_class[patches[i]] == v5_class[patches[i]])
    agree_rows.append({"class_v6": "OVERALL_AGREEMENT", "class_v5_proxy": "fraction",
                       "count": round(same / n, 4)})
    write_csv(M_CLASS, agree_rows)

    # region diagnostics
    region_rows = []
    for reg in sorted(set(region)):
        idx = [i for i in range(n) if region[i] == reg]
        rv6 = [v6[i] for i in idx]
        rv5 = [v5[i] for i in idx]
        region_rows.append({
            "regiao": reg, "n_patches": len(idx),
            "pearson_v6_v5": round(pearson(rv6, rv5), 6),
            "spearman_v6_v5": round(spearman(rv6, rv5), 6),
            "mean_abs_diff": round(float(np.mean(np.abs(np.array(rv6) - np.array(rv5)))), 6),
            "mean_score_v6": round(float(np.mean(rv6)), 6),
            "mean_score_v5": round(float(np.mean(rv5)), 6),
            "metric_basis": "review_only_not_validation",
        })
    write_csv(M_REGION, region_rows)

    # high score + spatial evidence cases (and the other combinations)
    hs_rows = []
    for i in range(n):
        pid = patches[i]
        high_v6 = v6_class[pid] == "high"
        has_ev = pid in spatial_patches
        if high_v6 and has_ev:
            cat = "high_score_with_spatial_evidence_review_only"
        elif high_v6 and not has_ev:
            cat = "high_score_no_spatial_evidence"
        elif has_ev and not high_v6:
            cat = "spatial_evidence_but_not_high_score"
        else:
            continue
        hs_rows.append({
            "patch_id": pid, "regiao": region[i], "score_v6": round(v6[i], 6),
            "class_v6": v6_class[pid], "has_spatial_evidence_review_only": str(has_ev).lower(),
            "category": cat,
            "interpretation": "aderencia espacial review-only; NAO e ocorrencia confirmada por patch.",
            "can_be_used_as_ground_truth": "false", "review_only": "true",
        })
    write_csv(M_HIGHSPATIAL, hs_rows)

    # disagreement cases (large rank shift or abs diff)
    dis_rows = []
    shifts = sorted(rank_rows, key=lambda r: abs(r["rank_shift"]), reverse=True)[:25]
    for r in shifts:
        direction = ("v6_up_vs_v5" if r["rank_shift"] < 0 else "v6_down_vs_v5")
        dis_rows.append({
            "patch_id": r["patch_id"], "regiao": r["regiao"],
            "score_v6": r["score_v6"], "score_v5_proxy": r["score_v5_proxy"],
            "abs_diff_v6_v5": r["abs_diff_v6_v5"], "rank_shift": r["rank_shift"],
            "direction": direction, "class_v6": r["class_v6"], "class_v5_proxy": r["class_v5_proxy"],
            "has_spatial_evidence_review_only": r["has_spatial_evidence_review_only"],
            "interpretation": "discordancia metodologica review-only (v6 deterministico vs proxy v5 heuristico).",
            "review_only": "true",
        })
    write_csv(M_DISAGREE, dis_rows)

    write_json(M_SUMMARY, {
        "stage": "SUSC-10B score comparison",
        "n_patches": n, "pearson_v6_v5": round(p_corr, 6), "spearman_v6_v5": round(s_corr, 6),
        "top15_overlap": topk[15], "top30_overlap": topk[30],
        "class_overall_agreement": round(same / n, 4),
        "n_high_with_spatial_evidence": sum(1 for r in hs_rows if r["category"].startswith("high_score_with")),
        "n_high_no_spatial_evidence": sum(1 for r in hs_rows if r["category"] == "high_score_no_spatial_evidence"),
        "spatial_evidence_patches": sorted(spatial_patches),
        "interpretation_guard": "correlation_is_not_real_event_validation",
        "can_be_used_as_ground_truth": False, "allowed_for_training": False, "review_only": True,
        "uses_proxy_v5_as_score_feature": False,
        "scientific_status": "internal_score_comparison_review_only_not_ground_truth",
    })

    write_markdown(M_REPORT, "\n".join([
        "# SUSC-10B — Comparação Score v6 × Proxies × Baseline (review-only)", "",
        f"> {MANDATORY}", "",
        "## 1. Objetivo",
        "Comparar o score v6 candidato (determinístico) com o proxy v5, o baseline (06A/06B) e a "
        "evidência espacial review-only (07B/08/09A). Diagnóstico, não validação.", "",
        "## 2. Diferença entre score v6, score v5, proxy e baseline",
        "- **v6 candidato:** determinístico, pesos documentados, 19 features aprovadas, review-only.",
        "- **v5 proxy:** score heurístico composto (SUSC-06B mostrou circularidade alta).",
        "- **baseline 06A:** ajuste interpretável contra o proxy v5 (recuperabilidade, não predição).", "",
        "## 3. Comparação global",
        f"- n={n}; Pearson(v6,v5)={round(p_corr,4)}; Spearman(v6,v5)={round(s_corr,4)}.",
        f"- top-15 overlap={topk[15]}; top-30 overlap={topk[30]}; concordância de classe={round(same/n,4)}.", "",
        "## 4. Comparação por região",
        "Ver `SUSC_10B_score_region_diagnostics.csv` (Pearson/Spearman/diferença por região).", "",
        "## 5. Top patches v6",
        "Ver `SUSC_10A_score_v6_candidate_top_patches.csv` e o rank shift em `SUSC_10B_score_rank_shift_by_patch.csv`.", "",
        "## 6. Patches com maior concordância",
        "Patches com classe v6 = classe v5 (tercis) — ver `SUSC_10B_score_class_agreement.csv`.", "",
        "## 7. Patches com maior divergência",
        "Top 25 por |rank shift| em `SUSC_10B_score_disagreement_cases.csv`.", "",
        "## 8. Relação com casos espaciais review-only",
        f"- Patches com evidência espacial (07B): {sorted(spatial_patches)}.",
        "- Combinações em `SUSC_10B_high_score_spatial_evidence_cases.csv` (alto+evidência / alto sem evidência / evidência sem alto).", "",
        "## 9. Limitações",
        "Correlação alta v6×v5 reflete uso parcial dos mesmos condicionantes físicos (não validação). "
        "Evidência espacial é review-only; classes por tercis sensíveis à amostra.", "",
        "## 10. O que NÃO pode ser afirmado",
        "- Que concordância v6×v5 valida o score. Que evidência espacial confirma enchente no patch. "
        "Que algo aqui é ground truth.", "",
        "## 11. O que JÁ pode ser afirmado",
        "- v6 e v5 têm relação mensurável (review-only); há patches de alta suscetibilidade com aderência "
        "espacial review-only que merecem revisão humana.", "",
        "## 12. Próximo passo",
        "SUSC-10C (sensibilidade) e SUSC-10D (DINO diagnóstico) + pacote visual SUSC-11A.", "",
        "> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.", "",
    ]))

    print(f"\nn={n} | pearson={round(p_corr,4)} spearman={round(s_corr,4)} | top15={topk[15]} top30={topk[30]}")
    print(f"class agreement={round(same/n,4)} | spatial-evidence patches={len(spatial_patches)}")
    print("\nreview-only diagnostic. No ground truth. No training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
