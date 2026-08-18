"""
SUSC-10C Score Sensitivity & Ablation (review-only diagnostic)

Recomputes the deterministic v6 score under weight/ablation/threshold scenarios
and measures stability against the baseline v6 (Spearman, class agreement, per-
patch score range). No training, no validation, no real-event confirmation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, read_json, write_csv, write_json, write_markdown  # noqa: E402

MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
SCORE = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_score_v6_candidate_manifest_v1.json"
CARDS = ROOT / "outputs_public" / "suscetibilidade" / "feature_cards" / "susc_feature_cards_v1.json"
OVERLAY_07B = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_refined_patch_event_overlay.csv"

OUT = ROOT / "outputs_public" / "suscetibilidade"
S_SCEN = OUT / "SUSC_10C_score_sensitivity_scenarios.csv"
S_PATCH = OUT / "SUSC_10C_score_sensitivity_by_patch.csv"
S_REGION = OUT / "SUSC_10C_score_sensitivity_by_region.csv"
S_UNSTABLE = OUT / "SUSC_10C_score_sensitivity_top_unstable_patches.csv"
S_SUMMARY = OUT / "SUSC_10C_score_sensitivity_summary.json"
S_REPORT = OUT / "SUSC_10C_score_sensitivity_report.md"

MITIGATION = {"ndvi_mean", "vegetation_prop"}
URBAN_SPECTRAL = {"ndbi_mean", "urban_prop", "mndwi_mean"}
TH_GROUPS = {"topography", "topography_hydrology", "hydrology"}
SPATIAL_RELS = {"bbox_overlap", "near_patch_buffer_candidate", "exact_patch_overlap"}
MANDATORY = ("O SUSC-10C avalia a estabilidade de um score determinístico candidato. A análise de "
             "sensibilidade não constitui treinamento, validação supervisionada ou confirmação de evento real.")

# scenario -> subindex weights (TH, RT, US, VM, ES)
SCENARIOS = {
    "baseline_v6": (0.40, 0.25, 0.20, -0.10, 0.05),
    "no_rainfall_trigger": (0.40, 0.00, 0.20, -0.10, 0.05),
    "no_urban_spectral": (0.40, 0.25, 0.00, -0.10, 0.05),
    "no_topography_hydrology": (0.00, 0.25, 0.20, -0.10, 0.05),
    "no_evidence_support": (0.40, 0.25, 0.20, -0.10, 0.00),
    "equal_group_weights": (0.20, 0.20, 0.20, -0.20, 0.20),
    "hydrology_heavy": (0.60, 0.15, 0.10, -0.10, 0.05),
    "urban_heavy": (0.20, 0.15, 0.50, -0.10, 0.05),
    "rainfall_heavy": (0.20, 0.50, 0.15, -0.10, 0.05),
    "strict_high_threshold": (0.40, 0.25, 0.20, -0.10, 0.05),  # threshold change below
    "loose_high_threshold": (0.40, 0.25, 0.20, -0.10, 0.05),
}
SUB_KEYS = ["topography_hydrology_index", "rainfall_trigger_index", "urban_spectral_index",
            "vegetation_mitigation_index", "evidence_support_index"]


def _f(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def winsorize(vals, lo=0.01, hi=0.99):
    s = sorted(vals)
    n = len(s)
    lo_v, hi_v = s[max(0, int(lo * n))], s[min(n - 1, int(hi * n))]
    return [min(max(v, lo_v), hi_v) for v in vals]


def robust_minmax(vals):
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [0.5 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def rankdata(a):
    a = np.asarray(a, float)
    order = a.argsort()
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def classes(vals, mode="tertile"):
    s = sorted(vals)
    n = len(vals)
    if mode == "strict":
        t1, t2 = s[int(0.40 * n)], s[int(0.80 * n)]
    elif mode == "loose":
        t1, t2 = s[int(0.25 * n)], s[int(0.50 * n)]
    else:
        t1, t2 = s[int(0.33 * n)], s[int(0.66 * n)]
    return ["low" if v <= t1 else ("high" if v > t2 else "medium") for v in vals]


def main() -> int:
    print("=" * 60)
    print("SUSC-10C Score Sensitivity & Ablation (review-only)")
    print("=" * 60)

    for p in (MATRIX, SCORE, MANIFEST, CARDS):
        if not p.exists():
            print(f"STOP: required input not found: {p}")
            return 1

    rows = read_csv(MATRIX)
    used = read_json(MANIFEST)["features_used"]
    cards = {c["feature_name"]: c for c in read_json(CARDS)["cards"]}

    # direction-normalized contribution per feature + subindex bucket
    feat_contrib, feat_sub = {}, {}
    for feat in used:
        raw = [_f(r.get(feat)) for r in rows]
        if any(v is None for v in raw):
            continue
        mm = robust_minmax(winsorize(raw))
        direction = cards.get(feat, {}).get("expected_direction", "higher_increases")
        contrib = [1.0 - v for v in mm] if direction in {"lower_increases", "higher_decreases"} else mm
        group = cards.get(feat, {}).get("feature_group", "")
        if feat in MITIGATION:
            sub = "vegetation_mitigation_index"
        elif feat in URBAN_SPECTRAL:
            sub = "urban_spectral_index"
        elif group in TH_GROUPS:
            sub = "topography_hydrology_index"
        elif group == "precipitation":
            sub = "rainfall_trigger_index"
        else:
            continue
        feat_contrib[feat] = contrib
        feat_sub[feat] = sub

    n = len(rows)
    # evidence support
    spatial = set()
    if OVERLAY_07B.exists():
        for o in read_csv(OVERLAY_07B):
            if o["spatial_relation"] in SPATIAL_RELS and o["patch_id"] != "REGION_LEVEL_NO_PATCH_RESOLUTION":
                spatial.add(o["patch_id"])
    support = [0.5 if rows[i]["patch_id"] in spatial else 0.0 for i in range(n)]

    # subindex value per patch
    sub_feats = {k: [f for f in feat_sub if feat_sub[f] == k] for k in SUB_KEYS if k != "evidence_support_index"}
    subval = {k: [] for k in SUB_KEYS}
    for i in range(n):
        for k, feats in sub_feats.items():
            subval[k].append(sum(feat_contrib[f][i] for f in feats) / len(feats) if feats else 0.0)
        subval["evidence_support_index"].append(support[i])

    # compute each scenario score
    scenario_scores = {}
    scenario_classes = {}
    for name, w in SCENARIOS.items():
        raw = [sum(w[j] * subval[SUB_KEYS[j]][i] for j in range(5)) for i in range(n)]
        final = robust_minmax(raw)
        mode = "strict" if name == "strict_high_threshold" else ("loose" if name == "loose_high_threshold" else "tertile")
        scenario_scores[name] = final
        scenario_classes[name] = classes(final, mode)

    base = scenario_scores["baseline_v6"]
    base_cls = scenario_classes["baseline_v6"]

    # scenarios.csv
    scen_rows = []
    for name in SCENARIOS:
        sc = scenario_scores[name]
        cl = scenario_classes[name]
        sp = spearman(sc, base)
        agree = sum(1 for i in range(n) if cl[i] == base_cls[i]) / n
        w = SCENARIOS[name]
        scen_rows.append({
            "scenario": name,
            "weights_TH_RT_US_VM_ES": ";".join(str(x) for x in w),
            "threshold_mode": "strict" if name == "strict_high_threshold" else ("loose" if name == "loose_high_threshold" else "tertile"),
            "spearman_vs_baseline": round(sp, 6),
            "class_agreement_vs_baseline": round(agree, 6),
            "n_high": sum(1 for c in cl if c == "high"),
            "review_only": "true",
        })
    write_csv(S_SCEN, scen_rows)

    # by patch. NOTE: each scenario rescales to [0,1] independently, so raw
    # score_range overstates instability. Stability is measured by CLASS
    # consistency vs baseline (rank/class is the meaningful invariant).
    patch_rows = []
    sranges = []
    class_stab = []
    for i in range(n):
        pid = rows[i]["patch_id"]
        vals = [scenario_scores[name][i] for name in SCENARIOS]
        rng = max(vals) - min(vals)
        sranges.append((pid, rng))
        same_class = sum(1 for name in SCENARIOS if scenario_classes[name][i] == base_cls[i]) / len(SCENARIOS)
        class_stab.append(same_class)
        distinct_cls = len({scenario_classes[name][i] for name in SCENARIOS})
        flag = "robust" if same_class >= 0.8 else ("moderate" if same_class >= 0.5 else "unstable")
        row = {"patch_id": pid, "regiao": rows[i].get("regiao", ""),
               "baseline_v6": round(base[i], 6), "baseline_class": base_cls[i],
               "class_stability_vs_baseline": round(same_class, 4),
               "score_range_across_scenarios_rescaled": round(rng, 6),
               "n_distinct_classes": distinct_cls, "stability_flag": flag,
               "review_only": "true"}
        for name in SCENARIOS:
            row[name] = round(scenario_scores[name][i], 6)
        patch_rows.append(row)
    write_csv(S_PATCH, patch_rows)

    # by region (class-stability based)
    region_rows = []
    for reg in sorted({r.get("regiao", "") for r in rows}):
        idx = [i for i in range(n) if rows[i].get("regiao", "") == reg]
        region_rows.append({
            "regiao": reg, "n_patches": len(idx),
            "mean_class_stability": round(float(np.mean([class_stab[i] for i in idx])), 6),
            "n_robust": sum(1 for i in idx if patch_rows[i]["stability_flag"] == "robust"),
            "n_unstable": sum(1 for i in idx if patch_rows[i]["stability_flag"] == "unstable"),
            "metric_basis": "review_only_not_validation",
        })
    write_csv(S_REGION, region_rows)

    # top unstable (lowest class stability)
    top_unstable = sorted(patch_rows, key=lambda r: r["class_stability_vs_baseline"])[:25]
    write_csv(S_UNSTABLE, [{
        "patch_id": r["patch_id"], "regiao": r["regiao"], "baseline_v6": r["baseline_v6"],
        "baseline_class": r["baseline_class"],
        "class_stability_vs_baseline": r["class_stability_vs_baseline"],
        "n_distinct_classes": r["n_distinct_classes"], "stability_flag": r["stability_flag"],
        "review_only": "true",
    } for r in top_unstable],
        ["patch_id", "regiao", "baseline_v6", "baseline_class", "class_stability_vs_baseline", "n_distinct_classes", "stability_flag", "review_only"])

    # which group changes ranking most: lowest spearman among ablations
    ablations = {k: v for k, v in scenario_scores.items() if k.startswith("no_")}
    most_impactful = min(ablations, key=lambda k: spearman(ablations[k], base)) if ablations else "n/a"

    robust_high = [patch_rows[i]["patch_id"] for i in range(n)
                   if base_cls[i] == "high" and patch_rows[i]["stability_flag"] == "robust"]
    n_unstable = sum(1 for r in patch_rows if r["stability_flag"] == "unstable")
    n_robust = sum(1 for r in patch_rows if r["stability_flag"] == "robust")

    write_json(S_SUMMARY, {
        "stage": "SUSC-10C sensitivity",
        "n_patches": n, "n_scenarios": len(SCENARIOS),
        "mean_class_stability": round(float(np.mean(class_stab)), 6),
        "global_mean_score_range_rescaled_informational": round(float(np.mean([r[1] for r in sranges])), 6),
        "min_spearman_vs_baseline": round(min(s["spearman_vs_baseline"] for s in scen_rows), 6),
        "most_impactful_ablation": most_impactful,
        "n_robust_patches": n_robust,
        "n_unstable_patches": n_unstable,
        "n_robust_high_patches": len(robust_high),
        "can_be_used_as_ground_truth": False, "allowed_for_training": False, "review_only": True,
        "trained_model": False, "model_persisted": False,
        "scientific_status": "deterministic_score_sensitivity_review_only_not_training",
    })

    write_markdown(S_REPORT, "\n".join([
        "# SUSC-10C — Sensibilidade e Ablação do Score v6 (review-only)", "",
        f"> {MANDATORY}", "",
        "## 1. O que foi testado",
        f"{len(SCENARIOS)} cenários: remoção de grupos (chuva/urbano-espectral/topo-hidro/evidência), "
        "pesos iguais, hidrologia/urbano/chuva pesados, e thresholds estrito/frouxo de classe.", "",
        "## 2. Estabilidade global",
        f"- Estabilidade de classe média (fração de cenários que mantêm a classe do baseline): {round(float(np.mean(class_stab)),4)}.",
        f"- Menor Spearman vs baseline entre cenários: {round(min(s['spearman_vs_baseline'] for s in scen_rows),4)}.",
        f"- Patches robustos: {n_robust}; instáveis: {n_unstable}.",
        "- (O range bruto do score por patch é inflado pelo reescalonamento por cenário; a métrica honesta é a estabilidade de classe/rank.)", "",
        "## 3. Estabilidade por região",
        "Ver `SUSC_10C_score_sensitivity_by_region.csv`.", "",
        "## 4. Patches robustamente altos",
        f"{len(robust_high)} patches `high` no baseline e `robust` sob cenários.", "",
        "## 5. Patches instáveis",
        "Top 25 em `SUSC_10C_score_sensitivity_top_unstable_patches.csv`.", "",
        "## 6. Qual grupo mais altera o ranking",
        f"Ablação mais impactante: **{most_impactful}** (menor Spearman vs baseline).", "",
        "## 7. Implicação metodológica",
        "O score é uma composição determinística; remover topo/hidrologia tende a alterar mais o ranking, "
        "coerente com o peso alto desse grupo (escolha documentada, não aprendida).", "",
        "## 8. Limitações",
        "Pesos e thresholds são escolhas metodológicas. Sensibilidade não é treino nem validação; "
        "evidência de suporte é review-only.", "",
        "> A matriz de suscetibilidade ≠ ocorrência confirmada de enchente.", "",
    ]))

    print(f"\nscenarios: {len(SCENARIOS)} | mean class stability={round(float(np.mean(class_stab)),4)} | robust={n_robust} unstable={n_unstable}")
    print(f"most impactful ablation: {most_impactful} | robust-high: {len(robust_high)}")
    print("\nreview-only diagnostic. No training. No ground truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
