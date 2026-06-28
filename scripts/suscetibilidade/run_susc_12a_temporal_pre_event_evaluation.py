"""SUSC-12A temporal pre-event evaluation runner."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, write_csv, write_json, write_markdown  # noqa: E402

DATASET = ROOT / "datasets" / "suscetibilidade" / "susc_12a_pre_event_evaluation_dataset_v1.csv"
SCORE = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"

OUT = ROOT / "outputs_public" / "suscetibilidade"
METRICS = OUT / "SUSC_12A_temporal_evaluation_metrics.csv"
EVENT_VS_CONTROL = OUT / "SUSC_12A_temporal_event_vs_control.csv"
HIT_RATE = OUT / "SUSC_12A_temporal_hit_rate_topk.csv"
REGION_SUMMARY = OUT / "SUSC_12A_temporal_region_summary.csv"
LEAKAGE_AUDIT = OUT / "SUSC_12A_temporal_leakage_audit.csv"
SUMMARY = OUT / "SUSC_12A_temporal_evaluation_summary.json"
REPORT = OUT / "SUSC_12A_temporal_pre_event_evaluation_report.md"

MANDATORY = (
    "O SUSC-12A avalia se patches com evidência observada apresentavam maior suscetibilidade "
    "antes do evento, quando a temporalidade dos dados permite. A análise é review-only, não "
    "cria ground truth, não treina modelo supervisionado e não constitui predição operacional."
)


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
    if len(s) % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _round(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(round(value, 6))


def _dist_rows(label: str, vals: list[float]) -> list[dict]:
    return [
        {"metric": f"{label}_count", "value": len(vals), "review_only": "true"},
        {"metric": f"{label}_mean", "value": _round(_mean(vals)), "review_only": "true"},
        {"metric": f"{label}_median", "value": _round(_median(vals)), "review_only": "true"},
        {"metric": f"{label}_min", "value": _round(min(vals) if vals else None), "review_only": "true"},
        {"metric": f"{label}_max", "value": _round(max(vals) if vals else None), "review_only": "true"},
    ]


def _top_threshold(scores: list[float], pct: int) -> float:
    if not scores:
        return 1.0
    s = sorted(scores, reverse=True)
    idx = max(0, min(len(s) - 1, math.ceil(len(s) * pct / 100.0) - 1))
    return s[idx]


def main() -> int:
    print("=" * 60)
    print("SUSC-12A Temporal Pre-Event Evaluation")
    print("=" * 60)
    for path in (DATASET, SCORE):
        if not path.exists():
            print(f"STOP: required input not found: {path}")
            return 1

    rows = read_csv(DATASET)
    scores = read_csv(SCORE)
    all_scores = [_f(r.get("susc_score_v6_candidate")) for r in scores]
    all_scores = [v for v in all_scores if v is not None]

    event_rows = [r for r in rows if r["row_role"] == "event_patch"]
    control_rows = [r for r in rows if r["row_role"] == "control_patch"]
    event_scores = [_f(r["score_v6"]) for r in event_rows if _f(r["score_v6"]) is not None]
    control_scores = [_f(r["score_v6"]) for r in control_rows if _f(r["score_v6"]) is not None]
    valid_event_rows = [r for r in event_rows if r["pre_event_valid"] == "true"]

    pair_rows = []
    controls_by_group = {r["control_group_id"]: r for r in control_rows}
    for er in event_rows:
        cr = controls_by_group.get(er["control_group_id"])
        ev = _f(er["score_v6"])
        cv = _f(cr["score_v6"]) if cr else None
        pair_rows.append({
            "control_group_id": er["control_group_id"],
            "event_id": er["event_id"],
            "event_patch_id": er["patch_id"],
            "control_patch_id": cr["patch_id"] if cr else "",
            "region": er["region"],
            "event_score_v6": er["score_v6"],
            "control_score_v6": cr["score_v6"] if cr else "",
            "score_difference_event_minus_control": _round(ev - cv) if ev is not None and cv is not None else "",
            "event_feature_temporal_status": er["feature_temporal_status"],
            "control_feature_temporal_status": cr["feature_temporal_status"] if cr else "",
            "pre_event_valid": er["pre_event_valid"],
            "control_patch_status": cr["control_patch_status"] if cr else "",
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
        })
    write_csv(EVENT_VS_CONTROL, pair_rows)

    metrics_rows = []
    metrics_rows.extend(_dist_rows("event_patch_score_distribution", event_scores))
    metrics_rows.extend(_dist_rows("control_patch_score_distribution", control_scores))
    diff_vals = [_f(r["score_difference_event_minus_control"]) for r in pair_rows]
    diff_vals = [v for v in diff_vals if v is not None]
    metrics_rows.extend(_dist_rows("score_difference_event_vs_control", diff_vals))
    event_mean = _mean(event_scores)
    control_mean = _mean(control_scores)
    enrichment = (event_mean / control_mean) if control_mean and not math.isnan(control_mean) else float("nan")
    metrics_rows.append({"metric": "event_enrichment_ratio", "value": _round(enrichment), "review_only": "true"})
    metrics_rows.append({"metric": "low_sample_size", "value": str(len(valid_event_rows) < 30).lower(), "review_only": "true"})
    write_csv(METRICS, metrics_rows, ["metric", "value", "review_only"])

    hit_rows = []
    for pct in (10, 20, 30):
        threshold = _top_threshold(all_scores, pct)
        event_hit = sum(1 for v in event_scores if v >= threshold)
        control_hit = sum(1 for v in control_scores if v >= threshold)
        event_rate = event_hit / len(event_scores) if event_scores else 0.0
        control_rate = control_hit / len(control_scores) if control_scores else 0.0
        ratio = event_rate / control_rate if control_rate else ""
        hit_rows.append({
            "top_k": f"top_{pct}",
            "score_threshold": _round(threshold),
            "event_hits": event_hit,
            "event_total": len(event_scores),
            "event_hit_rate": _round(event_rate),
            "control_hits": control_hit,
            "control_total": len(control_scores),
            "control_hit_rate": _round(control_rate),
            "event_enrichment_ratio": _round(ratio) if ratio != "" else "",
            "review_only": "true",
        })
    write_csv(HIT_RATE, hit_rows)

    region_rows = []
    for region in sorted({r["region"] for r in rows}):
        er = [r for r in event_rows if r["region"] == region]
        cr = [r for r in control_rows if r["region"] == region]
        es = [_f(r["score_v6"]) for r in er if _f(r["score_v6"]) is not None]
        cs = [_f(r["score_v6"]) for r in cr if _f(r["score_v6"]) is not None]
        region_rows.append({
            "region": region,
            "event_patch_rows": len(er),
            "control_patch_rows": len(cr),
            "event_mean_score_v6": _round(_mean(es)),
            "control_mean_score_v6": _round(_mean(cs)),
            "score_difference_event_minus_control": _round(_mean(es) - _mean(cs)) if es and cs else "",
            "pre_event_valid_event_rows": sum(1 for r in er if r["pre_event_valid"] == "true"),
            "post_event_risk_event_rows": sum(1 for r in er if r["feature_temporal_status"] == "post_event_risk"),
            "temporal_uncertain_event_rows": sum(1 for r in er if r["temporal_leakage_risk"] == "temporal_uncertain"),
            "review_only": "true",
        })
    write_csv(REGION_SUMMARY, region_rows)

    leakage_rows = []
    for role in ("event_patch", "control_patch"):
        subset = [r for r in rows if r["row_role"] == role]
        for status, n in sorted(Counter(r["feature_temporal_status"] for r in subset).items()):
            leakage_rows.append({"row_role": role, "audit_type": "feature_temporal_status", "status": status, "count": n, "review_only": "true"})
        for status, n in sorted(Counter(r["temporal_leakage_risk"] for r in subset).items()):
            leakage_rows.append({"row_role": role, "audit_type": "temporal_leakage_risk", "status": status, "count": n, "review_only": "true"})
    write_csv(LEAKAGE_AUDIT, leakage_rows)

    temporal_counts = Counter(r["feature_temporal_status"] for r in rows)
    leakage_counts = Counter(r["temporal_leakage_risk"] for r in rows)
    summary = {
        "artifact": "SUSC-12A temporal pre-event evaluation",
        "events_evaluated": len({r["event_id"] for r in event_rows}),
        "event_patch_rows": len(event_rows),
        "event_patches": len({r["patch_id"] for r in event_rows}),
        "controls": len({r["patch_id"] for r in control_rows}),
        "event_patch_score_distribution": {
            "count": len(event_scores),
            "mean": None if math.isnan(event_mean) else round(event_mean, 6),
            "median": None if math.isnan(_median(event_scores)) else round(_median(event_scores), 6),
        },
        "control_patch_score_distribution": {
            "count": len(control_scores),
            "mean": None if math.isnan(control_mean) else round(control_mean, 6),
            "median": None if math.isnan(_median(control_scores)) else round(_median(control_scores), 6),
        },
        "score_difference_event_vs_control": round(_mean(diff_vals), 6) if diff_vals else None,
        "event_enrichment_ratio": None if math.isnan(enrichment) else round(enrichment, 6),
        "hit_rate_topk": hit_rows,
        "region_level_summary": region_rows,
        "temporal_validity_counts": dict(temporal_counts),
        "leakage_risk_counts": dict(leakage_counts),
        "low_sample_size": len(valid_event_rows) < 30,
        "valid_pre_event_event_rows": len(valid_event_rows),
        "can_be_ground_truth": False,
        "allowed_for_training": False,
        "model_persisted": False,
        "review_only": True,
    }
    write_json(SUMMARY, summary)
    _write_report(summary, temporal_counts, leakage_counts)

    print(f"event rows: {len(event_rows)} | controls: {len(control_rows)}")
    print("event mean:", _round(event_mean), "control mean:", _round(control_mean))
    print("hit rate:", {r["top_k"]: r["event_hit_rate"] for r in hit_rows})
    print("temporal status:", dict(temporal_counts))
    print("review-only. No ground truth. No supervised training.")
    return 0


def _write_report(summary: dict, temporal_counts: Counter, leakage_counts: Counter) -> None:
    hit_lines = "\n".join(
        f"| {r['top_k']} | {r['score_threshold']} | {r['event_hit_rate']} | {r['control_hit_rate']} | {r['event_enrichment_ratio']} |"
        for r in summary["hit_rate_topk"]
    )
    region_lines = "\n".join(
        f"| {r['region']} | {r['event_patch_rows']} | {r['control_patch_rows']} | {r['event_mean_score_v6']} | {r['control_mean_score_v6']} | {r['score_difference_event_minus_control']} |"
        for r in summary["region_level_summary"]
    )
    temporal_lines = "\n".join(f"| {k} | {v} |" for k, v in sorted(temporal_counts.items()))
    leakage_lines = "\n".join(f"| {k} | {v} |" for k, v in sorted(leakage_counts.items()))
    md = f"""# SUSC-12A - Temporal Pre-Event Susceptibility Evaluation

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

{MANDATORY}

## 1. Objetivo
Avaliar se patches ligados a evidencias observadas/contextuais no SUSC-11C ja
apareciam com maior score v6 candidato antes do evento, sem transformar essa
aderencia em rotulo, ground truth ou predicao operacional.

## 2. O que e avaliacao temporal pre-evento
E uma checagem de coerencia temporal: uma feature so conta como pre-evento quando
sua data de referencia e comprovadamente anterior ao evento. Quando a data e
ausente, igual ao periodo do evento ou posterior, a analise fica incerta ou com
risco de vazamento.

## 3. Eventos usados
Eventos distintos avaliados: **{summary['events_evaluated']}**. Linhas
evento-patch: **{summary['event_patch_rows']}**. Patches distintos com ligacao:
**{summary['event_patches']}**.

## 4. Criterio de pre-evento
`pre_event_valid=true` exige que nenhuma feature temporal usada no score tenha
risco pos-evento ou metadado temporal ausente/incerto. Features fisicas estaticas
sao aceitas como `static_feature`.

## 5. Features estaticas
Topografia, hidrologia estrutural e distancia/forma de drenagem entram como
condicionantes fisicos estaticos ou quase estaticos, ainda review-only.

## 6. Features temporais
Chuva, runoff, indices espectrais e uso/cobertura urbana exigem metadado temporal
pre-evento. Quando a referencia disponivel e posterior ao evento, a linha e
marcada como `post_event_risk`.

## 7. Riscos de vazamento
| status | n |
|---|---:|
{leakage_lines}

Validade temporal agregada:

| feature_temporal_status | n |
|---|---:|
{temporal_lines}

## 8. Comparacao evento vs controle
Media evento: **{summary['event_patch_score_distribution']['mean']}**.
Media controle: **{summary['control_patch_score_distribution']['mean']}**.
Diferenca media evento-controle: **{summary['score_difference_event_vs_control']}**.

Controles sao `no_documented_event_control`; eles nao representam ausencia formal de evento.

## 9. Hit-rate top-k
| top-k | limiar score | hit-rate evento | hit-rate controle | razao |
|---|---:|---:|---:|---:|
{hit_lines}

## 10. Resultados por regiao
| regiao | evento-patch | controles | media evento | media controle | diferenca |
|---|---:|---:|---:|---:|---:|
{region_lines}

## 11. Limitacoes
- Amostra exploratoria e pequena para inferencia estatistica.
- As ligacoes SUSC-11C permanecem review-only.
- Varios eventos nao tem data suficiente para validar temporalidade.
- Score v6 usa componentes temporais/espectrais que podem ter referencia posterior ao evento.
- Controle sem evento documentado nao e ausencia formal de evento.

## 12. O que nao pode ser afirmado
- Nao ha ground truth por patch.
- Nao ha treino supervisionado.
- Nao ha predicao operacional.
- Controle sem evento documentado nao representa ausencia formal de evento.
- Nao se pode afirmar causalidade ou confirmacao automatica por patch.

## 13. O que ja pode ser afirmado
- A comparacao evento vs controle foi gerada de modo reproduzivel e auditavel.
- A auditoria temporal identifica quais linhas ficam bloqueadas por risco pos-evento
  ou metadado temporal insuficiente.
- A leitura quantitativa e exploratoria e review-only.

## 14. Proximo marco
SUSC-12B: readiness/gap report para preencher metadados temporais pre-evento e
separar features estaticas de features dinamicas em uma avaliacao sem vazamento.
"""
    write_markdown(REPORT, md)


if __name__ == "__main__":
    raise SystemExit(main())
