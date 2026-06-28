"""SUSC-12C score v7 candidate readiness gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, read_json, write_csv, write_json, write_markdown  # noqa: E402

CONTRAST_SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12B_observed_event_contrast_summary.json"
RECS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12C_proxy_calibration_recommendations.csv"
CAL_SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12C_proxy_calibration_summary.json"
SCORE_CONTRAST = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12B_observed_event_score_contrast.csv"
REGION_CONTRAST = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12B_observed_event_region_contrast.csv"

OUT = ROOT / "outputs_public" / "suscetibilidade"
NOT_READY = OUT / "SUSC_12C_score_v7_not_ready_report.md"
INTEGRATED = OUT / "SUSC_12B_12C_observed_event_contrast_and_calibration_report.md"

MANDATORY = (
    "O SUSC-12B/12C compara métricas de suscetibilidade com evidências observacionais "
    "de alagamento/inundação para calibração review-only do proxy. A análise não cria "
    "ground truth, não treina modelo supervisionado e não transforma ausência de registro "
    "em negativo verdadeiro."
)


def _table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "| sem registros |\n|---|"
    head = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|"
    body = "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)
    return head + "\n" + body


def main() -> int:
    print("=" * 60)
    print("SUSC-12C Score v7 Readiness Gate")
    print("=" * 60)
    for path in (CONTRAST_SUMMARY, RECS, CAL_SUMMARY, SCORE_CONTRAST, REGION_CONTRAST):
        if not path.exists():
            print(f"STOP: required input not found: {path}")
            return 1

    contrast = read_json(CONTRAST_SUMMARY)
    recs = read_csv(RECS)
    cal = read_json(CAL_SUMMARY)
    score_rows = read_csv(SCORE_CONTRAST)
    region_rows = read_csv(REGION_CONTRAST)

    ready = (
        contrast.get("event_patch_count", 0) >= 20
        and not contrast.get("low_sample_size", True)
        and cal.get("score_v7_ready") is True
    )
    if ready:
        print("STOP: readiness unexpectedly open; this conservative gate expects manual review before v7.")
        return 1

    md = f"""# SUSC-12C - Score v7 Not Ready Report

Status: **not ready** | `score_v7_created=false`

Motivo principal: amostra observacional insuficiente para recalibrar pesos de
modo deterministico sem risco metodologico alto. `event_patch_count` =
**{contrast.get('event_patch_count')}**; minimo operacional conservador = **20**.

Decisao: nenhum `score_v7_candidate_review_only` foi criado. O `score_v6_candidate`
permanece como score deterministico candidato.

Guardrails:
- sem ground truth;
- sem treino supervisionado;
- sem modelo persistido;
- controles `no_documented_event_control` nao representam ausencia formal;
- eventos fracos/contextuais nao calibram como evidencia forte.
"""
    write_markdown(NOT_READY, md)

    agree = contrast.get("features_direction_agree", [])
    diverge = contrast.get("features_direction_diverge", [])
    inconclusive = contrast.get("features_inconclusive", [])
    rec_counts = cal.get("recommendation_counts", {})
    rec_lines = "\n".join(f"| {k} | {v} |" for k, v in sorted(rec_counts.items()))
    integrated = f"""# SUSC-12B/12C - Observed Event Contrast and Proxy Calibration

Status: **review-only** | `score_v7_created=false`

{MANDATORY}

## 1. Objetivo
Comparar patches com evidencia observacional moderada a controles sem evento
documentado e gerar recomendacoes conservadoras para calibracao do proxy/score.

## 2. Diferenca entre contraste direto e avaliacao temporal
O SUSC-12A avaliou validade pre-evento e encontrou bloqueios temporais. O SUSC-12B
faz contraste observacional direto, sem afirmar temporalidade pre-evento.

## 3. Eventos usados
Patches evento/moderados: **{contrast.get('event_patch_count')}**. Patches de
contexto fraco: **{contrast.get('weak_context_patch_count')}**.

## 4. Controles
Controles criados: **{contrast.get('control_patch_count')}**. Todos sao
`no_documented_event_control` e nao representam ausencia formal de evento.

## 5. Resultado do score v6 em patches com evento
Media evento: **{contrast.get('score_v6_event_mean')}**. Media controle:
**{contrast.get('score_v6_control_mean')}**. Diferenca evento-controle:
**{contrast.get('score_v6_event_minus_control')}**.

## 6. Features que concordam com eventos observados
{', '.join(agree) if agree else 'Nenhuma feature com suporte suficiente para promocao automatica.'}

## 7. Features que divergem
{', '.join(diverge) if diverge else 'Nenhuma divergencia interpretada como decisao automatica devido a amostra pequena.'}

## 8. Features inconclusivas
{', '.join(inconclusive) if inconclusive else 'Sem features inconclusivas no contraste numerico, mas a confianca segue baixa.'}

## 9. Recomendacoes de calibracao
| recomendacao | n |
|---|---:|
{rec_lines}

## 10. Se score v7 foi criado ou nao
Nao foi criado. A saida oficial e `SUSC_12C_score_v7_not_ready_report.md`.

## 11. Limitacoes
- Amostra pequena de patches com evento moderado.
- Eventos fracos/contextuais nao foram usados como evidencia forte.
- Controles nao sao ausencia formal de evento.
- Nao ha treinamento, modelo persistido ou ground truth.

## 12. O que nao pode ser afirmado
- Nao se pode afirmar que o score v6 valida ocorrencia por patch.
- Nao se pode afirmar ausencia de evento nos controles.
- Nao se pode treinar modelo supervisionado com estes grupos.

## 13. O que ja pode ser afirmado
- O contraste evento/controle foi produzido de forma auditavel.
- A amostra atual nao justifica score v7.
- As recomendacoes sao conservadoras e review-only.

## 14. Proximo marco
SUSC-13A: ampliar evidencias observadas fortes/moderadas e revisar manualmente
os grupos antes de qualquer nova calibracao deterministica.

## Apêndice - score
{_table(score_rows, ['group', 'n', 'mean_score_v6', 'median_score_v6'])}

## Apêndice - regioes
{_table(region_rows, ['region', 'event_patch_count', 'control_patch_count', 'mean_score_event', 'mean_score_control', 'score_difference_event_minus_control'])}
"""
    write_markdown(INTEGRATED, integrated)

    print("score_v7_created: false")
    print("not ready report ->", NOT_READY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
