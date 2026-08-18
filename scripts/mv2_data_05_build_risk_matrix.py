"""MV2-DATA-05 Tarefa 7 — Matriz de risco do intake de janela temporal.

Registra os riscos de aceitar/usar janelas temporais, com mitigação fail-closed. Garante
todos os RISK_TYPES obrigatórios.

Saída: outputs_public/mv2_data_temporal_window_intake/mv2_data_05_risk_matrix.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_05_common import (
    PROJECT_ROOT,
    PUBLIC_DIR,
    RISK_TYPES,
    ensure_public_dir,
    write_csv,
)

COLUMNS = [
    "risk_id",
    "target_id",
    "severity",
    "risk_type",
    "evidence",
    "mitigation",
    "status",
    "blocks_next_step",
]

RISK_SPEC = {
    "RISK_TEMPORAL_WINDOW_INVENTED": (
        "ALTA", "preencher janela temporal sem evidencia (data inventada)",
        "intake preserva vazios; janela sem fonte -> INVALID; nunca inventa data"),
    "RISK_TEMPORAL_WINDOW_TOO_BROAD": (
        "ALTA", "janela ampla demais captura cena errada",
        "janela >45d sem justificativa -> REVIEW_REQUIRED, nunca STRONG"),
    "RISK_FUTURE_DATE": (
        "ALTA", "data futura tratada como aquisicao",
        "classify_datetime INVALID_FUTURE -> TEMPORAL_EVIDENCE_INVALID"),
    "RISK_START_AFTER_END": (
        "MEDIA", "start posterior a end",
        "window_days<0 -> INVALID"),
    "RISK_SOURCE_MISSING": (
        "ALTA", "janela aceita sem fonte",
        "source/source_ref ausentes -> INVALID; STRONG exige source_ref"),
    "RISK_REVIEW_STATUS_MISSING": (
        "MEDIA", "janela promovida a STRONG sem review",
        "review_status nao aprovado -> no maximo PARTIAL"),
    "RISK_EVENT_DATE_AS_SCENE_DATE": (
        "ALTA", "usar data do evento como data da cena Sentinel",
        "event_date e campo separado; janela exige start/end proprios + source"),
    "RISK_EXPORT_DATE_AS_ACQUISITION_DATE": (
        "ALTA", "usar data de export GEE como data de aquisicao",
        "export_task_hint e separado; janela de aquisicao exige evidencia propria"),
    "RISK_STAC_PROBE_WITH_WEAK_TEMPORALITY": (
        "ALTA", "habilitar STAC probe com temporalidade fraca",
        "GEE/STAC so habilitam com promocao STRONG/PARTIAL"),
    "RISK_DAY10_FALSE_UNLOCK": (
        "ALTA", "tratar intake temporal como desbloqueio do Dia 10",
        "can_unlock_day10_now=false sempre; corpus_day10_unlocked=false"),
    "RISK_SANDBOX_FALSE_UNLOCK": (
        "ALTA", "abrir sandbox/treino com base no intake",
        "sandbox_unlocked=false; can_train=false"),
}


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, rt in enumerate(RISK_TYPES, start=1):
        severity, evidence, mitigation = RISK_SPEC[rt]
        rows.append({
            "risk_id": f"MV2_DATA_05_RISK_{i:02d}",
            "target_id": "GLOBAL",
            "severity": severity,
            "risk_type": rt,
            "evidence": evidence,
            "mitigation": mitigation,
            "status": "MITIGADO_FAIL_CLOSED",
            "blocks_next_step": "true",
        })
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_05_risk_matrix.csv"
    write_csv(out, COLUMNS, rows)
    present = {str(r["risk_type"]) for r in rows}
    missing = [rt for rt in RISK_TYPES if rt not in present]
    assert not missing, f"risk_types obrigatorios ausentes: {missing}"
    print(
        f"[mv2_data_05_risk] {len(rows)} riscos | tipos={len(present)}/{len(RISK_TYPES)} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
