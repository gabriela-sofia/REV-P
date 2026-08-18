"""MV2-DATA-05 Tarefa 6 — Template corrigido para preenchimento humano.

Copia os targets do batch DATA-04, preserva os valores já preenchidos, anexa o status de
validação atual e instruções de correção por target. Não inventa data.

Saída: outputs_public/mv2_data_temporal_window_intake/mv2_data_05_temporal_window_correction_template.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_05_common import (
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    ensure_public_dir,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "target_id",
    "asset_id",
    "patch_id",
    "region",
    "temporal_window_start",
    "temporal_window_end",
    "temporal_window_source",
    "event_id",
    "event_date",
    "export_task_hint",
    "source_ref",
    "filled_by",
    "reviewed_by",
    "review_status",
    "current_validation_status",
    "correction_needed",
    "correction_instruction",
    "notes",
]

# blocking_reason -> instrução de correção objetiva.
CORRECTION_BY_REASON = {
    "janela_vazia_sem_evidencia": "preencher temporal_window_start/end com data de evento/export (ISO), + source_ref",
    "target_id_inexistente_no_corpus": "usar target_id valido do corpus DATA-01",
    "asset_id_ou_patch_id_nao_batem_com_corpus": "corrigir asset_id/patch_id para casar com o corpus",
    "data_em_formato_invalido": "usar formato ISO YYYY-MM-DD",
    "data_futura_nao_e_aquisicao": "remover data futura; usar data historica de aquisicao",
    "start_posterior_a_end": "garantir start <= end",
    "janela_sem_fonte": "preencher temporal_window_source e source_ref",
    "janela_ampla_demais_sem_justificativa": "estreitar janela (<=45d) ou justificar (multi_scene/janela_ampla)",
    "datas_e_fonte_ok_mas_falta_review_final": "marcar review_status=APPROVED apos revisao humana",
    "evidencia_insuficiente": "completar fonte + review_status",
}


def build_rows() -> list[dict[str, object]]:
    evidence = {clean(e.get("target_id")): e
                for e in read_csv_rows(PUBLIC_DIR / "mv2_data_05_temporal_evidence_validation.csv")}
    normalized = read_csv_rows(PUBLIC_DIR / "mv2_data_05_normalized_temporal_windows.csv")

    rows: list[dict[str, object]] = []
    for n in normalized:
        tid = clean(n.get("target_id"))
        ev = evidence.get(tid, {})
        status = clean(ev.get("evidence_status")) or "TEMPORAL_EVIDENCE_EMPTY"
        reason = clean(ev.get("blocking_reason"))
        needs = status not in {"TEMPORAL_EVIDENCE_VALID_STRONG"}
        instruction = CORRECTION_BY_REASON.get(reason, "") if needs else "nenhuma; janela ja STRONG"
        rows.append({
            "target_id": tid,
            "asset_id": clean(n.get("asset_id")),
            "patch_id": clean(n.get("patch_id")),
            "region": clean(n.get("region")),
            "temporal_window_start": clean(n.get("temporal_window_start")),
            "temporal_window_end": clean(n.get("temporal_window_end")),
            "temporal_window_source": clean(n.get("temporal_window_source")),
            "event_id": clean(n.get("event_id")),
            "event_date": clean(n.get("event_date")),
            "export_task_hint": clean(n.get("export_task_hint")),
            "source_ref": clean(n.get("source_ref")),
            "filled_by": clean(n.get("filled_by")),
            "reviewed_by": clean(n.get("reviewed_by")),
            "review_status": clean(n.get("review_status")) or "PENDING_HUMAN_FILL",
            "current_validation_status": status,
            "correction_needed": "true" if needs else "false",
            "correction_instruction": instruction,
            "notes": "corrigir SO com evidencia; nunca inventar data",
        })
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_05_temporal_window_correction_template.csv"
    write_csv(out, COLUMNS, rows)
    needs = sum(1 for r in rows if r["correction_needed"] == "true")
    print(
        f"[mv2_data_05_correction] {len(rows)} linhas | correcao_necessaria={needs} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
