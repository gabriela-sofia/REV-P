"""MV2-DATA-05 Tarefa 3 — Validar evidência temporal.

Classifica cada janela normalizada em STRONG/PARTIAL/REVIEW_REQUIRED/EMPTY/CONFLICT/INVALID,
checando: existência do target no corpus, match asset/patch, datas válidas e não futuras,
start<=end, janela não ampla demais, fonte presente e review aprovado. Nunca inventa data.

Saída: outputs_public/mv2_data_temporal_window_intake/mv2_data_05_temporal_evidence_validation.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_05_common import (
    APPROVED_REVIEW,
    ISO_DATE_RE,
    MAX_WINDOW_DAYS,
    PROJECT_ROOT,
    PUBLIC_DIR,
    classify_datetime,
    clean,
    corpus_index,
    ensure_public_dir,
    read_csv_rows,
    window_days,
    write_csv,
)

COLUMNS = [
    "validation_id",
    "target_id",
    "asset_id",
    "patch_id",
    "region",
    "temporal_window_start",
    "temporal_window_end",
    "temporal_window_source",
    "source_ref",
    "review_status",
    "window_days",
    "future_date",
    "start_after_end",
    "asset_patch_match",
    "source_present",
    "review_approved",
    "too_broad",
    "evidence_status",
    "blocking_reason",
    "notes",
]


def _has_justification(row: dict[str, str]) -> bool:
    """Janela ampla só é aceitável com justificativa explícita na fonte/notas."""
    txt = (clean(row.get("temporal_window_source")) + " " + clean(row.get("notes"))).lower()
    return any(k in txt for k in ("justif", "wide_window", "janela_ampla", "multi_scene"))


def evaluate(r: dict[str, str], corpus: dict[str, dict[str, str]]) -> dict[str, object]:
    """Classificação fail-closed pura de uma janela temporal (testável isoladamente)."""
    tid = clean(r.get("target_id"))
    asset_id = clean(r.get("asset_id"))
    patch_id = clean(r.get("patch_id"))
    start = clean(r.get("temporal_window_start"))
    end = clean(r.get("temporal_window_end"))
    source = clean(r.get("temporal_window_source"))
    source_ref = clean(r.get("source_ref"))
    review = clean(r.get("review_status")).upper()

    both_empty = not start and not end
    corpus_row = corpus.get(tid)
    target_exists = corpus_row is not None
    asset_patch_match = bool(
        corpus_row
        and clean(corpus_row.get("asset_id")) == asset_id
        and clean(corpus_row.get("patch_id")) == patch_id
    )

    start_fmt_ok = bool(start) and ISO_DATE_RE.match(start) is not None
    end_fmt_ok = bool(end) and ISO_DATE_RE.match(end) is not None
    future = (
        (start_fmt_ok and classify_datetime(start)[1] == "INVALID_FUTURE")
        or (end_fmt_ok and classify_datetime(end)[1] == "INVALID_FUTURE")
    )
    days = window_days(start, end)
    start_after_end = days is not None and days < 0
    too_broad = days is not None and days > MAX_WINDOW_DAYS and not _has_justification(r)
    source_present = bool(source) and bool(source_ref)
    review_approved = review in APPROVED_REVIEW

    if both_empty:
        status, blocking = "TEMPORAL_EVIDENCE_EMPTY", "janela_vazia_sem_evidencia"
    elif not target_exists:
        status, blocking = "TEMPORAL_EVIDENCE_INVALID", "target_id_inexistente_no_corpus"
    elif not asset_patch_match:
        status, blocking = "TEMPORAL_EVIDENCE_CONFLICT", "asset_id_ou_patch_id_nao_batem_com_corpus"
    elif not (start_fmt_ok and end_fmt_ok):
        status, blocking = "TEMPORAL_EVIDENCE_INVALID", "data_em_formato_invalido"
    elif future:
        status, blocking = "TEMPORAL_EVIDENCE_INVALID", "data_futura_nao_e_aquisicao"
    elif start_after_end:
        status, blocking = "TEMPORAL_EVIDENCE_INVALID", "start_posterior_a_end"
    elif not (source or source_ref):
        status, blocking = "TEMPORAL_EVIDENCE_INVALID", "janela_sem_fonte"
    elif too_broad:
        status, blocking = "TEMPORAL_EVIDENCE_REVIEW_REQUIRED", "janela_ampla_demais_sem_justificativa"
    elif source_present and review_approved:
        status, blocking = "TEMPORAL_EVIDENCE_VALID_STRONG", ""
    elif source_ref:
        status, blocking = "TEMPORAL_EVIDENCE_VALID_PARTIAL", "datas_e_fonte_ok_mas_falta_review_final"
    else:
        status, blocking = "TEMPORAL_EVIDENCE_REVIEW_REQUIRED", "evidencia_insuficiente"

    return {
        "target_id": tid,
        "asset_id": asset_id,
        "patch_id": patch_id,
        "region": clean(r.get("region")),
        "temporal_window_start": start,
        "temporal_window_end": end,
        "temporal_window_source": source,
        "source_ref": source_ref,
        "review_status": clean(r.get("review_status")),
        "window_days": "" if days is None else days,
        "future_date": "true" if future else "false",
        "start_after_end": "true" if start_after_end else "false",
        "asset_patch_match": "true" if asset_patch_match else "false",
        "source_present": "true" if source_present else "false",
        "review_approved": "true" if review_approved else "false",
        "too_broad": "true" if too_broad else "false",
        "evidence_status": status,
        "blocking_reason": blocking,
        "notes": "validacao fail-closed; nenhuma data inventada",
    }


def build_rows() -> list[dict[str, object]]:
    norm = read_csv_rows(PUBLIC_DIR / "mv2_data_05_normalized_temporal_windows.csv")
    corpus = corpus_index()
    rows: list[dict[str, object]] = []
    for i, r in enumerate(norm, start=1):
        row = evaluate(r, corpus)
        row["validation_id"] = f"MV2_DATA_05_VAL_{i:03d}"
        rows.append(row)
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_05_temporal_evidence_validation.csv"
    write_csv(out, COLUMNS, rows)
    from collections import Counter

    dist = Counter(str(r["evidence_status"]) for r in rows)
    print(
        f"[mv2_data_05_evidence] {len(rows)} | "
        + " ".join(f"{k.replace('TEMPORAL_EVIDENCE_', '')}={v}" for k, v in sorted(dist.items()))
        + f" | saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
