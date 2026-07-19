"""MV2-DATA-05 Tarefa 2 — Normalizar janelas temporais.

Lê o template preenchido (privado) se existir, ou o template público DATA-04 (vazio) como
fallback. Preserva vazios (nunca inventa data), normaliza datas para ISO e valida o formato
básico. Não promove ainda.

Saída: outputs_public/mv2_data_temporal_window_intake/mv2_data_05_normalized_temporal_windows.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_05_common import (
    ISO_DATE_RE,
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    ensure_public_dir,
    load_intake_source,
    write_csv,
)

COLUMNS = [
    "intake_id",
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
    "normalization_status",
    "normalization_notes",
]


def _norm_date(raw: str) -> tuple[str, bool]:
    """Normaliza uma data para ISO YYYY-MM-DD. Retorna (valor, ok_formato)."""
    txt = clean(raw)
    if not txt:
        return "", True  # vazio é permitido (preserva)
    # Aceita YYYY-MM-DD ou YYYY/MM/DD ou YYYYMMDD; converte para ISO.
    t = txt.replace("/", "-")
    if len(t) == 8 and t.isdigit():
        t = f"{t[:4]}-{t[4:6]}-{t[6:8]}"
    if ISO_DATE_RE.match(t):
        return t, True
    return txt, False


def build_rows() -> list[dict[str, object]]:
    source, _, _ = load_intake_source()
    rows: list[dict[str, object]] = []
    for i, r in enumerate(source, start=1):
        start, start_ok = _norm_date(r.get("temporal_window_start", ""))
        end, end_ok = _norm_date(r.get("temporal_window_end", ""))
        ev_date, ev_ok = _norm_date(r.get("event_date", ""))

        both_empty = not start and not end
        if both_empty:
            status = "EMPTY"
            notes = "janela vazia preservada; nada inventado"
        elif start_ok and end_ok and ev_ok:
            status = "NORMALIZED_OK"
            notes = "datas em ISO"
        else:
            status = "FORMAT_INVALID"
            bad = [n for n, ok in (("start", start_ok), ("end", end_ok), ("event_date", ev_ok)) if not ok]
            notes = "formato_invalido:" + ",".join(bad)

        rows.append({
            "intake_id": f"MV2_DATA_05_NORM_{i:03d}",
            "target_id": clean(r.get("target_id")),
            "asset_id": clean(r.get("asset_id")),
            "patch_id": clean(r.get("patch_id")),
            "region": clean(r.get("region")),
            "temporal_window_start": start,
            "temporal_window_end": end,
            "temporal_window_source": clean(r.get("temporal_window_source")),
            "event_id": clean(r.get("event_id")),
            "event_date": ev_date,
            "export_task_hint": clean(r.get("export_task_hint")),
            "source_ref": clean(r.get("source_ref")),
            "filled_by": clean(r.get("filled_by")),
            "reviewed_by": clean(r.get("reviewed_by")),
            "review_status": clean(r.get("review_status")),
            "normalization_status": status,
            "normalization_notes": notes,
        })
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_05_normalized_temporal_windows.csv"
    write_csv(out, COLUMNS, rows)
    from collections import Counter

    dist = Counter(str(r["normalization_status"]) for r in rows)
    print(
        f"[mv2_data_05_normalize] {len(rows)} | "
        + " ".join(f"{k}={v}" for k, v in sorted(dist.items()))
        + f" | saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
