"""MV2-DATA-04 Tarefa 8 — Template de janela temporal por target.

Gera o template preenchível para a janela temporal de cada target do batch. NÃO inventa
datas: os campos temporais ficam vazios se não houver evidência. Em re-execuções, PRESERVA
valores já preenchidos por revisão humana (não sobrescreve).

Saída: outputs_public/mv2_data_corpus_metadata_probe/mv2_data_04_temporal_window_template.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_04_common import (
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    ensure_public_dir,
    load_batch,
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
    "notes",
]

# Campos preenchíveis por humano que devem ser preservados em re-execuções.
PRESERVE = [
    "temporal_window_start", "temporal_window_end", "temporal_window_source",
    "event_id", "event_date", "export_task_hint", "source_ref",
    "filled_by", "reviewed_by",
]

OUT_NAME = "mv2_data_04_temporal_window_template.csv"


def _existing() -> dict[str, dict[str, str]]:
    return {clean(r.get("target_id")): r for r in read_csv_rows(PUBLIC_DIR / OUT_NAME)}


def build_rows() -> list[dict[str, object]]:
    batch = load_batch()
    prior = _existing()
    rows: list[dict[str, object]] = []
    for t in batch:
        tid = clean(t.get("target_id"))
        old = prior.get(tid, {})
        row: dict[str, object] = {col: "" for col in COLUMNS}
        # Campos já conhecidos.
        row["target_id"] = tid
        row["asset_id"] = clean(t.get("asset_id"))
        row["patch_id"] = clean(t.get("patch_id"))
        row["region"] = clean(t.get("region"))
        # Preserva valores humanos prévios; nunca inventa.
        for f in PRESERVE:
            if clean(old.get(f)):
                row[f] = clean(old.get(f))
        row["review_status"] = clean(old.get("review_status")) or "PENDING_HUMAN_FILL"
        row["notes"] = clean(old.get("notes")) or "preencher janela temporal SO com evidencia (evento/export); nao inventar data"
        rows.append(row)
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / OUT_NAME
    write_csv(out, COLUMNS, rows)
    filled = sum(1 for r in rows if r["temporal_window_start"] and r["temporal_window_end"])
    print(
        f"[mv2_data_04_template] {len(rows)} linhas | janelas_preenchidas={filled} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
