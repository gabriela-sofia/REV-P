"""MV2-DATA-04 Tarefa 7 — Matriz de lacuna temporal.

Para cada target do batch, registra o que existe (bbox/CRS) e o que falta (janela temporal,
data de evento, hint de data de export, scene_id) — provando que o dado crítico do corpus é
a janela temporal por target.

Saída: outputs_public/mv2_data_corpus_metadata_probe/mv2_data_04_temporal_window_gap_matrix.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_04_common import (
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    crs_status,
    ensure_public_dir,
    filled_temporal_windows,
    load_batch,
    write_csv,
)

COLUMNS = [
    "gap_id",
    "target_id",
    "asset_id",
    "patch_id",
    "region",
    "has_bbox",
    "has_crs",
    "has_temporal_window",
    "has_event_date",
    "has_export_date_hint",
    "has_scene_id",
    "required_for",
    "blocking_reason",
    "manual_recovery_action",
    "priority",
]


def build_rows() -> list[dict[str, object]]:
    batch = load_batch()
    windows = filled_temporal_windows()
    rows: list[dict[str, object]] = []
    for i, t in enumerate(batch, start=1):
        tid = clean(t.get("target_id"))
        has_window = tid in windows
        rows.append({
            "gap_id": f"MV2_DATA_04_GAP_{i:02d}",
            "target_id": tid,
            "asset_id": clean(t.get("asset_id")),
            "patch_id": clean(t.get("patch_id")),
            "region": clean(t.get("region")),
            "has_bbox": "true" if clean(t.get("bbox")) else "false",
            "has_crs": "true" if crs_status(clean(t.get("crs"))) != "ABSENT" else "false",
            "has_temporal_window": "true" if has_window else "false",
            # corpus não tem evento/export/scene vinculado por target ainda
            "has_event_date": "false",
            "has_export_date_hint": "false",
            "has_scene_id": "false",
            "required_for": "GEE_filterDate;STAC_datetime;OData_product_resolution",
            "blocking_reason": "temporal_window_ausente_bloqueia_todo_probe_metadata",
            "manual_recovery_action": "preencher_temporal_window_template_via_evento_ou_export_GEE",
            "priority": clean(t.get("priority")),
        })
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_04_temporal_window_gap_matrix.csv"
    write_csv(out, COLUMNS, rows)
    missing = sum(1 for r in rows if r["has_temporal_window"] == "false")
    print(
        f"[mv2_data_04_gap] {len(rows)} | sem_janela_temporal={missing} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
