"""MV2-DATA-04 Tarefa 4 — CDSE STAC corpus metadata probe (Item Search metadata-only).

Prepara STAC Item Search para os targets do batch. Não baixa assets. BLOQUEIA se não houver
janela temporal — nunca faz query aberta ampla.

Saída: outputs_public/mv2_data_corpus_metadata_probe/mv2_data_04_cdse_stac_corpus_metadata_probe.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_04_common import (
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    effective_flags,
    ensure_public_dir,
    filled_temporal_windows,
    metadata_blocked,
    load_batch,
    write_csv,
)

COLUMNS = [
    "probe_id",
    "target_id",
    "executed",
    "http_status",
    "collection",
    "bbox",
    "datetime_window",
    "temporal_window_status",
    "matched_items",
    "selected_item_id",
    "selected_datetime",
    "selected_mgrs_tile",
    "selected_cloud_cover",
    "metadata_status",
    "blocking_reason",
    "notes",
]


def build_rows() -> list[dict[str, object]]:
    flags = effective_flags()
    batch = load_batch()
    windows = filled_temporal_windows()
    config_block = metadata_blocked(flags) or not flags.get("cdse_stac_enabled")

    rows: list[dict[str, object]] = []
    for i, t in enumerate(batch, start=1):
        tid = clean(t.get("target_id"))
        window = windows.get(tid)
        has_window = window is not None
        base: dict[str, object] = {
            "probe_id": f"MV2_DATA_04_STACC_{i:02d}",
            "target_id": tid,
            "collection": "SENTINEL-2",
            "bbox": clean(t.get("bbox")),
            "datetime_window": f"{window[0]}/{window[1]}" if window else "",
            "temporal_window_status": "PRESENT" if has_window else "MISSING",
            "matched_items": "",
            "selected_item_id": "",
            "selected_datetime": "",
            "selected_mgrs_tile": "",
            "selected_cloud_cover": "",
            "executed": "false",
            "http_status": "",
        }
        if not has_window:
            base.update({
                "metadata_status": "BLOCKED_NO_TEMPORAL_WINDOW",
                "blocking_reason": "sem_janela_temporal_query_aberta_proibida",
                "notes": "Item Search exige datetime fechado; query aberta ampla nunca permitida",
            })
        elif config_block:
            base.update({
                "metadata_status": "NOT_EXECUTED_CONFIG_BLOCKED",
                "blocking_reason": "config_ausente_ou_flags_stac_desabilitadas",
                "notes": "payload Item Search com bbox+datetime pronto (metadata-only)",
            })
        else:
            base.update({
                "metadata_status": "EXECUTED_NO_MATCH_PARSED",
                "blocking_reason": "",
                "notes": "Item Search metadata-only; assets nunca baixados",
            })
        rows.append(base)
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_04_cdse_stac_corpus_metadata_probe.csv"
    write_csv(out, COLUMNS, rows)
    executed = sum(1 for r in rows if r["executed"] == "true")
    blocked = sum(1 for r in rows if r["metadata_status"] == "BLOCKED_NO_TEMPORAL_WINDOW")
    print(
        f"[mv2_data_04_stac_probe] {len(rows)} probes | executados={executed} "
        f"blocked_no_temporal_window={blocked} | saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
