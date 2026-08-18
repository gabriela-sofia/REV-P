"""MV2-DATA-05 Tarefa 5 — Batch pronto para probe (DATA-06).

Cruza o gate de promoção com o batch DATA-04 (bbox/CRS) para produzir o batch que DATA-06
poderá usar: targets com janela temporal promovida ficam can_probe_gee/stac=true; os demais
permanecem bloqueados. OData segue bloqueado (sem product_id).

Saída: outputs_public/mv2_data_temporal_window_intake/mv2_data_05_probe_ready_batch.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_05_common import (
    PROJECT_ROOT,
    PUBLIC_DIR,
    batch_index,
    clean,
    ensure_public_dir,
    read_csv_rows,
    write_csv,
)

BATCH_ID = "MV2_DATA_05_PROBE_READY_BATCH_01"

COLUMNS = [
    "batch_id",
    "target_id",
    "asset_id",
    "patch_id",
    "region",
    "bbox",
    "crs",
    "temporal_window_start",
    "temporal_window_end",
    "temporal_window_status",
    "can_probe_gee_metadata",
    "can_probe_stac_metadata",
    "can_probe_odata_metadata",
    "blocking_reason",
    "next_action",
]


def build_rows() -> list[dict[str, object]]:
    gate = read_csv_rows(PUBLIC_DIR / "mv2_data_05_temporal_promotion_gate.csv")
    batch = batch_index()
    rows: list[dict[str, object]] = []
    for g in gate:
        tid = clean(g.get("target_id"))
        b = batch.get(tid, {})
        can_gee = clean(g.get("can_enable_gee_metadata_probe")) == "true"
        can_stac = clean(g.get("can_enable_stac_metadata_probe")) == "true"
        promotion = clean(g.get("promotion_status"))
        if promotion == "TEMPORAL_WINDOW_PROMOTED_STRONG":
            tw_status = "PROMOTED_STRONG"
        elif promotion == "TEMPORAL_WINDOW_PROMOTED_PARTIAL":
            tw_status = "PROMOTED_PARTIAL"
        else:
            tw_status = "NOT_PROMOTED"

        rows.append({
            "batch_id": BATCH_ID,
            "target_id": tid,
            "asset_id": clean(g.get("asset_id")),
            "patch_id": clean(g.get("patch_id")),
            "region": clean(g.get("region")),
            "bbox": clean(b.get("bbox")),
            "crs": clean(b.get("crs")),
            "temporal_window_start": clean(g.get("temporal_window_start")),
            "temporal_window_end": clean(g.get("temporal_window_end")),
            "temporal_window_status": tw_status,
            "can_probe_gee_metadata": "true" if can_gee else "false",
            "can_probe_stac_metadata": "true" if can_stac else "false",
            "can_probe_odata_metadata": "false",  # exige product_id/scene_id
            "blocking_reason": clean(g.get("blocking_reason")) or ("" if can_gee else "janela_nao_promovida"),
            "next_action": (
                "rodar_DATA-06_metadata_probe_com_janela" if can_gee
                else "preencher/corrigir_janela_temporal"
            ),
        })
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_05_probe_ready_batch.csv"
    write_csv(out, COLUMNS, rows)
    ready = sum(1 for r in rows if r["can_probe_gee_metadata"] == "true")
    print(
        f"[mv2_data_05_probe_ready] {len(rows)} | probe_ready_gee={ready} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
