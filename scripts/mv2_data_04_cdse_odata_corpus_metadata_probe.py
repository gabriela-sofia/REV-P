"""MV2-DATA-04 Tarefa 5 — CDSE OData corpus metadata probe.

OData só opera com product_id/scene_id. Como o corpus tem 0 product_id, todos os targets do
batch bloqueiam de forma limpa (BLOCKED_NO_PRODUCT_ID). Nada é baixado.

Saída: outputs_public/mv2_data_corpus_metadata_probe/mv2_data_04_cdse_odata_metadata_probe.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_04_common import (
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    ensure_public_dir,
    load_batch,
    write_csv,
)

COLUMNS = [
    "probe_id",
    "target_id",
    "executed",
    "http_status",
    "product_id",
    "scene_id",
    "odata_id",
    "metadata_status",
    "blocking_reason",
    "notes",
]


def build_rows() -> list[dict[str, object]]:
    batch = load_batch()
    rows: list[dict[str, object]] = []
    for i, t in enumerate(batch, start=1):
        # Corpus não tem product_id/scene_id -> OData não tem como resolver.
        rows.append({
            "probe_id": f"MV2_DATA_04_ODATAC_{i:02d}",
            "target_id": clean(t.get("target_id")),
            "executed": "false",
            "http_status": "",
            "product_id": "",
            "scene_id": "",
            "odata_id": "",
            "metadata_status": "BLOCKED_NO_PRODUCT_ID",
            "blocking_reason": "OData_requer_product_id_ou_scene_id_corpus_tem_zero",
            "notes": "resolver product_id via GEE/STAC com janela temporal antes do OData",
        })
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_04_cdse_odata_metadata_probe.csv"
    write_csv(out, COLUMNS, rows)
    blocked = sum(1 for r in rows if r["metadata_status"] == "BLOCKED_NO_PRODUCT_ID")
    print(
        f"[mv2_data_04_odata_probe] {len(rows)} probes | blocked_no_product_id={blocked} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
