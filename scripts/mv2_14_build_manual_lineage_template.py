"""MV2-14 Tarefa 9 — Template manual preenchível de lineage.

Gera um CSV com uma linha por patch alvo (128), preenchendo APENAS os campos já
conhecidos e verdadeiros (asset_id, patch_id, region) e deixando TODOS os campos de
lineage VAZIOS para preenchimento humano após checagem no GEE. Nada é inventado.

Saída: outputs_public/mv2_gee_lineage_recovery/mv2_14_manual_lineage_template.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_14_gee_lineage_common import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    clean,
    ensure_output_dir,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "asset_id", "patch_id", "scene_id", "acquisition_datetime", "tile",
    "cloud_cover", "sensor", "platform", "gee_collection", "gee_export_task",
    "gee_script_path_or_name", "evidence_url_or_local_ref", "filled_by",
    "reviewed_by", "review_status", "notes",
]

# Campos de lineage que DEVEM permanecer vazios no template (preenchimento humano).
LINEAGE_FILL_FIELDS = [
    "scene_id", "acquisition_datetime", "tile", "cloud_cover", "gee_collection",
    "gee_export_task", "gee_script_path_or_name", "evidence_url_or_local_ref",
    "filled_by", "reviewed_by",
]


def build_rows() -> list[dict[str, object]]:
    lineage = read_csv_rows(OUTPUT_DIR / "mv2_14_lineage_binding_matrix.csv")
    rows: list[dict[str, object]] = []
    for lr in lineage:
        row: dict[str, object] = {c: "" for c in COLUMNS}
        # Apenas fatos já conhecidos — nunca lineage inventado.
        row["asset_id"] = clean(lr.get("asset_id"))
        row["patch_id"] = clean(lr.get("patch_id"))
        # sensor/platform conhecidos do binding ficam como dica, não como lineage temporal.
        row["sensor"] = ""  # preenchimento humano (S2 confirmado no GEE)
        row["platform"] = ""
        row["review_status"] = "PENDING_GEE_CHECK"
        row["notes"] = f"lineage_status_atual={clean(lr.get('lineage_status'))};preencher_apos_checagem_GEE"
        rows.append(row)
    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_14_manual_lineage_template.csv"
    write_csv(out, COLUMNS, rows)
    print(f"[mv2_14_template] linhas (preencher humano): {len(rows)} | campos lineage vazios por design")
    print(f"[mv2_14_template] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
