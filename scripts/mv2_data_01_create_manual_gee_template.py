"""MV2-DATA-01 Tarefa 2 — Template manual de preenchimento GEE.

Gera um template preenchível para recuperação manual de lineage no Google Earth Engine.
Preenche SOMENTE campos já conhecidos (target_id, asset_id, patch_id, bbox, crs, region)
e deixa VAZIOS todos os campos que exigem evidência (product_id, scene_id,
acquisition_datetime, mgrs_tile, cloud, datatake, etc.).

Regras fail-closed:
- não preencher product_id/scene_id/datetime/tile/cloud com inferência;
- não usar T23KPR (nem qualquer tile MGRS) como vínculo automático;
- não preencher dados inventados.

Saída: outputs_public/mv2_data_gee_lineage_targets/mv2_data_01_manual_gee_lineage_template.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_01_common import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    build_targets,
    ensure_output_dir,
    write_csv,
)

COLUMNS = [
    "target_id",
    "asset_id",
    "patch_id",
    "region",
    "bbox",
    "crs",
    "gee_collection",
    "product_id",
    "scene_id",
    "acquisition_datetime",
    "mgrs_tile",
    "cloudy_pixel_percentage",
    "cloud_coverage_assessment",
    "nodata_pixel_percentage",
    "datatake_identifier",
    "generation_time",
    "spacecraft_name",
    "export_task_name",
    "export_script_path_or_name",
    "evidence_ref",
    "filled_by",
    "reviewed_by",
    "review_status",
    "notes",
]

# Campos que NUNCA podem vir pré-preenchidos (só por checagem humana no GEE).
EMPTY_UNTIL_HUMAN = [
    "gee_collection",
    "product_id",
    "scene_id",
    "acquisition_datetime",
    "mgrs_tile",
    "cloudy_pixel_percentage",
    "cloud_coverage_assessment",
    "nodata_pixel_percentage",
    "datatake_identifier",
    "generation_time",
    "spacecraft_name",
    "export_task_name",
    "export_script_path_or_name",
    "evidence_ref",
    "filled_by",
    "reviewed_by",
]


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for t in build_targets():
        row: dict[str, object] = {col: "" for col in COLUMNS}
        # Apenas campos já conhecidos são preenchidos.
        row["target_id"] = t["target_id"]
        row["asset_id"] = t["asset_id"]
        row["patch_id"] = t["patch_id"]
        row["region"] = t["region"]
        row["bbox"] = t["bbox"]
        row["crs"] = t["crs"]
        row["review_status"] = "PENDING_HUMAN_FILL"
        row["notes"] = (
            "preencher SOMENTE com evidencia GEE; nao inferir tile/cidade/filename"
        )
        rows.append(row)
    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_data_01_manual_gee_lineage_template.csv"
    write_csv(out, COLUMNS, rows)
    print(
        f"[mv2_data_01_template] {len(rows)} linhas (lineage vazio) | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
