"""MV2-14 Tarefa 8 — Fila manual de recuperação GEE (antes de qualquer STAC real).

Saída: outputs_public/mv2_gee_lineage_recovery/mv2_14_gee_manual_recovery_queue.csv
"""

from __future__ import annotations

from collections import Counter
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
    "recovery_id", "priority", "target_id", "asset_id", "patch_id", "missing_field",
    "recovery_action", "expected_source", "exact_manual_instruction",
    "can_be_done_in_gee_ui", "can_be_done_locally", "blocks_stac_dry_run",
    "blocks_stac_real", "blocks_day10", "notes",
]


def build_rows() -> list[dict[str, object]]:
    lineage = read_csv_rows(OUTPUT_DIR / "mv2_14_lineage_binding_matrix.csv")
    by_region = Counter(clean(r.get("region")) for r in lineage)
    rows: list[dict[str, object]] = []
    idx = 0

    def add(priority, target, asset, patch, field, action, source, instruction,
            in_ui, local, b_dry, b_real, b_d10, notes):
        nonlocal idx
        idx += 1
        rows.append({
            "recovery_id": f"MV2_14_GREC_{idx:03d}", "priority": priority,
            "target_id": target, "asset_id": asset, "patch_id": patch,
            "missing_field": field, "recovery_action": action,
            "expected_source": source, "exact_manual_instruction": instruction,
            "can_be_done_in_gee_ui": in_ui, "can_be_done_locally": local,
            "blocks_stac_dry_run": b_dry, "blocks_stac_real": b_real,
            "blocks_day10": b_d10, "notes": notes,
        })

    # Ações estruturais GEE (fecham lineage dos 128 patches do corpus).
    add("P0", "ALL_128", "", "", "gee_task_history",
        "verificar_historico_de_tasks_GEE",
        "Google Earth Engine — Tasks tab / code.earthengine.google.com",
        "Abrir GEE Code Editor > aba Tasks > localizar exports dos patches do corpus (CUR/PET/REC)",
        "true", "false", "true", "true", "true",
        "maior alavanca: recupera scene_id+datetime+cloud dos 128 patches")
    add("P0", "ALL_128", "", "", "export_script",
        "abrir_task_export_original",
        "GEE script / repo local de export",
        "Abrir o script .js/.py que gerou os exports; ler COLLECTION, filterDate, bandas",
        "true", "true", "true", "true", "true", "fonte da proveniência de export")
    add("P0", "ALL_128", "", "", "scene_id|product_id",
        "recuperar_image_product_id",
        "system:index / PRODUCT_ID no GEE",
        "Para cada patch: image.get('system:index') ou PROPERTIES > PRODUCT_ID",
        "true", "false", "true", "true", "true", "scene_id por patch do corpus")
    add("P0", "ALL_128", "", "", "gee_collection",
        "recuperar_collection",
        "COPERNICUS/S2_SR_HARMONIZED (provável)",
        "Confirmar a ImageCollection usada no export",
        "true", "true", "true", "true", "false", "collection candidata")
    add("P0", "ALL_128", "", "", "acquisition_datetime",
        "recuperar_data_de_aquisicao",
        "image.date() / GENERATION_TIME",
        "Ler a data de aquisição real (não a de processamento/futuro)",
        "true", "false", "true", "true", "true", "Trilha A e STAC")
    add("P1", "ALL_128", "", "", "cloud_cover",
        "recuperar_cloud_cover",
        "CLOUDY_PIXEL_PERCENTAGE",
        "Ler CLOUDY_PIXEL_PERCENTAGE da imagem selecionada",
        "true", "false", "false", "true", "false", "controle de confound")
    add("P1", "ALL_128", "", "", "tile",
        "recuperar_tile_MGRS",
        "MGRS_TILE",
        "Ler MGRS_TILE da imagem (não inferir por zona)",
        "true", "false", "false", "true", "false", "tile sem auto-join")

    # Ações específicas por bucket.
    n_review = by_region.get("Petropolis", 0)
    if n_review:
        add("P1", "PETROPOLIS_REVIEW", "", "", "scene_to_patch_link",
            "validar_quais_cenas_T23_correspondem_a_cada_patch",
            "revisão humana + registry oficial Protocolo-C",
            "Existem cenas T23KPR locais (anchor Protocolo-C); confirmar manualmente se "
            "correspondem aos patches PET do corpus — NUNCA juntar por tile/zona",
            "false", "true", "true", "true", "true",
            f"{n_review} patches Petropolis em CANDIDATE_REVIEW")
    add("P0", "CUR_REC_NOT_FOUND", "", "", "local_scene_for_zone",
        "recuperar_export_GEE_proprio_dos_patches_Curitiba_e_Recife",
        "histórico GEE da usuária",
        "Não há cena local para zonas 22/25; recuperar o export GEE específico desses patches",
        "true", "true", "true", "true", "true",
        "Curitiba(43)+Recife(37) sem cena local")
    add("P2", "ALL_128", "", "", "manual_lineage_fill",
        "preencher_template_manual_de_lineage",
        "mv2_14_manual_lineage_template.csv",
        "Após checagem no GEE, preencher o template com valores REAIS (nunca inventados)",
        "false", "true", "true", "true", "true",
        "template versionável; só metadado leve, nunca raster")

    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_14_gee_manual_recovery_queue.csv"
    write_csv(out, COLUMNS, rows)
    print(f"[mv2_14_recovery] itens: {len(rows)}")
    print(f"[mv2_14_recovery] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
