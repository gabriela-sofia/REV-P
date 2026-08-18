"""MV2-13 Tarefa 11 — Fila objetiva de recuperação manual/local antes do STAC real.

Gera, a partir da matriz de binding, a lista do que precisa ser recuperado por
binding/grupo antes de qualquer STAC real. Não baixa nada.

Saída: outputs_public/mv2_sentinel_binding/mv2_13_manual_recovery_queue.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_13_sentinel_binding_common import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    clean,
    ensure_output_dir,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "recovery_id", "priority", "target_type", "related_anchor_id",
    "related_asset_id", "related_patch_id", "missing_field", "recovery_action",
    "expected_source", "local_or_external", "can_be_done_without_download",
    "blocks_stac_real", "blocks_day10", "notes",
]


def build_rows() -> list[dict[str, object]]:
    binding = read_csv_rows(OUTPUT_DIR / "mv2_13_binding_matrix.csv")
    rows: list[dict[str, object]] = []
    idx = 0

    def add(priority, target_type, anchor, asset, patch, field, action, source,
            loc, no_dl, blocks_real, blocks_d10, notes):
        nonlocal idx
        idx += 1
        rows.append({
            "recovery_id": f"MV2_13_REC_{idx:04d}",
            "priority": priority,
            "target_type": target_type,
            "related_anchor_id": anchor,
            "related_asset_id": asset,
            "related_patch_id": patch,
            "missing_field": field,
            "recovery_action": action,
            "expected_source": source,
            "local_or_external": loc,
            "can_be_done_without_download": no_dl,
            "blocks_stac_real": blocks_real,
            "blocks_day10": blocks_d10,
            "notes": notes,
        })

    # Ações estruturais (uma vez), independentes de binding individual.
    add("P0", "scene_lineage", "", "", "", "scene_id",
        "recuperar_historico_de_tasks_GEE", "Google Earth Engine task history",
        "local", "true", "true", "true",
        "destrava scene_id+datetime+cloud para os 128 patches espacialmente prontos")
    add("P0", "export_script", "", "", "", "export_script",
        "localizar_script_de_export_GEE", "workspace PROJETO / repo",
        "local", "true", "true", "true", "fonte do binding cena->asset")
    add("P0", "scene_lineage", "", "", "", "scene_id_in_export",
        "localizar_scene_id_no_export", "log/metadata de export GEE",
        "local", "true", "true", "true", "vincula asset->cena de forma auditável")

    # Grupo A: 128 MEDIUM (lado espacial pronto; falta cena/data).
    medium = [b for b in binding if clean(b.get("binding_strength")) == "BINDING_MEDIUM"]
    if medium:
        add("P0", "temporal", "GRUPO_128_MEDIUM", "", "128_patches",
            "scene_id|acquisition_datetime|cloud_cover",
            "recuperar_temporal_para_128_patches_espacialmente_prontos",
            "GEE export / Copernicus metadata", "local", "true", "true", "true",
            "binding MEDIUM: patch+asset+bbox+crs presentes; falta cena+data -> vira dry-run")

    # Grupo B: 10 WEAK strong-scene (cena presente; falta patch/asset/geometria).
    weak_scene = [b for b in binding
                  if clean(b.get("binding_strength")) == "BINDING_WEAK" and clean(b.get("scene_id"))]
    for b in weak_scene:
        add("P1", "spatial_binding", clean(b.get("anchor_id")), "", "",
            "asset_id|patch_id|geometry|crs",
            "vincular_cena_a_patch_e_asset_oficial_sem_inferencia_de_tile",
            "lineage manifest + revisão humana", "local", "true", "true", "true",
            f"cena {clean(b.get('scene_id'))[:31]} sem patch/asset; nao auto-juntar por tile")

    # Conflitos / inválidos.
    bad = [b for b in binding if clean(b.get("conflict_status")) in {"CONFLICT", "INVALID"}]
    for b in bad:
        add("P2", "resolve_conflict", clean(b.get("anchor_id")),
            clean(b.get("asset_id")), clean(b.get("patch_id")),
            "conflict_or_invalid_evidence",
            "validar_se_anchor_pertence_ao_patch_ou_descartar",
            "revisão humana", "local", "true", "true", "false",
            f"conflict_status={clean(b.get('conflict_status'))}")

    # Raster nativo (externo) — necessário para Dia 10, independe do binding.
    add("P1", "native_raster", "", "", "128_patches", "native_sentinel_raster",
        "adquirir_GeoTIFF_L2A_com_bandas_apos_scene_id",
        "Copernicus Dataspace / GEE", "external", "false", "true", "true",
        "Dia 10 exige raster nativo validado; nunca de PNG/NPZ/DINOv2")

    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_13_manual_recovery_queue.csv"
    write_csv(out, COLUMNS, rows)
    print(f"[mv2_13_recovery] itens na fila: {len(rows)}")
    print(f"[mv2_13_recovery] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
