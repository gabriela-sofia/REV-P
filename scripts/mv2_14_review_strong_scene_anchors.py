"""MV2-14 Tarefa 5 — Revisão das 10 cenas com scene_id (sem auto-join por tile/zona).

Lê as âncoras WEAK com scene_id do MV2-13, enriquece com metadados candidatos
(collection, datetime, cloud) das fontes locais, e produz fila de revisão manual.
auto_join_allowed=false para todas, salvo asset_id/patch_id explícito em fonte — que
não existe localmente; logo permanece false.

Saída: outputs_public/mv2_gee_lineage_recovery/mv2_14_strong_scene_anchor_review.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_14_gee_lineage_common import (
    MV2_13_BINDING_MATRIX,
    OUTPUT_DIR,
    PROJECT_ROOT,
    REGION_ZONE,
    clean,
    ensure_output_dir,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "scene_anchor_id", "scene_id", "tile", "datetime", "sensor",
    "possible_region_hint", "possible_crs_zone_hint", "auto_join_allowed",
    "review_required", "reason", "needed_evidence", "next_action",
]


def _region_from_tile(tile: str) -> tuple[str, str]:
    """Apenas HINT (zona), nunca vínculo. T23 -> zona 23 -> hint Petrópolis."""
    zone = tile[1:3] if tile.startswith("T") and len(tile) >= 3 else tile[:2]
    for region, (crs, tprefix) in REGION_ZONE.items():
        if tprefix == f"T{zone}":
            return region, crs
    return "DESCONHECIDO", ""


def build_rows() -> list[dict[str, object]]:
    binding = read_csv_rows(MV2_13_BINDING_MATRIX)
    weak_scene = [b for b in binding
                  if clean(b.get("binding_strength")) == "BINDING_WEAK" and clean(b.get("scene_id"))]
    # Enriquecimento por scene_id a partir dos candidatos locais.
    candidates = read_csv_rows(OUTPUT_DIR / "mv2_14_lineage_candidates.csv")
    cand_by_scene: dict[str, dict] = {}
    for c in candidates:
        sid = clean(c.get("normalized_value"))
        if sid and sid not in cand_by_scene and clean(c.get("inferred_datetime")):
            cand_by_scene[sid] = c

    rows: list[dict[str, object]] = []
    for i, b in enumerate(weak_scene, start=1):
        scene = clean(b.get("scene_id"))
        tile = clean(b.get("tile_id")) or (scene.split("_")[-1] if "_" in scene else "")
        cand = cand_by_scene.get(scene, {})
        dt = clean(b.get("acquisition_datetime")) or clean(cand.get("inferred_datetime"))
        sensor = clean(b.get("sensor")) or clean(cand.get("inferred_sensor")) or "S2_MSI"
        region_hint, crs_hint = _region_from_tile(tile)
        rows.append({
            "scene_anchor_id": f"MV2_14_SCENE_{i:02d}",
            "scene_id": scene,
            "tile": tile,
            "datetime": dt,
            "sensor": sensor,
            "possible_region_hint": region_hint,
            "possible_crs_zone_hint": crs_hint,
            # Sem asset_id/patch_id explícito em fonte: auto-join PROIBIDO.
            "auto_join_allowed": "false",
            "review_required": "true",
            "reason": "scene_id_forte_mas_sem_asset_id_patch_id_explicito;tile/zona_nao_basta",
            "needed_evidence": "asset_id_ou_patch_id_oficial_que_referencie_esta_cena",
            "next_action": "revisar_humano_se_cena_pertence_a_patch_do_corpus_sem_inferir_por_tile",
        })
    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_14_strong_scene_anchor_review.csv"
    write_csv(out, COLUMNS, rows)
    auto = sum(1 for r in rows if r["auto_join_allowed"] == "true")
    print(f"[mv2_14_scene_review] cenas revisadas: {len(rows)} | auto_join permitido: {auto}")
    print(f"[mv2_14_scene_review] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
