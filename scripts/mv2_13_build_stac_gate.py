"""MV2-13 Tarefa 9 — Gate STAC (matriz + plano dry-run). NÃO executa STAC.

Combina binding + geometria/AOI/CRS + temporal/scene lineage e classifica stac_status.
Produz um plano de dry-run hipotético: TODA linha tem would_download=false,
would_create_crop=false, execution_status=NOT_EXECUTED_DRY_RUN_ONLY.

Saídas:
- mv2_13_stac_gate_matrix.csv
- mv2_13_stac_dry_run_plan.csv
"""

from __future__ import annotations

from mv2_13_sentinel_binding_common import (
    OUTPUT_DIR,
    clean,
    ensure_output_dir,
    read_csv_rows,
    write_csv,
)

GATE_COLUMNS = [
    "binding_id", "anchor_id", "asset_id", "patch_id", "sensor",
    "collection_candidate", "scene_id", "datetime_or_range",
    "bbox_or_geometry_source", "crs", "binding_strength", "geometry_ok",
    "temporal_ok", "stac_status", "can_stac_dry_run", "can_stac_real_next_step",
    "reason_not_real", "required_before_real_execution",
]

DRY_RUN_COLUMNS = [
    "dry_run_query_id", "binding_id", "anchor_id", "patch_id", "asset_id",
    "collection_candidate", "bbox", "datetime_or_range", "query_constraints",
    "would_download", "would_create_crop", "execution_status", "reason_not_executed",
]

# Sentinel-first: coleção candidata padrão (apenas rótulo de plano, nada é consultado).
COLLECTION_S2 = "sentinel-2-l2a"


def _collection(sensor: str) -> str:
    s = sensor.upper()
    if "S2" in s or "OPTICAL" in s:
        return COLLECTION_S2
    if "S1" in s or "SAR" in s:
        return "sentinel-1-grd"
    return "UNKNOWN_PENDING_SENSOR_CONFIRMATION"


def build() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    binding = read_csv_rows(OUTPUT_DIR / "mv2_13_binding_matrix.csv")
    geom = {r["binding_id"]: r for r in read_csv_rows(OUTPUT_DIR / "mv2_13_geometry_aoi_crs_matrix.csv")}
    temporal = {r["binding_id"]: r for r in read_csv_rows(OUTPUT_DIR / "mv2_13_temporal_scene_lineage_matrix.csv")}

    gate_rows: list[dict[str, object]] = []
    dry_rows: list[dict[str, object]] = []
    dry_idx = 0

    for b in binding:
        bid = clean(b.get("binding_id"))
        g = geom.get(bid, {})
        t = temporal.get(bid, {})
        strength = clean(b.get("binding_strength"))
        sensor = clean(b.get("sensor"))
        scene = clean(b.get("scene_id"))
        bbox = clean(b.get("bbox"))
        crs = clean(b.get("crs"))
        geometry_ok = clean(g.get("has_patch_geometry")) == "true" and clean(g.get("crs_status")) != "ABSENT"
        temporal_ok = clean(t.get("datetime_status")) == "VALID_SINGLE" or bool(scene)
        stac_status = clean(b.get("stac_status"))

        datetime_or_range = clean(b.get("acquisition_datetime")) or (
            "OPEN_UNKNOWN_PENDING_RECOVERY" if geometry_ok else ""
        )
        collection = _collection(sensor)

        # Fail-closed absoluto: nada vira real/dry executável nesta etapa.
        can_dry = "false"
        can_real = "false"
        reason_not_real = clean(b.get("blocking_reason")) or "binding_insuficiente"
        required = []
        if not clean(b.get("patch_id")):
            required.append("patch_id")
        if not clean(b.get("asset_id")):
            required.append("asset_id")
        if not geometry_ok:
            required.append("geometry+crs")
        if not scene:
            required.append("scene_id")
        if clean(t.get("datetime_status")) != "VALID_SINGLE":
            required.append("acquisition_datetime")
        if sensor in {"", "UNKNOWN"}:
            required.append("sensor_confirmado")
        if strength != "BINDING_STRONG":
            required.append("binding_strong")

        gate_rows.append(
            {
                "binding_id": bid,
                "anchor_id": clean(b.get("anchor_id")),
                "asset_id": clean(b.get("asset_id")),
                "patch_id": clean(b.get("patch_id")),
                "sensor": sensor,
                "collection_candidate": collection,
                "scene_id": scene,
                "datetime_or_range": datetime_or_range,
                "bbox_or_geometry_source": clean(g.get("aoi_status")),
                "crs": crs,
                "binding_strength": strength,
                "geometry_ok": str(geometry_ok).lower(),
                "temporal_ok": str(temporal_ok).lower(),
                "stac_status": stac_status,
                "can_stac_dry_run": can_dry,
                "can_stac_real_next_step": can_real,
                "reason_not_real": reason_not_real,
                "required_before_real_execution": "|".join(required) or "none",
            }
        )

        # Plano de dry-run: rascunho espacial só para bindings com lado espacial pronto
        # (geometria+crs), explicitamente NÃO eligível e NÃO executado.
        if geometry_ok:
            dry_idx += 1
            dry_rows.append(
                {
                    "dry_run_query_id": f"MV2_13_DRY_{dry_idx:05d}",
                    "binding_id": bid,
                    "anchor_id": clean(b.get("anchor_id")),
                    "patch_id": clean(b.get("patch_id")),
                    "asset_id": clean(b.get("asset_id")),
                    "collection_candidate": collection,
                    "bbox": bbox,
                    "datetime_or_range": datetime_or_range,
                    "query_constraints": "spatial_bbox_only;temporal_open_pending_scene_id_recovery",
                    "would_download": "false",
                    "would_create_crop": "false",
                    "execution_status": "NOT_EXECUTED_DRY_RUN_ONLY",
                    "reason_not_executed": "rascunho_espacial_sem_scene_id_nem_datetime;nao_eligivel_dry_run_formal",
                }
            )
    return gate_rows, dry_rows


def main() -> None:
    ensure_output_dir()
    gate_rows, dry_rows = build()
    write_csv(OUTPUT_DIR / "mv2_13_stac_gate_matrix.csv", GATE_COLUMNS, gate_rows)
    write_csv(OUTPUT_DIR / "mv2_13_stac_dry_run_plan.csv", DRY_RUN_COLUMNS, dry_rows)
    real = sum(1 for r in gate_rows if r["can_stac_real_next_step"] == "true")
    print(f"[mv2_13_stac] gate: {len(gate_rows)} | dry-run drafts: {len(dry_rows)} | real eligible: {real}")
    print("[mv2_13_stac] saidas: mv2_13_stac_gate_matrix.csv, mv2_13_stac_dry_run_plan.csv")


if __name__ == "__main__":
    main()
