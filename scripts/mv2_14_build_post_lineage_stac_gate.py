"""MV2-14 Tarefa 6 — Gate STAC pós-lineage. NÃO executa STAC.

Combina o binding MV2-13 (geometria/CRS) com o lineage recuperado e decide a prontidão
STAC. Nada vira executável: can_stac_dry_run/can_stac_real_review_required são apenas
sinalizadores para etapa futura.

Saída: outputs_public/mv2_gee_lineage_recovery/mv2_14_post_lineage_stac_gate.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_14_gee_lineage_common import (
    MV2_13_BINDING_MATRIX,
    OUTPUT_DIR,
    PROJECT_ROOT,
    clean,
    ensure_output_dir,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "gate_id", "lineage_binding_id", "binding_id", "asset_id", "patch_id",
    "scene_id", "datetime", "tile", "cloud_cover", "bbox", "crs", "lineage_status",
    "geometry_ok", "temporal_ok", "provenance_ok", "stac_readiness",
    "can_stac_dry_run", "can_stac_real_review_required", "stac_real_executed",
    "download_executed", "reason_not_real", "next_action",
]


def build_rows() -> list[dict[str, object]]:
    lineage = read_csv_rows(OUTPUT_DIR / "mv2_14_lineage_binding_matrix.csv")
    binding = {clean(b.get("binding_id")): b for b in read_csv_rows(MV2_13_BINDING_MATRIX)}
    rows: list[dict[str, object]] = []
    for i, lr in enumerate(lineage, start=1):
        bid = clean(lr.get("binding_id"))
        b = binding.get(bid, {})
        bbox = clean(b.get("bbox"))
        crs = clean(b.get("crs"))
        geometry_ok = bool(bbox) and clean(b.get("crs_status")) != "ABSENT"
        scene = clean(lr.get("recovered_scene_id"))
        dt = clean(lr.get("recovered_datetime"))
        temporal_ok = bool(scene) and bool(dt)
        provenance_ok = bool(clean(lr.get("export_task_or_source")))
        status = clean(lr.get("lineage_status"))
        stac_ready = clean(lr.get("stac_readiness_after_lineage"))

        # Só lineage forte revisável poderia sinalizar review de STAC real (futuro).
        can_dry = "false"
        can_real_review = "false"
        if status == "LINEAGE_RECOVERED_STRONG" and geometry_ok and temporal_ok:
            can_dry = "true"
            can_real_review = "true"  # apenas REVIEW; nunca executa
            stac_ready = "READY_FOR_STAC_REAL_REVIEW_REQUIRED"

        reason = "lineage_incompleto_ou_sem_geometria" if can_real_review == "false" else "requer_revisao_humana_antes_de_executar"
        action = clean(lr.get("next_action"))

        rows.append({
            "gate_id": f"MV2_14_STAC_{i:05d}",
            "lineage_binding_id": clean(lr.get("lineage_binding_id")),
            "binding_id": bid,
            "asset_id": clean(lr.get("asset_id")),
            "patch_id": clean(lr.get("patch_id")),
            "scene_id": scene,
            "datetime": dt,
            "tile": clean(lr.get("recovered_tile")),
            "cloud_cover": clean(lr.get("recovered_cloud_cover")),
            "bbox": bbox,
            "crs": crs,
            "lineage_status": status,
            "geometry_ok": str(geometry_ok).lower(),
            "temporal_ok": str(temporal_ok).lower(),
            "provenance_ok": str(provenance_ok).lower(),
            "stac_readiness": stac_ready,
            "can_stac_dry_run": can_dry,
            "can_stac_real_review_required": can_real_review,
            # Fail-closed absoluto.
            "stac_real_executed": "false",
            "download_executed": "false",
            "reason_not_real": reason,
            "next_action": action,
        })
    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_14_post_lineage_stac_gate.csv"
    write_csv(out, COLUMNS, rows)
    dry = sum(1 for r in rows if r["can_stac_dry_run"] == "true")
    rev = sum(1 for r in rows if r["can_stac_real_review_required"] == "true")
    print(f"[mv2_14_stac] gate: {len(rows)} | dry_run: {dry} | real_review_required: {rev}")
    print(f"[mv2_14_stac] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
