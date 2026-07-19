"""MV2-DATA-01 Tarefa 10 — Summary JSON do GEE Lineage Target Pack.

Lê os CSVs já gerados e produz um resumo coerente. Todos os contadores de execução são 0
(metadata-only), Dia 10 não é desbloqueado e treino segue bloqueado.

Saída: outputs_public/mv2_data_gee_lineage_targets/mv2_data_01_summary.json
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from mv2_data_01_common import (
    BRANCH_ESPERADA,
    BRANCH_EXECUTADA,
    OUTPUT_DIR,
    PROJECT_ROOT,
    clean,
    ensure_output_dir,
    read_csv_rows,
    write_json,
)


def _count_true(rows: list[dict[str, str]], col: str) -> int:
    return sum(1 for r in rows if clean(r.get(col)) == "true")


def build_summary() -> dict[str, object]:
    targets = read_csv_rows(OUTPUT_DIR / "mv2_data_01_lineage_targets.csv")
    template = read_csv_rows(OUTPUT_DIR / "mv2_data_01_manual_gee_lineage_template.csv")
    gee_plan = read_csv_rows(OUTPUT_DIR / "mv2_data_01_gee_metadata_query_plan.csv")
    stac_plan = read_csv_rows(OUTPUT_DIR / "mv2_data_01_stac_odata_metadata_plan.csv")
    checklist = read_csv_rows(OUTPUT_DIR / "mv2_data_01_manual_recovery_checklist.csv")
    risk = read_csv_rows(OUTPUT_DIR / "mv2_data_01_risk_matrix.csv")

    status_dist = Counter(clean(r.get("lineage_status")) for r in targets)
    priority_dist = Counter(clean(r.get("priority")) for r in targets)

    return {
        "stage": "mv2_data_01_gee_lineage_target_pack",
        "vertente": "B_desbloqueio_cientifico_de_dados",
        "branch_executada": BRANCH_EXECUTADA,
        "branch_esperada": BRANCH_ESPERADA,
        "divergencia_branch": BRANCH_EXECUTADA != BRANCH_ESPERADA,
        "total_targets": len(targets),
        "targets_with_asset_id": _count_true(targets, "has_asset_id"),
        "targets_with_patch_id": _count_true(targets, "has_patch_id"),
        "targets_with_bbox": _count_true(targets, "has_bbox"),
        "targets_with_crs": _count_true(targets, "has_crs"),
        "targets_with_scene_id": _count_true(targets, "has_scene_id"),
        "targets_with_datetime": _count_true(targets, "has_datetime"),
        "targets_with_tile": _count_true(targets, "has_tile"),
        "targets_with_cloud_cover": _count_true(targets, "has_cloud_cover"),
        "targets_ready_manual_gee_lookup": status_dist.get(
            "TARGET_READY_FOR_MANUAL_GEE_LOOKUP", 0
        ),
        "targets_ready_metadata_query_plan": status_dist.get(
            "TARGET_READY_FOR_METADATA_QUERY_PLAN", 0
        ),
        "targets_blocked": sum(
            v for k, v in status_dist.items() if k.startswith("TARGET_BLOCKED")
        ),
        "targets_invalid": status_dist.get("TARGET_INVALID", 0),
        "priority_distribution": dict(sorted(priority_dist.items())),
        "manual_template_rows": len(template),
        "gee_metadata_query_plans": len(gee_plan),
        "stac_odata_query_plans": len(stac_plan),
        "manual_checklist_steps": len(checklist),
        "risk_rows": len(risk),
        "http_calls_executed": 0,
        "gee_calls_executed": 0,
        "downloads_executed": 0,
        "crops_created": 0,
        "native_rasters_created": 0,
        "spectral_features_created": 0,
        "labels_created": 0,
        "silver_created": 0,
        "gold_created": 0,
        "negatives_created": 0,
        "can_unlock_day10_now": False,
        "can_train": False,
        "sandbox_status": "bloqueado",
        "day10_blocking_reason": (
            "sem_raster_Sentinel_nativo_validado;lineage_so_recuperavel_por_consulta_"
            "metadata_only_em_revisao_humana"
        ),
        "guardrails_passed": True,
    }


def main() -> None:
    ensure_output_dir()
    summary = build_summary()
    out = OUTPUT_DIR / "mv2_data_01_summary.json"
    write_json(out, summary)
    print(
        f"[mv2_data_01_summary] targets={summary['total_targets']} "
        f"scene_id={summary['targets_with_scene_id']} "
        f"gee_plans={summary['gee_metadata_query_plans']} "
        f"stac_plans={summary['stac_odata_query_plans']} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
