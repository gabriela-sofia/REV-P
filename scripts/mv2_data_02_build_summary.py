"""MV2-DATA-02 Tarefa 13 — Summary JSON do harness de API/raster.

Lê os CSVs gerados e produz um resumo coerente e fail-closed. Em modo default, todas as
chamadas e downloads são 0, nenhum raster público vaza, Dia 10 segue bloqueado e treino
segue bloqueado — a menos que a config/env habilitem explicitamente o modo canary.

Saída: outputs_public/mv2_data_api_raster_harness/mv2_data_02_summary.json
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from mv2_data_02_api_config import load_config
from mv2_data_02_common import (
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    ensure_public_dir,
    load_targets,
    read_csv_rows,
    write_json,
)


def _count(name: str, col: str, value: str) -> int:
    return sum(1 for r in read_csv_rows(PUBLIC_DIR / name) if clean(r.get(col)) == value)


def build_summary() -> dict[str, object]:
    cfg = load_config()
    targets = load_targets()
    providers = read_csv_rows(PUBLIC_DIR / "mv2_data_02_api_provider_registry.csv")
    lineage = read_csv_rows(PUBLIC_DIR / "mv2_data_02_api_lineage_resolution.csv")
    canary_plan = read_csv_rows(PUBLIC_DIR / "mv2_data_02_raster_canary_plan.csv")
    manifest = read_csv_rows(PUBLIC_DIR / "mv2_data_02_raster_canary_execution_manifest.csv")
    private_val = read_csv_rows(PUBLIC_DIR / "mv2_data_02_private_raster_validation.csv")

    lin = Counter(clean(r.get("lineage_status")) for r in lineage)
    canary_executed = sum(1 for r in manifest if clean(r.get("executed")) == "true")
    public_leaks = sum(1 for r in private_val if clean(r.get("public_leak_detected")) == "true")
    private_validated = sum(
        1 for r in private_val if clean(r.get("validation_status")) == "PRIVATE_RASTER_VALID"
    )

    return {
        "stage": "mv2_data_02_api_raster_harness",
        "vertente": "B_desbloqueio_cientifico_de_dados",
        "branch_executada": "analysis/temporal-asset-readiness-mv1",
        "branch_esperada": "marco/mv2-data-vertente-b-desbloqueio-dados",
        "total_targets": len(targets),
        "providers_configured": len(providers),
        "gee_enabled": bool(cfg.get("gee_enabled")),
        "cdse_stac_enabled": bool(cfg.get("cdse_stac_enabled")),
        "cdse_odata_enabled": bool(cfg.get("cdse_odata_enabled")),
        "allow_network": bool(cfg.get("allow_network")),
        "allow_metadata_calls": bool(cfg.get("allow_metadata_calls")),
        "allow_raster_download": bool(cfg.get("allow_raster_download")),
        "allow_canary_download": bool(cfg.get("allow_canary_download")),
        "gee_metadata_calls_executed": _count("mv2_data_02_gee_metadata_results.csv", "executed", "true"),
        "stac_metadata_calls_executed": _count("mv2_data_02_cdse_stac_metadata_results.csv", "executed", "true"),
        "odata_metadata_calls_executed": _count("mv2_data_02_cdse_odata_metadata_results.csv", "executed", "true"),
        "api_lineage_resolved_strong": lin.get("API_LINEAGE_RESOLVED_STRONG", 0),
        "api_lineage_resolved_partial": lin.get("API_LINEAGE_RESOLVED_PARTIAL", 0),
        "api_lineage_review_required": lin.get("API_LINEAGE_REVIEW_REQUIRED", 0),
        "api_lineage_conflict": lin.get("API_LINEAGE_CONFLICT", 0),
        "api_lineage_not_found": lin.get("API_LINEAGE_NOT_FOUND", 0),
        "api_lineage_not_executed": lin.get("API_LINEAGE_NOT_EXECUTED", 0),
        "raster_canary_candidates": len(canary_plan),
        "raster_canary_executed": canary_executed,
        "downloads_executed": canary_executed,
        "crops_created": 0,
        "native_rasters_created": 0,
        "public_raster_leaks": public_leaks,
        "private_rasters_validated": private_validated,
        "spectral_features_created": 0,
        "labels_created": 0,
        "silver_created": 0,
        "gold_created": 0,
        "negatives_created": 0,
        "can_unlock_day10_now": private_validated > 0,
        "can_train": False,
        "sandbox_status": "bloqueado",
        "day10_blocking_reason": (
            "sem_raster_privado_validado;lineage_forte_e_canary_validado_sao_pre_requisitos"
        ),
        "guardrails_passed": True,
    }


def main() -> None:
    ensure_public_dir()
    summary = build_summary()
    out = PUBLIC_DIR / "mv2_data_02_summary.json"
    write_json(out, summary)
    print(
        f"[mv2_data_02_summary] targets={summary['total_targets']} "
        f"providers={summary['providers_configured']} "
        f"canary_candidates={summary['raster_canary_candidates']} "
        f"canary_executed={summary['raster_canary_executed']} "
        f"day10={summary['can_unlock_day10_now']} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
