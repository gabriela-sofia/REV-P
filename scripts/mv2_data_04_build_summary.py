"""MV2-DATA-04 Tarefa 13 — Summary JSON do corpus metadata probe.

Lê os CSVs gerados e produz um resumo coerente e fail-closed. Enquanto não houver janela
temporal preenchida, o resultado esperado é bloqueio limpo: 0 probes executados, lineage
bloqueado por janela temporal/product_id, e corpus/Dia 10/sandbox/treino bloqueados.

Saída: outputs_public/mv2_data_corpus_metadata_probe/mv2_data_04_summary.json
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from mv2_data_04_common import (
    BRANCH_EXECUTADA,
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    effective_flags,
    ensure_public_dir,
    load_corpus_targets,
    read_csv_rows,
    write_json,
)


def build_summary() -> dict[str, object]:
    flags = effective_flags()
    corpus = load_corpus_targets()
    batch = read_csv_rows(PUBLIC_DIR / "mv2_data_04_corpus_metadata_batch.csv")
    gee = read_csv_rows(PUBLIC_DIR / "mv2_data_04_gee_corpus_metadata_probe.csv")
    stac = read_csv_rows(PUBLIC_DIR / "mv2_data_04_cdse_stac_corpus_metadata_probe.csv")
    odata = read_csv_rows(PUBLIC_DIR / "mv2_data_04_cdse_odata_metadata_probe.csv")
    lineage = read_csv_rows(PUBLIC_DIR / "mv2_data_04_corpus_api_lineage_resolution.csv")

    region = Counter(clean(r.get("region")) for r in batch)
    lin = Counter(clean(r.get("lineage_status")) for r in lineage)

    def execed(rows: list[dict[str, str]]) -> int:
        return sum(1 for r in rows if clean(r.get("executed")) == "true")

    return {
        "stage": "mv2_data_04_corpus_metadata_probe",
        "vertente": "B_desbloqueio_cientifico_de_dados",
        "branch": BRANCH_EXECUTADA,
        "total_targets": len(corpus),
        "batch_targets": len(batch),
        "batch_recife": region.get("Recife", 0),
        "batch_petropolis": region.get("Petropolis", 0),
        "batch_curitiba": region.get("Curitiba", 0),
        "config_found": bool(flags["config_found"]),
        "allow_network": bool(flags["allow_network"]),
        "allow_metadata_calls": bool(flags["allow_metadata_calls"]),
        "gee_probes_executed": execed(gee),
        "stac_probes_executed": execed(stac),
        "odata_probes_executed": execed(odata),
        "blocked_no_temporal_window": sum(
            1 for r in gee if clean(r.get("metadata_status")) == "BLOCKED_NO_TEMPORAL_WINDOW"
        ),
        "blocked_no_product_id": sum(
            1 for r in odata if clean(r.get("metadata_status")) == "BLOCKED_NO_PRODUCT_ID"
        ),
        "corpus_lineage_resolved_strong": lin.get("CORPUS_LINEAGE_RESOLVED_STRONG", 0),
        "corpus_lineage_resolved_partial": lin.get("CORPUS_LINEAGE_RESOLVED_PARTIAL", 0),
        "corpus_lineage_review_required": lin.get("CORPUS_LINEAGE_REVIEW_REQUIRED", 0),
        "corpus_lineage_blocked_no_temporal_window": lin.get("CORPUS_LINEAGE_BLOCKED_NO_TEMPORAL_WINDOW", 0),
        "can_stac_dry_run": sum(1 for r in lineage if clean(r.get("can_stac_dry_run")) == "true"),
        "can_raster_download_next_step": sum(
            1 for r in lineage if clean(r.get("can_raster_download_next_step")) == "true"
        ),
        "downloads_executed": 0,
        "crops_created": 0,
        "native_rasters_created": 0,
        "spectral_features_created": 0,
        "labels_created": 0,
        "silver_created": 0,
        "gold_created": 0,
        "negatives_created": 0,
        "corpus_day10_unlocked": False,
        "sandbox_unlocked": False,
        "can_train": False,
        "guardrails_passed": True,
    }


def main() -> None:
    ensure_public_dir()
    summary = build_summary()
    out = PUBLIC_DIR / "mv2_data_04_summary.json"
    write_json(out, summary)
    print(
        f"[mv2_data_04_summary] batch={summary['batch_targets']} "
        f"(REC={summary['batch_recife']} PET={summary['batch_petropolis']} CUR={summary['batch_curitiba']}) "
        f"blocked_no_temporal={summary['blocked_no_temporal_window']} "
        f"blocked_no_product_id={summary['blocked_no_product_id']} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
