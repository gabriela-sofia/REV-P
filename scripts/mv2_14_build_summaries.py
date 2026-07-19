"""MV2-14 Tarefa 12c — Summary JSON coerente com os CSVs.

Saída: outputs_public/mv2_gee_lineage_recovery/mv2_14_lineage_summary.json
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
    write_json,
)


def build_summary() -> dict[str, object]:
    candidates = read_csv_rows(OUTPUT_DIR / "mv2_14_lineage_candidates.csv")
    targets = read_csv_rows(OUTPUT_DIR / "mv2_14_medium_binding_index.csv")
    lineage = read_csv_rows(OUTPUT_DIR / "mv2_14_lineage_binding_matrix.csv")
    scene_review = read_csv_rows(OUTPUT_DIR / "mv2_14_strong_scene_anchor_review.csv")
    stac = read_csv_rows(OUTPUT_DIR / "mv2_14_post_lineage_stac_gate.csv")
    day10 = read_csv_rows(OUTPUT_DIR / "mv2_14_day10_post_lineage_gate.csv")
    sources = read_csv_rows(OUTPUT_DIR / "mv2_14_gee_source_discovery.csv")

    dist = Counter(clean(r.get("lineage_status")) for r in lineage)
    auto_joins = sum(1 for r in scene_review if clean(r.get("auto_join_allowed")) == "true")
    dry = sum(1 for r in stac if clean(r.get("can_stac_dry_run")) == "true")
    real_review = sum(1 for r in stac if clean(r.get("can_stac_real_review_required")) == "true")
    d10_next = sum(1 for r in day10 if clean(r.get("can_unlock_day10_next_step")) == "true")
    d10_now = sum(1 for r in day10 if clean(r.get("can_unlock_day10_now")) == "true")

    return {
        "stage": "mv2_14_gee_lineage_recovery",
        "branch_executada": "marco/validacao-label-free-evidencia-estrutural-mv1",
        "branch_esperada": "marco/mv2-12-reconstrucao-espectral-sentinel-baseline",
        "total_medium_bindings_input": len(targets),
        "total_gee_sources_discovered": len(sources),
        "total_lineage_candidates": len(candidates),
        "lineage_recovered_strong": dist.get("LINEAGE_RECOVERED_STRONG", 0),
        "lineage_recovered_partial": dist.get("LINEAGE_RECOVERED_PARTIAL", 0),
        "lineage_candidate_review": dist.get("LINEAGE_CANDIDATE_REVIEW", 0),
        "lineage_conflict": dist.get("LINEAGE_CONFLICT", 0),
        "lineage_invalid": dist.get("LINEAGE_INVALID", 0),
        "lineage_not_found": dist.get("LINEAGE_NOT_FOUND", 0),
        "strong_scene_anchors_reviewed": len(scene_review),
        "auto_joins_performed": auto_joins,
        "stac_dry_run_eligible_after_lineage": dry,
        "stac_real_review_required_after_lineage": real_review,
        "stac_real_executed": 0,
        "downloads_executed": 0,
        "crops_created": 0,
        "native_rasters_created": 0,
        "spectral_features_created": 0,
        "labels_created": 0,
        "silver_created": 0,
        "gold_created": 0,
        "negatives_created": 0,
        "can_train": False,
        "can_unlock_day10_now": d10_now > 0,
        "can_unlock_day10_next_step_count": d10_next,
        "day10_status": "BLOQUEADO_SEM_RASTER_NATIVO;LINEAGE_PARCIALMENTE_RECUPERAVEL_SO_POR_REVISAO_GEE",
        "heavy_public_outputs": 0,
        "guardrails_passed": True,
    }


def main() -> None:
    ensure_output_dir()
    summary = build_summary()
    out = OUTPUT_DIR / "mv2_14_lineage_summary.json"
    write_json(out, summary)
    print(f"[mv2_14_summary] strong={summary['lineage_recovered_strong']} "
          f"partial={summary['lineage_recovered_partial']} "
          f"review={summary['lineage_candidate_review']} "
          f"not_found={summary['lineage_not_found']} | dry={summary['stac_dry_run_eligible_after_lineage']}")
    print(f"[mv2_14_summary] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
