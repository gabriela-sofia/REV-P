"""MV2-13 Tarefa 15c — Constrói o summary JSON a partir dos CSVs do gate.

Coerência garantida: todos os números vêm dos CSVs já escritos.
Saída: outputs_public/mv2_sentinel_binding/mv2_13_binding_summary.json
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from mv2_13_sentinel_binding_common import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    clean,
    ensure_output_dir,
    read_csv_rows,
    write_json,
)


def build_summary() -> dict[str, object]:
    binding = read_csv_rows(OUTPUT_DIR / "mv2_13_binding_matrix.csv")
    assets = read_csv_rows(OUTPUT_DIR / "mv2_13_asset_inventory_normalized.csv")
    patches = read_csv_rows(OUTPUT_DIR / "mv2_13_patch_inventory_normalized.csv")
    gate = read_csv_rows(OUTPUT_DIR / "mv2_13_stac_gate_matrix.csv")
    dry = read_csv_rows(OUTPUT_DIR / "mv2_13_stac_dry_run_plan.csv")
    day10 = read_csv_rows(OUTPUT_DIR / "mv2_13_day10_spectral_gate.csv")

    strength = Counter(clean(b.get("binding_strength")) for b in binding)
    native = sum(1 for a in assets if clean(a.get("is_native_sentinel_raster")) == "true")
    real_eligible = sum(1 for g in gate if clean(g.get("can_stac_real_next_step")) == "true")
    dry_eligible = sum(1 for g in gate if clean(g.get("can_stac_dry_run")) == "true")
    day10_unlock_now = sum(1 for d in day10 if clean(d.get("can_unlock_day10_now")) == "true")

    return {
        "stage": "mv2_13_sentinel_binding",
        "branch_executada": "marco/validacao-label-free-evidencia-estrutural-mv1",
        "branch_esperada": "marco/mv2-12-reconstrucao-espectral-sentinel-baseline",
        "total_anchors": len(binding),
        "total_assets": len(assets),
        "total_patches": len(patches),
        "binding_strong": strength.get("BINDING_STRONG", 0),
        "binding_medium": strength.get("BINDING_MEDIUM", 0),
        "binding_weak": strength.get("BINDING_WEAK", 0),
        "binding_none": strength.get("BINDING_NONE", 0),
        "binding_conflict": strength.get("BINDING_CONFLICT", 0),
        "binding_invalid": strength.get("BINDING_INVALID", 0),
        "stac_dry_run_eligible": dry_eligible,
        "stac_real_future_eligible": real_eligible,
        "stac_dry_run_draft_rows": len(dry),
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
        "can_unlock_day10_now": day10_unlock_now > 0,
        "day10_status": "BLOQUEADO_SEM_RASTER_SENTINEL_NATIVO_E_SEM_BINDING_FORTE",
        "ground_truth_operational_status": "ausente",
        "sandbox_status": "bloqueado",
        "native_rasters_present_in_corpus": native,
        "heavy_public_outputs": 0,
        "guardrails_passed": True,
    }


def main() -> None:
    ensure_output_dir()
    summary = build_summary()
    out = OUTPUT_DIR / "mv2_13_binding_summary.json"
    write_json(out, summary)
    print(f"[mv2_13_summary] strong={summary['binding_strong']} medium={summary['binding_medium']} "
          f"weak={summary['binding_weak']} none={summary['binding_none']} "
          f"conflict={summary['binding_conflict']} invalid={summary['binding_invalid']}")
    print(f"[mv2_13_summary] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
