"""MV2-13 Tarefa 14 — Orquestrador do Sentinel Binding & Spectral Eligibility Gate.

Executa a cadeia completa em ordem e roda a validação de guardrails ao final.
Falha (exit != 0) se qualquer guardrail fail-closed for violado. NÃO executa STAC
real, NÃO baixa raster, NÃO cria crop, NÃO faz git add/commit.
"""

from __future__ import annotations

import sys

import mv2_13_build_binding_risk_matrix as risk
import mv2_13_build_day10_spectral_gate as day10
import mv2_13_build_manual_recovery_queue as recovery
import mv2_13_build_stac_gate as stac
import mv2_13_build_summaries as summaries
import mv2_13_discover_binding_inputs as discovery
import mv2_13_normalize_asset_inventory as assets
import mv2_13_normalize_patch_inventory as patches
import mv2_13_normalize_sentinel_anchors as anchors
import mv2_13_reconcile_anchor_asset_patch as reconcile
import mv2_13_resolve_geometry_aoi_crs as geometry
import mv2_13_resolve_temporal_scene_lineage as temporal
import mv2_13_validate_sentinel_binding_guardrails as validator
from mv2_13_sentinel_binding_common import ensure_output_dir

STEPS = [
    ("discovery", discovery.main),
    ("normalize_anchors", anchors.main),
    ("normalize_patches", patches.main),
    ("normalize_assets", assets.main),
    ("reconcile", reconcile.main),
    ("geometry_aoi_crs", geometry.main),
    ("temporal_scene_lineage", temporal.main),
    ("stac_gate", stac.main),
    ("day10_gate", day10.main),
    ("manual_recovery_queue", recovery.main),
    ("risk_matrix", risk.main),
    ("summaries", summaries.main),
]


def main() -> int:
    ensure_output_dir()
    for name, fn in STEPS:
        print(f"--- MV2-13 step: {name} ---")
        fn()
    print("--- MV2-13 step: validate_guardrails ---")
    rc = validator.main()
    if rc != 0:
        print("[mv2_13_run] FALHA: guardrails violados")
        return rc
    print("[mv2_13_run] cadeia MV2-13 concluida com guardrails preservados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
