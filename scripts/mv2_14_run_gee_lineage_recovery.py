"""MV2-14 Tarefa 14 — Orquestrador do GEE Scene Lineage Recovery.

Executa a cadeia completa e valida guardrails ao final. NÃO baixa raster, NÃO executa
STAC, NÃO cria crop, NÃO faz git add/commit.
"""

from __future__ import annotations

import sys

import mv2_14_build_day10_post_lineage_gate as day10
import mv2_14_build_gee_manual_recovery_queue as recovery
import mv2_14_build_lineage_risk_matrix as risk
import mv2_14_build_manual_lineage_template as template
import mv2_14_build_medium_binding_index as medium
import mv2_14_build_post_lineage_stac_gate as stac
import mv2_14_build_summaries as summaries
import mv2_14_discover_gee_lineage_sources as discovery
import mv2_14_extract_lineage_candidates as extract
import mv2_14_reconcile_lineage_to_bindings as reconcile
import mv2_14_review_strong_scene_anchors as scene_review
import mv2_14_validate_gee_lineage_guardrails as validator
from mv2_14_gee_lineage_common import ensure_output_dir

STEPS = [
    ("discover_sources", discovery.main),
    ("extract_candidates", extract.main),
    ("build_medium_binding_index", medium.main),
    ("reconcile_lineage_to_bindings", reconcile.main),
    ("review_strong_scene_anchors", scene_review.main),
    ("build_post_lineage_stac_gate", stac.main),
    ("build_day10_post_lineage_gate", day10.main),
    ("build_gee_manual_recovery_queue", recovery.main),
    ("create_manual_template", template.main),
    ("build_lineage_risk_matrix", risk.main),
    ("build_summaries", summaries.main),
]


def main() -> int:
    ensure_output_dir()
    for name, fn in STEPS:
        print(f"--- MV2-14 step: {name} ---")
        fn()
    print("--- MV2-14 step: validate_guardrails ---")
    rc = validator.main()
    if rc != 0:
        print("[mv2_14_run] FALHA: guardrails violados")
        return rc
    print("[mv2_14_run] cadeia MV2-14 concluida com guardrails preservados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
