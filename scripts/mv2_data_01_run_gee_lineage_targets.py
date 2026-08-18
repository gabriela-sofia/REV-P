"""MV2-DATA-01 Tarefa 9 — Orquestrador do GEE Lineage Target Pack.

Executa a cadeia completa e valida guardrails ao final. NÃO baixa raster, NÃO executa
GEE/STAC/OData real, NÃO cria crop, NÃO faz git add/commit.
"""

from __future__ import annotations

import sys

import mv2_data_01_build_gee_metadata_query_plan as gee_plan
import mv2_data_01_build_lineage_targets as targets
import mv2_data_01_build_manual_recovery_checklist as checklist
import mv2_data_01_build_risk_matrix as risk
import mv2_data_01_build_stac_odata_metadata_plan as stac_plan
import mv2_data_01_build_summary as summary
import mv2_data_01_create_manual_gee_template as template
import mv2_data_01_validate_lineage_targets as validator
from mv2_data_01_common import ensure_output_dir

STEPS = [
    ("build_lineage_targets", targets.main),
    ("create_manual_gee_template", template.main),
    ("build_gee_metadata_query_plan", gee_plan.main),
    ("build_stac_odata_metadata_plan", stac_plan.main),
    ("build_manual_recovery_checklist", checklist.main),
    ("build_risk_matrix", risk.main),
    ("build_summary", summary.main),
]


def main() -> int:
    ensure_output_dir()
    for name, fn in STEPS:
        print(f"--- MV2-DATA-01 step: {name} ---")
        fn()
    print("--- MV2-DATA-01 step: validate_lineage_targets ---")
    rc = validator.main()
    if rc != 0:
        print("[mv2_data_01_run] FALHA: guardrails violados")
        return rc
    print("[mv2_data_01_run] cadeia MV2-DATA-01 concluida com guardrails preservados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
