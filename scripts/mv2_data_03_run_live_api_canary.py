"""MV2-DATA-03 Tarefa 13 — Orquestrador do live API/canary.

Executa a cadeia completa e valida guardrails ao final. Sem config local/env, termina
fail-closed (0 probes, 0 canary). NÃO faz git add/commit.
"""

from __future__ import annotations

import sys

import mv2_data_03_build_corpus_impact_matrix as impact
import mv2_data_03_build_live_api_canary_risk_matrix as risk
import mv2_data_03_build_summary as summary
import mv2_data_03_config_env_preflight as preflight
import mv2_data_03_execute_private_raster_canary as canary_exec
import mv2_data_03_live_cdse_odata_metadata_probe as odata
import mv2_data_03_live_cdse_stac_metadata_probe as stac
import mv2_data_03_live_gee_metadata_probe as gee
import mv2_data_03_resolve_canary_metadata_consensus as consensus
import mv2_data_03_select_official_canary_candidates as candidates
import mv2_data_03_validate_live_api_canary as validator
import mv2_data_03_validate_private_raster_canary as private_val
from mv2_data_03_common import ensure_public_dir

STEPS = [
    ("config_env_preflight", preflight.main),
    ("select_official_canary_candidates", candidates.main),
    ("live_gee_metadata_probe", gee.main),
    ("live_cdse_stac_metadata_probe", stac.main),
    ("live_cdse_odata_metadata_probe", odata.main),
    ("resolve_canary_metadata_consensus", consensus.main),
    ("execute_private_raster_canary", canary_exec.main),
    ("validate_private_raster_canary", private_val.main),
    ("build_corpus_impact_matrix", impact.main),
    ("build_live_api_canary_risk_matrix", risk.main),
    ("build_summary", summary.main),
]


def main() -> int:
    ensure_public_dir()
    for name, fn in STEPS:
        print(f"--- MV2-DATA-03 step: {name} ---")
        fn()
    print("--- MV2-DATA-03 step: validate_live_api_canary ---")
    rc = validator.main()
    if rc != 0:
        print("[mv2_data_03_run] FALHA: guardrails violados")
        return rc
    print("[mv2_data_03_run] cadeia MV2-DATA-03 concluida com guardrails preservados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
