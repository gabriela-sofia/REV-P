"""MV2-DATA-02 Tarefa 12 — Orquestrador do harness de API/raster.

Executa a cadeia completa e valida guardrails ao final. Por padrão termina com GEE/HTTP/
downloads/crops/raster em 0 e Dia 10 bloqueado — a menos que config/env habilitem
explicitamente o modo canary. NÃO faz git add/commit.
"""

from __future__ import annotations

import sys

import mv2_data_02_api_config as config
import mv2_data_02_build_api_lineage_resolution as lineage
import mv2_data_02_build_api_provider_registry as providers
import mv2_data_02_build_api_raster_risk_matrix as risk
import mv2_data_02_build_raster_canary_plan as canary_plan
import mv2_data_02_build_summary as summary
import mv2_data_02_cdse_odata_client as odata
import mv2_data_02_cdse_stac_client as stac
import mv2_data_02_execute_raster_canary as canary_exec
import mv2_data_02_gee_metadata_client as gee
import mv2_data_02_validate_api_raster_harness as validator
import mv2_data_02_validate_private_raster_intake as private_val
from mv2_data_02_common import ensure_public_dir

STEPS = [
    ("write_example_config", config.write_example_config),
    ("build_provider_registry", providers.main),
    ("cdse_stac_metadata_client", stac.main),
    ("cdse_odata_metadata_client", odata.main),
    ("gee_metadata_client", gee.main),
    ("build_api_lineage_resolution", lineage.main),
    ("build_raster_canary_plan", canary_plan.main),
    ("execute_raster_canary", canary_exec.main),
    ("validate_private_raster_intake", private_val.main),
    ("build_api_raster_risk_matrix", risk.main),
    ("build_summary", summary.main),
]


def main() -> int:
    ensure_public_dir()
    for name, fn in STEPS:
        print(f"--- MV2-DATA-02 step: {name} ---")
        fn()
    print("--- MV2-DATA-02 step: validate_api_raster_harness ---")
    rc = validator.main()
    if rc != 0:
        print("[mv2_data_02_run] FALHA: guardrails violados")
        return rc
    print("[mv2_data_02_run] cadeia MV2-DATA-02 concluida com guardrails preservados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
