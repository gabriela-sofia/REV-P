"""MV2-DATA-04 Tarefa 12 — Orquestrador do corpus metadata probe.

Executa a cadeia completa e valida guardrails ao final. Sem janela temporal/config, termina
em bloqueio limpo (0 probes executados, lineage bloqueado). NÃO faz git add/commit.
"""

from __future__ import annotations

import sys

import mv2_data_04_build_risk_matrix as risk
import mv2_data_04_build_summary as summary
import mv2_data_04_build_temporal_window_gap_matrix as gap
import mv2_data_04_cdse_odata_corpus_metadata_probe as odata
import mv2_data_04_cdse_stac_corpus_metadata_probe as stac
import mv2_data_04_config_env_preflight as preflight
import mv2_data_04_create_temporal_window_template as template
import mv2_data_04_gee_corpus_metadata_probe as gee
import mv2_data_04_resolve_corpus_api_lineage as lineage
import mv2_data_04_select_corpus_metadata_batch as batch
import mv2_data_04_validate_corpus_metadata_probe as validator
from mv2_data_04_common import ensure_public_dir

STEPS = [
    ("config_env_preflight", preflight.main),
    ("select_corpus_metadata_batch", batch.main),
    ("gee_corpus_metadata_probe", gee.main),
    ("cdse_stac_corpus_metadata_probe", stac.main),
    ("cdse_odata_corpus_metadata_probe", odata.main),
    ("resolve_corpus_api_lineage", lineage.main),
    ("build_temporal_window_gap_matrix", gap.main),
    ("create_temporal_window_template", template.main),
    ("build_risk_matrix", risk.main),
    ("build_summary", summary.main),
]


def main() -> int:
    ensure_public_dir()
    for name, fn in STEPS:
        print(f"--- MV2-DATA-04 step: {name} ---")
        fn()
    print("--- MV2-DATA-04 step: validate_corpus_metadata_probe ---")
    rc = validator.main()
    if rc != 0:
        print("[mv2_data_04_run] FALHA: guardrails violados")
        return rc
    print("[mv2_data_04_run] cadeia MV2-DATA-04 concluida com guardrails preservados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
