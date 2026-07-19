"""MV2-DATA-05 Tarefa 10 — Orquestrador do intake de janela temporal.

Executa a cadeia completa e valida guardrails ao final. Sem template preenchido, termina com
0 promovidas e DATA-06 bloqueado, mas com gates e template de correção prontos. NÃO faz git
add/commit.
"""

from __future__ import annotations

import sys

import mv2_data_05_build_probe_ready_batch as probe_ready
import mv2_data_05_build_risk_matrix as risk
import mv2_data_05_build_summary as summary
import mv2_data_05_build_temporal_promotion_gate as gate
import mv2_data_05_create_temporal_window_correction_template as correction
import mv2_data_05_discover_temporal_inputs as discover
import mv2_data_05_normalize_temporal_windows as normalize
import mv2_data_05_validate_temporal_evidence as evidence
import mv2_data_05_validate_temporal_window_intake as validator
from mv2_data_05_common import ensure_public_dir

STEPS = [
    ("discover_temporal_inputs", discover.main),
    ("normalize_temporal_windows", normalize.main),
    ("validate_temporal_evidence", evidence.main),
    ("build_temporal_promotion_gate", gate.main),
    ("build_probe_ready_batch", probe_ready.main),
    ("create_temporal_window_correction_template", correction.main),
    ("build_risk_matrix", risk.main),
    ("build_summary", summary.main),
]


def main() -> int:
    ensure_public_dir()
    for name, fn in STEPS:
        print(f"--- MV2-DATA-05 step: {name} ---")
        fn()
    print("--- MV2-DATA-05 step: validate_temporal_window_intake ---")
    rc = validator.main()
    if rc != 0:
        print("[mv2_data_05_run] FALHA: guardrails violados")
        return rc
    print("[mv2_data_05_run] cadeia MV2-DATA-05 concluida com guardrails preservados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
