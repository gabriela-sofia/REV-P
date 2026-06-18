from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ground_truth"))

from revp_v2ez_to_v2ff_comum import read_csv, run_integrated, table


def ensure_outputs() -> None:
    if not table(ROOT, "revp_painel_perda_recuperacao_base_original_v2ff.csv").exists():
        run_integrated(ROOT, True)


ensure_outputs()
ROW = read_csv(table(ROOT, "revp_registro_decisao_recuperacao_forense_v2fe.csv"))[0]


def test_v2fe_decision_row_exists() -> None:
    assert ROW["decision_id"] == "DECISION_v2fe_0001"


def test_v2fe_original_recovery_flag_is_boolean_text() -> None:
    assert ROW["original_53_recoverable"] in {"true", "false"}


def test_v2fe_ground_truth_remains_absent() -> None:
    assert ROW["ground_truth_operational_status"] == "ABSENT"


def test_v2fe_manual_action_when_original_not_recoverable() -> None:
    if ROW["original_53_recoverable"] == "false":
        assert ROW["manual_action_required"] == "true"


def test_v2fe_decision_status_is_forensic_not_operational() -> None:
    assert ROW["decision_status"] in {
        "ORIGINAL_BASE_FOUND_READY_FOR_CONTROLLED_RESTORE",
        "ORIGINAL_BASE_PARTIAL_ONLY",
        "ONLY_REFERENCES_FOUND",
        "ONLY_FALLBACK_AVAILABLE",
        "REQUIRES_MANUAL_BACKUP_SEARCH",
        "ORIGINAL_BASE_NOT_FOUND",
    }

