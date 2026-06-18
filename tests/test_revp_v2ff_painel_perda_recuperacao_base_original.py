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
ROW = read_csv(table(ROOT, "revp_painel_perda_recuperacao_base_original_v2ff.csv"))[0]
ACTIONS = read_csv(table(ROOT, "revp_proximas_acoes_base_original_v2ff.csv"))


def test_v2ff_painel_exists() -> None:
    assert ROW["painel_id"] == "DASH_v2ff_0001"


def test_v2ff_ground_truth_absent() -> None:
    assert ROW["ground_truth_operational_status"] == "ABSENT"


def test_v2ff_continuity_is_blocked_or_pending_restore() -> None:
    assert ROW["continuity_status"] in {
        "CONTINUITY_BLOCKED_ORIGINAL_BASE_NOT_FOUND",
        "CONTINUITY_RECOVERABLE_PENDING_CONTROLLED_RESTORE",
    }


def test_v2ff_original_recoverable_flag_is_boolean_text() -> None:
    assert ROW["original_53_recoverable"] in {"true", "false"}


def test_v2ff_next_actions_are_non_empty() -> None:
    assert ACTIONS and ACTIONS[0]["recommended_next_action"]


