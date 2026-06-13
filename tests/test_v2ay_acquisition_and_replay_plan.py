"""v2ay - acquisition, query and controlled replay plan tests."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with open(ROOT / "datasets" / name, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_acquisition_targets_are_specific_and_point_to_v2ax_manual_files():
    targets = rows("v2ay_geometry_acquisition_targets.csv")
    assert len(targets) == 56
    assert sum(row["target_type"] == "patch_boundary" for row in targets) == 55
    assert sum(row["target_type"] == "event_observed_polygon" for row in targets) == 1
    assert all("manual_intake/recife_p1" in row["feeds_file"] for row in targets)
    assert all(row["validation_command"] == "python scripts/run_v2ax_recife_geometry_intake_pack.py"
               for row in targets)


def test_query_plan_never_claims_source_can_promote_alone():
    queries = rows("v2ay_external_source_query_plan.csv")
    assert queries
    assert all(row["can_promote_alone"] == "false" for row in queries)
    assert any("SGB/CPRM" in row["source_name"] for row in queries)


def test_replay_plan_orders_v2ax_v2aw_v2av_v2au_and_human_review():
    replay = rows("v2ay_pipeline_replay_plan.csv")
    assert len(replay) == 8
    stages = [row["stage"] for row in replay]
    assert stages.index("v2ax_validate") < stages.index("v2aw") < stages.index("v2av") < stages.index("v2au")
    assert stages[-2:] == ["human_review", "no_training"]
