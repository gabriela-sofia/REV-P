"""v2az - docs, public outputs and schema tests."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docs_describe_dry_run_replay_and_tp4_guardrail():
    paths = [
        ROOT / "docs" / "v2az_turning_point_replay_orchestrator.md",
        ROOT / "docs" / "v2az_minimal_real_geometry_handoff.md",
        ROOT / "docs" / "v2az_how_to_reach_tp4.md",
    ]
    combined = ""
    for path in paths:
        assert path.is_file()
        combined += path.read_text(encoding="utf-8")
    for term in ("dry_run", "replay", "CRS", "C4_CANDIDATE_REQUIRES_HUMAN_REVIEW", "label"):
        assert term.lower() in combined.lower()


def test_ten_v2az_schemas_are_valid_json():
    paths = sorted((ROOT / "datasets" / "schemas").glob("v2az_*.schema.json"))
    assert len(paths) == 10
    for path in paths:
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["required"] and schema["properties"]


def test_public_summary_matches_tp0_dry_run():
    summary = json.loads((ROOT / "outputs_public" / "execution_reports" /
                          "v2az_turning_point_replay_orchestrator_summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "dry_run"
    assert summary["turning_point_level"] == "TP0_DOCUMENTED_ABSENCE"
    assert summary["can_attempt_replay"] is False
