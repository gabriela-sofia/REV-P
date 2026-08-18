from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_cronograma_engine"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mv2_15_run_cronograma_engine as runner  # noqa: E402
import mv2_15_validate_cronograma_engine as validator  # noqa: E402


def _read_csv(name: str) -> list[dict[str, str]]:
    with (OUTPUT_DIR / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _summary() -> dict[str, object]:
    return json.loads((OUTPUT_DIR / "mv2_15_cronograma_summary.json").read_text(encoding="utf-8"))


def setup_module() -> None:
    assert runner.main() == 0


def test_output_discovery_exists() -> None:
    rows = _read_csv("mv2_15_mv2_output_discovery.csv")
    assert rows
    assert {row["stage"] for row in rows} >= {
        "MV2-10_GAP_AUDIT",
        "MV2-11_REGIONAL_REBALANCING",
        "MV2-12_DATA_READINESS",
        "MV2-12_SPECTRAL_RECONSTRUCTION",
        "MV2-13_SENTINEL_BINDING",
        "MV2-14_GEE_LINEAGE_RECOVERY",
    }


def test_schedule_state_contains_days_8_to_22() -> None:
    rows = _read_csv("mv2_15_schedule_state.csv")
    assert {row["day_id"] for row in rows} == {str(day) for day in range(8, 23)}


def test_day10_blocked() -> None:
    by_day = {row["day_id"]: row for row in _read_csv("mv2_15_schedule_state.csv")}
    assert by_day["10"]["real_status"] == "BLOCKED_DATA"


def test_day18_blocked() -> None:
    by_day = {row["day_id"]: row for row in _read_csv("mv2_15_schedule_state.csv")}
    assert by_day["18"]["real_status"] == "BLOCKED_EVIDENCE"


def test_day19_blocked() -> None:
    by_day = {row["day_id"]: row for row in _read_csv("mv2_15_schedule_state.csv")}
    assert by_day["19"]["real_status"] == "BLOCKED_EVIDENCE"


def test_day21_blocked() -> None:
    by_day = {row["day_id"]: row for row in _read_csv("mv2_15_schedule_state.csv")}
    assert by_day["21"]["real_status"] == "BLOCKED_METHOD"


def test_day22_blocked() -> None:
    by_day = {row["day_id"]: row for row in _read_csv("mv2_15_schedule_state.csv")}
    assert by_day["22"]["real_status"] == "BLOCKED_GUARDRAIL"


def test_forbidden_actions_not_allowed() -> None:
    rows = _read_csv("mv2_15_allowed_programming_actions.csv")
    prohibited = {
        "STAC_REAL",
        "RASTER_DOWNLOAD",
        "PRIVATE_CROP",
        "SPECTRAL_FEATURE_EXTRACTION",
        "SUPERVISED_BASELINE",
        "TRAINING",
        "SILVER_PROMOTION",
        "NEGATIVE_CREATION",
        "SPLIT_TRAINABLE",
        "SANDBOX_SUPERVISED",
    }
    for row in rows:
        if row["action_category"] in prohibited:
            assert row["allowed_now"] == "false", row["action_category"]


def test_allowed_actions_include_heavy_engineering_without_external_data() -> None:
    rows = _read_csv("mv2_15_allowed_programming_actions.csv")
    allowed = {row["action_category"]: row for row in rows if row["allowed_now"] == "true"}
    for category in ["CLI_ORCHESTRATION", "TEST_HARDENING", "MANIFEST_VALIDATION", "QUALITY_DASHBOARD"]:
        assert category in allowed
        assert allowed[category]["depends_on_external_data"] == "false"


def test_gate_engine_contains_required_gates() -> None:
    gates = {row["gate_name"] for row in _read_csv("mv2_15_gate_engine.csv")}
    for gate in [
        "GATE_NO_LABELS_CREATED",
        "GATE_NO_SILVER_FORMAL",
        "GATE_NO_GOLD",
        "GATE_NO_NEGATIVES_FORMAL",
        "GATE_NO_TRAINING",
        "GATE_NO_SANDBOX",
        "GATE_NO_STAC_REAL",
        "GATE_NO_DOWNLOAD",
        "GATE_NO_CROP",
        "GATE_NO_NATIVE_RASTER",
        "GATE_NO_SPECTRAL_FEATURES",
        "GATE_UNKNOWN_NOT_NEGATIVE",
        "GATE_CITY_NOT_LABEL",
        "GATE_PNG_NOT_RASTER",
        "GATE_DINOV2_REVIEW_ONLY",
        "GATE_DAY10_BLOCKED",
        "GATE_DAY18_BLOCKED",
        "GATE_DAY19_BLOCKED",
        "GATE_DAY21_BLOCKED",
        "GATE_DAY22_BLOCKED",
    ]:
        assert gate in gates


def test_artifact_registry_exists() -> None:
    rows = _read_csv("mv2_15_artifact_registry.csv")
    assert rows
    assert any(row["stage"] == "MV2-15_CRONOGRAMA_ENGINE" for row in rows)


def test_worktree_integration_no_private_absolute_paths() -> None:
    text = (OUTPUT_DIR / "mv2_15_worktree_integration_matrix.csv").read_text(encoding="utf-8")
    assert "C:/Users" not in text
    assert "C:\\Users" not in text
    assert "\\\\?\\C:" not in text


def test_risk_summary_contains_required_risks() -> None:
    risks = {row["risk_name"] for row in _read_csv("mv2_15_scientific_risk_summary.csv")}
    for risk in [
        "claim operacional indevido",
        "Dia 10 falso desbloqueio",
        "PNG como raster",
        "DINOv2 como evidencia operacional",
        "cidade como label",
        "unknown como negativo",
        "silver IA como silver formal",
        "split treinavel sem label",
        "sandbox sem silver",
        "STAC sem binding",
        "crop sem CRS",
        "lineage sem patch",
        "cena T23KPR auto-vinculada",
        "relatorio publico exagerado",
    ]:
        assert risk in risks


def test_summary_json_consistent() -> None:
    summary = _summary()
    actions = _read_csv("mv2_15_allowed_programming_actions.csv")
    backlog = _read_csv("mv2_15_engineering_backlog.csv")
    assert summary["allowed_programming_actions_count"] == sum(1 for row in actions if row["allowed_now"] == "true")
    assert summary["blocked_actions_count"] == sum(1 for row in actions if row["allowed_now"] == "false")
    assert summary["engineering_backlog_count"] == len(backlog)


def test_outputs_public_light() -> None:
    heavy = [path for path in OUTPUT_DIR.rglob("*") if path.is_file() and path.stat().st_size > 5_000_000]
    assert heavy == []


def test_can_train_false() -> None:
    assert _summary()["can_train"] is False


def test_can_stac_real_false() -> None:
    assert _summary()["can_stac_real"] is False


def test_can_create_sandbox_false() -> None:
    assert _summary()["can_create_sandbox"] is False


def test_backlog_has_at_least_10_implementable_items() -> None:
    rows = _read_csv("mv2_15_engineering_backlog.csv")
    assert sum(1 for row in rows if row["can_implement_now"] == "true") >= 10


def test_validator_passes() -> None:
    assert validator.main() == 0
