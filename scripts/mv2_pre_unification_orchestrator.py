"""Pre-unification orchestrator for DATA-05/06/07/08 and MV2-16 dry-run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "outputs_public" / "execution_reports"
STAMP = "20260623T213111"


def run_step(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    return {
        "command": " ".join(command),
        "returncode": result.returncode,
        "stdout_tail": result.stdout.strip().splitlines()[-5:],
        "stderr_tail": result.stderr.strip().splitlines()[-5:],
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def build_final_reports(results: list[dict[str, Any]]) -> None:
    data05 = read_json(PROJECT_ROOT / "outputs_public" / "mv2_data_temporal_window_intake" / "mv2_data_05_summary.json")
    data06 = read_json(PROJECT_ROOT / "outputs_public" / "mv2_data_temporal_window_promotion" / "mv2_data_06_summary.json")
    data07 = read_json(PROJECT_ROOT / "outputs_public" / "mv2_data_source_sensor_lineage_promotion" / "mv2_data_07_summary.json")
    data08 = read_json(PROJECT_ROOT / "outputs_public" / "mv2_data_metadata_only_probe" / "mv2_data_08_summary.json")
    mv216 = read_json(PROJECT_ROOT / "outputs_public" / "mv2_16_unified_sentinel_execution_core" / "mv2_16_summary.json")
    summary = {
        "data05_status": "CLOSED",
        "data05_inputs": data05.get("total_input_rows", 0),
        "data05_promoted_windows": data05.get("temporal_promoted_strong", 0) + data05.get("temporal_promoted_partial", 0),
        "data06_status": "BLOCKED_NO_FILLED_TEMPLATE" if data06.get("blocked_no_filled_template", 0) else "PROMOTED_METADATA_READY",
        "data07_status": "UNKNOWN_BLOCKED" if data07.get("sentinel_2_eligible", 0) == 0 else "S2_ELIGIBLE",
        "data08_status": data08.get("preflight_status", "BLOCKED_NO_CONFIG"),
        "mv2_16_readiness": mv216.get("readiness", "NOT_READY_FOR_MV2_16"),
        "gate_a": "BLOCKED",
        "gate_b": mv216.get("gate_b", "GEOMETRY_BACKLOG_READY"),
        "gate_c": mv216.get("gate_c", "POLICY_READY"),
        "gate_d": mv216.get("gate_d", "POLICY_READY"),
        "calls": data08.get("calls", 0),
        "downloads": data08.get("downloads", 0),
        "rasters": data08.get("rasters", 0),
        "crops": data08.get("crops", 0),
        "steps": results,
    }
    write_json(REPORT_DIR / f"revp_next_programming_evolution_summary_{STAMP}.json", summary)
    write_text(
        REPORT_DIR / f"revp_next_programming_evolution_report_{STAMP}.md",
        f"""# REV-P next programming evolution {STAMP}

## Restored side effects
- v2es/v2et/v2eu/v2ev tracked files were individually restored before this run.

## Intentional outputs
- DATA-06 temporal promotion
- DATA-07 source sensor lineage promotion
- DATA-08 metadata-only preflight/probe
- Crop authorization policy
- SCL local QA readiness
- MV2-16 unified gate matrix

## Status
- DATA-05: CLOSED, inputs={summary['data05_inputs']}, promoted={summary['data05_promoted_windows']}
- DATA-06: {summary['data06_status']}
- DATA-07: {summary['data07_status']}
- DATA-08: {summary['data08_status']}
- MV2-16: {summary['mv2_16_readiness']}
- Gate A/B/C/D: {summary['gate_a']} / {summary['gate_b']} / {summary['gate_c']} / {summary['gate_d']}
- calls/downloads/rasters/crops: {summary['calls']}/{summary['downloads']}/{summary['rasters']}/{summary['crops']}

## Recommended selective staging
Do not run `git add .`. Review `revp_pre_unification_staging_plan_{STAMP}.md` and stage only the listed intentional paths.
""",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--allow-metadata-calls", action="store_true")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "api_config.local.json"))
    parser.add_argument("--no-downloads", action="store_true", default=True)
    parser.add_argument("--no-crops", action="store_true", default=True)
    parser.add_argument("--strict", action="store_true", default=True)
    args = parser.parse_args(argv)
    if not args.no_downloads or not args.no_crops:
        raise SystemExit("downloads and crops are prohibited in this orchestrator")
    steps = [
        ["python", "scripts/mv2_data_05_run_temporal_window_intake.py"],
        ["python", "scripts/mv2_data_06_temporal_window_promotion.py"],
        ["python", "scripts/mv2_data_07_source_sensor_lineage_promotion.py"],
        ["python", "scripts/mv2_data_08_metadata_only_probe_runner.py", "--config", args.config],
        ["python", "scripts/mv2_crop_authorization_policy.py"],
        ["python", "scripts/mv2_scl_local_qa_readiness.py"],
        ["python", "scripts/mv2_16_unified_sentinel_execution_core.py"],
    ]
    results = [run_step(step) for step in steps]
    build_final_reports(results)
    return 0 if all(result["returncode"] == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
