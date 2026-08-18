"""MV2 DATA-06/07/08 local promotion orchestrator.

Runs local input intake, promotes only metadata readiness statuses when local
evidence is valid, evaluates DATA-08 preflight flags, and recomputes MV2-16
readiness. It writes public summaries only and does not call providers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

import sys

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mv2_data_06_08_local_input_intake as intake

OUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_local_promotion"
EXEC_REPORTS = PROJECT_ROOT / "outputs_public" / "execution_reports"
DEFAULT_TIMESTAMP = "20260623T213111"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def data06_public_status(local_status: str) -> str:
    if local_status == "PROMOTED_METADATA_READY":
        return "PROMOTED_METADATA_READY"
    if local_status == "BLOCKED_INVALID_TEMPLATE":
        return "BLOCKED_INVALID_TEMPLATE"
    return "BLOCKED_NO_FILLED_TEMPLATE"


def data07_public_status(local_status: str) -> str:
    if local_status == "SENTINEL_2_ELIGIBLE_FOUND":
        return "SENTINEL_2_ELIGIBLE_FOUND"
    if local_status == "BLOCKED_INVALID_SENSOR_LINEAGE":
        return "BLOCKED_INVALID_SENSOR_LINEAGE"
    return "UNKNOWN_BLOCKED"


def data08_public_status(local_status: str) -> str:
    if local_status == "READY_METADATA_ONLY_PREFLIGHT":
        return "READY_METADATA_ONLY_PREFLIGHT"
    if local_status == "BLOCKED_BY_FLAGS":
        return "BLOCKED_BY_FLAGS"
    return "BLOCKED_NO_CONFIG"


def compute_mv2_16_status(d06_status: str, d07_status: str, d08_status: str) -> str:
    if (
        d06_status == "PROMOTED_METADATA_READY"
        and d07_status == "SENTINEL_2_ELIGIBLE_FOUND"
        and d08_status == "READY_METADATA_ONLY_PREFLIGHT"
    ):
        return "READY_FOR_MV2_16_METADATA_ONLY"
    return "READY_FOR_MV2_16_DRY_RUN"


def live_metadata_allowed(
    *,
    allow_live_metadata: bool,
    d06_status: str,
    d07_status: str,
    d08_status: str,
    safe_flags: dict[str, bool],
) -> bool:
    return (
        allow_live_metadata
        and d06_status == "PROMOTED_METADATA_READY"
        and d07_status == "SENTINEL_2_ELIGIBLE_FOUND"
        and d08_status == "READY_METADATA_ONLY_PREFLIGHT"
        and safe_flags.get("allow_network") is True
        and safe_flags.get("allow_metadata_calls") is True
        and safe_flags.get("allow_raster_download") is False
        and safe_flags.get("allow_canary_download") is False
    )


def next_human_inputs(d06_status: str, d07_status: str, d08_status: str) -> list[str]:
    inputs: list[str] = []
    if d06_status != "PROMOTED_METADATA_READY":
        inputs.append("DATA-06: preencher template temporal local com fonte rastreavel.")
    if d07_status != "SENTINEL_2_ELIGIBLE_FOUND":
        inputs.append("DATA-07: preencher lineage sensorial local com sensor_source_ref.")
    if d08_status != "READY_METADATA_ONLY_PREFLIGHT":
        inputs.append("DATA-08: criar config local metadata-only com flags seguras.")
    return inputs


def build_local_promotion(
    readiness: dict[str, Any],
    *,
    allow_live_metadata: bool = False,
) -> dict[str, Any]:
    d06_status = data06_public_status(readiness["data_06_status"])
    d07_status = data07_public_status(readiness["data_07_status"])
    d08_status = data08_public_status(readiness["data_08_status"])
    mv2_16_status = compute_mv2_16_status(d06_status, d07_status, d08_status)
    live_allowed = live_metadata_allowed(
        allow_live_metadata=allow_live_metadata,
        d06_status=d06_status,
        d07_status=d07_status,
        d08_status=d08_status,
        safe_flags=readiness["data_08"]["safe_flags"],
    )
    return {
        "stage": "DATA-06/07/08 local promotion",
        "branch": "dados/promocao-local-metadados-data-06-08",
        "worktree": "REV-P-promocao-local-metadados-data-06-08",
        "fail_closed": True,
        "allow_live_metadata_requested": allow_live_metadata,
        "live_metadata_allowed_by_preflight": live_allowed,
        "local_inputs_found": readiness["local_input_counts"],
        "data_06_status": d06_status,
        "data_07_status": d07_status,
        "data_08_status": d08_status,
        "mv2_16_status": mv2_16_status,
        "day10_status": "BLOCKED",
        "live_calls": 0,
        "downloads": 0,
        "rasters": 0,
        "crops": 0,
        "ready_for_real_metadata_only": live_allowed,
        "next_human_inputs": next_human_inputs(d06_status, d07_status, d08_status),
    }


def write_local_promotion_outputs(summary: dict[str, Any], out_dir: Path = OUT_DIR) -> None:
    write_json(out_dir / "mv2_data_06_08_local_promotion_summary.json", summary)
    next_inputs = "\n".join("- " + item for item in summary["next_human_inputs"]) or "- nenhum bloqueador metadata-only local pendente"
    write_text(
        out_dir / "mv2_data_06_08_local_promotion_report.md",
        f"""# Promocao local DATA-06/07/08

## Estado consolidado
- DATA-06: {summary['data_06_status']}
- DATA-07: {summary['data_07_status']}
- DATA-08: {summary['data_08_status']}
- MV2-16: {summary['mv2_16_status']}
- Dia 10: {summary['day10_status']}

## Execucao
- --allow-live-metadata solicitado: {summary['allow_live_metadata_requested']}
- metadata-only real permitido pelo preflight: {summary['live_metadata_allowed_by_preflight']}
- chamadas/downloads/rasters/crops: 0/0/0/0

## Proximos inputs humanos
{next_inputs}
""",
    )
    write_text(
        out_dir / "commands.txt",
        "python scripts/mv2_data_06_08_local_promotion_orchestrator.py\n"
        "# somente apos autorizacao e flags seguras:\n"
        "python scripts/mv2_data_06_08_local_promotion_orchestrator.py --allow-live-metadata",
    )


def write_execution_reports(summary: dict[str, Any], stamp: str) -> None:
    next_inputs = "\n".join("- " + item for item in summary["next_human_inputs"]) or "- nenhum bloqueador metadata-only local pendente"
    write_json(EXEC_REPORTS / f"revp_data_06_08_local_promotion_summary_{stamp}.json", summary)
    write_text(
        EXEC_REPORTS / f"revp_data_06_08_local_promotion_report_{stamp}.md",
        f"""# Relatorio de promocao local DATA-06/07/08

## Branch / worktree
- branch: {summary['branch']}
- worktree: {summary['worktree']}
- base: dados/desbloqueio-metadados-data-06-08

## Inputs locais
- DATA-06 templates encontrados: {summary['local_inputs_found']['data_06_templates']}
- DATA-07 templates encontrados: {summary['local_inputs_found']['data_07_templates']}
- DATA-08 configs encontradas: {summary['local_inputs_found']['data_08_configs']}

## Status
- DATA-06: {summary['data_06_status']}
- DATA-07: {summary['data_07_status']}
- DATA-08: {summary['data_08_status']}
- MV2-16: {summary['mv2_16_status']}
- Dia 10: {summary['day10_status']}

## Chamadas / downloads / rasters / crops
- chamadas: {summary['live_calls']}
- downloads: {summary['downloads']}
- rasters: {summary['rasters']}
- crops: {summary['crops']}

## Pronto para metadata-only real?
- {summary['ready_for_real_metadata_only']}

## Proximos inputs humanos necessarios
{next_inputs}
""",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-live-metadata", action="store_true")
    parser.add_argument("--timestamp", default=DEFAULT_TIMESTAMP)
    args = parser.parse_args(argv)

    intake.ensure_local_input_dirs(PROJECT_ROOT)
    intake.write_schema(PROJECT_ROOT)
    intake.write_public_examples(intake.OUT_DIR)
    readiness = intake.classify_local_input_readiness(PROJECT_ROOT)
    intake.write_intake_manifest(readiness, intake.OUT_DIR)
    intake.write_intake_summary(readiness, intake.OUT_DIR)
    intake.write_intake_report(readiness, intake.OUT_DIR)
    write_text(intake.OUT_DIR / "commands.txt", "python scripts/mv2_data_06_08_local_input_intake.py")

    summary = build_local_promotion(readiness, allow_live_metadata=args.allow_live_metadata)
    write_local_promotion_outputs(summary, OUT_DIR)
    write_execution_reports(summary, args.timestamp)
    print(
        "[mv2_data_06_08_local_promotion] "
        f"DATA-06={summary['data_06_status']} "
        f"DATA-07={summary['data_07_status']} "
        f"DATA-08={summary['data_08_status']} "
        f"MV2-16={summary['mv2_16_status']} "
        f"Dia10={summary['day10_status']} "
        "calls/downloads/rasters/crops=0/0/0/0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
