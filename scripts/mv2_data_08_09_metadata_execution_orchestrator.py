"""MV2 DATA-08/09 metadata-only execution orchestrator.

Wires the local intake, the mockable provider clients, the replay fixtures and
the lineage consensus engine into a single fail-closed run. The default is
``--strict --replay-only``: no network, no live call, no download, no raster, no
crop. A real metadata-only pass only happens with ``--allow-live-metadata`` AND a
local ``api_config.local.json`` whose flags allow network + metadata calls while
keeping raster/canary downloads disabled, AND targets with a valid temporal
window, Sentinel-2 lineage and a valid AOI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mv2_data_06_08_local_input_intake as intake
import mv2_data_06_08_metadata_only_readiness_orchestrator as readiness
import mv2_data_08_metadata_only_probe_runner as probe
import mv2_data_08_metadata_replay as replay
import mv2_data_08_lineage_consensus_engine as consensus
from mv2_data_08_metadata_clients import build_clients
from mv2_data_08_metadata_provider_contracts import (
    PROVIDERS,
    MetadataQueryTarget,
)

OUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_08_09_metadata_execution"
EXEC_REPORTS = PROJECT_ROOT / "outputs_public" / "execution_reports"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "api_config.local.json"
DEFAULT_TIMESTAMP = "20260623T213111"
BRANCH = "dados/motor-metadados-sentinel-data-08-09"
WORKTREE = "REV-P-motor-metadados-sentinel-data-08-09"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def load_config(config_path: Path) -> dict[str, Any] | None:
    if not config_path.exists():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def build_targets(max_targets: int) -> list[MetadataQueryTarget]:
    """Build query targets from eligible (promoted + Sentinel-2) rows only."""
    rows = probe.eligible_targets()
    targets: list[MetadataQueryTarget] = []
    for row in rows[: max(0, max_targets)]:
        targets.append(
            MetadataQueryTarget(
                patch_id=row.get("patch_id", ""),
                asset_id=row.get("asset_id", ""),
                sensor_family=row.get("sensor_family", ""),
                temporal_window_start=row.get("temporal_window_start", ""),
                temporal_window_end=row.get("temporal_window_end", ""),
                aoi_wgs84=None,  # AOI only via DATA-07 geometry inputs; absent by default
                mgrs_tile=row.get("mgrs_tile", ""),
                collection=row.get("collection", ""),
                source_ref=row.get("source_ref", ""),
            )
        )
    return targets


def resolve_providers(selection: str) -> list[str]:
    if selection == "ALL":
        return list(PROVIDERS)
    return [selection] if selection in PROVIDERS else []


def run_metadata_execution(
    targets: list[MetadataQueryTarget],
    *,
    providers: list[str],
    live: bool,
    replay_mode: bool,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    clients = build_clients(providers, live=live, replay_mode=replay_mode, config=config)
    results_by_target: dict[tuple[str, str], list[Any]] = {}
    for target in targets:
        results: list[Any] = []
        for client in clients.values():
            results.extend(client.query(target))
        results_by_target[(target.patch_id, target.asset_id)] = results
    live_calls = sum(getattr(client, "call_count", 0) for client in clients.values())
    records = consensus.consensus_for_targets(results_by_target)
    return {
        "results_by_target": results_by_target,
        "records": records,
        "live_calls": live_calls,
        "providers_configured": providers,
    }


def metadata_execution_status(targets: list[MetadataQueryTarget], live: bool, replay_mode: bool, live_calls: int) -> str:
    if not targets:
        return "NO_CALL"
    if live and live_calls > 0:
        return "METADATA_ONLY_EXECUTED"
    if replay_mode:
        return "REPLAY_ONLY"
    return "NO_CALL"


def lineage_consensus_status(records: list[Any]) -> str:
    if not records:
        return "NO_CALL"
    statuses = {record.consensus_status for record in records}
    if statuses == {"NO_CALL"}:
        return "NO_CALL"
    for priority in ("CONFLICT", "STRONG", "MEDIUM_REVIEW", "WEAK_BLOCKED", "NO_MATCH"):
        if priority in statuses:
            return priority
    return "NO_CALL"


def build_summary(
    args: argparse.Namespace,
    targets: list[MetadataQueryTarget],
    execution: dict[str, Any],
    d06: dict[str, Any],
    d07: dict[str, Any],
    d08: dict[str, Any],
    mv216: dict[str, Any],
) -> dict[str, Any]:
    live = args.allow_live_metadata
    replay_mode = not live
    records = execution["records"]
    return {
        "stage": "DATA-08/09 metadata-only execution",
        "branch": BRANCH,
        "worktree": WORKTREE,
        "mode": "live-metadata-only" if live else "replay-only",
        "strict": True,
        "fail_closed": True,
        "providers_configured": execution["providers_configured"],
        "providers_blocked": [] if live else list(execution["providers_configured"]),
        "eligible_targets": len(targets),
        "data_06_status": d06["status"],
        "data_07_status": d07["status"],
        "data_08_status": d08["status"],
        "metadata_execution_status": metadata_execution_status(targets, live, replay_mode, execution["live_calls"]),
        "lineage_consensus_status": lineage_consensus_status(records),
        "mv2_16_readiness": mv216["readiness"],
        "day10_status": mv216["day10_status"],
        "live_calls": execution["live_calls"],
        "downloads": 0,
        "rasters": 0,
        "crops": 0,
        "consensus_summary": consensus.summarize(records),
    }


def write_execution_report(stamp: str, summary: dict[str, Any]) -> None:
    json_out = EXEC_REPORTS / f"revp_data_08_09_metadata_engine_summary_{stamp}.json"
    write_json(json_out, summary)
    md_out = EXEC_REPORTS / f"revp_data_08_09_metadata_engine_report_{stamp}.md"
    write_text(
        md_out,
        f"""# Relatorio de execucao - motor metadata-only Sentinel DATA-08/09

## 1. Branch / worktree
- branch: {summary['branch']}
- worktree: {summary['worktree']}
- modo: {summary['mode']} (strict={summary['strict']})

## 2. Targets elegiveis
- elegiveis (promovido + Sentinel-2): {summary['eligible_targets']}

## 3. Motivos de bloqueio
- DATA-06: {summary['data_06_status']}
- DATA-07: {summary['data_07_status']}
- DATA-08: {summary['data_08_status']}

## 4. Providers configurados
- {', '.join(summary['providers_configured']) or 'nenhum'}

## 5. Providers bloqueados
- {', '.join(summary['providers_blocked']) or 'nenhum'}

## 6. Chamadas / downloads / rasters / crops
- live_calls: {summary['live_calls']}
- downloads: {summary['downloads']}
- rasters: {summary['rasters']}
- crops: {summary['crops']}

## 7. Consenso de lineage
- status: {summary['lineage_consensus_status']}
- contagens: {json.dumps(summary['consensus_summary']['counts'], ensure_ascii=True)}
- lineage confirmado (STRONG): {summary['consensus_summary']['confirmed_lineage']}

## 8. MV2-16
- readiness: {summary['mv2_16_readiness']}

## 9. Dia 10
- {summary['day10_status']}

## 10. Execucao de metadados
- {summary['metadata_execution_status']}

## 11. Proximos inputs humanos
- DATA-06: preencher janela temporal com fonte rastreavel e rodar a promocao.
- DATA-07: confirmar lineage Sentinel-2 elegivel com referencia de fonte.
- DATA-08: criar configs/api_config.local.json local (nao versionado) habilitando
  apenas allow_network + allow_metadata_calls; exportar variaveis de ambiente do
  provedor (sem versionar segredo).
- Somente com os tres acima o motor sai de replay-only para metadata-only real.
""",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Motor metadata-only Sentinel DATA-08/09")
    parser.add_argument("--strict", action="store_true", help="reforça política fail-closed (sempre ativo)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--allow-live-metadata", action="store_true")
    parser.add_argument(
        "--provider",
        default="ALL",
        choices=["GEE", "CDSE_STAC", "CDSE_ODATA", "TRACEABILITY", "ALL"],
    )
    parser.add_argument("--max-targets", type=int, default=10)
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--timestamp", default=DEFAULT_TIMESTAMP)
    args = parser.parse_args(argv)

    live = args.allow_live_metadata
    replay_mode = not live
    config = load_config(Path(args.config)) if live else None

    # Always publish fixtures + schemas so the engine is reproducible offline.
    replay.write_fixtures()
    replay.write_readme()
    consensus.write_schema()

    # 1. Intake + canonical status recompute (no overwrite of committed outputs).
    intake.classify_local_input_readiness()
    d06 = readiness.evaluate_data06()
    d07 = readiness.evaluate_data07()
    d08 = readiness.evaluate_data08()
    mv216 = readiness.evaluate_mv2_16()

    # 2-7. Build targets, run clients (replay/no-call by default), consensus.
    providers = resolve_providers(args.provider)
    targets = build_targets(args.max_targets)
    execution = run_metadata_execution(
        targets,
        providers=providers,
        live=live,
        replay_mode=replay_mode,
        config=config,
    )
    consensus.write_outputs(execution["records"])

    summary = build_summary(args, targets, execution, d06, d07, d08, mv216)
    write_json(OUT_DIR / "mv2_data_08_09_metadata_execution_summary.json", summary)
    write_text(
        OUT_DIR / "mv2_data_08_09_metadata_execution_report.md",
        f"""# Execucao metadata-only DATA-08/09

- modo: {summary['mode']} (strict)
- DATA-06: {summary['data_06_status']}
- DATA-07: {summary['data_07_status']}
- DATA-08: {summary['data_08_status']}
- metadata_execution: {summary['metadata_execution_status']}
- lineage_consensus: {summary['lineage_consensus_status']}
- MV2-16: {summary['mv2_16_readiness']}
- Dia 10: {summary['day10_status']}
- chamadas/downloads/rasters/crops: {summary['live_calls']}/0/0/0
""",
    )
    write_text(
        OUT_DIR / "commands.txt",
        "python scripts/mv2_data_08_09_metadata_execution_orchestrator.py --strict --replay-only",
    )
    write_execution_report(args.timestamp, summary)

    print(
        "[mv2_data_08_09_metadata_execution_orchestrator] "
        f"DATA-06={d06['status']} DATA-07={d07['status']} DATA-08={d08['status']} "
        f"metadata={summary['metadata_execution_status']} "
        f"consensus={summary['lineage_consensus_status']} "
        f"MV2-16={mv216['readiness']} Dia10={mv216['day10_status']} "
        f"calls/downloads/rasters/crops={summary['live_calls']}/0/0/0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
