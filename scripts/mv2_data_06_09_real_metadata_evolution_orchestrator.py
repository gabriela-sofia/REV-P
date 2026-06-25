"""MV2 DATA-06/07/08/09 real metadata evolution orchestrator.

Pursues real progress on the metadata-only front: it discovers real local inputs,
validates them, and either (a) emits operational acquisition queues when inputs
are missing, or (b) prepares metadata-only targets when all three gates exist
locally. The default is ``--strict --replay-only``: no network, no live call, no
download, no raster, no crop. A live metadata-only pass only happens with
``--allow-live-metadata`` AND valid DATA-06 (temporal window + source), DATA-07
(Sentinel-2 + sensor_source_ref) and DATA-08 (local config, network + metadata
calls on, raster/canary downloads off). Even live, nothing is downloaded.
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

import mv2_data_06_09_real_input_discovery as discovery
import mv2_data_06_09_real_acquisition_queue as acq_queue
import mv2_data_08_metadata_replay as replay
import mv2_data_08_lineage_consensus_engine as consensus
import mv2_data_08_metadata_only_probe_runner as probe
from mv2_data_08_metadata_clients import build_clients
from mv2_data_08_metadata_provider_contracts import PROVIDERS, MetadataQueryTarget

OUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_06_09_real_metadata_evolution"
EXEC_REPORTS = PROJECT_ROOT / "outputs_public" / "execution_reports"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "api_config.local.json"
DEFAULT_TIMESTAMP = "20260623T213111"
BRANCH = "dados/evolucao-real-metadados-data-06-09"
WORKTREE = "REV-P-evolucao-real-metadados-data-06-09"


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


def final_data06_status(disc_status: str) -> str:
    if disc_status == "DATA_06_PROMOTABLE":
        return "PROMOTED_METADATA_READY"
    return "BLOCKED_NO_REAL_TEMPORAL_WINDOW"


def final_data07_status(disc_status: str) -> str:
    if disc_status == "DATA_07_S2_ELIGIBLE":
        return "SENTINEL_2_ELIGIBLE_FOUND"
    return "BLOCKED_NO_REAL_SENSOR_LINEAGE"


def final_data08_status(disc_status: str) -> str:
    if disc_status == "DATA_08_READY_METADATA_ONLY":
        return "READY_METADATA_ONLY_PREFLIGHT"
    if disc_status == "DATA_08_BLOCKED_BY_FLAGS":
        return "BLOCKED_BY_FLAGS"
    return "BLOCKED_NO_CONFIG"


def all_gates_ready(d06_final: str, d07_final: str, d08_final: str) -> bool:
    return (
        d06_final == "PROMOTED_METADATA_READY"
        and d07_final == "SENTINEL_2_ELIGIBLE_FOUND"
        and d08_final == "READY_METADATA_ONLY_PREFLIGHT"
    )


def build_targets_from_real_inputs(max_targets: int) -> list[MetadataQueryTarget]:
    """Targets only from promoted + Sentinel-2 eligible rows (probe join)."""
    rows = probe.eligible_targets()
    geometry = acq_queue._geometry_lookup()
    targets: list[MetadataQueryTarget] = []
    for row in rows[: max(0, max_targets)]:
        patch_id = row.get("patch_id", "")
        bbox, _crs = acq_queue._bbox_crs_for(patch_id, geometry)
        aoi = None
        if bbox:
            try:
                aoi = [float(part) for part in bbox.split(",")]
            except ValueError:
                aoi = None
        targets.append(
            MetadataQueryTarget(
                patch_id=patch_id,
                asset_id=row.get("asset_id", ""),
                sensor_family=row.get("sensor_family", ""),
                temporal_window_start=row.get("temporal_window_start", ""),
                temporal_window_end=row.get("temporal_window_end", ""),
                aoi_wgs84=aoi,
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


def run_metadata(targets, providers, live, replay_mode, config) -> dict[str, Any]:
    clients = build_clients(providers, live=live, replay_mode=replay_mode, config=config)
    results_by_target: dict[tuple[str, str], list[Any]] = {}
    for target in targets:
        results: list[Any] = []
        for client in clients.values():
            results.extend(client.query(target))
        results_by_target[(target.patch_id, target.asset_id)] = results
    live_calls = sum(getattr(client, "call_count", 0) for client in clients.values())
    records = consensus.consensus_for_targets(results_by_target)
    return {"records": records, "live_calls": live_calls}


def metadata_execution_status(all_ready: bool, live: bool, live_calls: int) -> str:
    if not all_ready:
        return "NO_CALL"
    if live and live_calls > 0:
        return "METADATA_ONLY_DONE"
    return "READY_BUT_NOT_EXECUTED"


def lineage_consensus_status(records: list[Any], live: bool, live_calls: int) -> str:
    if not live or live_calls == 0 or not records:
        return "NO_CALL"
    statuses = {record.consensus_status for record in records}
    for priority in ("CONFLICT", "STRONG", "MEDIUM_REVIEW", "WEAK_BLOCKED", "NO_MATCH"):
        if priority in statuses:
            return priority
    return "NO_CALL"


def mv2_16_readiness(all_ready: bool) -> str:
    return "READY_FOR_MV2_16_METADATA_ONLY" if all_ready else "READY_FOR_MV2_16_DRY_RUN"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evolucao real de metadados DATA-06/09")
    parser.add_argument("--strict", action="store_true", help="reforça política fail-closed (sempre ativo)")
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--allow-live-metadata", action="store_true")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--max-targets", type=int, default=10)
    parser.add_argument(
        "--provider",
        default="ALL",
        choices=["GEE", "CDSE_STAC", "CDSE_ODATA", "TRACEABILITY", "ALL"],
    )
    parser.add_argument("--timestamp", default=DEFAULT_TIMESTAMP)
    args = parser.parse_args(argv)

    live = args.allow_live_metadata
    replay_mode = not live
    config = load_config(Path(args.config)) if live else None

    # Always publish fixtures + schemas (offline, reproducible).
    replay.write_fixtures()
    replay.write_readme()
    consensus.write_schema()

    # 1-4. Discover + validate real local inputs.
    disc = discovery.classify_real_inputs(PROJECT_ROOT)
    discovery.write_schema(PROJECT_ROOT)
    discovery.write_outputs(disc)

    d06_final = final_data06_status(disc["data_06_status"])
    d07_final = final_data07_status(disc["data_07_status"])
    d08_final = final_data08_status(disc["data_08_status"])
    all_ready = all_gates_ready(d06_final, d07_final, d08_final)

    # 5. Acquisition queues whenever any input is missing/invalid.
    queues_generated = not all_ready
    acq_summary: dict[str, Any] = {}
    if queues_generated:
        acq_queue.write_schema(PROJECT_ROOT)
        acq_summary = acq_queue.write_outputs(acq_queue.OUT_DIR, args.max_targets)

    # 6-10. Targets + metadata pass (replay/no-call by default) + consensus.
    targets = build_targets_from_real_inputs(args.max_targets) if all_ready else []
    providers = resolve_providers(args.provider)
    execution = run_metadata(targets, providers, live, replay_mode, config)
    consensus.write_outputs(execution["records"])

    meta_status = metadata_execution_status(all_ready, live, execution["live_calls"])
    cons_status = lineage_consensus_status(execution["records"], live, execution["live_calls"])
    mv216 = mv2_16_readiness(all_ready)
    day10 = "BLOCKED"  # always blocked until local-only raster + SCL QA exist

    summary = {
        "stage": "DATA-06/07/08/09 real metadata evolution",
        "branch": BRANCH,
        "worktree": WORKTREE,
        "mode": "live-metadata-only" if live else "replay-only",
        "strict": True,
        "fail_closed": True,
        "real_local_input_found": disc["overall_status"] == "REAL_LOCAL_INPUT_FOUND",
        "data_06_status": d06_final,
        "data_07_status": d07_final,
        "data_08_status": d08_final,
        "acquisition_queues_generated": queues_generated,
        "acquisition_queue_summary": acq_summary,
        "ready_targets": len(targets),
        "providers_configured": providers,
        "providers_blocked": [] if live and all_ready else list(providers),
        "metadata_execution_status": meta_status,
        "lineage_consensus_status": cons_status,
        "mv2_16_readiness": mv216,
        "day10_status": day10,
        "live_calls": execution["live_calls"],
        "downloads": 0,
        "rasters": 0,
        "crops": 0,
        "consensus_summary": consensus.summarize(execution["records"]),
    }

    write_json(OUT_DIR / "mv2_data_06_09_real_metadata_evolution_summary.json", summary)
    write_text(
        OUT_DIR / "mv2_data_06_09_real_metadata_evolution_report.md",
        _report_body(summary),
    )
    write_text(
        OUT_DIR / "commands.txt",
        "python scripts/mv2_data_06_09_real_metadata_evolution_orchestrator.py --strict --replay-only",
    )
    write_execution_report(args.timestamp, summary)

    print(
        "[mv2_data_06_09_real_metadata_evolution_orchestrator] "
        f"input_real={summary['real_local_input_found']} "
        f"DATA-06={d06_final} DATA-07={d07_final} DATA-08={d08_final} "
        f"filas={queues_generated} metadata={meta_status} consensus={cons_status} "
        f"MV2-16={mv216} Dia10={day10} "
        f"calls/downloads/rasters/crops={execution['live_calls']}/0/0/0"
    )
    return 0


def _report_body(summary: dict[str, Any]) -> str:
    return f"""# Evolucao real de metadados DATA-06/07/08/09

- modo: {summary['mode']} (strict)
- input local real encontrado: {summary['real_local_input_found']}
- DATA-06: {summary['data_06_status']}
- DATA-07: {summary['data_07_status']}
- DATA-08: {summary['data_08_status']}
- filas de aquisicao geradas: {summary['acquisition_queues_generated']}
- targets prontos para metadata-only: {summary['ready_targets']}
- metadata_execution: {summary['metadata_execution_status']}
- lineage_consensus: {summary['lineage_consensus_status']}
- MV2-16: {summary['mv2_16_readiness']}
- Dia 10: {summary['day10_status']}
- chamadas/downloads/rasters/crops: {summary['live_calls']}/0/0/0
"""


def write_execution_report(stamp: str, summary: dict[str, Any]) -> None:
    write_json(EXEC_REPORTS / f"revp_data_06_09_real_metadata_evolution_summary_{stamp}.json", summary)
    proxima_acao = (
        "Preencher inputs_local/data_06_temporal_windows/ e data_07_sensor_lineage/ com janela temporal "
        "e lineage Sentinel-2 reais (fonte rastreavel), criar configs/api_config.local.json local "
        "(network+metadata on, raster/canary off) e re-rodar com --allow-live-metadata."
        if not summary["real_local_input_found"]
        else "Validar inputs reais e, se gates verdes, re-rodar com --allow-live-metadata."
    )
    write_text(
        EXEC_REPORTS / f"revp_data_06_09_real_metadata_evolution_report_{stamp}.md",
        f"""# Relatorio de execucao - evolucao real de metadados DATA-06/07/08/09

## 1. Branch / worktree
- branch: {summary['branch']}
- worktree: {summary['worktree']}
- modo: {summary['mode']} (strict={summary['strict']})

## 2. Inputs reais encontrados
- input local real encontrado: {summary['real_local_input_found']}

## 3. Inputs faltantes
- DATA-06: {summary['data_06_status']}
- DATA-07: {summary['data_07_status']}
- DATA-08: {summary['data_08_status']}

## 4. Filas de aquisicao geradas
- geradas: {summary['acquisition_queues_generated']}
- outputs_public/mv2_data_06_09_real_acquisition_queue/

## 5. Targets prontos para metadata-only
- {summary['ready_targets']}

## 6. Providers disponiveis / bloqueados
- configurados: {', '.join(summary['providers_configured']) or 'nenhum'}
- bloqueados: {', '.join(summary['providers_blocked']) or 'nenhum'}

## 7. Metadata execution
- {summary['metadata_execution_status']}

## 8. Lineage consensus
- {summary['lineage_consensus_status']}

## 9. MV2-16
- {summary['mv2_16_readiness']}

## 10. Dia 10
- {summary['day10_status']} (bloqueado ate raster local-only + SCL QA existirem)

## 11. Chamadas / downloads / rasters / crops
- {summary['live_calls']} / {summary['downloads']} / {summary['rasters']} / {summary['crops']}

## 12. Proxima acao humana objetiva
- {proxima_acao}
""",
    )


if __name__ == "__main__":
    raise SystemExit(main())
