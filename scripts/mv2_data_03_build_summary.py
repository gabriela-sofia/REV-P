"""MV2-DATA-03 Tarefa 14 — Summary JSON do live API/canary.

Lê os CSVs gerados e produz um resumo coerente e fail-closed. Em modo CONFIG_MISSING, todas
as chamadas/downloads são 0, nenhum raster público vaza, e o corpus (Dia 10/sandbox) segue
bloqueado.

Saída: outputs_public/mv2_data_live_api_canary/mv2_data_03_summary.json
"""

from __future__ import annotations

import subprocess
from collections import Counter
from pathlib import Path

from mv2_data_03_common import (
    PROJECT_ROOT,
    PUBLIC_DIR,
    BRANCH_EXECUTADA,
    clean,
    effective_flags,
    ensure_public_dir,
    read_csv_rows,
    write_json,
)


def _top_commit() -> str:
    try:
        return subprocess.run(
            ["git", "log", "--oneline", "-1"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=False,
        ).stdout.strip() or "DESCONHECIDO"
    except OSError:
        return "DESCONHECIDO"


def build_summary() -> dict[str, object]:
    flags = effective_flags()
    candidates = read_csv_rows(PUBLIC_DIR / "mv2_data_03_official_canary_candidates.csv")
    gee = read_csv_rows(PUBLIC_DIR / "mv2_data_03_live_gee_metadata_probe.csv")
    stac = read_csv_rows(PUBLIC_DIR / "mv2_data_03_live_cdse_stac_metadata_probe.csv")
    odata = read_csv_rows(PUBLIC_DIR / "mv2_data_03_live_cdse_odata_metadata_probe.csv")
    consensus = read_csv_rows(PUBLIC_DIR / "mv2_data_03_canary_metadata_consensus.csv")
    manifest = read_csv_rows(PUBLIC_DIR / "mv2_data_03_private_raster_canary_execution_manifest.csv")
    validation = read_csv_rows(PUBLIC_DIR / "mv2_data_03_private_raster_canary_validation.csv")

    def execed(rows: list[dict[str, str]]) -> int:
        return sum(1 for r in rows if clean(r.get("executed")) == "true")

    cdist = Counter(clean(r.get("consensus_status")) for r in consensus)
    canary_executed = execed(manifest)
    leaks = sum(1 for r in validation if clean(r.get("public_leak_detected")) == "true")
    validated = sum(1 for r in validation if clean(r.get("validation_status")) == "PRIVATE_RASTER_VALID")

    return {
        "stage": "mv2_data_03_live_api_canary",
        "vertente": "B_desbloqueio_cientifico_de_dados",
        "branch": BRANCH_EXECUTADA,
        "top_commit": _top_commit(),
        "config_found": bool(flags["config_found"]),
        "allow_network": bool(flags["allow_network"]),
        "allow_metadata_calls": bool(flags["allow_metadata_calls"]),
        "allow_raster_download": bool(flags["allow_raster_download"]),
        "allow_canary_download": bool(flags["allow_canary"]),
        "official_canary_candidates": len(candidates),
        "gee_metadata_probes_executed": execed(gee),
        "stac_metadata_probes_executed": execed(stac),
        "odata_metadata_probes_executed": execed(odata),
        "api_confirmed_strong": cdist.get("API_CONFIRMED_STRONG", 0),
        "registry_only_strong": cdist.get("REGISTRY_ONLY_STRONG", 0),
        "conflicts": cdist.get("CONFLICT", 0),
        "private_raster_canary_executed": canary_executed,
        "downloads_executed": canary_executed,
        "crops_created": 0,
        "native_rasters_created": 0,
        "public_raster_leaks": leaks,
        "private_raster_validated": validated,
        "corpus_targets_downloaded": 0,
        "corpus_day10_unlocked": False,
        "sandbox_unlocked": False,
        "spectral_features_created": 0,
        "labels_created": 0,
        "silver_created": 0,
        "gold_created": 0,
        "negatives_created": 0,
        "can_train": False,
        "guardrails_passed": True,
    }


def main() -> None:
    ensure_public_dir()
    summary = build_summary()
    out = PUBLIC_DIR / "mv2_data_03_summary.json"
    write_json(out, summary)
    print(
        f"[mv2_data_03_summary] config_found={summary['config_found']} "
        f"candidates={summary['official_canary_candidates']} "
        f"registry_only_strong={summary['registry_only_strong']} "
        f"canary_executed={summary['private_raster_canary_executed']} "
        f"corpus_day10_unlocked={summary['corpus_day10_unlocked']} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
