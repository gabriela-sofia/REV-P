"""MV2 DATA-06/07/08/09 real local input discovery.

Looks for *real* human-filled local inputs (temporal windows, sensor lineage and
the metadata-only API config) without ever copying their raw content. Only
redacted public artifacts are written: statuses, counts, sha256 hashes, redacted
file names and validation errors. Secrets, tokens, ``.env`` and private paths are
never printed; ``api_config.local.json`` is validated by flags only.
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

OUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_06_09_real_input_discovery"

# Discovery-level statuses (distinct from the canonical promotion statuses).
DATA06_STATUSES = ["DATA_06_REAL_INPUT_FOUND", "DATA_06_NO_REAL_INPUT", "DATA_06_INVALID", "DATA_06_PROMOTABLE"]
DATA07_STATUSES = ["DATA_07_REAL_INPUT_FOUND", "DATA_07_NO_REAL_INPUT", "DATA_07_INVALID", "DATA_07_S2_ELIGIBLE"]
DATA08_STATUSES = ["DATA_08_CONFIG_FOUND", "DATA_08_NO_CONFIG", "DATA_08_BLOCKED_BY_FLAGS", "DATA_08_READY_METADATA_ONLY"]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def discover_env_files(project_root: Path = PROJECT_ROOT) -> list[Path]:
    """Detect .env presence only (never its content)."""
    candidates = [project_root / ".env", project_root / ".env.local"]
    return [path for path in candidates if path.exists()]


def map_data06_status(intake_status: str) -> str:
    return {
        "NO_LOCAL_INPUT_FOUND": "DATA_06_NO_REAL_INPUT",
        "PROMOTED_METADATA_READY": "DATA_06_PROMOTABLE",
        "BLOCKED_INVALID_TEMPLATE": "DATA_06_INVALID",
    }.get(intake_status, "DATA_06_REAL_INPUT_FOUND")


def map_data07_status(intake_status: str) -> str:
    return {
        "NO_LOCAL_INPUT_FOUND": "DATA_07_NO_REAL_INPUT",
        "SENTINEL_2_ELIGIBLE_FOUND": "DATA_07_S2_ELIGIBLE",
        "BLOCKED_INVALID_SENSOR_LINEAGE": "DATA_07_INVALID",
    }.get(intake_status, "DATA_07_REAL_INPUT_FOUND")


def map_data08_status(intake_status: str) -> str:
    return {
        "BLOCKED_NO_CONFIG": "DATA_08_NO_CONFIG",
        "READY_METADATA_ONLY_PREFLIGHT": "DATA_08_READY_METADATA_ONLY",
        "BLOCKED_BY_FLAGS": "DATA_08_BLOCKED_BY_FLAGS",
    }.get(intake_status, "DATA_08_CONFIG_FOUND")


def classify_real_inputs(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    discovered = intake.discover_local_inputs(project_root)
    d06_path = discovered["data_06_templates"][0] if discovered["data_06_templates"] else None
    d07_path = discovered["data_07_templates"][0] if discovered["data_07_templates"] else None
    d08_path = discovered["data_08_configs"][0] if discovered["data_08_configs"] else None

    d06 = intake.validate_data_06_template(d06_path)
    d07 = intake.validate_data_07_template(d07_path)
    d08 = intake.validate_data_08_config_presence(d08_path)

    env_files = discover_env_files(project_root)
    manifest = {
        "data_06_templates": [intake._file_manifest(path, project_root) for path in discovered["data_06_templates"]],
        "data_07_templates": [intake._file_manifest(path, project_root) for path in discovered["data_07_templates"]],
        "data_08_configs": [intake._file_manifest(path, project_root) for path in discovered["data_08_configs"]],
        "env_files_present": [intake.redact_local_paths(path, project_root) for path in env_files],
    }

    any_input = bool(discovered["data_06_templates"] or discovered["data_07_templates"] or discovered["data_08_configs"])
    overall = "REAL_LOCAL_INPUT_FOUND" if any_input else "NO_REAL_LOCAL_INPUT_FOUND"

    return {
        "stage": "DATA-06/07/08/09 real input discovery",
        "fail_closed": True,
        "overall_status": overall,
        "data_06_status": map_data06_status(d06["status"]),
        "data_07_status": map_data07_status(d07["status"]),
        "data_08_status": map_data08_status(d08["status"]),
        "data_06_detail": d06,
        "data_07_detail": d07,
        "data_08_detail": d08,
        "local_input_counts": {key: len(value) for key, value in discovered.items()},
        "env_files_present": len(env_files),
        "public_manifest": manifest,
        "live_calls": 0,
        "downloads": 0,
        "rasters": 0,
        "crops": 0,
    }


def write_schema(project_root: Path = PROJECT_ROOT) -> None:
    write_json(
        project_root / "datasets" / "schemas" / "schema_mv2_data_06_09_real_input_discovery.json",
        {
            "schema_id": "schema_mv2_data_06_09_real_input_discovery",
            "public_outputs_only": ["status", "counts", "sha256", "redacted_paths", "validation_errors"],
            "forbidden_public_outputs": ["filled_rows", "credentials", "tokens", "secrets", "env_content", "api_config_contents", "private_paths"],
            "data_06_statuses": DATA06_STATUSES,
            "data_07_statuses": DATA07_STATUSES,
            "data_08_statuses": DATA08_STATUSES,
            "side_effects": {"live_calls": 0, "downloads": 0, "rasters": 0, "crops": 0},
        },
    )


def write_outputs(discovery: dict[str, Any], out_dir: Path = OUT_DIR) -> None:
    write_json(out_dir / "mv2_data_06_09_real_input_manifest.json", discovery["public_manifest"])
    write_json(
        out_dir / "mv2_data_06_09_real_input_summary.json",
        {key: value for key, value in discovery.items() if key != "public_manifest"},
    )
    write_text(
        out_dir / "mv2_data_06_09_real_input_report.md",
        f"""# Discovery de inputs reais locais DATA-06/07/08/09

## Estado geral
- {discovery['overall_status']}
- DATA-06: {discovery['data_06_status']}
- DATA-07: {discovery['data_07_status']}
- DATA-08: {discovery['data_08_status']}
- templates DATA-06 encontrados: {discovery['local_input_counts']['data_06_templates']}
- templates DATA-07 encontrados: {discovery['local_input_counts']['data_07_templates']}
- configs DATA-08 encontradas: {discovery['local_input_counts']['data_08_configs']}
- arquivos .env presentes: {discovery['env_files_present']}

## Garantias
- nenhum input local e copiado para outputs publicos;
- apenas hashes, contagens, nomes redigidos, status e erros sao publicados;
- segredos, tokens, .env e paths privados nunca sao impressos;
- api_config.local.json e validado por flags, nunca por conteudo;
- chamadas/downloads/rasters/crops: 0/0/0/0.
""",
    )
    write_text(out_dir / "commands.txt", "python scripts/mv2_data_06_09_real_input_discovery.py")


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser().parse_args(argv)
    intake.ensure_local_input_dirs(PROJECT_ROOT)
    write_schema(PROJECT_ROOT)
    discovery = classify_real_inputs(PROJECT_ROOT)
    write_outputs(discovery, OUT_DIR)
    print(
        "[mv2_data_06_09_real_input_discovery] "
        f"{discovery['overall_status']} "
        f"DATA-06={discovery['data_06_status']} "
        f"DATA-07={discovery['data_07_status']} "
        f"DATA-08={discovery['data_08_status']} "
        "calls/downloads/rasters/crops=0/0/0/0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
