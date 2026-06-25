"""MV2 DATA-06/07/08 local input intake.

Discovers human-filled local inputs under ``inputs_local/`` and writes only
redacted public manifests: statuses, counts, hashes, sizes, and non-sensitive
flags. Filled local inputs are never copied to ``outputs_public``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_INPUT_ROOT = PROJECT_ROOT / "inputs_local"
OUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_local_input_intake"

DATA06_REQUIRED = {
    "patch_id",
    "asset_id",
    "temporal_window_start",
    "temporal_window_end",
    "temporal_window_source",
    "source_ref",
}
DATA07_REQUIRED = {"patch_id", "asset_id", "sensor_family", "sensor_source_ref"}
SAFE_CONFIG_FLAGS = [
    "allow_network",
    "allow_metadata_calls",
    "allow_raster_download",
    "allow_canary_download",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_iso(value: str) -> date | None:
    try:
        return date.fromisoformat((value or "").strip())
    except ValueError:
        return None


def redact_local_paths(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(project_root.resolve())
    except ValueError:
        return f"<external_local>/{path.name}"
    parts = rel.parts
    if parts and parts[0] == "inputs_local":
        return "<inputs_local>/" + "/".join(parts[1:])
    if parts == ("configs", "api_config.local.json"):
        return "<configs>/api_config.local.json"
    return "<repo_local>/" + "/".join(parts)


def ensure_local_input_dirs(project_root: Path = PROJECT_ROOT) -> None:
    for rel in [
        "inputs_local/data_06_temporal_windows",
        "inputs_local/data_07_sensor_lineage",
        "inputs_local/data_08_metadata_config",
    ]:
        (project_root / rel).mkdir(parents=True, exist_ok=True)


def discover_local_inputs(project_root: Path = PROJECT_ROOT) -> dict[str, list[Path]]:
    root = project_root / "inputs_local"
    data06 = sorted((root / "data_06_temporal_windows").glob("*.csv"))
    data07 = sorted((root / "data_07_sensor_lineage").glob("*.csv"))
    config_candidates = sorted((root / "data_08_metadata_config").glob("*.json"))
    explicit_config = project_root / "configs" / "api_config.local.json"
    if explicit_config.exists():
        config_candidates.append(explicit_config)
    return {
        "data_06_templates": data06,
        "data_07_templates": data07,
        "data_08_configs": config_candidates,
    }


def validate_data_06_template(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "NO_LOCAL_INPUT_FOUND", "rows": 0, "promoted_rows": 0, "blocked_rows": 0, "errors": []}
    try:
        rows = read_csv(path)
    except Exception as exc:  # pragma: no cover
        return {"status": "BLOCKED_INVALID_TEMPLATE", "rows": 0, "promoted_rows": 0, "blocked_rows": 0, "errors": [str(exc)]}
    headers = set(rows[0].keys()) if rows else set()
    missing = sorted(DATA06_REQUIRED - headers)
    errors: list[str] = []
    if missing:
        errors.append("missing_fields:" + ",".join(missing))
    promoted = 0
    blocked = 0
    for index, row in enumerate(rows, 1):
        start = (row.get("temporal_window_start") or "").strip()
        end = (row.get("temporal_window_end") or "").strip()
        source = (row.get("temporal_window_source") or "").strip()
        source_ref = (row.get("source_ref") or "").strip()
        if not start and not end:
            blocked += 1
            errors.append(f"row_{index}:missing_temporal_window")
            continue
        start_date = _parse_iso(start)
        end_date = _parse_iso(end)
        if not start_date or not end_date or start_date > end_date:
            blocked += 1
            errors.append(f"row_{index}:invalid_temporal_window")
            continue
        if not source or not source_ref:
            blocked += 1
            errors.append(f"row_{index}:date_without_traceable_source")
            continue
        promoted += 1
    status = "PROMOTED_METADATA_READY" if promoted and not missing and not errors else "BLOCKED_INVALID_TEMPLATE"
    return {
        "status": status,
        "rows": len(rows),
        "promoted_rows": promoted,
        "blocked_rows": blocked,
        "errors": errors,
    }


def validate_data_07_template(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "NO_LOCAL_INPUT_FOUND", "rows": 0, "sentinel_2_eligible": 0, "blocked_rows": 0, "errors": []}
    try:
        rows = read_csv(path)
    except Exception as exc:  # pragma: no cover
        return {"status": "BLOCKED_INVALID_SENSOR_LINEAGE", "rows": 0, "sentinel_2_eligible": 0, "blocked_rows": 0, "errors": [str(exc)]}
    headers = set(rows[0].keys()) if rows else set()
    missing = sorted(DATA07_REQUIRED - headers)
    errors: list[str] = []
    if missing:
        errors.append("missing_fields:" + ",".join(missing))
    eligible = 0
    blocked = 0
    for index, row in enumerate(rows, 1):
        family = (row.get("sensor_family") or "UNKNOWN").strip().upper()
        source_ref = (row.get("sensor_source_ref") or "").strip()
        if family == "SENTINEL_2" and source_ref:
            eligible += 1
            continue
        blocked += 1
        if family == "SENTINEL_2" and not source_ref:
            errors.append(f"row_{index}:sentinel_2_without_sensor_source_ref")
        elif family in {"", "UNKNOWN"}:
            errors.append(f"row_{index}:unknown_sensor_lineage")
        else:
            errors.append(f"row_{index}:non_sentinel_2_or_blocked_family")
    status = "SENTINEL_2_ELIGIBLE_FOUND" if eligible and not missing and not errors else "BLOCKED_INVALID_SENSOR_LINEAGE"
    return {
        "status": status,
        "rows": len(rows),
        "sentinel_2_eligible": eligible,
        "blocked_rows": blocked,
        "errors": errors,
    }


def _safe_config_flags(config: dict[str, Any]) -> dict[str, bool]:
    return {flag: bool(config.get(flag, False)) for flag in SAFE_CONFIG_FLAGS}


def validate_data_08_config_presence(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "BLOCKED_NO_CONFIG",
            "config_present": False,
            "safe_flags": {flag: False for flag in SAFE_CONFIG_FLAGS},
            "errors": [],
        }
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "BLOCKED_BY_FLAGS",
            "config_present": True,
            "safe_flags": {flag: False for flag in SAFE_CONFIG_FLAGS},
            "errors": [f"invalid_json:{exc}"],
        }
    flags = _safe_config_flags(config)
    if flags["allow_raster_download"] or flags["allow_canary_download"]:
        status = "BLOCKED_BY_FLAGS"
    elif flags["allow_network"] and flags["allow_metadata_calls"]:
        status = "READY_METADATA_ONLY_PREFLIGHT"
    else:
        status = "BLOCKED_BY_FLAGS"
    return {"status": status, "config_present": True, "safe_flags": flags, "errors": []}


def _file_manifest(path: Path, project_root: Path) -> dict[str, Any]:
    return {
        "redacted_path": redact_local_paths(path, project_root),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def classify_local_input_readiness(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    discovered = discover_local_inputs(project_root)
    data06_path = discovered["data_06_templates"][0] if discovered["data_06_templates"] else None
    data07_path = discovered["data_07_templates"][0] if discovered["data_07_templates"] else None
    data08_path = discovered["data_08_configs"][0] if discovered["data_08_configs"] else None
    data06 = validate_data_06_template(data06_path)
    data07 = validate_data_07_template(data07_path)
    data08 = validate_data_08_config_presence(data08_path)
    manifest = {
        "data_06_templates": [_file_manifest(path, project_root) for path in discovered["data_06_templates"]],
        "data_07_templates": [_file_manifest(path, project_root) for path in discovered["data_07_templates"]],
        "data_08_configs": [_file_manifest(path, project_root) for path in discovered["data_08_configs"]],
    }
    return {
        "stage": "DATA-06/07/08 local input intake",
        "fail_closed": True,
        "data_06_status": data06["status"],
        "data_07_status": data07["status"],
        "data_08_status": data08["status"],
        "data_06": data06,
        "data_07": data07,
        "data_08": data08,
        "local_input_counts": {key: len(value) for key, value in discovered.items()},
        "public_manifest": manifest,
        "live_calls": 0,
        "downloads": 0,
        "rasters": 0,
        "crops": 0,
    }


def write_intake_manifest(readiness: dict[str, Any], out_dir: Path = OUT_DIR) -> Path:
    path = out_dir / "mv2_data_06_08_local_input_manifest.json"
    write_json(path, readiness["public_manifest"])
    return path


def write_intake_summary(readiness: dict[str, Any], out_dir: Path = OUT_DIR) -> Path:
    path = out_dir / "mv2_data_06_08_local_input_summary.json"
    write_json(path, {key: value for key, value in readiness.items() if key != "public_manifest"})
    return path


def write_intake_report(readiness: dict[str, Any], out_dir: Path = OUT_DIR) -> Path:
    path = out_dir / "mv2_data_06_08_local_input_report.md"
    write_text(
        path,
        f"""# Intake local DATA-06/07/08

## Estado
- DATA-06: {readiness['data_06_status']}
- DATA-07: {readiness['data_07_status']}
- DATA-08: {readiness['data_08_status']}
- inputs DATA-06 encontrados: {readiness['local_input_counts']['data_06_templates']}
- inputs DATA-07 encontrados: {readiness['local_input_counts']['data_07_templates']}
- configs DATA-08 encontradas: {readiness['local_input_counts']['data_08_configs']}

## Garantias
- inputs locais preenchidos nao sao copiados para outputs publicos;
- manifestos publicos contem somente caminhos redigidos, contagens, hashes, tamanhos e flags nao sensiveis;
- chamadas/downloads/rasters/crops: 0/0/0/0.
""",
    )
    return path


def write_schema(project_root: Path = PROJECT_ROOT) -> None:
    write_json(
        project_root / "datasets" / "schemas" / "schema_mv2_data_06_08_local_input_intake.json",
        {
            "schema_id": "schema_mv2_data_06_08_local_input_intake",
            "public_outputs_only": ["status", "counts", "hashes", "redacted_paths", "non_sensitive_flags"],
            "forbidden_public_outputs": ["filled_local_rows", "credentials", "tokens", "api_config_contents", "private_paths"],
            "data_06_statuses": ["NO_LOCAL_INPUT_FOUND", "BLOCKED_INVALID_TEMPLATE", "PROMOTED_METADATA_READY"],
            "data_07_statuses": ["NO_LOCAL_INPUT_FOUND", "BLOCKED_INVALID_SENSOR_LINEAGE", "SENTINEL_2_ELIGIBLE_FOUND"],
            "data_08_statuses": ["BLOCKED_NO_CONFIG", "BLOCKED_BY_FLAGS", "READY_METADATA_ONLY_PREFLIGHT"],
            "side_effects": {"live_calls": 0, "downloads": 0, "rasters": 0, "crops": 0},
        },
    )


def write_public_examples(out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "DATA_06_TEMPLATE_PREENCHIDO_EXEMPLO.csv").write_text(
        "patch_id,asset_id,temporal_window_start,temporal_window_end,temporal_window_source,source_ref,review_status\n"
        "EXEMPLO_PATCH_001,EXEMPLO_ASSET_001,2022-01-01,2022-01-03,BOLETIM_OFICIAL_FICTICIO,EXEMPLO_REF_PUBLICA_FICTICIA,APPROVED\n",
        encoding="utf-8",
    )
    (out_dir / "DATA_07_TEMPLATE_PREENCHIDO_EXEMPLO.csv").write_text(
        "patch_id,asset_id,sensor_family,sensor_source_ref,review_status\n"
        "EXEMPLO_PATCH_001,EXEMPLO_ASSET_001,SENTINEL_2,EXEMPLO_PRODUTO_S2_FICTICIO,APPROVED\n",
        encoding="utf-8",
    )
    write_text(
        out_dir / "README_PREENCHIMENTO_LOCAL_PTBR.md",
        """# Preenchimento local DATA-06/07/08

Os CSVs de exemplo desta pasta sao ficticios e publicos. Nao copie dados reais
para `outputs_public`.

## Onde colocar inputs reais
- DATA-06: `inputs_local/data_06_temporal_windows/`
- DATA-07: `inputs_local/data_07_sensor_lineage/`
- DATA-08: `inputs_local/data_08_metadata_config/` ou `configs/api_config.local.json`

`inputs_local/`, `.env`, `configs/api_config.local.json`, secrets, credenciais e
tokens permanecem gitignored. O intake publico grava somente status, contagens,
hashes, caminhos redigidos e flags nao sensiveis.
""",
    )


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser().parse_args(argv)
    ensure_local_input_dirs(PROJECT_ROOT)
    write_schema(PROJECT_ROOT)
    write_public_examples(OUT_DIR)
    readiness = classify_local_input_readiness(PROJECT_ROOT)
    write_intake_manifest(readiness, OUT_DIR)
    write_intake_summary(readiness, OUT_DIR)
    write_intake_report(readiness, OUT_DIR)
    write_text(OUT_DIR / "commands.txt", "python scripts/mv2_data_06_08_local_input_intake.py")
    print(
        "[mv2_data_06_08_local_input_intake] "
        f"DATA-06={readiness['data_06_status']} "
        f"DATA-07={readiness['data_07_status']} "
        f"DATA-08={readiness['data_08_status']} "
        "calls/downloads/rasters/crops=0/0/0/0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
