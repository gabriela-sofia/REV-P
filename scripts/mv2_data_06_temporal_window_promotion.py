"""MV2-DATA-06 temporal window promotion.

Consumes a private filled temporal-window template only when it exists. Without
that input, DATA-06 remains blocked and no metadata probe is opened.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_temporal_window_promotion"
SEED_PATH = PROJECT_ROOT / "outputs_public" / "mv2_pre_unification_seed" / "revp_temporal_window_seed_10.csv"
CORRECTION_PATH = PROJECT_ROOT / "outputs_public" / "mv2_data_temporal_window_intake" / "mv2_data_05_temporal_window_correction_template.csv"
FILLED_CANDIDATES = [
    PROJECT_ROOT / "local_only" / "mv2_data_temporal_window" / "mv2_data_06_temporal_window_filled.csv",
    PROJECT_ROOT / "data_local" / "mv2_data_temporal_window" / "mv2_data_06_temporal_window_filled.csv",
    PROJECT_ROOT / "private_outputs" / "mv2_data_temporal_window" / "mv2_data_06_temporal_window_filled.csv",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def find_filled_template(explicit_path: str | None = None) -> Path | None:
    candidates = [Path(explicit_path)] if explicit_path else FILLED_CANDIDATES
    for path in candidates:
        if path.exists():
            return path
    return None


def _parse_iso(value: str) -> date | None:
    try:
        return date.fromisoformat((value or "").strip())
    except ValueError:
        return None


def classify_temporal_row(row: dict[str, str], template_found: bool) -> tuple[str, str]:
    if not template_found:
        return "NO_FILLED_TEMPLATE_FOUND", "BLOCKED_NO_FILLED_TEMPLATE"
    start = (row.get("temporal_window_start") or "").strip()
    end = (row.get("temporal_window_end") or "").strip()
    source = (row.get("temporal_window_source") or "").strip()
    source_ref = (row.get("source_ref") or "").strip()
    review = (row.get("review_status") or "").strip().upper()
    if not start or not end:
        return "BLOCKED_NO_TEMPORAL_WINDOW", "janela_temporal_ausente"
    start_date = _parse_iso(start)
    end_date = _parse_iso(end)
    if not start_date or not end_date or start_date > end_date:
        return "INVALID_TEMPORAL_WINDOW", "janela_temporal_invalida"
    if not source or not source_ref:
        return "BLOCKED_NO_SOURCE", "fonte_rastreavel_ausente"
    if review not in {"APPROVED", "REVIEWED", "CONFIRMED", "APROVADO", "REVISADO"}:
        return "BLOCKED_WEAK", "revisao_humana_ausente_ou_fraca"
    return "PROMOTED_METADATA_READY", ""


def build_rows(template_path: Path | None) -> list[dict[str, Any]]:
    seed_rows = read_csv(SEED_PATH)
    filled_rows = read_csv(template_path) if template_path is not None else []
    filled_by_key = {
        (row.get("patch_id", ""), row.get("asset_id", "")): row
        for row in filled_rows
    }
    rows: list[dict[str, Any]] = []
    for idx, seed in enumerate(seed_rows, 1):
        key = (seed.get("patch_id", ""), seed.get("asset_id", ""))
        source = filled_by_key.get(key, seed)
        status, reason = classify_temporal_row(source, template_path is not None)
        rows.append(
            {
                "promotion_id": f"MV2_DATA_06_PROM_{idx:03d}",
                "patch_id": seed.get("patch_id", ""),
                "asset_id": seed.get("asset_id", ""),
                "temporal_window_start": source.get("temporal_window_start", ""),
                "temporal_window_end": source.get("temporal_window_end", ""),
                "temporal_window_source": source.get("temporal_window_source", ""),
                "source_ref": source.get("source_ref", ""),
                "review_status": source.get("review_status", "PENDING"),
                "promotion_status": status,
                "blocked_reason": reason,
                "can_probe_metadata": str(status == "PROMOTED_METADATA_READY").lower(),
            }
        )
    return rows


def write_schema() -> None:
    write_json(
        PROJECT_ROOT / "datasets" / "schemas" / "schema_mv2_data_06_temporal_window_promotion.json",
        {
            "schema_id": "schema_mv2_data_06_temporal_window_promotion",
            "required_fields": [
                "patch_id",
                "asset_id",
                "temporal_window_start",
                "temporal_window_end",
                "temporal_window_source",
                "source_ref",
                "review_status",
            ],
            "allowed_statuses": [
                "NO_FILLED_TEMPLATE_FOUND",
                "BLOCKED_NO_TEMPORAL_WINDOW",
                "BLOCKED_NO_SOURCE",
                "INVALID_TEMPORAL_WINDOW",
                "BLOCKED_WEAK",
                "PROMOTED_METADATA_READY",
            ],
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filled-template")
    args = parser.parse_args(argv)
    template = find_filled_template(args.filled_template)
    rows = build_rows(template)
    fields = [
        "promotion_id",
        "patch_id",
        "asset_id",
        "temporal_window_start",
        "temporal_window_end",
        "temporal_window_source",
        "source_ref",
        "review_status",
        "promotion_status",
        "blocked_reason",
        "can_probe_metadata",
    ]
    write_schema()
    write_csv(OUT_DIR / "mv2_data_06_temporal_window_promotion.csv", fields, rows)
    write_csv(
        OUT_DIR / "mv2_data_06_probe_ready_batch.csv",
        fields,
        [row for row in rows if row["promotion_status"] == "PROMOTED_METADATA_READY"],
    )
    write_csv(OUT_DIR / "mv2_data_06_correction_template.csv", fields, rows)
    summary = {
        "stage": "DATA-06",
        "filled_template_found": template is not None,
        "total_targets": len(rows),
        "promoted_metadata_ready": sum(1 for row in rows if row["promotion_status"] == "PROMOTED_METADATA_READY"),
        "blocked_no_filled_template": sum(1 for row in rows if row["promotion_status"] == "NO_FILLED_TEMPLATE_FOUND"),
        "api_calls": 0,
        "downloads": 0,
        "rasters": 0,
        "crops": 0,
    }
    write_json(OUT_DIR / "mv2_data_06_summary.json", summary)
    write_text(
        OUT_DIR / "mv2_data_06_report.md",
        f"# DATA-06 temporal window promotion\n\n- filled template found: {summary['filled_template_found']}\n- promoted metadata ready: {summary['promoted_metadata_ready']}\n- blocked no filled template: {summary['blocked_no_filled_template']}\n- calls/downloads/rasters/crops: 0/0/0/0",
    )
    write_text(OUT_DIR / "commands.txt", "python scripts/mv2_data_06_temporal_window_promotion.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
