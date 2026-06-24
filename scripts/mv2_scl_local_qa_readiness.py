"""MV2 SCL local QA readiness.

Does not run QA unless an explicit local-only raster manifest exists.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_pre_unification_scl_qa"
LOCAL_MANIFEST = PROJECT_ROOT / "local_only" / "mv2_raster_manifest.csv"
CROP_PATH = PROJECT_ROOT / "outputs_public" / "mv2_pre_unification_crop_policy" / "revp_crop_authorization_candidates.csv"


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


def scl_status(has_local_raster: bool) -> str:
    return "READY" if has_local_raster else "NOT_RUN_NO_LOCAL_RASTER"


def build_rows(manifest_path: Path = LOCAL_MANIFEST) -> list[dict[str, Any]]:
    manifest = {row.get("asset_id", ""): row for row in read_csv(manifest_path)}
    crops = read_csv(CROP_PATH)
    rows: list[dict[str, Any]] = []
    for row in crops:
        local = manifest.get(row.get("asset_id", ""), {})
        has_local = bool(local.get("local_only_path"))
        rows.append(
            {
                "patch_id": row.get("patch_id", ""),
                "asset_id": row.get("asset_id", ""),
                "scl_qa_status": scl_status(has_local),
                "cloud_local_ratio": "",
                "shadow_local_ratio": "",
                "valid_local_ratio": "",
                "scl_class_histogram": "",
                "blocked_reason": "" if has_local else "BLOCKED_NO_NATIVE_RASTER",
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(LOCAL_MANIFEST))
    args = parser.parse_args(argv)
    rows = build_rows(Path(args.manifest))
    fields = ["patch_id", "asset_id", "scl_qa_status", "cloud_local_ratio", "shadow_local_ratio", "valid_local_ratio", "scl_class_histogram", "blocked_reason"]
    write_csv(OUT_DIR / "revp_scl_qa_readiness.csv", fields, rows)
    summary = {
        "targets": len(rows),
        "ready": sum(1 for row in rows if row["scl_qa_status"] == "READY"),
        "not_run_no_local_raster": sum(1 for row in rows if row["scl_qa_status"] == "NOT_RUN_NO_LOCAL_RASTER"),
        "downloads": 0,
        "rasters": 0,
        "crops": 0,
    }
    write_json(OUT_DIR / "revp_scl_qa_summary.json", summary)
    write_text(OUT_DIR / "revp_scl_qa_report.md", f"# SCL local QA readiness\n\n- ready: {summary['ready']}\n- not run no local raster: {summary['not_run_no_local_raster']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
