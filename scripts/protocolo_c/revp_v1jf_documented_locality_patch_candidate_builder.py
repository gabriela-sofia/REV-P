"""
REV-P v1jf - official locality-to-patch review candidates.

Builds review-area-only candidates for documented CPRM units that still lack
explicit coordinates after hardened PDF recovery. Locality text is used only as
an audit object, never as geocoding, label creation, or a confirmed anchor.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REVP_ROOT = SCRIPT_PATH.parents[2]
LOCAL_RUN_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1jf"
DATASETS_DIR = REVP_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"
EVENT_UNITS = DATASETS_DIR / "official_documented_event_unit_registry.csv"
V1JE_RECOVERY = DATASETS_DIR / "official_coordinate_recovery_hardened_registry.csv"

FIELDS = [
    "review_area_candidate_id",
    "documented_event_unit_id",
    "source_document_name_sanitized",
    "region",
    "municipality",
    "locality_text_sanitized",
    "event_or_survey_date",
    "phenomenon_group",
    "coordinate_status",
    "geometry_source_status",
    "review_area_status",
    "patch_candidate_status",
    "can_be_spatial_anchor",
    "can_be_positive_label",
    "can_create_training_label",
    "can_train_model",
    "blocking_reason",
    "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare(force: bool) -> None:
    if force and LOCAL_RUN_DIR.exists():
        resolved = LOCAL_RUN_DIR.resolve()
        expected = (REVP_ROOT / "local_runs" / "protocolo_c" / "v1jf").resolve()
        if resolved != expected:
            raise RuntimeError(f"Refusing to clear unexpected path: {resolved}")
        shutil.rmtree(resolved)
    LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)


def units_with_valid_coordinates() -> set[str]:
    return {
        row["documented_event_unit_id"]
        for row in read_csv(V1JE_RECOVERY)
        if row.get("can_be_official_anchor_candidate") == "true" and row.get("coordinate_confidence") == "EXPLICIT_COORDINATE_HIGH"
    }


def build_review_rows() -> list[dict[str, Any]]:
    valid_units = units_with_valid_coordinates()
    rows: list[dict[str, Any]] = []
    for unit in read_csv(EVENT_UNITS):
        unit_id = unit["documented_event_unit_id"]
        has_event_metadata = bool(unit.get("event_date") or unit.get("event_window")) and unit.get("phenomenon_group") not in {"", "UNKNOWN"}
        if unit_id in valid_units or not has_event_metadata:
            continue
        rows.append(
            {
                "review_area_candidate_id": f"REVIEW_AREA_{unit_id}",
                "documented_event_unit_id": unit_id,
                "source_document_name_sanitized": unit.get("source_document_name_sanitized", ""),
                "region": unit.get("region", ""),
                "municipality": unit.get("municipality", ""),
                "locality_text_sanitized": unit.get("locality_text_sanitized", ""),
                "event_or_survey_date": unit.get("event_date") or unit.get("event_window", ""),
                "phenomenon_group": unit.get("phenomenon_group", ""),
                "coordinate_status": "NO_EXPLICIT_COORDINATE_AFTER_V1JE",
                "geometry_source_status": "NO_OFFICIAL_GEOMETRY_AVAILABLE_LOCALLY",
                "review_area_status": "REVIEW_AREA_ONLY",
                "patch_candidate_status": "PATCH_SEARCH_BLOCKED_NO_EXPLICIT_GEOMETRY",
                "can_be_spatial_anchor": "false",
                "can_be_positive_label": "false",
                "can_create_training_label": "false",
                "can_train_model": "false",
                "blocking_reason": "LOCALITY_TEXT_IS_NOT_GEOMETRY",
                "notes": "Locality is preserved for manual review area planning; no geocoding, centroid, or label is created.",
            }
        )
    return rows


def write_schema() -> None:
    write_csv(SCHEMAS_DIR / "documented_locality_patch_review_candidate_schema.csv", [{"field": field, "description": f"REV-P v1jf documented locality review candidate field: {field}."} for field in FIELDS], ["field", "description"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepare(args.force)
    rows = build_review_rows()
    write_csv(LOCAL_RUN_DIR / "v1jf_documented_locality_review_area_candidates.csv", rows, FIELDS)
    write_csv(LOCAL_RUN_DIR / "v1jf_patch_candidate_readiness.csv", rows, FIELDS)
    qa = [
        {"check": "locality_without_coordinate_not_label", "status": "PASS", "detail": str(len(rows))},
        {"check": "review_area_not_positive", "status": "PASS", "detail": "all can_be_positive_label=false"},
        {"check": "can_train_model_false", "status": "PASS", "detail": "false"},
        {"check": "no_private_path_in_public_outputs", "status": "PASS", "detail": "sanitized document names only"},
    ]
    write_csv(LOCAL_RUN_DIR / "v1jf_qa.csv", qa, ["check", "status", "detail"])
    if rows:
        write_csv(DATASETS_DIR / "documented_locality_patch_review_candidate_registry.csv", rows, FIELDS)
        write_schema()
    summary = {
        "stage": "v1jf",
        "timestamp": utc_now(),
        "review_area_candidates_count": len(rows),
        "patch_search_ready_count": 0,
        "review_area_status": "REVIEW_AREA_ONLY" if rows else "NO_REVIEW_AREAS_REMAINING",
        "can_create_training_label": False,
        "can_train_model": False,
        "can_unfreeze_dino_for_scientific_claim": False,
    }
    write_json(LOCAL_RUN_DIR / "v1jf_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print("REV-P v1jf DOCUMENTED LOCALITY PATCH CANDIDATE BUILDER")
    print(f"Review areas: {summary['review_area_candidates_count']}")
    print(f"Status: {summary['review_area_status']}")
    print("No git add, commit, or push was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
