"""Helpers for isolated v2az synthetic turning-point tests."""

from __future__ import annotations

import csv

import scripts.v2ax_recife_geometry_intake_pack_engine as v2ax


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{col: row.get(col, "") for col in columns} for row in rows])


def patch_row(valid=False, patch_id="REC_SYNTHETIC"):
    row = {col: "" for col in v2ax.PATCH_COLUMNS}
    row.update({
        "intake_id": f"INTAKE_{patch_id}", "patch_id": patch_id, "region": "Recife",
        "city": "Recife", "priority_rank": "1", "package_count": "1",
        "required_geometry_kind": "patch_boundary_polygon", "source_type": "missing",
        "provenance_type": "unknown", "review_status": "not_started",
    })
    if valid:
        row.update({
            "source_type": "bbox", "geometry_value": "0,0,10,10", "crs": "EPSG:3857",
            "provenance_type": "manual_digitization", "provenance_note": "synthetic test source",
            "license_status": "test_license", "review_status": "approved_for_v2av",
        })
    return row


def event_row(valid=False, event_id="REC_2022_05_24_30", point=False):
    row = {col: "" for col in v2ax.EVENT_COLUMNS}
    row.update({
        "event_intake_id": f"INTAKE_{event_id}", "event_id": event_id, "region": "Recife",
        "city": "Recife", "hazard_type": "urban_flood", "linked_packages_count": "1",
        "required_geometry_kind": "observed_event_polygon", "source_type": "missing",
        "event_geometry_role": "observed_event_polygon", "provenance_type": "unknown",
        "review_status": "not_started",
    })
    if valid or point:
        row.update({
            "source_type": "wkt",
            "geometry_value": "POINT(1 2)" if point else "POLYGON((0 0,10 0,10 10,0 10,0 0))",
            "crs": "EPSG:3857", "provenance_type": "manual_digitization",
            "provenance_note": "synthetic test source", "license_status": "test_license",
            "review_status": "approved_for_v2au",
        })
    return row


def package_row(patch_id="REC_SYNTHETIC", event_id="REC_2022_05_24_30"):
    return {
        "package_id": "PKG_SYNTHETIC", "event_id": event_id, "patch_id": patch_id,
        "region": "Recife", "city": "Recife", "hazard_type": "urban_flood",
        "evidence_score": "0.9", "allowed_use": "candidate_reference",
        "promotion_decision": "C3_CANDIDATE_REFERENCE_HOLD_FOR_OVERLAY",
    }


def make_dataset(tmp_path, *, valid_patch=False, valid_event=False, event_point=False,
                 package_patch="REC_SYNTHETIC", package_event="REC_2022_05_24_30"):
    dataset = tmp_path / "datasets"
    output = tmp_path / "outputs"
    config = tmp_path / "configs"
    docs = tmp_path / "docs"
    manual = dataset / "manual_intake" / "recife_p1"
    write_csv(manual / "recife_p1_patch_geometry_intake.csv", v2ax.PATCH_COLUMNS,
              [patch_row(valid_patch)])
    write_csv(manual / "recife_p1_event_geometry_intake.csv", v2ax.EVENT_COLUMNS,
              [event_row(valid_event, point=event_point)])
    package_cols = ["package_id", "event_id", "patch_id", "region", "city", "hazard_type",
                    "evidence_score", "allowed_use", "promotion_decision"]
    write_csv(dataset / "v2at_event_patch_package_registry.csv", package_cols,
              [package_row(package_patch, package_event)])
    return dataset, output, config, docs
